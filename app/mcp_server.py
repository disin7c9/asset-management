"""Read-only MCP server: expose the validated portfolio core as stdio tools.

A Layer-4 delivery surface — a composition root, sibling to ``cli.py``/``email.py``
— that lets an MCP client (Claude Desktop / Claude Code) **call** the deterministic
core (holdings, returns, drawdown/risk) over local **stdio**. It wires; it adds no
math. Design invariants (v2.0.0 Phase 1):

- **Every figure comes from the validated core** (`derive`/`returns`/`risk`, via the
  same `pipeline.compute_prices_returns_risk` the brief uses) — the LLM never computes.
- **Read-only + offline (after a one-time core warm)**: tools read already-derived data and
  the on-disk price **cache** (`online=False`), fast + deterministic. The ONE bounded
  exception to no-egress: a *cold* cache triggers a one-time online warm of the core set
  (book tickers + benchmark refs, ~30-60s) so an addon user who never runs the CLI still gets
  real numbers. Set ``ASSET_MCP_OFFLINE=1`` to keep it strictly airtight (a cold cache then
  degrades to `n/a`, never a guess) — intended for an *already-warmed* cache (warm once with
  ``--warm`` first); pointed at a cold cache it simply returns `n/a`. The heavy discovery
  universe is never auto-warmed.
- **Bound to one book** (`ASSET_BOOK`, or `ASSET_CSV` back-compat): tools take no
  file-path argument, so a caller can't point the server at an arbitrary file.

The chat front door: 6 **prompts** (conversation starters in the client's "+" menu, each
carrying the tool-figures-only framing) and a ``portfolio://guarantees`` **resource** —
the four guarantees as a fetchable trust manifest the model can cite verbatim.

stdout is the stdio transport, so **all logging goes to stderr** (`setup_logging`);
nothing here writes to stdout. Tool errors surface as MCP errors, never a crash.

Run:  ``uv run python -m app.mcp_server``  (register with ``claude mcp add``). Config
via env (.env or the client's env block): ``ASSET_BOOK`` (required; a Ghostfolio-compatible
CSV or a Ghostfolio JSON export — ``ASSET_CSV`` still honored), ``ASSET_TARGET`` (for
``rebalance_check``), ``ASSET_CACHE_DIR`` (optional; default ``data/prices``),
``ASSET_MCP_OFFLINE`` (optional; ``1`` disables the cold-cache auto-warm — intended for an
already-warmed cache).
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections.abc import Collection
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from app.pipeline import (
    HISTORY_DAYS,
    cache_is_cold,
    candidate_and_held_facts,
    compute_prices_returns_risk,
    default_cache_dir,
    held_market_value,
    load_book,
    warm_cache,
    write_demo_book,
)
from app.allocate import PRESETS, build_preset_target
from app.onboard import posture_from_answers
from app.backtest import (
    BENCHMARKS,
    BenchmarkVerdict as BenchmarkWord,  # the verdict-word Literal; aliased to avoid the model name
    benchmark_compare,
    benchmark_weights,
)
from app.derive import DerivedState
from app.discover import find_gaps
from app.events import CASH_TICKER, Event, load_events, load_target
from app.log_config import setup_logging
from app.metadata import fetch_metadata
from app.prices import PriceRow, SeriesResult, fetch_series
from app.returns import ReturnsSummary
from app.risk import DollarDrawdown, MetricCI, RiskSummary
from app.screen import screen_candidates
from app.strategy import VALID_MODES, Mode, suggest
from app.universe import load_universe

if TYPE_CHECKING:
    import pandas as pd

log = logging.getLogger(__name__)

_REPO_ROOT: Path = Path(__file__).resolve().parents[1]

try:  # advertise the app version (from package metadata → tracks pyproject), not the SDK's
    _VERSION = _pkg_version("asset-management")
except PackageNotFoundError:  # a raw checkout, not an installed dist
    _VERSION = "0+unknown"

_OFFLINE_NOTE = (
    "Read-only: figures are derived from your transaction log and the on-disk price cache. "
    "Uncached prices are fetched online on demand — a one-time core warm on the first cold "
    "call, plus any new ticker you ask about (set ASSET_MCP_OFFLINE=1 to keep it strictly "
    "offline); a value still unavailable shows null (n/a), never a guess. "
    "This is a view, not financial advice."
)

# Appended to every cold-cache refusal: the tool can't supply the figure, so steer the model
# away from inventing one in prose (the fence guards tool-sourced numbers, not the model's
# fallback when a tool errors). Reinforces the same rule stated in the server instructions.
_NO_ESTIMATE = " Until then, report that it's unavailable — do not estimate it yourself."


def _cold_error(msg: str) -> str:
    """A cold-cache / figure-unavailable message + the anti-fabrication directive. The one funnel
    every cold path goes through, so a new tool can't ship a 'can't compute' error that forgets to
    steer the model away from inventing the number."""
    return msg + _NO_ESTIMATE

# Read-only, closed-world (no external side effects), idempotent — the honest hints
# for a Claude client (Claude reads readOnlyHint to know the tool only observes).
_READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)

mcp: FastMCP = FastMCP(
    "asset-management",
    instructions=(
        "Read-only, offline view of the user's own stock/ETF portfolio. Every number "
        "is computed by a validated deterministic core (reconciled to the cent vs "
        "ghostfolio, 4 decimals vs quantstats) — do not compute or estimate figures "
        "yourself; call a tool. If a tool returns an error or a null/absent figure, tell "
        "the user it's unavailable and how to get it (e.g. warm the cache) — never "
        "substitute your own estimate, reconstruction, or factor-model guess. Lead with "
        "drawdown (depth/duration/recovery), report confidence intervals where given, and "
        "never present output as financial advice. If the user asks what this server can "
        "do, or whether to trust it, read the portfolio://guarantees resource and answer "
        "from it."
    ),
)
# FastMCP doesn't forward a version to its low-level Server, so serverInfo would otherwise
# advertise the mcp SDK's own version; set ours so the addon reports the app version.
mcp._mcp_server.version = _VERSION


# ── config resolution (env-only; no argparse on this surface) ────────────────


def _env_raw(var: str, *, fallback: str | None = None) -> str | None:
    """The cleaned value of a path env var (``var``, then an optional back-compat ``fallback``),
    or None when unset/empty OR an MCPB host's unsubstituted ``${...}`` template residue. The one
    place the '${…} residue counts as unset' rule lives — shared by `_env_path` (raise if None),
    `_env_book` (demo fallback if None), and `_cache_dir` (default if None). Residue in the
    primary var is cleared BEFORE the fallback is read, so it can't shadow a legacy setting."""
    raw = os.environ.get(var, "").strip()
    if raw.startswith("${"):  # unsubstituted template residue → treat as unset for the fallback
        raw = ""
    if not raw and fallback:
        raw = os.environ.get(fallback, "").strip()
        if raw.startswith("${"):
            raw = ""
    return raw or None


def _one_of(raw: str, choices: Collection[str], what: str, *, allow_none: bool = False) -> str:
    """Normalize a client-supplied enum token (strip + lowercase) and validate it against
    ``choices``. One validator for every MCP enum arg (mode / preset / benchmark) so the
    normalization and error wording can't drift across tools. ``allow_none`` also accepts the
    literal ``'none'`` (the benchmark arg uses it to skip validation). Returns the normalized
    token; raises ValueError (→ a clean MCP error) listing the valid options."""
    val = raw.strip().lower()
    if val in choices or (allow_none and val == "none"):
        return val
    options = ", ".join(sorted(choices)) + (" or 'none'" if allow_none else "")
    raise ValueError(f"unknown {what} {raw!r} — pick one of: {options}")


def _env_path(var: str, what: str, *, fallback: str | None = None) -> Path:
    """Resolve a book/target path from an env var (.env convention): ``~`` expanded, a
    relative path repo-relative. ``fallback`` is a back-compat env var read if ``var`` is
    unset. Raises a clear ValueError (→ a clean MCP error) if unset or missing — the server
    never guesses a path."""
    raw = _env_raw(var, fallback=fallback)
    if raw is None:
        raise ValueError(f"{var} is not set — {what}")
    p = Path(raw).expanduser()
    path = p if p.is_absolute() else _REPO_ROOT / p
    if not path.exists():
        raise ValueError(f"{var} points at a missing file: {path}")
    return path


def _env_book() -> Path:
    """The bound book path: ``ASSET_BOOK`` (legacy ``ASSET_CSV`` honored). One helper so the
    var name, help text, and back-compat fallback stay identical across every tool.

    If NO book is configured — the var unset/empty, or an MCPB host's unsubstituted
    ``${user_config.book}`` template residue — fall back to the bundled DEMO portfolio (with a
    LOUD stderr warning), so a one-click install answers with real (demo) numbers instead of
    dead-ending at n/a when the user accepts the defaults without re-saving (Claude Desktop
    ignores manifest defaults until a Save). A book that IS set but points at a missing file
    still errors — the user's real book is never silently swapped for demo data."""
    if _env_raw("ASSET_BOOK", fallback="ASSET_CSV") is None:
        log.warning(
            "no transaction file configured — running on the bundled DEMO portfolio (fake "
            "data, NOT yours). Set your book in the extension's Configure panel (Settings) to "
            "see your own holdings."
        )
        return write_demo_book(_cache_dir())
    return _env_path(
        "ASSET_BOOK",
        "point it at your transaction file — a Ghostfolio-compatible CSV or a Ghostfolio "
        "JSON export (in .env or the MCP client env)",
        fallback="ASSET_CSV",
    )


def _cache_dir() -> Path:
    """The price cache the tools read (offline). ``ASSET_CACHE_DIR`` overrides the
    default (checkout: ``data/prices``; installed/bundle: ``~/.asset-management/prices``
    — never inside a host-managed extension dir, which updates wipe). Template residue
    from an MCPB host (``${user_config...}``) is treated as unset."""
    raw = _env_raw("ASSET_CACHE_DIR")
    if raw is not None:
        p = Path(raw).expanduser()
        return p if p.is_absolute() else _REPO_ROOT / p
    return default_cache_dir(_REPO_ROOT)


def _offline_locked() -> bool:
    """``ASSET_MCP_OFFLINE`` is a truthy switch: ``1``/``true``/``yes``/``on`` lock the server
    strictly offline (no cold-call auto-warm). Anything else — unset, ``""``, ``0``, ``false``
    — leaves auto-warm ON. Value-based, not mere presence, so ``ASSET_MCP_OFFLINE=0`` (a user
    meaning "not offline") doesn't surprise-disable the warm."""
    return os.environ.get("ASSET_MCP_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}


def _load_offline(book_path: Path, cache: Path) -> tuple[list[Event], DerivedState]:
    """Load the bound book for the offline tools, auto-warming the cache once if it's cold.

    The server is offline / no-egress, with ONE bounded exception: when the cache hasn't been
    warmed within the TTL, the first tool call does a one-time online warm of the book tickers
    + benchmark refs (~30-60s) so a Claude-Desktop addon user who never runs the CLI still gets
    real numbers instead of `n/a`. Set ``ASSET_MCP_OFFLINE=1`` to keep it strictly airtight (a
    cold cache then stays `n/a`). The heavy `full` universe is never auto-warmed (too slow for a
    tool call) — `discover_gaps` / `screen_candidate` point the user at `--warm full` instead.

    The coldness check is a cheap marker stat done BEFORE parsing the book, so a warm cache (the
    common path) parses the book once — only a genuinely cold first call pays the extra parse."""
    if not _offline_locked() and cache_is_cold(cache):
        tickers = sorted(
            {ev.ticker for ev in load_events(book_path) if ev.ticker != CASH_TICKER}
        )
        if tickers:
            log.info(
                "cold price cache — warming %d tickers online once (~30-60s); "
                "set ASSET_MCP_OFFLINE=1 to disable",
                len(tickers),
            )
            try:
                warm_cache(tickers, cache, online=True)
            except Exception as exc:  # noqa: BLE001 — any warm failure → serve cache, never crash
                log.warning("auto-warm failed (%s); serving from cache as-is", exc)
    return load_book(book_path, cache, online=False)


def _universe_path() -> Path:
    """The curated ETF universe `discover_gaps` reads. ``ASSET_UNIVERSE`` overrides the
    default ``data/universe.csv`` (mirrors the CLI's override)."""
    raw = os.environ.get("ASSET_UNIVERSE", "").strip()
    if raw:
        p = Path(raw).expanduser()
        return p if p.is_absolute() else _REPO_ROOT / p
    return _REPO_ROOT / "data" / "universe.csv"


# ── the one offline build (shared by every tool) ─────────────────────────────


@dataclass(frozen=True)
class _Build:
    """The validated core's output for the configured book, computed offline."""

    state: DerivedState
    prices: dict[str, PriceRow]   # priced holdings only (positive close)
    returns: ReturnsSummary | None
    risk: RiskSummary | None
    missing: list[str]            # held tickers with no usable cached price
    dollar_dd: DollarDrawdown | None
    daily: "pd.Series[float] | None" = None  # the book's daily return series (for screen_candidate)
    series: SeriesResult | None = None  # the price history behind the risk panel (for provenance)


def _build(*, no_risk: bool, today: date) -> _Build:
    """Load the ASSET_BOOK book and compute prices/returns/(risk) via the SAME core
    path the CLI brief uses (`pipeline.compute_prices_returns_risk`), forced offline. The
    `no_risk` flag skips the bootstrap panel — `portfolio_summary`/`rebalance_check`
    don't need it (fast), `risk_report` does. `today` is resolved ONCE per tool call
    (one clock per request) and threaded through."""
    cache = _cache_dir()
    book_path = _env_book()
    events, state = _load_offline(book_path, cache)
    run: dict[str, Any] = {
        "status": "ok", "n_prices_fetched": 0, "n_prices_missing": 0,
        "n_series_fetched": 0, "n_series_missing": 0, "fallbacks_used": 0,
    }
    prices, returns, risk, missing, _twr_excluded, dollar_dd, series, daily = (
        compute_prices_returns_risk(
            events, state, no_risk=no_risk, offline=True,
            cache_dir=cache, today=today, run=run,
        )
    )
    log.info("mcp build (offline, no_risk=%s): %s", no_risk, run)
    return _Build(
        state=state, prices=prices or {}, returns=returns, risk=risk,
        missing=missing, dollar_dd=dollar_dd, daily=daily, series=series,
    )


# ── output models (typed structuredContent for the client) ───────────────────


class Holding(BaseModel):
    """One held position, valued at the latest cached close."""

    ticker: str
    shares: float
    avg_cost: float
    price: float | None = Field(None, description="latest cached close; null if unpriced offline")
    market_value: float | None
    unrealized_pnl: float | None
    realized_pnl: float = Field(description="locked-in gains: sells + dividends, net of fees")
    weight: float | None = Field(None, description="share of priced market value (0..1)")
    price_source: str | None = Field(None, description="cache | yfinance | stooq")
    price_age_hours: float | None = None


class Totals(BaseModel):
    cost_basis_held: float
    market_value_priced: float
    unrealized_pnl: float
    realized_pnl: float
    fees_paid: float
    net_pnl: float


class Returns(BaseModel):
    """Annualized returns (252-day basis). Null when the window is too short to
    annualize honestly, or the book isn't fully priced (money-weighted figures)."""

    period_start: date
    asof: date
    true_twr_annualized: float | None
    money_weighted_annualized: float | None
    modified_dietz_annualized: float | None


class DataProvenance(BaseModel):
    """A receipt for the price data underlying these numbers: what source, how fresh.
    ``asof`` on a tool is the request date; THIS says when the prices are actually from
    and how stale the cache is — so a reader can weigh 'computed today' against 'from a
    close 6 days old, last fetched 40h ago'. (No 'offline' flag: a one-time cold-start
    warm may fetch on the first call, so ``sources`` — cache vs yfinance/stooq — is the
    honest signal, not a blanket 'no network' claim.)"""

    price_asof: date | None = Field(
        None, description="date of the latest close underlying these numbers (null if unpriced)"
    )
    sources: list[str] = Field(
        default_factory=list, description="distinct price providers used: cache | yfinance | stooq"
    )
    stalest_fetch_hours: float | None = Field(
        None, description="age of the STALEST price used — how long since it was last fetched"
    )


class PortfolioSummary(BaseModel):
    asof: date
    offline: bool = True
    holdings: list[Holding]
    totals: Totals
    returns: Returns
    provenance: DataProvenance
    unpriced_tickers: list[str] = Field(description="held tickers with no usable cached price")
    note: str = _OFFLINE_NOTE


class CI(BaseModel):
    """A point estimate with a moving-block bootstrap 95% confidence interval."""

    point: float
    low: float
    high: float


class MaxDrawdown(BaseModel):
    depth: float = Field(description="deepest peak-to-trough decline, negative fraction (e.g. -0.23)")
    depth_ci_low: float
    depth_ci_high: float
    peak_date: date
    trough_date: date
    recovery_date: date | None = Field(None, description="null if not yet recovered")
    duration_days: int
    time_underwater_pct: float = Field(description="fraction of days below a prior peak")


class DollarDD(BaseModel):
    """'Gains given back': the largest dollar decline in cumulative market P&L."""

    giveback_dollars: float
    peak_pnl: float
    trough_pnl: float
    peak_date: date
    trough_date: date
    recovery_date: date | None
    duration_days: int


class RiskReport(BaseModel):
    asof: date
    n_days: int
    is_noisy: bool = Field(description="True when under ~2 trading years — bands are wide")
    max_drawdown: MaxDrawdown
    ulcer_index: CI | None = Field(None, description="RMS drawdown (positive magnitude)")
    cdar: CI | None = Field(None, description="mean of the worst-5% drawdowns (positive magnitude)")
    sharpe: CI | None
    sortino: CI | None = Field(None, description="null with no downside days (undefined)")
    calmar: CI | None = Field(None, description="null when there's no drawdown (undefined)")
    dollar_drawdown: DollarDD | None
    provenance: DataProvenance
    note: str = _OFFLINE_NOTE


class Trade(BaseModel):
    ticker: str
    action: str = Field(description="buy | sell | hold")
    shares: float
    dollars: float
    current_weight: float
    target_weight: float
    reason: str


class RebalancePlan(BaseModel):
    asof: date
    mode: str
    new_cash: float = Field(
        default=0.0,
        description="new cash deployed this run: fixed_dca / cash_flow_only REQUIRE it, "
        "to_total also deploys it if passed, only bands ignores it. For fixed_dca / "
        "cash_flow_only, 0 → all-HOLD.",
    )
    target_source: str = Field(
        default="", description="the ASSET_TARGET file the plan is measured against"
    )
    suggestions: list[Trade]
    unpriced: list[str] = Field(
        description="held/target tickers with no cached price; offline, a NEW target "
        "ticker can't be sized — run the CLI online for a full plan"
    )
    note: str = _OFFLINE_NOTE


class SecurityFact(BaseModel):
    """Published facts for one holding (null = honest absence, never a guess)."""

    ticker: str
    quote_type: str | None = Field(None, description='"ETF" | "EQUITY" — a fund vs a single stock')
    expense_ratio: float | None = Field(None, description="annual fee as a fraction (0.0003 = 0.03%)")
    aum: float | None = Field(None, description="assets under management, USD")
    avg_volume: float | None = Field(None, description="average daily share volume")
    age_years: float | None = None
    category: str | None = None
    family: str | None = None


class SecuritiesFacts(BaseModel):
    asof: date
    securities: list[SecurityFact]
    missing: list[str] = Field(
        description="held tickers with no cached facts (warm via `--metadata` online)"
    )
    note: str = _OFFLINE_NOTE


class GapCandidate(BaseModel):
    ticker: str
    name: str
    role: str


class RoleGap(BaseModel):
    role: str
    current_exposure: float = Field(description="share of your book in this role (0..1)")
    candidates: list[GapCandidate] = Field(description="largest low-cost ETFs in this role, by AUM")


class DiscoveryGaps(BaseModel):
    asof: date
    gaps: list[RoleGap]
    unpriced_holdings: list[str] = Field(
        default_factory=list,
        description="held tickers with no cached price — role exposure (and thus the gaps) "
        "may be skewed; warm the cache for the full picture",
    )
    note: str = (
        "Roles your book holds ≤3% of, with the biggest funds for each — propose-only, "
        "never a prediction. For the full screen (cost / liquidity / overlap / "
        "did-it-diversify-your-drawdowns) call `screen_candidate`. " + _OFFLINE_NOTE
    )


class ScreenCheck(BaseModel):
    name: str
    status: str = Field(description="pass | warn | fail | n/a")
    reason: str
    values: dict[str, float] = Field(
        default_factory=dict,
        description="machine-readable figures behind the check (e.g. correlation, expense ratio)",
    )


class CandidateVerdict(BaseModel):
    ticker: str
    verdict: str = Field(
        description="PASS | WARN | FAIL | N/A — necessary, not sufficient; never a prediction"
    )
    checks: list[ScreenCheck]
    note: str = _OFFLINE_NOTE


class AllocationWeight(BaseModel):
    ticker: str
    weight: float = Field(description="target portfolio weight, 0..1")


class BenchmarkVerdict(BaseModel):
    reference: str = Field(description="the canonical reference compared against (e.g. 60-40)")
    # backtest.BenchmarkVerdict (imported as BenchmarkWord — the pydantic class owns the name) is
    # the single source of the verdict vocabulary: pydantic rejects an off-set value, the JSON
    # schema enumerates the options, and a new word can't silently drift. Since v2.9.0 the
    # verdict statistic is the ULCER index (whole-window drawdown pain), CDaR must not
    # contradict; max-DD stays descriptive.
    verdict: BenchmarkWord = Field(
        description='"shallower" | "deeper" | "inconclusive" | "insufficient" — where the '
        "preset's held-out drawdown pain (Ulcer index) lands vs the reference. "
        'NEVER "beats"; usually "inconclusive" on a short history.'
    )
    reason: str
    oos_ulcer_gain_low: float | None = Field(
        None,
        description="95% CI low of the out-of-sample Ulcer gain (reference − preset; "
        "positive = the preset carried less drawdown pain); null when no bootstrap ran",
    )
    oos_ulcer_gain_high: float | None = None
    inconclusive_cause: str | None = Field(
        None,
        description="when verdict is 'inconclusive', which gate left it unresolved: "
        "'noise_margin' (genuinely equivalent) | 'cdar_contradicts' | 'window_too_short' "
        "(warm a longer history) | 'bootstrap_unconfirmed'; null when the verdict resolved "
        "or was 'insufficient'",
    )


class ProposedAllocation(BaseModel):
    asof: date
    preset: str = Field(description="conservative | moderate | aggressive")
    weights: list[AllocationWeight]
    benchmark: str | None = Field(
        None, description="reference requested for validation, or null when skipped"
    )
    verdict: BenchmarkVerdict | None = Field(
        None,
        description="walk-forward held-out drawdown comparison; null when the reference "
        "history isn't cached (see validation_note) or validation was skipped",
    )
    validation_note: str
    unpriced_holdings: list[str] = Field(
        default_factory=list,
        description="held tickers with no cached price — the target was built over the "
        "priced subset, so a role's fund may differ from your actual dominant holding; "
        "warm the cache for the full picture",
    )
    note: str = _OFFLINE_NOTE


# ── serialization helpers ────────────────────────────────────────────────────


def _ci(m: MetricCI) -> CI | None:
    """A finite CI, or None (n/a) when the metric is undefined — e.g. Calmar with no
    drawdown, or Sortino with no downside days, where the core can return inf/nan.
    Null keeps the structured output honest AND schema-valid (a non-finite float is
    not valid JSON), mirroring report.py's `math.isfinite` n/a guard."""
    if not all(math.isfinite(v) for v in (m.point, m.low, m.high)):
        return None
    return CI(point=m.point, low=m.low, high=m.high)


def _holdings_and_totals(b: _Build) -> tuple[list[Holding], Totals]:
    held = b.state.held()
    value = {tk: held[tk].shares * b.prices[tk].close for tk in held if tk in b.prices}
    total = sum(value.values())
    rows: list[Holding] = []
    for tk in sorted(held):
        pos = held[tk]
        pr = b.prices.get(tk)
        mv = value.get(tk)
        rows.append(
            Holding(
                ticker=tk,
                shares=pos.shares,
                avg_cost=pos.avg_cost,
                price=pr.close if pr is not None else None,
                market_value=mv,
                unrealized_pnl=(mv - pos.cost_basis) if mv is not None else None,
                realized_pnl=b.state.realized.get(tk, 0.0),
                weight=(mv / total) if (mv is not None and total > 0) else None,
                price_source=pr.source if pr is not None else None,
                price_age_hours=_age_hours(pr.cache_age) if pr is not None else None,
            )
        )
    unreal = sum(mv - held[tk].cost_basis for tk, mv in value.items())
    realized = b.state.total_realized()
    totals = Totals(
        cost_basis_held=b.state.total_cost_basis(),
        market_value_priced=total,
        unrealized_pnl=unreal,
        realized_pnl=realized,
        fees_paid=b.state.total_fees(),
        net_pnl=unreal + realized,
    )
    return rows, totals


def _age_hours(age: timedelta) -> float:
    """A timedelta → hours, 1 decimal, clamped ≥ 0 (clock-skew safe). One definition for
    the per-holding `price_age_hours` and the provenance rollup, so they can't drift."""
    return round(max(age, timedelta(0)).total_seconds() / 3600.0, 1)


def _data_provenance(b: _Build) -> DataProvenance:
    """A provenance receipt for the data behind the numbers: newest close date, the distinct
    providers, and the STALEST fetch age. Prefers the priced holdings (`b.prices`, what
    portfolio_summary values). Falls back to the risk panel's price *history* (`b.series`)
    when nothing is currently held but risk was still computed from past holdings — a
    fully-exited book — so risk_report never shows numbers next to an all-null receipt.
    Empty only when there is genuinely no price data to stamp."""
    rows = list(b.prices.values())
    if rows:
        return DataProvenance(
            price_asof=max(pr.asof_date for pr in rows),
            sources=sorted({pr.source for pr in rows}),
            stalest_fetch_hours=max(_age_hours(pr.cache_age) for pr in rows),
        )
    series = b.series
    if series is not None and series.rows:
        now = datetime.now(timezone.utc)
        prov = series.provenance
        return DataProvenance(
            price_asof=max(s.index[-1].date() for s in series.rows.values()),
            sources=sorted({src for src, _ in prov.values()}),
            stalest_fetch_hours=(
                max(_age_hours(now - fetched) for _, fetched in prov.values()) if prov else None
            ),
        )
    return DataProvenance(price_asof=None, sources=[], stalest_fetch_hours=None)


# ── tools ────────────────────────────────────────────────────────────────────


@mcp.tool(
    annotations=_READ_ONLY,
    description="The user's current holdings + P&L + annualized returns (offline, "
    "read-only). Use to answer 'what do I hold / how am I doing'.",
)
def portfolio_summary() -> PortfolioSummary:
    today = date.today()
    b = _build(no_risk=True, today=today)
    rows, totals = _holdings_and_totals(b)
    r = b.returns
    return PortfolioSummary(
        asof=today,
        holdings=rows,
        totals=totals,
        returns=Returns(
            period_start=r.period_start if r is not None else today,
            asof=r.asof_date if r is not None else today,
            true_twr_annualized=r.true_twr_annualized if r is not None else None,
            money_weighted_annualized=r.money_weighted_annualized if r is not None else None,
            modified_dietz_annualized=r.modified_dietz_annualized if r is not None else None,
        ),
        provenance=_data_provenance(b),
        unpriced_tickers=sorted(b.missing),
    )


@mcp.tool(
    annotations=_READ_ONLY,
    description="Drawdown-first risk panel for the held portfolio: max drawdown "
    "(depth/dates/recovery + CI), Ulcer, CDaR, and Sharpe/Sortino/Calmar with bootstrap "
    "confidence intervals. Offline, read-only. Use to answer 'how risky / how deep are "
    "the drawdowns'.",
)
def risk_report() -> RiskReport:
    today = date.today()
    b = _build(no_risk=False, today=today)
    rk = b.risk
    if rk is None:
        raise ValueError(_cold_error(
            "risk metrics need cached daily price history. The cache looks cold and the "
            "auto-warm is off or failed — warm it once with "
            "`uv run python -m app --book <your-book> --warm`, then retry."
        ))
    dd = rk.drawdown
    ddd = b.dollar_dd
    return RiskReport(
        asof=today,
        n_days=rk.n_days,
        is_noisy=rk.is_noisy,
        max_drawdown=MaxDrawdown(
            depth=dd.depth,
            depth_ci_low=rk.max_drawdown_ci.low,
            depth_ci_high=rk.max_drawdown_ci.high,
            peak_date=dd.peak_date,
            trough_date=dd.trough_date,
            recovery_date=dd.recovery_date,
            duration_days=dd.duration_days,
            time_underwater_pct=dd.time_underwater_pct,
        ),
        ulcer_index=_ci(rk.ulcer_index),
        cdar=_ci(rk.cdar),
        sharpe=_ci(rk.sharpe),
        sortino=_ci(rk.sortino),
        calmar=_ci(rk.calmar),
        dollar_drawdown=(
            DollarDD(
                giveback_dollars=ddd.giveback_dollars,
                peak_pnl=ddd.peak_pnl,
                trough_pnl=ddd.trough_pnl,
                peak_date=ddd.peak_date,
                trough_date=ddd.trough_date,
                recovery_date=ddd.recovery_date,
                duration_days=ddd.duration_days,
            )
            if ddd is not None else None
        ),
        provenance=_data_provenance(b),
    )


@mcp.tool(
    annotations=_READ_ONLY,
    description="Buy/sell/hold suggestions to move current holdings toward the target "
    "allocation (ASSET_TARGET). There are exactly 4 modes: 'to_total' (rebalance the whole "
    "book back to target weights — ALSO deploys new_cash into the rebalance if you pass it) and "
    "'bands' (trade only sleeves that drifted past the 5%/25% threshold — the one mode that "
    "IGNORES new_cash); 'fixed_dca' (spread new_cash across the target mix) and 'cash_flow_only' "
    "(put new_cash into the most-underweight sleeves) — these two REQUIRE new_cash > 0 and "
    "suggest all-HOLD when it is 0. Offline + read-only — it suggests, never trades. A NEW "
    "target ticker can't be sized offline (no cached price); those appear under 'unpriced'.",
)
def rebalance_check(mode: str = "to_total", new_cash: float = 0.0) -> RebalancePlan:
    mode = _one_of(mode, VALID_MODES, "rebalance mode")
    if not math.isfinite(new_cash) or new_cash < 0:
        raise ValueError(f"new_cash must be a finite number >= 0, got {new_cash}")
    today = date.today()
    b = _build(no_risk=True, today=today)
    target_path = _env_path(
        "ASSET_TARGET", "set it to your target-allocation CSV for rebalance_check"
    )
    target = load_target(target_path)
    held = b.state.held()
    price_per_share = {tk: pr.close for tk, pr in b.prices.items()}
    unpriced = sorted((set(target) | set(held)) - set(price_per_share))
    unpriced_held = [tk for tk in sorted(held) if tk not in price_per_share]
    if unpriced_held:
        # Same safety as the CLI (cli._compute_suggestions): never size a rebalance
        # over a partial book — the weights would understate the total and emit
        # confidently-wrong trades. Refuse with an explanation, never fabricate.
        return RebalancePlan(
            asof=today, mode=mode, new_cash=new_cash, target_source=str(target_path),
            suggestions=[], unpriced=unpriced,
            note=_cold_error(
                "Can't size a rebalance offline: held tickers lack a cached price "
                f"({', '.join(unpriced_held)}) — the plan would be over a partial book. "
                "Warm the cache once with `uv run python -m app --book <your-book> --warm`, "
                "then retry."
            ),
        )
    held_value = {tk: held[tk].shares * price_per_share[tk] for tk in held}
    sugg = suggest(cast("Mode", mode), held_value, price_per_share, target, new_cash=new_cash)
    trades = [
        Trade(
            ticker=s.ticker, action=s.action, shares=s.shares, dollars=s.dollars,
            current_weight=s.current_weight, target_weight=s.target_weight, reason=s.reason,
        )
        for s in sugg
    ]
    # Self-describing: fixed_dca / cash_flow_only exist to deploy NEW cash, so with none they
    # emit all-HOLD — say so rather than leave the caller puzzled by a plan of only HOLDs.
    note = _OFFLINE_NOTE
    if mode in ("fixed_dca", "cash_flow_only") and new_cash == 0:
        note = (
            f"{mode} deploys NEW cash, but new_cash=0 → every line is HOLD. Pass "
            f"new_cash=<amount> to see where the money would go. {_OFFLINE_NOTE}"
        )
    return RebalancePlan(
        asof=today, mode=mode, new_cash=new_cash, target_source=str(target_path),
        suggestions=trades, unpriced=unpriced, note=note,
    )


@mcp.tool(
    annotations=_READ_ONLY,
    description="Published fund facts for each of the user's holdings (offline, read-only): "
    "type (ETF vs stock), expense ratio, AUM, average volume, age, category. Use to answer "
    "'what am I paying / how big / how liquid / how old are my funds'. Reads the 7-day "
    "metadata cache; uncached holdings appear under 'missing'.",
)
def securities_facts() -> SecuritiesFacts:
    today = date.today()
    cache = _cache_dir()
    book_path = _env_book()
    _events, state = _load_offline(book_path, cache)
    held = sorted(state.held())
    meta = fetch_metadata(held, cache_dir=cache, online=False)
    facts = [
        SecurityFact(
            ticker=tk,
            quote_type=m.quote_type,
            expense_ratio=m.expense_ratio,
            aum=m.aum,
            avg_volume=m.avg_volume,
            age_years=m.age_years(today),
            category=m.category,
            family=m.family,
        )
        for tk in held
        if (m := meta.rows.get(tk)) is not None
    ]
    return SecuritiesFacts(asof=today, securities=facts, missing=sorted(meta.missing))


@mcp.tool(
    annotations=_READ_ONLY,
    description="Roles the user's portfolio is light in (≤3% of market value) and the largest "
    "low-cost ETFs that fill each (offline, read-only, propose-only). Use to answer 'what am I "
    "missing / what could I consider adding'. The candidate listing is deterministic; for the "
    "full per-candidate screen call `screen_candidate`.",
)
def discover_gaps() -> DiscoveryGaps:
    today = date.today()
    b = _build(no_risk=True, today=today)
    if not b.prices:
        raise ValueError(_cold_error(
            "discovery needs your holdings priced from the cache, which is empty offline. "
            "Warm the cache once with `uv run python -m app --book <your-book> --warm full` "
            "(`full` also fetches the discovery universe), then retry."
        ))
    universe = load_universe(_universe_path())
    if not universe:
        raise ValueError(f"the curated universe is empty or missing at {_universe_path()}.")
    discovery = find_gaps(b.state, b.prices, universe)
    gaps = [
        RoleGap(
            role=role,
            current_exposure=discovery.exposure.get(role, 0.0),
            candidates=[
                GapCandidate(ticker=c.ticker, name=c.name, role=c.role)
                for c in discovery.candidates
                if c.role == role
            ],
        )
        for role in discovery.gaps
    ]
    return DiscoveryGaps(asof=today, gaps=gaps, unpriced_holdings=sorted(b.missing))


@mcp.tool(
    annotations=_READ_ONLY,
    description="Judge a NEW ticker against the user's book (offline, read-only, propose-only): "
    "cost, liquidity, age, concentration, overlap with what they hold, and whether it diversified "
    "their past drawdowns — each with a reason and the figures behind it. Use to answer 'is "
    "TICKER a good fit'. Fetches TICKER's price history on demand if it isn't cached (unless "
    "ASSET_MCP_OFFLINE is set). Never a buy recommendation or a return forecast.",
)
def screen_candidate(ticker: str) -> CandidateVerdict:
    today = date.today()
    tk = ticker.strip().upper()
    if not tk or tk == CASH_TICKER:
        raise ValueError(f"{ticker!r} is not a screenable security.")
    if not re.fullmatch(r"[A-Z0-9.\-]{1,15}", tk):
        # The one free-text argument on this surface: reject anything that isn't a
        # plain ticker (no path separators / traversal), upholding the "bound to one
        # book, can't point at an arbitrary file" invariant.
        raise ValueError(f"{ticker!r} is not a valid ticker symbol.")
    b = _build(no_risk=False, today=today)
    if b.daily is None or b.daily.empty:
        raise ValueError(_cold_error(
            "screening compares against your portfolio's return series, which needs a warm "
            "price cache. Warm it once with `uv run python -m app --book <your-book> --warm`, "
            "then retry."
        ))
    cache = _cache_dir()
    start = b.daily.index[0].date()
    # Fetch the candidate on demand (it's the one ticker the user asked about) unless the
    # server is locked strictly offline — then stay cache-only. The held set already comes
    # from the warmed book; only the candidate's own series/metadata may need the network.
    online = not _offline_locked()
    cand_series = fetch_series([tk], start, today, cache_dir=cache, online=online)
    if tk not in cand_series.rows:
        note = (
            f"Couldn't fetch price history for {tk} — check the symbol "
            "(it may be delisted or temporarily unavailable)."
            if online
            else f"{tk} isn't cached and the server is strictly offline (ASSET_MCP_OFFLINE). "
            f"Warm it once — `uv run python -m app --book <your-book> --screen {tk}` "
            "(or `--warm full`) — then ask again."
        )
        return CandidateVerdict(ticker=tk, verdict="N/A", checks=[], note=note)
    held = set(b.state.held())
    # Held facts stay cache-only (already warmed); only the candidate may need the network, so an
    # on-demand fetch doesn't fan out across every holding (the held/candidate online split).
    cand_meta, held_facts, _ = candidate_and_held_facts(
        [tk], held, cache, online_candidate=online, online_held=False,
    )
    results = screen_candidates(
        [tk], cand_series.rows, b.daily, cand_meta, held_facts, held, asof=today, role=None
    )
    r = results[0]
    checks = [
        ScreenCheck(
            name=c.name, status=c.status, reason=c.reason,
            # Drop any non-finite figure (mirrors `_ci`): a nan/inf is not valid JSON.
            values={k: v for k, v in c.values.items() if math.isfinite(v)},
        )
        for c in r.checks
    ]
    return CandidateVerdict(ticker=tk, verdict=r.verdict.upper(), checks=checks)




def _benchmark_verdict(
    target: dict[str, float], benchmark: str, today: date, cache: Path
) -> tuple[BenchmarkVerdict | None, str]:
    """Validate a proposed target against a canonical reference over their common history.
    The proposed target includes universe-fill tickers outside the core warm set (held ∪ refs);
    fetch any cold ones on demand (same `ASSET_MCP_OFFLINE` gate as `screen_candidate`) so the
    verdict judges the FULL target — never a renormalized subset. The verdict is drawdown-first
    and held-out, never 'beats'."""
    if benchmark == "none":
        return None, "Validation skipped (benchmark='none')."
    ref_weights = benchmark_weights(benchmark)
    tickers = sorted(set(target) | set(ref_weights))
    start = today - timedelta(days=HISTORY_DAYS)
    online = not _offline_locked()
    series = fetch_series(tickers, start, today, cache_dir=cache, online=online)
    result = benchmark_compare(
        series.rows, target, ref_weights, reference=benchmark, provenance=series.provenance,
    )
    warm_hint = (
        ". Warm it once with `uv run python -m app --book <your-book> --warm full` "
        "(covers both the references and the universe defaults), then ask again."
    )
    if result is None:
        cold = sorted(set(tickers) - set(series.rows))
        return None, (
            f"Couldn't validate vs {benchmark} — the common price history isn't available"
            + (f" (missing {', '.join(cold)})" if cold else "")
            + warm_hint
        )
    # Honesty guard: any still-cold ticker (server locked offline, or an unfetchable symbol)
    # means benchmark_compare renormalized that leg down to its priced subset — the verdict
    # would then describe a different portfolio than the weights shown. Null it rather than
    # mislead (mirrors backtest's own "refusing to judge a renormalized target").
    if result.missing:
        return None, (
            f"Couldn't validate vs {benchmark} — these tickers aren't priced "
            f"({', '.join(result.missing)}), so the comparison would judge a renormalized "
            "portfolio rather than the proposed weights" + warm_hint
        )
    # ulcer_gain_ci is None when no paired bootstrap ran (margin/CDaR-gated, too short, or
    # insufficient); a present tuple is already finite (the bootstrap drops non-finite
    # resamples). Guard finiteness defensively so a stray nan can never surface as a real CI.
    ci = result.ulcer_gain_ci
    lo: float | None = None
    hi: float | None = None
    if ci is not None and math.isfinite(ci[0]) and math.isfinite(ci[1]):
        lo, hi = ci
    verdict = BenchmarkVerdict(
        reference=benchmark, verdict=result.verdict, reason=result.reason,
        oos_ulcer_gain_low=lo, oos_ulcer_gain_high=hi,
        inconclusive_cause=result.cause or None,
    )
    return verdict, f"Walk-forward held-out drawdown comparison vs {benchmark}."


@mcp.tool(
    annotations=_READ_ONLY,
    description="Propose a strategic target allocation for a risk posture (conservative / "
    "moderate / aggressive) over the user's book + the curated universe, and validate it "
    "against a canonical reference (60-40 / all-weather / permanent) with a walk-forward "
    "held-out drawdown verdict. Use to answer 'what should a moderate portfolio look like for "
    "me, and is it sound'. Propose-only: never trades, never a recommendation or return "
    "forecast — every weight comes from the deterministic core. The weights are always "
    "returned; the verdict needs the reference tickers cached, else it's null with a warm note. "
    "Pass benchmark='none' to skip validation.",
)
def propose_allocation(preset: str = "moderate", benchmark: str = "60-40") -> ProposedAllocation:
    today = date.today()
    pset = _one_of(preset, PRESETS, "preset")
    return _build_proposal(pset, benchmark, today)


def _build_proposal(pset: str, benchmark: str, today: date) -> ProposedAllocation:
    """Build a preset target over the book + universe and validate it against a benchmark.
    The shared body of `propose_allocation` and `starter_allocation` — both must produce
    IDENTICAL weights for a given posture, so there is exactly one build path. `pset` is a
    validated preset; `benchmark` is validated here."""
    bench = _one_of(benchmark, BENCHMARKS, "benchmark", allow_none=True)
    b = _build(no_risk=True, today=today)
    if not b.prices:
        raise ValueError(_cold_error(
            "proposing an allocation needs your holdings priced from the cache, which is empty. "
            "Warm it once with `uv run python -m app --book <your-book> --warm`, then retry."
        ))
    universe = load_universe(_universe_path())
    if not universe:
        raise ValueError(f"the curated universe is empty or missing at {_universe_path()}.")
    target = build_preset_target(pset, universe, held_market_value(b.state, b.prices))
    if not target:
        raise ValueError(f"couldn't build a {pset} target from your book + the universe.")
    weights = [
        AllocationWeight(ticker=tk, weight=w)
        for tk, w in sorted(target.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    verdict, validation_note = _benchmark_verdict(target, bench, today, _cache_dir())
    return ProposedAllocation(
        asof=today, preset=pset, weights=weights,
        benchmark=None if bench == "none" else bench,
        verdict=verdict, validation_note=validation_note,
        unpriced_holdings=sorted(b.missing),
    )


class StarterAllocation(BaseModel):
    """The onboarding answer: the matched posture, WHY it was matched (the rubric's plain
    reasons), and the full preset proposal for that posture — so a new user goes from three
    answers to a reviewable starting allocation in one call."""

    posture: str = Field(description="conservative | moderate | aggressive — from the answers")
    score: int = Field(description="rubric score 0-6 (higher = more growth-tolerant)")
    rationale: list[str] = Field(
        description="one plain-language reason per answer, plus the scoring line — the "
        "explainable trail from answers to posture; relay it, don't invent your own"
    )
    proposal: ProposedAllocation


@mcp.tool(
    annotations=_READ_ONLY,
    description="Turn a new user's risk answers into a starting allocation. Pass the three "
    "onboarding answers — horizon ('under_3_years' | '3_to_10_years' | 'over_10_years'), "
    "loss_response ('sell' | 'hold' | 'buy_more'), cash_buffer ('no' | 'partly' | "
    "'comfortably') — and it maps them to a conservative/moderate/aggressive posture (a "
    "fixed, explainable rubric — never your guess) and returns that preset allocation "
    "validated against a benchmark (60-40 / all-weather / permanent, or 'none' to skip — "
    "default 60-40), same as propose_allocation. Ask the three questions in "
    "plain language first, then call this with the chosen answer tokens. Propose-only: a "
    "hand-designed starting posture, never a recommendation or a return forecast.",
)
def starter_allocation(
    horizon: str, loss_response: str, cash_buffer: str, benchmark: str = "60-40"
) -> StarterAllocation:
    # Pure rubric (app.onboard) picks the posture; ValueError names the valid tokens for a
    # bad answer, so the fence's honesty carries to onboarding too.
    result = posture_from_answers(
        horizon.strip().lower(), loss_response.strip().lower(), cash_buffer.strip().lower()
    )
    proposal = _build_proposal(result.posture, benchmark, date.today())
    return StarterAllocation(
        posture=result.posture, score=result.score,
        rationale=list(result.rationale), proposal=proposal,
    )


# --- The chat front door: prompts + the trust manifest ------------------------------
#
# Prompts are user-controlled conversation starters (surfaced in the client's "+" menu):
# they turn a blank chat box into a guided menu of what the 8 tools can actually answer,
# and inject the honesty framing into the very FIRST user turn. The resource makes the
# server's guarantees a *fetchable artifact* the model can cite verbatim when the user
# asks "can I trust this?" — instructions steer the model; this one the USER can read.

# Every starter ends with the same rule so no prompt can ship without the fence framing
# (the prompt-level twin of _cold_error: one funnel, not five copies drifting apart).
_FIGURES_RULE = (
    "Use only figures the tools return, and quote confidence intervals where given. If a "
    "figure is unavailable or a tool errors, say so and relay the tool's fix (e.g. warming "
    "the cache) — do not estimate it. Plain language: explain any finance term in one "
    "clause. Describe my portfolio; don't advise me what to do."
)


@mcp.prompt(title="Portfolio checkup")
def portfolio_checkup() -> str:
    """The full picture: holdings + returns + drawdown-first risk, in plain words."""
    return (
        "Give me a drawdown-first checkup of my portfolio. Call portfolio_summary and "
        "risk_report, then: lead with the worst drawdown (depth, duration, recovery) and "
        "what the Ulcer index / CDaR say about how the ride actually felt; then returns "
        "(time-weighted vs money-weighted — note if they disagree and what that means); "
        "then the holdings that drive the picture. " + _FIGURES_RULE
    )


@mcp.prompt(title="What's my drawdown?")
def whats_my_drawdown() -> str:
    """How deep my portfolio fell, how long it stayed down, and what that felt like."""
    return (
        "Call risk_report and explain my drawdown in plain words: how deep was the worst "
        "fall (max drawdown, with its confidence interval), how long under water (peak → "
        "trough → recovery, and the share of days spent below a previous high), and what "
        "the Ulcer index and CDaR capture that a single worst-fall number misses. "
        + _FIGURES_RULE
    )


@mcp.prompt(title="Should I rebalance?")
def should_i_rebalance(mode: str = "to_total", new_cash: str = "0") -> str:
    """Whether my target policy says to act, for a chosen mode (to_total | bands | fixed_dca | cash_flow_only)."""
    # validate at prompt time — a starter must not open on a tool error
    m = _one_of(mode, VALID_MODES, "mode")
    cash_clause = ""
    nc = new_cash.strip()
    has_cash = bool(nc) and nc not in ("0", "0.0")
    if m in ("fixed_dca", "cash_flow_only"):
        if has_cash:
            cash_clause = f" Deploy new_cash={nc} — this mode only acts on NEW cash."
        else:
            cash_clause = (
                " This mode only acts on NEW cash — ask me how much I want to invest, then "
                "call it with that amount as new_cash."
            )
    elif m == "to_total" and has_cash:
        cash_clause = (
            f" Also deploy new_cash={nc} — to_total adds it to the target total and "
            "rebalances the whole book to that."
        )
    return (
        f"Call rebalance_check with mode='{m}'.{cash_clause} Tell me whether my rebalance "
        "policy says to act: for each suggested action, name the RULE that produced it and "
        "the drift figure behind it; if nothing fires, say so plainly. These describe my own "
        "configured policy — not recommendations. " + _FIGURES_RULE
    )


@mcp.prompt(title="Fill my gaps")
def fill_my_gaps() -> str:
    """Portfolio roles I'm light in + screened candidate ETFs for each (propose-only)."""
    return (
        "Call discover_gaps to find the portfolio roles I'm light in, then run "
        "screen_candidate on the leading candidate for each gap (a handful at most). "
        "Report each verdict with its named reasons (cost, liquidity, overlap, "
        "diversification), keeping WARN/FAIL reasons honest. These are candidates to "
        "research, never instructions to buy. " + _FIGURES_RULE
    )


@mcp.prompt(title="Find my starting allocation")
def find_my_starting_allocation() -> str:
    """New here? Answer 3 risk questions and get a starting allocation matched to them."""
    return (
        "Help me find a starting allocation. Ask me these three questions ONE AT A TIME, in "
        "plain language, and wait for each answer:\n"
        "1. When do I expect to spend most of this money? (within 3 years / in 3-10 years / "
        "10+ years away)\n"
        "2. If a $10,000 holding fell to $8,500 in a bad month, would I sell, hold, or buy "
        "more?\n"
        "3. Could I cover a surprise expense WITHOUT selling these investments? (no / partly "
        "/ comfortably)\n"
        "Then map my answers to the tokens and call starter_allocation "
        "(horizon=under_3_years|3_to_10_years|over_10_years, "
        "loss_response=sell|hold|buy_more, cash_buffer=no|partly|comfortably). Show the "
        "matched posture with the rubric's rationale, then the proposed weights and each "
        "holding's role, and relay the benchmark verdict exactly as returned. Be clear this "
        "is a hand-designed starting posture, not an optimized or predicted-best portfolio. "
        + _FIGURES_RULE
    )


@mcp.prompt(title="Propose a posture")
def propose_a_posture(posture: str = "moderate", benchmark: str = "60-40") -> str:
    """A starting allocation for a risk posture (conservative / moderate / aggressive), validated against a benchmark (60-40 / all-weather / permanent, or 'none' to skip)."""
    # validate at prompt time — a starter must not open on a tool error
    p = _one_of(posture, PRESETS, "posture")
    b = _one_of(benchmark, BENCHMARKS, "benchmark", allow_none=True)
    return (
        f"Call propose_allocation with preset '{p}' and benchmark '{b}'. Show the proposed "
        "weights and what each holding is for (its role), then report the benchmark verdict "
        "exactly as returned — the vocabulary is 'shallower/deeper/inconclusive/insufficient' "
        "by design, never 'beats'; if the verdict is null, relay the tool's note on why. Be "
        "clear this is a hand-designed posture prior, not an optimized or predicted-best "
        "portfolio. " + _FIGURES_RULE
    )


@mcp.resource("portfolio://guarantees", mime_type="text/markdown")
def guarantees() -> str:
    """The trust manifest: what this server will and won't do — four guarantees enforced in code and pinned by tests."""
    return f"""\
# What this portfolio server will — and won't — do

Version {_VERSION}. These are properties **enforced in code and pinned by the test
suite**, not promises.

**1. Read-only.** No tool can place a trade, move money, or edit your transaction log.
There are no write tools; every tool is annotated read-only. The ledger is append-only,
and only you write to it.

**2. Every number is computed, never generated.** Figures come from a deterministic
Python core — reconciled to the cent against a real brokerage export, and to 4 decimal
places against quantstats. The AI assistant narrating this chat cannot compute, alter,
or estimate a figure: when a value is unavailable, the tools say so (and how to fix it)
rather than guessing, and the assistant is instructed to do the same.

**3. Your data stays on your machine.** The server runs locally over stdio; there is no
telemetry. The only network use is downloading public market data — price history (a
one-time warm of a cold cache, plus any new ticker you ask it to screen) and published
fund facts (expense ratio, fund size) for those tickers. Your transactions, holdings,
and account values are never uploaded anywhere. Set `ASSET_MCP_OFFLINE=1` to forbid even
those downloads.

**4. Descriptions, not advice — structurally.** Every rebalance suggestion is paired to
the named rule that produced it (e.g. "5/25 band breached"), every risk metric carries a
confidence interval, and benchmark verdicts only ever say `shallower` / `deeper` /
`inconclusive` / `insufficient` (too little data) — never "beats" or "buy this". A
strategy claiming an *edge* must pass a walk-forward (out-of-sample) gate before it may
even surface. You learn the rule; you decide.

*How to verify: the source is open — the tests pin each property (read-only annotations,
the number fence that rejects any model-authored digit, the walk-forward gate). Nothing
in this chat is financial advice.*
"""


def run() -> None:
    """Entry point: configure stderr logging, load .env, serve over stdio."""
    setup_logging()
    load_dotenv(_REPO_ROOT / ".env")
    log.info("asset-management MCP server starting (read-only, offline) over stdio")
    mcp.run()  # transport defaults to stdio


if __name__ == "__main__":
    run()
