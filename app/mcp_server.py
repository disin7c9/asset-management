"""Read-only MCP server: expose the validated portfolio core as stdio tools.

A Layer-4 delivery surface — a composition root, sibling to ``cli.py``/``email.py``
— that lets an MCP client (Claude Desktop / Claude Code) **call** the deterministic
core (holdings, returns, drawdown/risk) over local **stdio**. It wires; it adds no
math. Design invariants (v2.0.0 Phase 1):

- **Every figure comes from the validated core** (`derive`/`returns`/`risk`, via the
  same `pipeline.compute_prices_returns_risk` the brief uses) — the LLM never computes.
- **Read-only + offline**: tools read already-derived data and the on-disk price
  **cache** (`online=False`), so a call never reaches the network (no egress) and is
  fast + deterministic. A cold cache degrades to `n/a`, never a guess.
- **Bound to one book** (`ASSET_CSV`): tools take no file-path argument, so a caller
  can't point the server at an arbitrary file.

stdout is the stdio transport, so **all logging goes to stderr** (`setup_logging`);
nothing here writes to stdout. Tool errors surface as MCP errors, never a crash.

Run:  ``uv run python -m app.mcp_server``  (register with ``claude mcp add``). Config
via env (.env or the client's env block): ``ASSET_CSV`` (required), ``ASSET_TARGET``
(for ``rebalance_check``), ``ASSET_CACHE_DIR`` (optional; default ``data/prices``).
"""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from app.pipeline import compute_prices_returns_risk, load_book
from app.derive import DerivedState
from app.discover import find_gaps
from app.events import CASH_TICKER, load_target
from app.log_config import setup_logging
from app.metadata import fetch_metadata
from app.prices import PriceRow, fetch_series
from app.returns import ReturnsSummary
from app.risk import DollarDrawdown, MetricCI, RiskSummary
from app.screen import screen_candidates
from app.strategy import VALID_MODES, Mode, suggest
from app.universe import load_universe

if TYPE_CHECKING:
    import pandas as pd

log = logging.getLogger(__name__)

_REPO_ROOT: Path = Path(__file__).resolve().parents[1]

_OFFLINE_NOTE = (
    "Offline/cache-only and read-only: figures are derived from your transaction log "
    "and the on-disk price cache; a cold cache shows null (n/a), never a guess. "
    "This is a view, not financial advice."
)

# Read-only, closed-world (no external side effects), idempotent — the honest hints
# for a Claude client (Claude reads readOnlyHint to know the tool only observes).
_READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)

mcp: FastMCP = FastMCP(
    "asset-management",
    instructions=(
        "Read-only, offline view of the user's own stock/ETF portfolio. Every number "
        "is computed by a validated deterministic core (reconciled to the cent vs "
        "ghostfolio, 4 decimals vs quantstats) — do not compute or estimate figures "
        "yourself; call a tool. Lead with drawdown (depth/duration/recovery), report "
        "confidence intervals where given, and never present output as financial advice."
    ),
)


# ── config resolution (env-only; no argparse on this surface) ────────────────


def _env_csv(var: str, what: str) -> Path:
    """Resolve a CSV path from an env var (.env convention): ``~`` expanded, a
    relative path repo-relative. Raises a clear ValueError (→ a clean MCP error)
    if unset or missing — the server never guesses a path."""
    raw = os.environ.get(var, "").strip()
    if not raw:
        raise ValueError(f"{var} is not set — {what}")
    p = Path(raw).expanduser()
    path = p if p.is_absolute() else _REPO_ROOT / p
    if not path.exists():
        raise ValueError(f"{var} points at a missing file: {path}")
    return path


def _cache_dir() -> Path:
    """The price cache the tools read (offline). ``ASSET_CACHE_DIR`` overrides the
    default ``data/prices`` (used by tests to point at a warmed temp cache)."""
    raw = os.environ.get("ASSET_CACHE_DIR", "").strip()
    if raw:
        p = Path(raw).expanduser()
        return p if p.is_absolute() else _REPO_ROOT / p
    return _REPO_ROOT / "data" / "prices"


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


def _build(*, no_risk: bool, today: date) -> _Build:
    """Load the ASSET_CSV book and compute prices/returns/(risk) via the SAME core
    path the CLI brief uses (`pipeline.compute_prices_returns_risk`), forced offline. The
    `no_risk` flag skips the bootstrap panel — `portfolio_summary`/`rebalance_check`
    don't need it (fast), `risk_report` does. `today` is resolved ONCE per tool call
    (one clock per request) and threaded through."""
    cache = _cache_dir()
    csv_path = _env_csv(
        "ASSET_CSV", "point it at your transaction CSV (in .env or the MCP client env)"
    )
    events, state = load_book(csv_path, cache, online=False)
    run: dict[str, Any] = {
        "status": "ok", "n_prices_fetched": 0, "n_prices_missing": 0,
        "n_series_fetched": 0, "n_series_missing": 0, "fallbacks_used": 0,
    }
    prices, returns, risk, missing, _twr_excluded, dollar_dd, _series, daily = (
        compute_prices_returns_risk(
            events, state, no_risk=no_risk, offline=True,
            cache_dir=cache, today=today, run=run,
        )
    )
    log.info("mcp build (offline, no_risk=%s): %s", no_risk, run)
    return _Build(
        state=state, prices=prices or {}, returns=returns, risk=risk,
        missing=missing, dollar_dd=dollar_dd, daily=daily,
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


class PortfolioSummary(BaseModel):
    asof: date
    offline: bool = True
    holdings: list[Holding]
    totals: Totals
    returns: Returns
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
                price_age_hours=(
                    round(pr.cache_age.total_seconds() / 3600.0, 1)
                    if pr is not None else None
                ),
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
        raise ValueError(
            "risk metrics need cached daily price history, which isn't available offline. "
            "Warm the cache by running the brief online once "
            "(uv run python -m app --csv <your.csv>), then retry."
        )
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
    )


@mcp.tool(
    annotations=_READ_ONLY,
    description="Buy/sell/hold suggestions to move current holdings toward the target "
    "allocation (ASSET_TARGET), for the named discipline rule (default to_total). "
    "Offline + read-only — it suggests, never trades. A NEW target ticker can't be "
    "sized offline (no cached price); those appear under 'unpriced'.",
)
def rebalance_check(mode: str = "to_total") -> RebalancePlan:
    if mode not in VALID_MODES:
        raise ValueError(f"unknown rebalance mode {mode!r}; valid: {sorted(VALID_MODES)}")
    today = date.today()
    b = _build(no_risk=True, today=today)
    target = load_target(
        _env_csv("ASSET_TARGET", "set it to your target-allocation CSV for rebalance_check")
    )
    held = b.state.held()
    price_per_share = {tk: pr.close for tk, pr in b.prices.items()}
    unpriced = sorted((set(target) | set(held)) - set(price_per_share))
    unpriced_held = [tk for tk in sorted(held) if tk not in price_per_share]
    if unpriced_held:
        # Same safety as the CLI (cli._compute_suggestions): never size a rebalance
        # over a partial book — the weights would understate the total and emit
        # confidently-wrong trades. Refuse with an explanation, never fabricate.
        return RebalancePlan(
            asof=today, mode=mode, suggestions=[], unpriced=unpriced,
            note="Can't size a rebalance offline: held tickers lack a cached price "
            f"({', '.join(unpriced_held)}) — the plan would be over a partial book. "
            "Warm the cache (run the brief online), then retry.",
        )
    held_value = {tk: held[tk].shares * price_per_share[tk] for tk in held}
    sugg = suggest(cast("Mode", mode), held_value, price_per_share, target)
    trades = [
        Trade(
            ticker=s.ticker, action=s.action, shares=s.shares, dollars=s.dollars,
            current_weight=s.current_weight, target_weight=s.target_weight, reason=s.reason,
        )
        for s in sugg
    ]
    return RebalancePlan(asof=today, mode=mode, suggestions=trades, unpriced=unpriced)


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
    csv_path = _env_csv(
        "ASSET_CSV", "point it at your transaction CSV (in .env or the MCP client env)"
    )
    _events, state = load_book(csv_path, cache, online=False)
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
        raise ValueError(
            "discovery needs your holdings priced from the cache, which is empty offline. "
            "Warm the cache (run the brief online once), then retry."
        )
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
    "TICKER a good fit'. Needs TICKER's price history in the cache; if it isn't there, returns "
    "an N/A verdict with how to warm it. Never a buy recommendation or a return forecast.",
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
        raise ValueError(
            "screening compares against your portfolio's return series, which needs a warm "
            "price cache. Run the brief online once, then retry."
        )
    cache = _cache_dir()
    start = b.daily.index[0].date()
    cand_series = fetch_series([tk], start, today, cache_dir=cache, online=False)
    if tk not in cand_series.rows:
        return CandidateVerdict(
            ticker=tk, verdict="N/A", checks=[],
            note=f"{tk} isn't in your price cache. Warm it once online — "
            f"`uv run python -m app --csv <your.csv> --screen {tk}` — then ask again.",
        )
    held = set(b.state.held())
    meta = fetch_metadata(sorted({tk} | held), cache_dir=cache, online=False)
    held_meta = {t: m for t, m in meta.rows.items() if t in held}
    cand_meta = {t: m for t, m in meta.rows.items() if t == tk}
    results = screen_candidates(
        [tk], cand_series.rows, b.daily, cand_meta, held_meta, held, asof=today, role=None
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


def run() -> None:
    """Entry point: configure stderr logging, load .env, serve over stdio."""
    setup_logging()
    load_dotenv(_REPO_ROOT / ".env")
    log.info("asset-management MCP server starting (read-only, offline) over stdio")
    mcp.run()  # transport defaults to stdio


if __name__ == "__main__":
    run()
