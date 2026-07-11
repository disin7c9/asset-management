"""Shared compute pipeline: book → derived holdings → prices → returns → risk.

Layer 2 (computation) — a *composite* over the Layer-2 leaves plus the `prices`
adapter, holding the one bundle every delivery surface (`cli`, `mcp_server`, and
discovery) needs. Lifted out of `cli` so no composition root imports another
(architecture cleanup R4). Like `backtest` it composes its Layer-2 siblings
(`returns` + `risk`); unlike the pure leaves it triggers the `prices` ★ adapter
(`fetch_series`/`fetch_latest`/`fetch_splits`) — its only I/O, delegated to the
one named market-data adapter, so it adds no new I/O concern, it just orchestrates.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from pathlib import Path

if TYPE_CHECKING:
    import pandas as pd

from app.backtest import BENCHMARKS, benchmark_weights
from app.corporate_actions import adjust_for_splits
from app.derive import DerivedState, derive
from app.events import CASH_TICKER, Event, load_events
from app.metadata import MetadataResult, SecurityMeta, fetch_metadata
from app.prices import (
    STALE_PRICE_FLOOR,
    PriceRow,
    PricesResult,
    SeriesResult,
    ensure_cache_dir,
    fetch_latest,
    fetch_series,
    fetch_splits,
    usable_price,
)
from app.returns import (
    ReturnsSummary,
    build_daily_returns,
    pnl_curve,
    price_basis_mismatches,
    summarize,
    true_twr_annualized,
    twr_index,
    value_curve,
)
from app.risk import DollarDrawdown, RiskSummary, dollar_drawdown, summarize_risk

log = logging.getLogger(__name__)

def default_cache_dir(repo_root: Path) -> Path:
    """The price-cache default for a surface rooted at ``repo_root``.

    A repo checkout keeps the historical ``data/prices``. But when the package runs
    INSTALLED (uvx / pip / the Claude-Desktop bundle), ``repo_root`` is site-packages or
    the unpacked extension dir — writable-ish, but ephemeral (wiped by ``uv cache clean``
    or an extension update) and invisible to the user. Fall back to a stable per-user
    location instead. The checkout marker is ``.git``, NOT ``pyproject.toml`` — the
    .mcpb bundle ships pyproject.toml, so that test would misroute the bundle's cache
    into the host-managed extension dir.
    """
    if (repo_root / ".git").exists():  # a checkout, not an installed dist
        return repo_root / "data" / "prices"
    return Path.home() / ".asset-management" / "prices"


# The bundled example book (--demo): a package constant rather than a repo data file, so
# an installed / `uvx` run with no checkout can still materialize it. Kept byte-identical
# to data/sample_data/transactions.csv (the browsable copy) — pinned by a test.
DEMO_BOOK_CSV = """\
Date,Code,DataSource,Currency,Price,Quantity,Action,Fee,Note
2023-01-05,VOO,YAHOO,USD,365.00,10,buy,1.00,initial position
2023-03-15,BND,YAHOO,USD,72.50,30,buy,1.00,
2023-06-01,VOO,YAHOO,USD,395.00,5,buy,1.00,adding on dip
2023-09-20,VOO,YAHOO,USD,22.40,0,dividend,0.00,quarterly dividend
2024-02-10,IAU,YAHOO,USD,37.20,40,buy,1.00,gold hedge
2024-04-01,VOO,YAHOO,USD,460.00,3,sell,1.00,trim winner
2024-07-15,VOO,YAHOO,USD,19.80,0,dividend,0.00,
2024-11-05,BND,YAHOO,USD,71.80,20,buy,1.00,
2025-01-20,BND,YAHOO,USD,45.10,0,dividend,0.00,
2025-05-10,IAU,YAHOO,USD,52.00,15,sell,1.00,take some profit
2025-06-21,VEA,YAHOO,USD,40.2,15,buy,1.00,"""


def write_demo_book(cache_dir: Path) -> Path:
    """Materialize the bundled example book into the cache dir and return its path.

    Rewritten on every call so the demo always matches the installed version. Raises
    OSError when the cache dir can't be used — a demo has no book to degrade to, so
    unlike the price caches it must fail loudly rather than run without a file.
    """
    resolved = ensure_cache_dir(cache_dir)
    if resolved is None:
        raise OSError(f"cache dir {cache_dir} is not writable")
    path = resolved / "demo_book.csv"
    path.write_text(DEMO_BOOK_CSV, encoding="utf-8")
    return path


def load_book(
    book_path: Path, cache_dir: Path, *, online: bool
) -> tuple[list[Event], DerivedState]:
    """Replay a transaction book (our CSV or a Ghostfolio JSON export) → split-adjusted
    events + derived holdings.

    The shared book loader for every surface (the CLI brief and the MCP server).
    Split-adjust raw share counts (yfinance prices are split-adjusted) so holdings
    AND the time-weighted series share one basis — the real fix for the NVDA-style
    corruption; splits are fetched cache-only when `online` is False, and the
    price-basis-mismatch guard in `compute_prices_returns_risk` stays the net for
    any split we couldn't fetch. The tickers we actually adjusted are logged.
    """
    events = load_events(book_path)
    splits = fetch_splits(
        sorted({ev.ticker for ev in events}), cache_dir=cache_dir, online=online
    )
    adjusted = adjust_for_splits(events, splits)
    changed = sorted(
        {a.ticker for a, b in zip(events, adjusted, strict=True) if a != b}
    )
    if changed:
        log.info("split-adjusted share counts for: %s", ", ".join(changed))
    return adjusted, derive(adjusted)


def record_series_fetch(run: dict[str, Any], series: SeriesResult) -> None:
    """Tally one fetch_series result into the run summary — counts, fallbacks, and
    the partial flip — so every fetching handler reports identically (the one
    bookkeeping gate; copies had already drifted on which keys they touched)."""
    run["n_series_fetched"] += len(series.rows)
    run["n_series_missing"] += len(series.missing)
    run["fallbacks_used"] += series.fallbacks_used
    if series.missing and run["status"] == "ok":
        run["status"] = "partial"


def candidate_and_held_facts(
    candidates: Iterable[str],
    held: set[str],
    cache_dir: Path | None,
    *,
    online_candidate: bool,
    online_held: bool,
    held_meta: MetadataResult | None = None,
) -> tuple[dict[str, SecurityMeta], dict[str, SecurityMeta], list[str]]:
    """Fetch the per-security facts the overlap/structure screen needs, as
    ``(candidate_facts, held_facts, missing)``. Candidate facts are always fetched (gated by
    ``online_candidate``); held facts reuse a prior ``MetadataResult`` when given, else are
    fetched (gated by ``online_held``). The split is the one shared by ``--screen``/``--discover``
    and the MCP ``screen_candidate``: the CLI fetches both online, while the MCP on-demand path
    fetches only the candidate online so the warmed held set stays cache-only — one new ticker
    can't fan a network call out across every holding."""
    cand_set = set(candidates)
    cand_res = fetch_metadata(sorted(cand_set), cache_dir=cache_dir, online=online_candidate)
    cand_facts = {tk: m for tk, m in cand_res.rows.items() if tk in cand_set}
    if held_meta is not None:
        held_facts = {tk: m for tk, m in held_meta.rows.items() if tk in held}
        missing = list(cand_res.missing)
    else:
        held_res = fetch_metadata(sorted(held), cache_dir=cache_dir, online=online_held)
        held_facts = dict(held_res.rows)
        missing = [*cand_res.missing, *held_res.missing]
    return cand_facts, held_facts, missing


def held_market_value(
    state: DerivedState, prices: dict[str, PriceRow]
) -> dict[str, float]:
    """Per-held-ticker market value (shares × close), priced tickers only.

    The single source of held-value math (holdings sizing, returns, suggestions,
    and --dump-target all go through here), so a future valuation change lands in
    one place. A ticker absent from `prices`, or whose close is not usable (`usable_price`:
    finite and > 0), is omitted. The provider/cache ingest points now reject bad closes at
    the source, but this sink applies the SAME rule so it can't disagree with a sibling sink
    (`cli` rebalance) about which tickers are priced."""
    held = state.held()
    return {
        tk: held[tk].shares * prices[tk].close
        for tk in held
        if tk in prices and usable_price(prices[tk].close)
    }


def compute_prices_returns_risk(
    events: list[Event],
    state: DerivedState,
    *,
    no_risk: bool,
    offline: bool,
    cache_dir: Path,
    today: date,
    run: dict[str, Any],
) -> tuple[
    dict[str, PriceRow] | None,
    ReturnsSummary | None,
    RiskSummary | None,
    list[str],
    list[str],
    DollarDrawdown | None,
    SeriesResult | None,
    "pd.Series[float] | None",
]:
    """Fetch prices, derive returns + risk. Single price source per mode.

    When the risk panel is on we fetch price *history* once and derive each
    held ticker's latest price from its series tail — so MWR/Modified Dietz
    and the true-TWR share one price universe (no spot-vs-history mismatch)
    and we avoid a second network round-trip. With --no-risk we fetch only
    the latest prices.
    """
    held = state.held()
    online = not offline
    prices: dict[str, PriceRow] = {}
    risk: RiskSummary | None = None
    true_twr: float | None = None
    twr_excluded: list[str] = []
    stale_tails: list[str] = []  # held tickers whose series tail is beyond the floor
    dollar_dd: DollarDrawdown | None = None
    daily: pd.Series[float] | None = None

    series = None
    if not no_risk and events:
        # Real securities only — the CASH pseudo-ticker (deposit/withdraw legs) has
        # no price history; fetching it would land in series.missing and falsely
        # flip the run status to "partial" on every book that holds cash flows.
        traded = sorted({ev.ticker for ev in events} - {CASH_TICKER})
        start = min(ev.date for ev in events) - timedelta(days=5)
        series = fetch_series(traded, start, today, cache_dir=cache_dir, online=online)
        record_series_fetch(run, series)
        if series.rows:
            # A currently-HELD ticker whose series tail is older than the floor (a
            # delisted / halted position) must leave the time-weighted series too, not
            # just the spot-price dict below: `value_curve` forward-fills the last close
            # across the gap, so risk/TWR would otherwise carry the position flat at the
            # very quote the pricing loop refuses — biasing volatility and drawdown
            # shallower, unflagged. Sold positions are exempt: zero shares after the
            # sale contribute nothing to the curve, and their history is real history.
            for tk in sorted(held):
                s = series.rows.get(tk)
                if s is None or s.empty:
                    continue
                tail = s.index[-1].date()
                if today - tail > STALE_PRICE_FLOOR:
                    stale_tails.append(tk)
                    log.warning(
                        "%s: last available close is from %s (%d days before %s) — "
                        "delisted, halted, or renamed? Refusing to value it at that "
                        "stale price; the holding is reported unpriced and excluded "
                        "from TWR & risk instead.",
                        tk, tail, (today - tail).days, today,
                    )
            # Safety net for a trade the price history can't reconcile: either a basis
            # mismatch (the log holds raw share counts while yfinance prices are split-
            # adjusted; splits ARE adjusted upstream, but an unfetched split or an
            # off-market entry price can still disagree) or a trade that predates the
            # history (a left-truncated series). Both would fabricate a return into the
            # time-weighted series, so exclude such a ticker from TWR + risk (and warn);
            # MWR / Modified Dietz, computed from cash flows, are unaffected.
            mismatched = price_basis_mismatches(events, series.rows)
            if mismatched:
                log.warning(
                    "excluding %s from TWR & risk: a trade can't be reconciled with the "
                    "price history — the recorded execution price disagrees with the "
                    "split-adjusted history (an unfetched/unavailable split or an "
                    "off-market entry price), or the trade predates the history's first "
                    "day (a truncated feed). Either would fabricate a return into the "
                    "time-weighted series; MWR / Modified Dietz are unaffected.",
                    ", ".join(mismatched),
                )
            twr_excluded = sorted(set(mismatched) | set(stale_tails))
            twr_series = series.rows
            if twr_excluded:
                twr_series = {
                    tk: s for tk, s in series.rows.items() if tk not in twr_excluded
                }
            # Build the holdings value curve once; both the TWR series and the
            # dollar P&L curve share it (same priced universe, no double work).
            value = value_curve(events, twr_series, today)
            daily = build_daily_returns(events, twr_series, asof_date=today, value=value)
            true_twr = true_twr_annualized(daily)
            risk = summarize_risk(daily, twr_index(daily))
            # 'Gains given back' — the flow-neutral dollar P&L drawdown over the
            # same priced universe (deposits/withdrawals/trades cancel).
            dollar_dd = dollar_drawdown(pnl_curve(events, twr_series, today, value=value))

    if series is not None and series.rows:
        # Derive each held ticker's latest price from its series tail, carrying the
        # series' REAL provenance (cache/yfinance/tiingo + true fetch time) — not a
        # fabricated "series"/now() stamp, so the footer's source + age are honest.
        for tk in held:
            s = series.rows.get(tk)
            if s is not None and not s.empty:
                if tk in stale_tails:
                    # The same staleness floor the offline latest path enforces (already
                    # detected and warned above): a tail that old is a last-EVER close,
                    # not a current price. Minting a PriceRow from it would value the
                    # holding — and size rebalance trades — at a dead quote. Leaving it
                    # unpriced routes it into the honesty machinery below (missing →
                    # status "partial", MWR/Dietz and dollar-DD suppressed, rebalance
                    # refused), and the exclusion above keeps it out of TWR & risk.
                    continue
                source, fetched_at = series.provenance.get(
                    tk, ("cache", datetime.now(timezone.utc))
                )
                prices[tk] = PriceRow(
                    ticker=tk,
                    asof_date=s.index[-1].date(),
                    close=float(s.iloc[-1]),
                    source=source,
                    fetched_at=fetched_at,
                )
        # (fallbacks already tallied by record_series_fetch at fetch time)
    else:
        # No risk panel (or series unavailable): fetch latest prices for held.
        result: PricesResult = fetch_latest(
            list(held), cache_dir=cache_dir, online=online
        )
        prices = result.rows
        run["fallbacks_used"] = result.fallbacks_used

    # One definition of "priced" for the whole report: a held ticker with a
    # positive, usable close — what market value (and so MWR / Modified Dietz)
    # actually need. A ticker present in `prices` but whose close is not finite and
    # positive (non-positive, NaN, or ±inf) is dropped from value, so it must count as
    # unpriced here too; else a partial book would be scored as fully priced and the
    # money-weighted figures would print a confident wrong number. Counters + partial-status are set here
    # once, for both branches (the series counts are recorded by the caller).
    priced_held = held_market_value(state, prices)
    prices = {tk: prices[tk] for tk in priced_held}  # drop non-finite/non-positive quotes
    missing = [tk for tk in held if tk not in priced_held]
    run["n_prices_fetched"] = len(priced_held)
    run["n_prices_missing"] = len(missing)
    if missing and run["status"] == "ok":
        run["status"] = "partial"
    if missing or twr_excluded:
        # Incomplete P&L curve → suppress the felt-dollar drawdown (as we do MWR /
        # Modified Dietz). Either a held ticker is unpriced (`missing`), or a split-
        # mismatched ticker was dropped from the curve (`twr_excluded`); in both
        # cases "Gains given back" would silently omit a holding, so print n/a
        # rather than a confident number missing part of the book.
        dollar_dd = None
    mkt_value = sum(priced_held.values())
    returns = summarize(
        events, mkt_value, asof_date=today, true_twr=true_twr, fully_priced=not missing
    )
    return prices, returns, risk, missing, twr_excluded, dollar_dd, series, daily


# ── cache warming (shared by the CLI --warm flag and the MCP cold-call auto-warm) ──

HISTORY_DAYS = 3653  # ~10y; the shared lookback so a warm covers the backtest/benchmark window
_WARM_MARKER = ".warmed"  # touched after each warm; throttles the MCP cold-call re-warm
_WARM_TTL_SECONDS = 6 * 3600  # re-attempt the cold-call warm at most ~4×/day


def benchmark_ref_tickers() -> list[str]:
    """The reference-portfolio tickers a benchmark validation needs priced (VOO, BND, …) —
    the union of every canonical reference in `backtest.BENCHMARKS`."""
    return sorted({tk for name in BENCHMARKS for tk in benchmark_weights(name)})


def cache_is_cold(cache_dir: Path) -> bool:
    """True when the offline cache hasn't been warmed within the TTL — the MCP cold-call
    auto-warm signal. Keyed on the `.warmed` marker's age, NOT per-ticker file existence:
    a book ticker that can never be fetched (delisted/typo) leaves no series file, so an
    existence check would read 'cold' forever and re-warm online on EVERY call. The marker
    makes warm one-time-per-TTL (a missing ref / cold metadata rides the same signal) and
    self-heals each TTL. A pre-existing series cache with no marker reads cold once, then
    the first warm stamps it."""
    try:
        age = time.time() - (cache_dir / _WARM_MARKER).stat().st_mtime
    except OSError:
        return True  # no marker yet → never warmed → cold
    return age > _WARM_TTL_SECONDS


def warm_cache(
    book_tickers: Iterable[str],
    cache_dir: Path | None,
    *,
    extra_tickers: Iterable[str] = (),
    online: bool = True,
) -> dict[str, int]:
    """Pre-fetch the offline cache so the brief / backtest / MCP tools work without network.

    Always warms the book tickers + the benchmark references (the 'core' set). `extra_tickers`
    (the ~375-ETF universe, for `--warm full`) is layered on for offline discovery. Fetches
    ~10y of daily history for every ticker, plus latest + splits for the book (held positions
    need a spot price) and metadata for book + extras (the securities / screen panels). Reuses
    the named price/metadata adapters, so a fresh cache is left untouched — re-running fills
    only the gaps. Returns counts including `book_total`/`book_missing` (YOUR holdings
    specifically) so a caller can tell a book-wide failure from a stray reference miss."""
    book = sorted({tk for tk in book_tickers if tk != CASH_TICKER})
    extras = sorted({tk for tk in extra_tickers if tk != CASH_TICKER})
    series_tickers = sorted(set(book) | set(benchmark_ref_tickers()) | set(extras))

    today = date.today()
    start = today - timedelta(days=HISTORY_DAYS)
    series = fetch_series(series_tickers, start, today, cache_dir=cache_dir, online=online)
    if book:  # held positions need a spot price + split history; refs/universe need only series
        fetch_latest(book, cache_dir=cache_dir, online=online)
        fetch_splits(book, cache_dir=cache_dir, online=online)
    meta_tickers = sorted(set(book) | set(extras))  # refs are price-only, no metadata needed
    meta = (
        fetch_metadata(meta_tickers, cache_dir=cache_dir, online=online)
        if meta_tickers
        else None
    )

    cdir = ensure_cache_dir(cache_dir)
    if cdir is not None:
        (cdir / _WARM_MARKER).touch()  # record the attempt → throttles the cold-call re-warm

    counts = {
        "tickers": len(series_tickers),
        "series_missing": len(series.missing),
        "book_total": len(book),
        "book_missing": len(set(book) & set(series.missing)),
        "meta_missing": len(meta.missing) if meta is not None else 0,
    }
    log.info(
        "warm_cache: %d tickers (%d series missing, %d metadata missing)",
        counts["tickers"], counts["series_missing"], counts["meta_missing"],
    )
    return counts
