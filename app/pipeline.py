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
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import date
    from pathlib import Path

    import pandas as pd

from app.corporate_actions import adjust_for_splits
from app.derive import DerivedState, derive
from app.events import CASH_TICKER, Event, load_events
from app.prices import (
    PriceRow,
    PricesResult,
    SeriesResult,
    fetch_latest,
    fetch_series,
    fetch_splits,
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


def load_book(
    csv_path: Path, cache_dir: Path, *, online: bool
) -> tuple[list[Event], DerivedState]:
    """Replay a transaction-log CSV → split-adjusted events + derived holdings.

    The shared book loader for every surface (the CLI brief and the MCP server).
    Split-adjust raw share counts (yfinance prices are split-adjusted) so holdings
    AND the time-weighted series share one basis — the real fix for the NVDA-style
    corruption; splits are fetched cache-only when `online` is False, and the
    price-basis-mismatch guard in `compute_prices_returns_risk` stays the net for
    any split we couldn't fetch. The tickers we actually adjusted are logged.
    """
    events = load_events(csv_path)
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


def held_market_value(
    state: DerivedState, prices: dict[str, PriceRow]
) -> dict[str, float]:
    """Per-held-ticker market value (shares × close), priced tickers only.

    The single source of held-value math (holdings sizing, returns, suggestions,
    and --dump-target all go through here), so a future valuation change lands in
    one place. A ticker absent from `prices` or with a non-positive close is
    omitted (no usable price)."""
    held = state.held()
    return {
        tk: held[tk].shares * prices[tk].close
        for tk in held
        if tk in prices and prices[tk].close > 0
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
            # Guard against unhandled stock splits: the log holds raw share counts
            # but yfinance prices are split-adjusted, so a ticker that split during
            # its holding period would fabricate a return in the time-weighted
            # series. Exclude such a ticker from TWR + risk (and warn) — the honest
            # stopgap until v1.x adjusts share counts for corporate actions.
            twr_series = series.rows
            twr_excluded = price_basis_mismatches(events, series.rows)
            if twr_excluded:
                log.warning(
                    "excluding %s from TWR & risk: execution price disagrees with the "
                    "split-adjusted price history (likely an unhandled stock split). The "
                    "time-weighted series mixes raw share counts with adjusted prices and "
                    "would fabricate a return. MWR / Modified Dietz are unaffected; full "
                    "corporate-action handling is a v1.x item.",
                    ", ".join(twr_excluded),
                )
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
        # series' REAL provenance (cache/yfinance/stooq + true fetch time) — not a
        # fabricated "series"/now() stamp, so the footer's source + age are honest.
        for tk in held:
            s = series.rows.get(tk)
            if s is not None and not s.empty:
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
    # actually need. A ticker present in `prices` but with a non-positive or NaN
    # close is dropped from value, so it must count as unpriced here too; else a
    # partial book would be scored as fully priced and the money-weighted figures
    # would print a confident wrong number. Counters + partial-status are set here
    # once, for both branches (the series counts are recorded by the caller).
    priced_held = held_market_value(state, prices)
    prices = {tk: prices[tk] for tk in priced_held}  # drop non-positive/NaN quotes
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
