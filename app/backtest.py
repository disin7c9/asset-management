"""Backtest harness (v1): notional historical simulation of a target allocation,
rebalanced on a schedule vs. buy-and-hold.

**Notional.** Starts a round `initial` ($10k) at the target weights on the first
date every target ticker has a price, then walks the daily closes forward. This
isolates the *strategy* (target + rebalance policy) from the user's actual
contribution timing — a clean lab for "is this allocation + rebalance rule worth
following?".

**Honesty / walk-forward.** A *fixed* rebalance policy fits no parameters, so the
whole history is an out-of-sample-clean simulation (nothing was optimized → there
is nothing to overfit). The walk-forward train/test *selection* machinery is only
needed once a strategy **searches** — tunes parameters or picks among candidates
(an optimizer, or an *edge* strategy). That is deferred; the
`strategy.may_suggest` gate keeps any future edge strategy from suggesting until a
walk-forward backtest validates it.

Pure compute over price Series. Composes `risk` + `returns` (a Layer-2 module that
uses its Layer-2 siblings — acyclic: neither imports back).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from app.returns import true_twr_annualized, twr_index
from app.risk import RiskSummary, summarize_risk

INITIAL_CAPITAL = 10_000.0
SCHEDULES: tuple[str, ...] = ("never", "monthly", "quarterly", "annually")


@dataclass(frozen=True)
class BacktestLeg:
    """One simulated path (rebalanced, or buy-and-hold)."""

    label: str
    annualized_return: float | None  # true TWR, 252-basis; None if window too short
    final_value: float
    risk: RiskSummary


@dataclass(frozen=True)
class BacktestResult:
    start: date
    end: date
    initial: float
    schedule: str
    legs: tuple[BacktestLeg, ...]  # (rebalanced, buy_and_hold)
    missing: tuple[str, ...]       # target tickers with no usable price history


def _period_key(ts: pd.Timestamp, schedule: str) -> object:
    if schedule == "monthly":
        return (ts.year, ts.month)
    if schedule == "quarterly":
        return (ts.year, (ts.month - 1) // 3)
    return ts.year  # annually


def _rebalance_dates(index: pd.DatetimeIndex, schedule: str) -> set[pd.Timestamp]:
    """First trading day of each period (never the very first day — already allocated)."""
    if schedule == "never" or len(index) == 0:
        return set()
    out: set[pd.Timestamp] = set()
    seen: set[object] = set()
    for ts in index:
        k = _period_key(ts, schedule)
        if k not in seen:
            seen.add(k)
            out.add(ts)
    out.discard(index[0])
    return out


def _priced_tickers(series: dict[str, "pd.Series[float]"], target: dict[str, float]) -> list[str]:
    return [
        tk for tk in target
        if tk in series and series[tk].first_valid_index() is not None
    ]


def simulate(
    series: dict[str, "pd.Series[float]"],
    target: dict[str, float],
    *,
    schedule: str,
    initial: float = INITIAL_CAPITAL,
    start: date | None = None,
    end: date | None = None,
) -> "pd.Series[float]":
    """Daily equity curve of `initial` allocated to `target` and rebalanced on
    `schedule` ('never' = buy-and-hold). Empty Series if no priced target ticker.

    Rebalancing is value-preserving (no cash in/out): on a rebalance day the
    holdings are reset to the target weights at that day's prices.
    """
    tickers = _priced_tickers(series, target)
    if not tickers:
        return pd.Series(dtype=float)

    idx = pd.DatetimeIndex([])
    for tk in tickers:
        idx = idx.union(series[tk].index)
    lo = max(series[tk].first_valid_index() for tk in tickers)
    if start is not None:
        lo = max(lo, pd.Timestamp(start))
    hi = pd.Timestamp(end) if end is not None else idx.max()
    idx = idx[(idx >= lo) & (idx <= hi)].sort_values()
    if len(idx) == 0:
        return pd.Series(dtype=float)

    px = {tk: series[tk].reindex(idx, method="ffill") for tk in tickers}
    p0 = {tk: float(px[tk].iloc[0]) for tk in tickers}
    # Drop tickers without a positive starting price (bad/placeholder data); a 0
    # would divide by zero when sizing shares. Renormalize over what's left, and
    # bail if nothing usable remains (degenerate target → caller skips gracefully).
    tickers = [tk for tk in tickers if p0[tk] > 0]
    wsum = sum(target[tk] for tk in tickers)
    if not tickers or wsum <= 0:
        return pd.Series(dtype=float)
    weight = {tk: target[tk] / wsum for tk in tickers}
    reb = _rebalance_dates(idx, schedule)

    shares = {tk: weight[tk] * initial / p0[tk] for tk in tickers}
    values: list[float] = []
    for ts in idx:
        price = {tk: float(px[tk].loc[ts]) for tk in tickers}
        value = sum(shares[tk] * price[tk] for tk in tickers)
        values.append(value)
        if ts in reb and value > 0:
            # Guard a (rare) zero price mid-history: keep that ticker's shares.
            shares = {
                tk: (weight[tk] * value / price[tk] if price[tk] > 0 else shares[tk])
                for tk in tickers
            }
    return pd.Series(values, index=idx, dtype=float)


def _leg(label: str, curve: "pd.Series[float]", *, bootstrap_n: int, seed: int) -> BacktestLeg | None:
    daily = curve.pct_change().dropna()
    risk = summarize_risk(daily, twr_index(daily), bootstrap_n=bootstrap_n, seed=seed)
    if risk is None:  # too few return-days to compute a risk panel
        return None
    return BacktestLeg(label, true_twr_annualized(daily), float(curve.iloc[-1]), risk)


def backtest_compare(
    series: dict[str, "pd.Series[float]"],
    target: dict[str, float],
    *,
    schedule: str = "quarterly",
    initial: float = INITIAL_CAPITAL,
    start: date | None = None,
    end: date | None = None,
    bootstrap_n: int = 1000,
    seed: int = 42,
) -> BacktestResult | None:
    """Simulate rebalanced-to-target vs buy-and-hold and bundle both legs with
    their drawdown-first risk + return. Returns None if there is no usable price
    history for any target ticker."""
    priced = _priced_tickers(series, target)
    missing = tuple(sorted(tk for tk in target if tk not in priced))
    if not priced:
        return None

    rebalanced = simulate(series, target, schedule=schedule, initial=initial, start=start, end=end)
    buyhold = simulate(series, target, schedule="never", initial=initial, start=start, end=end)
    if rebalanced.empty or buyhold.empty:
        return None

    reb = _leg(f"rebalanced ({schedule})", rebalanced, bootstrap_n=bootstrap_n, seed=seed)
    bh = _leg("buy & hold", buyhold, bootstrap_n=bootstrap_n, seed=seed)
    if reb is None or bh is None:  # window too short to score either leg
        return None

    return BacktestResult(
        start=rebalanced.index[0].date(),
        end=rebalanced.index[-1].date(),
        initial=initial,
        schedule=schedule,
        legs=(reb, bh),
        missing=missing,
    )
