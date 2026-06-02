"""Portfolio returns: money-weighted (true IRR) + Modified Dietz (approx TWR).

Money-weighted return (MWR):
    Annualized IRR over the user's cash flows + current portfolio value.
    Solved by Newton-Raphson on the XIRR equation. Returns None if Newton
    does not converge — better silence-free than a fabricated number.

Modified Dietz return:
    Approximation of time-weighted return needing only the cash flows and
    start/end portfolio values. Returned as a period rate; use
    `annualize_return` to express per-year. Returns None when the period is
    degenerate (e.g. net withdrawals so large that the weighted-contribution
    denominator is non-positive — Modified Dietz cannot represent that case).

Sign convention:
    cash_flow < 0  → user paid out  (a buy, a fee)
    cash_flow > 0  → user received  (sell proceeds, dividend, interest)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

import pandas as pd

from app.events import Event

log = logging.getLogger(__name__)

_DAYS_PER_YEAR = 365.25
_TRADING_DAYS_PER_YEAR = 252.0  # matches empyrical's daily annualization basis
_VALUE_DUST = 1e-6  # portfolio values below this are treated as "no position"

# Annualizing a return over a very short window explodes nonsensically
# (e.g. 21% over 2 days → millions of % per year). Below these floors we
# refuse to annualize and return None so the report shows "n/a".
_MIN_ANNUALIZE_DAYS = 30      # calendar-day floor for MWR / Modified Dietz
_MIN_ANNUALIZE_OBS = 20       # trading-day-observation floor for true TWR


class IRRError(ValueError):
    """Newton-Raphson failed to converge inside the iteration cap."""


@dataclass(frozen=True)
class CashFlow:
    date: date
    amount: float  # +inflow to user / -outflow from user


@dataclass(frozen=True)
class ReturnsSummary:
    """Per-period return numbers the report needs. Optional fields are None
    when the underlying computation was degenerate (so the report can render
    `n/a` instead of silently printing a fabricated percentage)."""

    period_start: date
    asof_date: date
    money_weighted_annualized: float | None  # None if Newton didn't converge
    modified_dietz_annualized: float | None  # None if denominator was degenerate

    @property
    def period_days(self) -> int:
        return (self.asof_date - self.period_start).days


def cash_flows_from_events(events: list[Event]) -> list[CashFlow]:
    """Translate the transaction log into signed cash flows.

    Buy/fee = outflow (negative). Sell/dividend/interest = inflow (positive).
    """
    cfs: list[CashFlow] = []
    for ev in events:
        if ev.action == "buy":
            cfs.append(CashFlow(ev.date, -(ev.quantity * ev.price + ev.fee)))
        elif ev.action == "sell":
            cfs.append(CashFlow(ev.date, ev.quantity * ev.price - ev.fee))
        elif ev.action in ("dividend", "interest"):
            cfs.append(CashFlow(ev.date, ev.cash))
        elif ev.action == "fee":
            cfs.append(CashFlow(ev.date, -ev.fee))
    return cfs


def money_weighted_return(
    events: list[Event],
    current_value: float,
    asof_date: date,
) -> float | None:
    """Annualized IRR over event cash flows + current portfolio value.

    Returns 0.0 when there are no events. Returns None if all flows share a
    sign (no real IRR exists) or if Newton-Raphson fails to converge.
    """
    cfs = cash_flows_from_events(events)
    return _mwr_from_cfs(cfs, current_value, asof_date)


def _mwr_from_cfs(
    cfs: list[CashFlow],
    current_value: float,
    asof_date: date,
) -> float | None:
    if not cfs:
        return 0.0
    cfs_full = [*cfs, CashFlow(asof_date, current_value)]
    t0 = min(cf.date for cf in cfs_full)
    span_days = (asof_date - t0).days
    if span_days < _MIN_ANNUALIZE_DAYS:
        # Annualizing an IRR over a sub-month window is meaningless.
        return None
    pairs = [((cf.date - t0).days / _DAYS_PER_YEAR, cf.amount) for cf in cfs_full]
    if all(amt <= 0 for _, amt in pairs) or all(amt >= 0 for _, amt in pairs):
        # IRR has no real solution when all flows share a sign.
        return None
    try:
        return _xirr_newton(pairs)
    except IRRError as exc:
        log.warning("MWR did not converge: %s", exc)
        return None


def modified_dietz_return(
    events: list[Event],
    current_value: float,
    asof_date: date,
) -> float | None:
    """Modified Dietz period return (NOT annualized — feed to `annualize_return`).

    Returns None when the weighted-contributions denominator is non-positive
    (which happens when net withdrawals + their timing make the formula
    nonsensical). Returns 0.0 on an empty or degenerate period.
    """
    return _md_from_cfs(cash_flows_from_events(events), current_value, asof_date)


def _md_from_cfs(
    cfs: list[CashFlow],
    current_value: float,
    asof_date: date,
) -> float | None:
    if not cfs:
        return 0.0
    t0 = min(cf.date for cf in cfs)
    period_days = (asof_date - t0).days
    if period_days <= 0:
        return 0.0
    net_in = -sum(cf.amount for cf in cfs)  # cash IN to portfolio (sign flip)
    weighted_in = sum(
        -cf.amount * ((period_days - (cf.date - t0).days) / period_days) for cf in cfs
    )
    if weighted_in <= 0.0:
        # Net withdrawals dominate; Modified Dietz cannot honestly represent.
        return None
    return (current_value - net_in) / weighted_in


def annualize_return(period_return: float | None, days: int) -> float | None:
    """Convert a period return to an annualized rate.

    Returns None when the input is None or the window is shorter than
    `_MIN_ANNUALIZE_DAYS` (annualizing a sub-month return explodes nonsensically).
    """
    if period_return is None:
        return None
    if days < _MIN_ANNUALIZE_DAYS:
        return None
    years = days / _DAYS_PER_YEAR
    if 1.0 + period_return <= 0.0:
        # Total loss; annualized is -100%.
        return -1.0
    return float((1.0 + period_return) ** (1.0 / years) - 1.0)


def summarize(
    events: list[Event],
    current_value: float,
    asof_date: date,
) -> ReturnsSummary:
    """Compute the per-period numbers the report layer renders.

    Computes cash flows ONCE and shares them between MWR and Modified Dietz.
    """
    if not events:
        return ReturnsSummary(asof_date, asof_date, 0.0, 0.0)
    cfs = cash_flows_from_events(events)
    period_start = min(ev.date for ev in events)
    period_days = (asof_date - period_start).days
    mwr = _mwr_from_cfs(cfs, current_value, asof_date)
    md_period = _md_from_cfs(cfs, current_value, asof_date)
    md_ann = annualize_return(md_period, period_days)
    return ReturnsSummary(
        period_start=period_start,
        asof_date=asof_date,
        money_weighted_annualized=mwr,
        modified_dietz_annualized=md_ann,
    )


def build_daily_returns(
    events: list[Event],
    series_by_ticker: dict[str, "pd.Series[float]"],
    asof_date: date,
) -> "pd.Series[float]":
    """Reconstruct the daily time-weighted return series from the log + prices.

    Time-weighted: external cash flows (buys/sells) are neutralized so the
    series reflects investment performance, not contribution timing. Dividend
    and interest income is added back as return the holdings earned.

    Daily return on day d (for days where the prior value exceeds dust):
        gain(d) = V(d) - V(d-1) - buy_cost(d) + sell_proceeds(d) + income(d)
        r(d)    = gain(d) / V(d-1)

    where V(d) is the market value of replayed holdings at that day's close.
    Returns an empty Series if there is no priced history.

    Only events for tickers present in `series_by_ticker` are used: a ticker
    with no price history contributes neither value NOR cash flow, so the TWR
    is computed consistently over the *priced* sub-portfolio. (A ticker whose
    buy cost was counted but whose value was absent would corrupt the curve.)
    """
    if not events or not series_by_ticker:
        return pd.Series(dtype=float)

    # Drop events whose ticker has no price series (keeps cash flows and market
    # value over the same ticker set — see docstring).
    events = [ev for ev in events if ev.ticker in series_by_ticker]
    if not events:
        return pd.Series(dtype=float)

    start = min(ev.date for ev in events)
    # Master trading-day index = union of all price dates within [start, asof].
    idx = pd.DatetimeIndex([])
    for s in series_by_ticker.values():
        idx = idx.union(s.index)
    lo, hi = pd.Timestamp(start), pd.Timestamp(asof_date)
    idx = idx[(idx >= lo) & (idx <= hi)].sort_values()
    if len(idx) == 0:
        return pd.Series(dtype=float)

    # Per-ticker: cumulative shares timeline + price, aligned to the master index.
    shares_by_day: dict[str, pd.Series[float]] = {}
    price_by_day: dict[str, pd.Series[float]] = {}
    for tk, series in series_by_ticker.items():
        deltas: dict[pd.Timestamp, float] = defaultdict(float)
        for ev in events:
            if ev.ticker != tk:
                continue
            if ev.action == "buy":
                deltas[pd.Timestamp(ev.date)] += ev.quantity
            elif ev.action == "sell":
                deltas[pd.Timestamp(ev.date)] -= ev.quantity
        step = pd.Series(deltas, dtype=float).sort_index().cumsum()
        shares_by_day[tk] = step.reindex(idx, method="ffill").fillna(0.0)
        price_by_day[tk] = series.reindex(idx, method="ffill")

    # Market value per day = sum_ticker shares(d) * price(d).
    value = pd.Series(0.0, index=idx)
    for tk in series_by_ticker:
        value = value.add(shares_by_day[tk] * price_by_day[tk], fill_value=0.0)

    # Per-day external flows + income, keyed to the master index.
    buy_cost = pd.Series(0.0, index=idx)
    sell_proceeds = pd.Series(0.0, index=idx)
    income = pd.Series(0.0, index=idx)
    for ev in events:
        d = pd.Timestamp(ev.date)
        if d not in idx:
            # Snap an off-calendar event date to the next trading day; if the
            # event falls after the last trading day (e.g. a Saturday asof with
            # a Saturday-dated dividend), clamp to the last day so its cash is
            # not silently dropped.
            later = idx[idx >= d]
            d = later[0] if len(later) > 0 else idx[-1]
        if ev.action == "buy":
            buy_cost[d] += ev.quantity * ev.price + ev.fee
        elif ev.action == "sell":
            sell_proceeds[d] += ev.quantity * ev.price - ev.fee
        elif ev.action in ("dividend", "interest"):
            income[d] += ev.cash
        elif ev.action == "fee":
            income[d] -= ev.fee

    prev_value = value.shift(1)
    gain = value - prev_value - buy_cost + sell_proceeds + income
    daily = gain / prev_value
    # Only keep days where the prior value was a real position.
    daily = daily[prev_value > _VALUE_DUST]
    return daily.dropna()


def twr_index(daily_returns: "pd.Series[float]") -> "pd.Series[float]":
    """Growth-of-1 index from a daily return series (starts at 1.0)."""
    if daily_returns.empty:
        return pd.Series(dtype=float)
    return (1.0 + daily_returns).cumprod()


def true_twr_annualized(daily_returns: "pd.Series[float]") -> float | None:
    """Annualized true time-weighted return from the daily return series.

    Annualized on a **252-trading-day basis** using the COUNT of return
    observations — the same clock empyrical uses for Sharpe/Sortino/Calmar, so
    return and risk in the report are comparable. Counting observations (not
    calendar span) also means a cash gap mid-history neither dilutes nor
    inflates the figure.

    Returns None when there are fewer than `_MIN_ANNUALIZE_OBS` return days
    (annualizing a sub-month window explodes nonsensically).
    """
    n = len(daily_returns)
    if daily_returns.empty or n < _MIN_ANNUALIZE_OBS:
        return None
    total_growth = float((1.0 + daily_returns).prod())
    if total_growth <= 0.0:
        return None
    return float(total_growth ** (_TRADING_DAYS_PER_YEAR / n) - 1.0)


def _xirr_newton(pairs: list[tuple[float, float]], guess: float = 0.1) -> float:
    """Solve sum(amt / (1+r)^t) = 0 by Newton-Raphson.

    `pairs` is a list of (years_from_first_flow, amount). Raises `IRRError`
    if the solver does not converge within the iteration cap. Keeps `1+r`
    strictly positive at the top of every iteration so f and df are always
    computed against a consistent base (fixes the mid-iteration mix bug).
    """
    r: float = guess
    for _ in range(100):
        # Hold r in the safe (1+r > 0) region BEFORE summing f and df,
        # so every pair in the iteration uses the same base.
        if 1.0 + r <= 0.0:
            r = -0.999
        base: float = 1.0 + r
        f: float = 0.0
        df: float = 0.0
        for t, amt in pairs:
            pv = float(amt / base**t)
            f += pv
            df += -t * pv / base
        if abs(df) < 1e-14:
            break
        r_next: float = r - f / df
        if abs(r_next - r) < 1e-10:
            return r_next
        r = r_next
    msg = f"Newton-Raphson did not converge within 100 iterations (last r={r:.4f})"
    raise IRRError(msg)
