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
    # True time-weighted return (252-basis), computed from the daily-return
    # series in the composition root and folded in here. A plain value (no CI):
    # the bootstrapped-metric convention (MetricCI triples) is for risk.py only.
    true_twr_annualized: float | None = None

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
            cfs.append(CashFlow(ev.date, ev.cash - ev.fee))  # net of any withholding fee
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
    *,
    true_twr: float | None = None,
    fully_priced: bool = True,
) -> ReturnsSummary:
    """Compute the per-period numbers the report layer renders.

    Computes cash flows ONCE and shares them between MWR and Modified Dietz.
    `true_twr` is computed upstream from the daily-return series (it needs the
    price history this function doesn't see) and folded into the summary so the
    report layer reads a single object instead of a side parameter.

    `fully_priced` MUST be False when any held ticker lacks a usable price:
    money-weighted returns (MWR, Modified Dietz) need the *whole* portfolio's
    current value, and `current_value` here sums only the priced holdings — so a
    partial book would yield a confidently-wrong figure. We return None for both
    (rendered `n/a`) rather than understate. (true TWR is a priced-subset, time-
    weighted measure and is left as-is.)
    """
    if not events:
        return ReturnsSummary(asof_date, asof_date, 0.0, 0.0, true_twr_annualized=true_twr)
    # Period starts at the first *investment* event, not a funding deposit that may
    # precede the first buy — an idle cash gap would lengthen the annualization
    # window with no offsetting flow and dilute the money-weighted figures. (This
    # matches the cash-flow window Modified Dietz / MWR are actually computed over,
    # since deposits/withdrawals are not cash flows there.)
    invested = [ev.date for ev in events if ev.action not in ("deposit", "withdraw")]
    period_start = min(invested) if invested else min(ev.date for ev in events)
    if not fully_priced:
        return ReturnsSummary(period_start, asof_date, None, None, true_twr_annualized=true_twr)
    cfs = cash_flows_from_events(events)
    period_days = (asof_date - period_start).days
    mwr = _mwr_from_cfs(cfs, current_value, asof_date)
    md_period = _md_from_cfs(cfs, current_value, asof_date)
    md_ann = annualize_return(md_period, period_days)
    return ReturnsSummary(
        period_start=period_start,
        asof_date=asof_date,
        money_weighted_annualized=mwr,
        modified_dietz_annualized=md_ann,
        true_twr_annualized=true_twr,
    )


def _snap_to_index(d: date, idx: "pd.DatetimeIndex") -> "pd.Timestamp":
    """Map an *income* event date (dividend/interest/fee) onto the trading day its
    cash lands on: the first index day on/after the date, clamped to idx[-1] so a
    late dividend's cash is never dropped. Income has no share/value counterpart, so
    clamping it only shifts a constant on the cumulative curve — it cannot fabricate
    a return the way a clamped *buy* would (the buy's cash would land with no
    matching shares). A date before idx[0] snaps to idx[0] for the same reason.

    Share-bearing events (buy/sell) instead use `_share_index_day`, which DROPS a
    trade past the priced window to stay consistent with value_curve.
    """
    ts = pd.Timestamp(d)
    if ts in idx:
        return ts
    later = idx[idx >= ts]
    return later[0] if len(later) > 0 else idx[-1]


def _share_index_day(d: date, idx: "pd.DatetimeIndex") -> "pd.Timestamp | None":
    """Trading day a *share-bearing* event (buy/sell) dated ``d`` lands on: the
    first index day on/after ``d`` (a date before idx[0] snaps to idx[0]).

    Returns None when ``d`` is past the last index day — the position can't be
    valued yet (no price reaches that far), so the trade is dropped from BOTH the
    value curve and the flow series. This single placement rule is shared by
    value_curve, build_daily_returns, and pnl_curve, so the share side (shares
    added) and the flow side (cash subtracted) can never disagree about where a
    trade lands: a buy whose cost is counted but whose shares are not would
    fabricate a phantom ~-100% day and poison the whole risk panel.
    """
    later = idx[idx >= pd.Timestamp(d)]
    return later[0] if len(later) > 0 else None


def value_curve(
    events: list[Event],
    series_by_ticker: dict[str, "pd.Series[float]"],
    asof_date: date,
) -> "pd.Series[float]":
    """Daily portfolio market value Σ shares(d)×price(d) over the priced master
    index — contributions included (that day's buys are in shares(d)).

    The shared basis for the time-weighted return series AND the felt dollar
    drawdown; the cli builds it once and passes it to both. Only tickers present in
    `series_by_ticker` contribute (a ticker priced-but-absent, or absent-but-traded,
    would corrupt the curve), so the value is over the *priced* sub-portfolio. Empty
    Series if no priced history.
    """
    if not events or not series_by_ticker:
        return pd.Series(dtype=float)
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

    value = pd.Series(0.0, index=idx)
    for tk, series in series_by_ticker.items():
        deltas: dict[pd.Timestamp, float] = defaultdict(float)
        for ev in events:
            if ev.ticker != tk or ev.action not in ("buy", "sell"):
                continue
            day = _share_index_day(ev.date, idx)
            if day is None:
                continue  # trade past the priced window → not valuable yet (dropped on the flow side too)
            deltas[day] += ev.quantity if ev.action == "buy" else -ev.quantity
        if deltas:
            shares = pd.Series(deltas, dtype=float).sort_index().cumsum().reindex(idx, method="ffill").fillna(0.0)
        else:
            shares = pd.Series(0.0, index=idx)  # priced but never bought/sold → 0 shares
        value = value.add(shares * series.reindex(idx, method="ffill"), fill_value=0.0)
    return value


def _cash_delta(ev: Event) -> float:
    """Signed USD effect of one event on the cash balance."""
    if ev.action == "deposit":
        return ev.cash
    if ev.action == "withdraw":
        return -ev.cash
    if ev.action == "buy":
        return -(ev.quantity * ev.price + ev.fee)
    if ev.action == "sell":
        return ev.quantity * ev.price - ev.fee
    if ev.action in ("dividend", "interest"):
        return ev.cash - ev.fee
    if ev.action == "fee":
        return -ev.fee
    return 0.0


def pnl_curve(
    events: list[Event],
    series_by_ticker: dict[str, "pd.Series[float]"],
    asof_date: date,
    *,
    value: "pd.Series[float] | None" = None,
) -> "pd.Series[float]":
    """Daily cumulative market P&L in dollars = holdings value + cumulative
    investment cash flows (sells − buys + income − fees), with **external** flows
    (deposits/withdrawals) excluded.

    This is flow-neutral: funding and broker transfers cancel out (they move both
    the balance and the cost base equally), so a drawdown of this curve is the
    real 'dollars of profit given back' — undistorted by trades OR transfers, the
    confounds that wreck the raw value / account-value curves. Computed over the
    same priced-securities universe as the time-weighted return (`build_daily_
    returns`): only tickers with a price series count, so CASH-tickered income
    (broker interest on idle cash) is excluded here exactly as it is from TWR —
    it's a cash earning, not a market gain, and still shows in Net P&L / MWR.
    Empty if no priced history.
    """
    holdings = value_curve(events, series_by_ticker, asof_date) if value is None else value
    if holdings.empty:
        return pd.Series(dtype=float)
    idx = holdings.index
    flows = pd.Series(0.0, index=idx)
    for ev in events:
        if ev.action in ("deposit", "withdraw"):
            continue  # external cash flow → cancels in P&L (raises balance AND cost base)
        if ev.ticker not in series_by_ticker:
            continue  # priced securities only (matches value_curve / TWR; drops CASH)
        if ev.action in ("buy", "sell"):
            day = _share_index_day(ev.date, idx)
            if day is None:
                continue  # trade past the priced window → its shares are dropped from value_curve too
        else:
            day = _snap_to_index(ev.date, idx)  # income: clamp a late date onto idx[-1] (keep the cash)
        flows[day] += _cash_delta(ev)
    return holdings.add(flows.cumsum(), fill_value=0.0)


def build_daily_returns(
    events: list[Event],
    series_by_ticker: dict[str, "pd.Series[float]"],
    asof_date: date,
    *,
    value: "pd.Series[float] | None" = None,
) -> "pd.Series[float]":
    """Reconstruct the daily time-weighted return series from the log + prices.

    Time-weighted: external cash flows (buys/sells) are neutralized so the series
    reflects investment performance, not contribution timing. Dividend/interest
    income is added back as return the holdings earned.

        gain(d) = V(d) - V(d-1) - buy_cost(d) + sell_proceeds(d) + income(d)
        r(d)    = gain(d) / V(d-1)        (only where V(d-1) > dust)

    V(d) is `value_curve` (the priced sub-portfolio's market value); the cli passes
    it in precomputed (shared with the dollar drawdown). Empty Series if there is no
    priced history.
    """
    value = value_curve(events, series_by_ticker, asof_date) if value is None else value
    if value.empty:
        return pd.Series(dtype=float)
    idx = value.index
    priced = [ev for ev in events if ev.ticker in series_by_ticker]

    # Per-day external flows + income, keyed to the master index.
    buy_cost = pd.Series(0.0, index=idx)
    sell_proceeds = pd.Series(0.0, index=idx)
    income = pd.Series(0.0, index=idx)
    for ev in priced:
        if ev.action in ("buy", "sell"):
            day = _share_index_day(ev.date, idx)
            if day is None:
                continue  # trade past the priced window → its shares are dropped from value_curve too
            if ev.action == "buy":
                buy_cost[day] += ev.quantity * ev.price + ev.fee
            else:
                sell_proceeds[day] += ev.quantity * ev.price - ev.fee
        else:
            d = _snap_to_index(ev.date, idx)  # income: clamp a late date onto idx[-1] (keep the cash)
            if ev.action in ("dividend", "interest"):
                income[d] += ev.cash - ev.fee  # net of any withholding fee
            elif ev.action == "fee":
                income[d] -= ev.fee

    prev_value = value.shift(1)
    gain = value - prev_value - buy_cost + sell_proceeds + income
    daily = gain / prev_value
    daily = daily[prev_value > _VALUE_DUST]  # only where prior value was a real position
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


def price_basis_mismatches(
    events: list[Event],
    series_by_ticker: dict[str, "pd.Series[float]"],
    *,
    factor: float = 2.0,
) -> list[str]:
    """Tickers whose trade execution price disagrees with the price *history* by
    more than `factor`× on the trade date — the fingerprint of an unhandled stock
    split.

    The log records **raw** share counts and execution prices; yfinance closes
    are **split-adjusted**. So for a ticker that split during its holding period,
    shares × price is inconsistent across the split and `build_daily_returns`
    fabricates a return (e.g. a 10:1 split shows the buy at ~10× the adjusted
    close → a spurious +900% day). A clean ≥2:1 split leaves a ratio far beyond
    any intraday fill-vs-close gap, so a generous `factor` flags splits without
    false-positiving on normal moves. The caller excludes the flagged ticker from
    the time-weighted series (TWR + risk) — the honest stopgap until v1.x adjusts
    share counts for corporate actions. Buys and sells carry an execution price;
    dividend/fee rows do not. Returns a sorted list (empty when nothing suspect).
    Pure.
    """
    suspect: set[str] = set()
    for ev in events:
        if ev.action not in ("buy", "sell") or ev.price <= 0:
            continue
        s = series_by_ticker.get(ev.ticker)
        if s is None or s.empty:
            continue
        at_or_before = s.index[s.index <= pd.Timestamp(ev.date)]
        if len(at_or_before) == 0:
            continue
        close = float(s.loc[at_or_before[-1]])
        if close <= 0:
            continue
        ratio = ev.price / close
        if ratio >= factor or ratio <= 1.0 / factor:
            suspect.add(ev.ticker)
    return sorted(suspect)


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
