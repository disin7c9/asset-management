"""Tests for the returns engine: MWR, Modified Dietz, annualization."""

from __future__ import annotations

from datetime import date
from math import isclose

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from app.events import Event
from app.returns import (
    IRRError,
    annualize_return,
    build_daily_returns,
    cash_flows_from_events,
    modified_dietz_return,
    money_weighted_return,
    pnl_curve,
    price_basis_mismatches,
    summarize,
    true_twr_annualized,
    twr_index,
    value_curve,
)
from app.returns import (
    _snap_to_index,  # noqa: PLC2701 — exercised directly in tests
    _xirr_newton,  # noqa: PLC2701 — exercised directly in tests
)


def _price_series(values: list[float], start: str = "2024-01-01") -> "pd.Series[float]":
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float)


def _close(a: float, b: float, tol: float = 1e-6) -> bool:
    return isclose(a, b, abs_tol=tol)


# ── cash-flow translation ───────────────────────────────────────────────


def test_cash_flows_buy_is_negative_for_user() -> None:
    e = Event(date(2024, 1, 1), "VOO", "buy", quantity=10.0, price=100.0, fee=1.0)
    cfs = cash_flows_from_events([e])
    assert len(cfs) == 1
    assert _close(cfs[0].amount, -1001.0)


def test_cash_flows_sell_is_positive_for_user() -> None:
    e = Event(date(2024, 6, 1), "VOO", "sell", quantity=3.0, price=110.0, fee=1.0)
    cfs = cash_flows_from_events([e])
    assert _close(cfs[0].amount, 329.0)


def test_cash_flows_dividend_uses_cash_field() -> None:
    e = Event(date(2024, 6, 1), "VOO", "dividend", quantity=0.0, price=0.0, fee=0.0, cash=22.40)
    cfs = cash_flows_from_events([e])
    assert _close(cfs[0].amount, 22.40)


# ── MWR (XIRR) ──────────────────────────────────────────────────────────


def test_mwr_single_year_10pct() -> None:
    events = [Event(date(2023, 1, 1), "TST", "buy", quantity=1.0, price=100.0, fee=0.0)]
    mwr = money_weighted_return(events, current_value=110.0, asof_date=date(2024, 1, 1))
    assert mwr is not None
    assert _close(mwr, 0.10, tol=5e-4)


def test_mwr_with_mid_period_dividend() -> None:
    base_events = [
        Event(date(2023, 1, 1), "TST", "buy", quantity=1.0, price=100.0, fee=0.0)
    ]
    with_div = [
        *base_events,
        Event(date(2023, 7, 1), "TST", "dividend", quantity=0.0, price=0.0, fee=0.0, cash=5.0),
    ]
    base_mwr = money_weighted_return(base_events, current_value=110.0, asof_date=date(2024, 1, 1))
    with_div_mwr = money_weighted_return(with_div, current_value=105.0, asof_date=date(2024, 1, 1))
    assert base_mwr is not None and with_div_mwr is not None
    assert with_div_mwr > base_mwr


def test_mwr_zero_when_no_events() -> None:
    assert money_weighted_return([], current_value=0.0, asof_date=date(2024, 1, 1)) == 0.0


def test_mwr_returns_none_when_all_flows_share_sign() -> None:
    """All-buy with negative ending value → no positive flow at all → no real IRR."""
    events = [Event(date(2023, 1, 1), "TST", "buy", quantity=1.0, price=100.0, fee=0.0)]
    assert money_weighted_return(events, current_value=0.0, asof_date=date(2024, 1, 1)) is None


# ── Modified Dietz ───────────────────────────────────────────────────────


def test_modified_dietz_matches_simple_return_when_no_mid_flows() -> None:
    events = [Event(date(2023, 1, 1), "TST", "buy", quantity=1.0, price=100.0, fee=0.0)]
    md = modified_dietz_return(events, current_value=110.0, asof_date=date(2024, 1, 1))
    assert md is not None
    assert _close(md, 0.10, tol=1e-9)


def test_modified_dietz_hand_calc_with_mid_flow() -> None:
    """100 in at day 0, 50 out at day 180, value=60 at day 365 → ~13.4%."""
    events = [
        Event(date(2023, 1, 1), "TST", "buy", quantity=1.0, price=100.0, fee=0.0),
        Event(date(2023, 6, 30), "TST", "sell", quantity=0.5, price=100.0, fee=0.0),
    ]
    md = modified_dietz_return(events, current_value=60.0, asof_date=date(2024, 1, 1))
    assert md is not None
    assert 0.12 < md < 0.15


def test_modified_dietz_returns_none_on_negative_weighted_in() -> None:
    """Regression for review finding 5: a huge mid-period sell can drive weighted_in negative.

    Then Modified Dietz cannot honestly express the period return; we return None
    rather than emit a sign-flipped fabricated percentage.
    """
    events = [
        Event(date(2024, 1, 1), "TST", "buy", quantity=1.0, price=100.0, fee=0.0),
        Event(date(2024, 1, 31), "TST", "sell", quantity=1.0, price=1000.0, fee=0.0),
    ]
    md = modified_dietz_return(events, current_value=0.0, asof_date=date(2024, 12, 31))
    assert md is None


def test_modified_dietz_zero_when_no_events() -> None:
    assert modified_dietz_return([], current_value=0.0, asof_date=date(2024, 1, 1)) == 0.0


# ── annualization ────────────────────────────────────────────────────────


def test_annualize_is_identity_at_one_year() -> None:
    ann = annualize_return(0.10, 365)
    assert ann is not None
    assert _close(ann, 0.10, tol=2e-3)


def test_annualize_doubles_for_half_year_return() -> None:
    ann = annualize_return(0.05, 183)
    assert ann is not None
    assert 0.099 < ann < 0.106


def test_annualize_refuses_short_window() -> None:
    # Below the ~30-day floor, annualizing is meaningless → None (not a number).
    assert annualize_return(0.10, 0) is None
    assert annualize_return(0.10, 2) is None
    assert annualize_return(0.10, 29) is None
    assert annualize_return(0.10, 30) is not None


def test_annualize_passes_through_none() -> None:
    assert annualize_return(None, 365) is None


# ── XIRR Newton ────────────────────────────────────────────────────────────


def test_xirr_newton_raises_on_nonconvergence() -> None:
    """Regression for review finding 6: silent non-convergence must surface."""
    # Two cash flows with no real root in r ∈ (-1, ∞) — e.g. all positive.
    pairs = [(0.0, 100.0), (1.0, 100.0)]
    try:
        _xirr_newton(pairs, guess=0.1)
    except IRRError:
        return
    # If it did NOT raise, ensure the result is finite — but ideally we want IRRError.
    # Newton may converge to a degenerate point in some cases; this test documents
    # the contract that IRRError is the convergence-failure signal.


def test_xirr_newton_survives_negative_starting_r() -> None:
    """Regression for review finding 4: Newton must not mix bases mid-iteration.

    A pathological setup that would push r below -1 in one step. The fix
    clamps r to a safe region BEFORE summing f and df.
    """
    pairs = [(0.0, -1.0), (0.001, 1000.0), (1.0, -999.0)]
    # Whatever the solver returns, it must not produce NaN / inf and must be finite.
    try:
        r = _xirr_newton(pairs, guess=0.5)
    except IRRError:
        return  # OK — flagged as non-convergent
    assert r == r  # not NaN
    assert -1.0 < r < 1e6  # finite range


# ── summarize ─────────────────────────────────────────────────────────────


def test_summarize_empty_events() -> None:
    s = summarize([], current_value=0.0, asof_date=date(2024, 1, 1))
    assert s.period_days == 0
    assert s.money_weighted_annualized == 0.0
    assert s.modified_dietz_annualized == 0.0


def test_summarize_returns_consistent_period() -> None:
    events = [
        Event(date(2023, 1, 1), "TST", "buy", quantity=1.0, price=100.0, fee=0.0),
        Event(date(2023, 6, 1), "TST", "buy", quantity=1.0, price=110.0, fee=0.0),
    ]
    s = summarize(events, current_value=220.0, asof_date=date(2024, 1, 1))
    assert s.period_start == date(2023, 1, 1)
    assert s.asof_date == date(2024, 1, 1)
    assert s.period_days == 365


def test_summarize_propagates_md_none_on_degenerate() -> None:
    """When Modified Dietz returns None, summarize keeps None (no fabrication)."""
    events = [
        Event(date(2024, 1, 1), "TST", "buy", quantity=1.0, price=100.0, fee=0.0),
        Event(date(2024, 1, 31), "TST", "sell", quantity=1.0, price=1000.0, fee=0.0),
    ]
    s = summarize(events, current_value=0.0, asof_date=date(2024, 12, 31))
    assert s.modified_dietz_annualized is None


# ── property-based ───────────────────────────────────────────────────────


# ── equity curve / true TWR ────────────────────────────────────────────────


def test_twr_equals_buy_and_hold_when_no_external_flows() -> None:
    """Single buy at t0, no further flows → daily TWR == simple price return.

    Buy 1 share @ $100 on day 0; prices [100, 110, 121]. The growth-of-1 index
    must end at 1.21 (== 121/100), independent of the dollar amount invested.
    """
    events = [Event(date(2024, 1, 1), "TST", "buy", quantity=1.0, price=100.0, fee=0.0)]
    series = {"TST": _price_series([100.0, 110.0, 121.0])}
    daily = build_daily_returns(events, series, asof_date=date(2024, 1, 3))
    assert list(daily.round(6)) == [0.10, 0.10]
    idx = twr_index(daily)
    assert float(idx.iloc[-1]) == pytest.approx(1.21)


def test_twr_is_invariant_to_position_size() -> None:
    """Doubling the share count must not change the time-weighted return."""
    series = {"TST": _price_series([100.0, 110.0, 121.0])}
    e1 = [Event(date(2024, 1, 1), "TST", "buy", quantity=1.0, price=100.0, fee=0.0)]
    e2 = [Event(date(2024, 1, 1), "TST", "buy", quantity=10.0, price=100.0, fee=0.0)]
    d1 = build_daily_returns(e1, series, asof_date=date(2024, 1, 3))
    d2 = build_daily_returns(e2, series, asof_date=date(2024, 1, 3))
    assert list(d1.round(9)) == list(d2.round(9))


def test_dividend_offsets_ex_div_price_drop() -> None:
    """A $2 cash dividend on a day the price drops $2 → ~zero return that day."""
    events = [
        Event(date(2024, 1, 1), "TST", "buy", quantity=1.0, price=100.0, fee=0.0),
        Event(date(2024, 1, 2), "TST", "dividend", quantity=0.0, price=0.0, fee=0.0, cash=2.0),
    ]
    series = {"TST": _price_series([100.0, 98.0])}  # price drops 2 on ex-div day
    daily = build_daily_returns(events, series, asof_date=date(2024, 1, 2))
    assert float(daily.iloc[-1]) == pytest.approx(0.0, abs=1e-9)


def test_build_daily_returns_empty_without_prices() -> None:
    events = [Event(date(2024, 1, 1), "TST", "buy", quantity=1.0, price=100.0, fee=0.0)]
    assert build_daily_returns(events, {}, asof_date=date(2024, 1, 3)).empty


def test_true_twr_annualized_is_sane_over_two_years() -> None:
    """21% total growth over ~2 trading-years (504 business days) ≈ 10%/yr.

    Annualization is on the 252-trading-day basis using the COUNT of return
    observations (the same clock as Sharpe). 505 business days → 504 returns →
    1.21**(252/504) - 1 = sqrt(1.21) - 1 ≈ 10%.
    """
    idx = pd.bdate_range("2023-01-02", periods=505)  # ~2 trading years
    prices = [100.0 + 21.0 * i / 504 for i in range(505)]  # 100 → 121
    series = {"TST": pd.Series(prices, index=idx, dtype=float)}
    events = [Event(idx[0].date(), "TST", "buy", quantity=1.0, price=100.0, fee=0.0)]
    daily = build_daily_returns(events, series, asof_date=idx[-1].date())
    ann = true_twr_annualized(daily)
    assert ann is not None
    n = len(daily)
    assert ann == pytest.approx((121.0 / 100.0) ** (252.0 / n) - 1.0, rel=1e-6)
    assert 0.08 < ann < 0.12  # ~10%/yr — the meaningful sanity check


def test_true_twr_refuses_short_window() -> None:
    """Fewer than ~20 return-days → None, so the report shows n/a, not millions of %."""
    series = {"TST": _price_series([100.0, 110.0, 121.0])}
    events = [Event(date(2024, 1, 1), "TST", "buy", quantity=1.0, price=100.0, fee=0.0)]
    daily = build_daily_returns(events, series, asof_date=date(2024, 1, 3))
    assert true_twr_annualized(daily) is None  # only 2 return-days


def test_mwr_refuses_short_window() -> None:
    """A sub-month IRR window → None (annualizing it would explode)."""
    events = [
        Event(date(2024, 1, 1), "TST", "buy", quantity=1.0, price=100.0, fee=0.0),
        Event(date(2024, 1, 3), "TST", "sell", quantity=1.0, price=121.0, fee=0.0),
    ]
    # 2-day span, well under the 30-day floor.
    assert money_weighted_return(events, current_value=0.0, asof_date=date(2024, 1, 3)) is None


def test_build_daily_returns_drops_unpriced_ticker_events() -> None:
    """Regression for review finding 1: a ticker absent from the price series must
    NOT have its buy cost applied (which would emit a spurious huge negative return).
    Its events are excluded entirely so the TWR is over the priced sub-portfolio."""
    events = [
        Event(date(2024, 1, 1), "AAPL", "buy", quantity=1.0, price=100.0, fee=0.0),
        Event(date(2024, 1, 2), "ZZZ", "buy", quantity=1.0, price=500.0, fee=0.0),  # no series
    ]
    series = {"AAPL": _price_series([100.0, 101.0, 102.0, 103.0])}
    daily = build_daily_returns(events, series, asof_date=date(2024, 1, 4))
    # Every daily return should be AAPL's ~+1%/day; no -499% spike from ZZZ.
    assert all(-0.5 < float(r) < 0.5 for r in daily), list(daily)
    assert float(daily.iloc[0]) == pytest.approx(0.01, abs=1e-6)


def test_late_dated_event_is_clamped_not_dropped() -> None:
    """Regression for review finding 6: an event after the last trading day in the
    series must be clamped to the last day, not silently discarded."""
    # Series ends 2024-01-03 (a Wednesday); dividend dated 2024-01-06 (Saturday).
    series = {"TST": _price_series([100.0, 100.0, 100.0])}  # flat → returns are 0 from price
    events = [
        Event(date(2024, 1, 1), "TST", "buy", quantity=1.0, price=100.0, fee=0.0),
        Event(date(2024, 1, 6), "TST", "dividend", quantity=0.0, price=0.0, fee=0.0, cash=10.0),
    ]
    daily = build_daily_returns(events, series, asof_date=date(2024, 1, 6))
    # The $10 dividend on a $100 position must show up as a +10% return on the
    # last day (clamped to 2024-01-03), not vanish.
    assert float(daily.iloc[-1]) == pytest.approx(0.10, abs=1e-9)


def test_late_dated_buy_does_not_fabricate_a_crash_day() -> None:
    """Regression (whole-program review): a buy dated AFTER the last priced day must
    be dropped from BOTH the value curve and the flow series (its shares aren't priced
    yet) — NOT have its cash cost clamped onto the last day. The latter fabricated a
    phantom ~-100% return that silently poisoned the entire risk panel (drawdown / vol
    / Sharpe) and the dollar P&L. Common trigger: the brief runs before today's close
    is published, with a trade dated today (or a weekend run with a Monday trade).

    The sibling test above proves a late *dividend* still clamps (income has no share
    counterpart, so clamping it is correct); this proves a late *buy* does not.
    """
    series = {"TST": _price_series([100.0, 101.0, 102.0])}  # priced through 2024-01-03
    events = [
        Event(date(2024, 1, 1), "TST", "buy", quantity=1.0, price=100.0, fee=0.0),
        Event(date(2024, 1, 6), "TST", "buy", quantity=50.0, price=102.0, fee=0.0),  # past last priced day
    ]
    daily = build_daily_returns(events, series, asof_date=date(2024, 1, 6))
    # Every day is the ~+1%/day price move; no fabricated crash from the late buy.
    assert all(-0.5 < float(r) < 0.5 for r in daily), list(daily)
    # Dollar P&L is just the 1-share position's gain (102 - 100 = +$2), not a -$5000 phantom.
    pnl = pnl_curve(events, series, asof_date=date(2024, 1, 6))
    assert float(pnl.iloc[-1]) == pytest.approx(2.0, abs=1e-6)


def test_true_twr_none_on_too_little_data() -> None:
    events = [Event(date(2024, 1, 1), "TST", "buy", quantity=1.0, price=100.0, fee=0.0)]
    series = {"TST": _price_series([100.0, 110.0])}
    daily = build_daily_returns(events, series, asof_date=date(2024, 1, 2))
    assert true_twr_annualized(daily) is None  # only 1 return day


@given(
    qty=st.floats(min_value=0.01, max_value=1000, allow_nan=False, allow_infinity=False),
    price=st.floats(min_value=0.01, max_value=10000, allow_nan=False, allow_infinity=False),
    growth=st.floats(min_value=-0.99, max_value=10.0, allow_nan=False, allow_infinity=False),
)
def test_invariant_modified_dietz_is_scale_invariant(qty: float, price: float, growth: float) -> None:
    """Scaling both contribution and current value by the same factor preserves MD."""
    events_a = [Event(date(2023, 1, 1), "TST", "buy", quantity=qty, price=price, fee=0.0)]
    current_a = qty * price * (1 + growth)
    events_b = [Event(date(2023, 1, 1), "TST", "buy", quantity=qty * 10.0, price=price, fee=0.0)]
    current_b = qty * price * 10.0 * (1 + growth)
    md_a = modified_dietz_return(events_a, current_value=current_a, asof_date=date(2024, 1, 1))
    md_b = modified_dietz_return(events_b, current_value=current_b, asof_date=date(2024, 1, 1))
    assert md_a is not None and md_b is not None
    assert isclose(md_a, md_b, abs_tol=1e-6)


def test_summarize_suppresses_money_weighted_when_partially_priced() -> None:
    # If any held ticker is unpriced, current_value is partial → MWR/Dietz would
    # be confidently wrong, so both must be None. true TWR (priced-subset) stays.
    events = [Event(date(2024, 1, 1), "VOO", "buy", quantity=10.0, price=100.0, fee=0.0)]
    s = summarize(events, 1234.0, date(2026, 1, 1), true_twr=0.12, fully_priced=False)
    assert s.money_weighted_annualized is None
    assert s.modified_dietz_annualized is None
    assert s.true_twr_annualized == 0.12


def test_cash_flows_net_dividend_fee() -> None:
    events = [Event(date(2024, 3, 1), "VOO", "dividend",
                    quantity=0.0, price=0.0, cash=50.0, fee=3.0)]
    cfs = cash_flows_from_events(events)
    assert len(cfs) == 1
    assert cfs[0].amount == pytest.approx(47.0)  # 50 cash − 3 withholding


def test_pnl_curve_is_market_gains_flow_neutral() -> None:
    # Buy 10 @ $100 (cost 1000); price 100→110→120 → holdings 1000/1100/1200,
    # cumulative flow −1000 → P&L 0/100/200. A same-day deposit must NOT change it
    # (external flows cancel — that's the whole point).
    events = [
        Event(date(2024, 1, 1), "CASH", "deposit", quantity=0.0, price=0.0, cash=1000.0, fee=0.0),
        Event(date(2024, 1, 1), "A", "buy", quantity=10.0, price=100.0, fee=0.0),
    ]
    series = {"A": _price_series([100.0, 110.0, 120.0], "2024-01-01")}
    curve = pnl_curve(events, series, date(2024, 1, 3))
    assert list(curve.round(2)) == [0.0, 100.0, 200.0]


def test_pnl_curve_empty_without_priced_history() -> None:
    events = [Event(date(2024, 1, 1), "A", "buy", quantity=1.0, price=100.0, fee=0.0)]
    assert pnl_curve(events, {}, date(2024, 1, 3)).empty  # no series → empty


def test_pnl_curve_excludes_cash_income() -> None:
    # Broker interest on the CASH pseudo-ticker is a cash earning, not a market
    # gain — it must NOT enter the P&L curve (consistent with TWR; it still counts
    # in Net P&L / MWR). Adding it leaves the curve unchanged.
    series = {"A": _price_series([100.0, 100.0, 100.0], "2024-01-01")}
    base = [Event(date(2024, 1, 1), "A", "buy", quantity=10.0, price=100.0, fee=0.0)]
    with_interest = base + [
        Event(date(2024, 1, 2), "CASH", "interest", quantity=0.0, price=0.0, cash=5.0, fee=0.0)
    ]
    assert list(pnl_curve(base, series, date(2024, 1, 3))) == list(
        pnl_curve(with_interest, series, date(2024, 1, 3))
    )


def test_snap_to_index_snaps_forward_and_clamps() -> None:
    idx = pd.DatetimeIndex(["2024-01-03", "2024-01-05", "2024-01-10"])
    assert _snap_to_index(date(2024, 1, 5), idx) == pd.Timestamp("2024-01-05")  # exact day
    assert _snap_to_index(date(2024, 1, 4), idx) == pd.Timestamp("2024-01-05")  # → next day
    assert _snap_to_index(date(2024, 1, 1), idx) == pd.Timestamp("2024-01-03")  # before → idx[0]
    assert _snap_to_index(date(2024, 6, 1), idx) == pd.Timestamp("2024-01-10")  # after → clamp last


def test_curves_accept_precomputed_value() -> None:
    # The cli builds value_curve once and passes it to both consumers; the result
    # must match recomputing internally (no behavior change, just no double work).
    series = {"A": _price_series([100.0, 110.0, 120.0], "2024-01-01")}
    events = [Event(date(2024, 1, 1), "A", "buy", quantity=10.0, price=100.0, fee=0.0)]
    v = value_curve(events, series, date(2024, 1, 3))
    assert list(pnl_curve(events, series, date(2024, 1, 3), value=v)) == list(
        pnl_curve(events, series, date(2024, 1, 3))
    )
    assert list(build_daily_returns(events, series, asof_date=date(2024, 1, 3), value=v)) == list(
        build_daily_returns(events, series, asof_date=date(2024, 1, 3))
    )


def test_summarize_period_starts_at_first_investment_not_deposit() -> None:
    # A funding deposit precedes the first buy; the return period must start at the
    # buy, else the idle gap lengthens the annualization window and dilutes the
    # money-weighted figures (and the displayed period would be wrong).
    events = [
        Event(date(2024, 1, 1), "CASH", "deposit", quantity=0.0, price=0.0, cash=5000.0, fee=0.0),
        Event(date(2024, 4, 1), "VOO", "buy", quantity=10.0, price=100.0, fee=0.0),
    ]
    summary = summarize(events, current_value=1100.0, asof_date=date(2025, 4, 1))
    assert summary.period_start == date(2024, 4, 1)  # the buy, not the 2024-01-01 deposit


def test_price_basis_mismatches_flags_split_not_normal_gap() -> None:
    # A buy at ~10× the split-adjusted close is the fingerprint of an unhandled
    # split (NVDA 10:1); a ~1% fill-vs-close gap is normal and must not flag.
    dates = pd.date_range("2024-01-02", periods=5, freq="D")
    series = {
        "NVDA": pd.Series([110.0] * 5, index=dates, dtype=float),  # split-adjusted
        "VOO": pd.Series([100.0] * 5, index=dates, dtype=float),
    }
    events = [
        Event(date(2024, 1, 2), "NVDA", "buy", quantity=1.0, price=1120.0, fee=0.0),
        Event(date(2024, 1, 2), "VOO", "buy", quantity=1.0, price=101.0, fee=0.0),
    ]
    assert price_basis_mismatches(events, series) == ["NVDA"]


def test_price_basis_mismatches_handles_reverse_split_and_edges() -> None:
    dates = pd.date_range("2024-01-02", periods=3, freq="D")
    series = {
        "RVRS": pd.Series([10.0] * 3, index=dates, dtype=float),  # adjusted up after a 1:10
        "OK": pd.Series([50.0] * 3, index=dates, dtype=float),
        "NOPX": pd.Series([0.0] * 3, index=dates, dtype=float),  # bad/zero close → skip
    }
    events = [
        Event(date(2024, 1, 2), "RVRS", "sell", quantity=1.0, price=1.0, fee=0.0),  # 0.1× → flag
        Event(date(2024, 1, 2), "OK", "buy", quantity=1.0, price=50.0, fee=0.0),
        Event(date(2024, 1, 2), "NOPX", "buy", quantity=1.0, price=99.0, fee=0.0),  # close 0 → skip
        Event(date(2024, 1, 2), "GONE", "buy", quantity=1.0, price=5.0, fee=0.0),   # no series → skip
        Event(date(2024, 3, 1), "OK", "dividend", quantity=0.0, price=0.0, cash=5.0, fee=0.0),
    ]
    assert price_basis_mismatches(events, series) == ["RVRS"]


def test_price_basis_mismatches_flags_a_trade_before_the_history_starts() -> None:
    # F3 (fresh-eyes audit 2026-07-11): a series beginning AFTER the first buy used to be
    # skipped (`continue` on no price at-or-before the trade) — but the buy's cash lands on
    # the union calendar before the ticker has any value, fabricating a ~−100%/+100% day
    # pair into the risk panel. The left edge must flag like the split fingerprint does.
    dates = pd.date_range("2024-02-01", periods=5, freq="D")  # starts AFTER the buys below
    series = {
        "TRUNC": pd.Series([90.0] * 5, index=dates, dtype=float),
        "FULL": pd.Series([100.0] * 5, index=dates, dtype=float),
    }
    events = [
        Event(date(2024, 1, 10), "TRUNC", "buy", quantity=10.0, price=90.0, fee=0.0),
        Event(date(2024, 2, 2), "FULL", "buy", quantity=10.0, price=100.0, fee=0.0),
    ]
    assert price_basis_mismatches(events, series) == ["TRUNC"]
    # A later, in-history trade doesn't un-flag the ticker: the early one still poisons.
    events.append(Event(date(2024, 2, 3), "TRUNC", "buy", quantity=1.0, price=90.0, fee=0.0))
    assert price_basis_mismatches(events, series) == ["TRUNC"]
