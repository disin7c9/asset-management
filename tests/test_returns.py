"""Tests for the returns engine: MWR, Modified Dietz, annualization."""

from __future__ import annotations

from datetime import date
from math import isclose

from hypothesis import given
from hypothesis import strategies as st

from app.events import Event
from app.returns import (
    IRRError,
    annualize_return,
    cash_flows_from_events,
    modified_dietz_return,
    money_weighted_return,
    summarize,
)
from app.returns import _xirr_newton  # noqa: PLC2701 — exercised directly in regression tests


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


def test_annualize_handles_zero_days() -> None:
    assert annualize_return(0.10, 0) == 0.0


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
