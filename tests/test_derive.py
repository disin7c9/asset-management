"""Tests for the event-replay engine.

- `test_golden_*` — reconcile against hand-verified numbers from `examples/07_derive_from_log.py`.
- `test_invariant_*` — property-based tests for invariants that must always hold.
- `test_load_*` / `test_derive_*` — regression tests covering each bug fixed in slice 1's review pass.
"""

from __future__ import annotations

import logging
from datetime import date
from math import isclose
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from app.derive import DerivedState, derive
from app.events import CASH_TICKER, Event, load_events

# --- Golden values: hand-verified from examples/07_derive_from_log.py ---
GOLDEN_HELD = {
    # ticker: (shares, cost_basis, realized)
    "VOO": (12.0, 4501.6, 295.80),
    "BND": (50.0, 3613.0, 45.10),
    "IAU": (25.0, 930.625, 220.625),
    "VEA": (15.0, 604.0, 0.0),
}
# 8 trade rows × $1 (3 VOO + 2 BND + 2 IAU + 1 VEA), plus the $25 annual account fee row.
GOLDEN_TOTAL_FEES = 33.0
# Interest $12.85 − the $25 account fee. Realized on the CASH side, with no position behind it.
GOLDEN_CASH_REALIZED = -12.15


def _abs_close(a: float, b: float, tol: float = 1e-6) -> bool:
    return isclose(a, b, abs_tol=tol)


def _example_csv() -> Path:
    """Resolve the example transactions CSV path relative to repo root."""
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "data" / "sample_data" / "transactions.csv"


# --- Golden reconciliation ---
def test_golden_holdings_match_script_07_numbers() -> None:
    """The reference example produced these values; v0 must agree exactly."""
    state = derive(load_events(_example_csv()))
    held = state.held()
    assert set(held) == set(GOLDEN_HELD), held
    for tk, (shares, cost_basis, realized) in GOLDEN_HELD.items():
        p = held[tk]
        assert _abs_close(p.shares, shares), (tk, p.shares, shares)
        assert _abs_close(p.cost_basis, cost_basis), (tk, p.cost_basis, cost_basis)
        assert _abs_close(state.realized[tk], realized), (tk, state.realized[tk])


def test_golden_total_fees() -> None:
    state = derive(load_events(_example_csv()))
    assert _abs_close(state.total_fees(), GOLDEN_TOTAL_FEES)


def test_derived_state_helpers_match() -> None:
    """Cross-check helper sums against per-ticker numbers."""
    state: DerivedState = derive(load_events(_example_csv()))
    assert _abs_close(
        state.total_cost_basis(),
        sum(cb for _, cb, _ in GOLDEN_HELD.values()),
    )
    # total_realized spans BOTH sides: per-ticker sale/dividend gains and the cash-side
    # interest/fee rows. The example book exercises all seven action types, so this sum
    # would silently drop the cash leg if the two were ever wired apart.
    assert _abs_close(
        state.total_realized(),
        sum(r for _, _, r in GOLDEN_HELD.values()) + GOLDEN_CASH_REALIZED,
    )
    assert _abs_close(state.cash_realized, GOLDEN_CASH_REALIZED)


# --- Property-based invariants ---
@given(
    buys=st.lists(
        st.tuples(
            st.floats(min_value=0.001, max_value=1000, allow_nan=False, allow_infinity=False),
            st.floats(min_value=0.01, max_value=10000, allow_nan=False, allow_infinity=False),
        ),
        min_size=1,
        max_size=20,
    ),
    sell_qty=st.floats(min_value=0.0, max_value=500, allow_nan=False, allow_infinity=False),
)
def test_invariant_share_balance_after_buys_then_sell(
    buys: list[tuple[float, float]], sell_qty: float
) -> None:
    """Shares after replay = total bought - total sold (within ε), for any ordering."""
    events: list[Event] = []
    for i, (qty, price) in enumerate(buys):
        events.append(
            Event(date=date(2024, 1, (i % 28) + 1), ticker="TST",
                  action="buy", quantity=qty, price=price, fee=0.0)
        )
    total_bought = sum(q for q, _ in buys)
    sell_qty = min(sell_qty, total_bought)  # cannot sell more than held
    if sell_qty > 0:
        events.append(
            Event(date=date(2024, 12, 31), ticker="TST",
                  action="sell", quantity=sell_qty, price=1.0, fee=0.0)
        )
    state = derive(events)
    expected = total_bought - sell_qty
    pos = state.positions["TST"]
    # After a full sell the engine snaps shares to 0; both sides become 0.
    assert isclose(pos.shares, expected, abs_tol=1e-6) or (
        isclose(expected, 0.0, abs_tol=1e-6) and pos.shares == 0.0
    ), (pos.shares, expected)


def test_dividend_fee_is_netted_into_realized() -> None:
    # A withholding fee on a dividend row reduces realized income (and is still
    # tracked informationally in total_fees) — was previously dropped.
    events = [
        Event(date=date(2024, 1, 1), ticker="VOO",
              action="buy", quantity=10.0, price=100.0, fee=0.0),
        Event(date=date(2024, 3, 1), ticker="VOO",
              action="dividend", quantity=0.0, price=0.0, cash=50.0, fee=3.0),
    ]
    state = derive(events)
    assert isclose(state.realized["VOO"], 47.0)  # 50 cash − 3 withholding
    assert isclose(state.total_fees(), 3.0)


def test_standalone_fee_event_nets_into_realized() -> None:
    # A separate fee/tax row (e.g. foreign withholding exported as its own line)
    # reduces realized P&L for that ticker, and is still counted in total_fees.
    events = [
        Event(date=date(2024, 1, 1), ticker="SCHP",
              action="buy", quantity=10.0, price=27.0, fee=0.0),
        Event(date=date(2024, 3, 1), ticker="SCHP",
              action="dividend", quantity=0.0, price=0.0, cash=5.0, fee=0.0),
        Event(date=date(2024, 3, 8), ticker="SCHP",
              action="fee", quantity=0.0, price=0.0, fee=0.30),  # foreign withholding tax
    ]
    state = derive(events)
    assert isclose(state.realized["SCHP"], 4.70)  # 5.0 dividend − 0.30 tax
    assert isclose(state.total_fees(), 0.30)


# --- Cash routed by kind (the slice-8 deferred fix, landed v1.5.0) ---
# Rows on the CASH pseudo-ticker are cash-account effects: they net into
# cash_realized (counted in total_realized) but never create a Position or a
# per-security realized entry. Surfaced by the 2026-06-09 whole-program review.


def test_unmatched_tax_on_cash_ticker_is_a_cash_account_cost() -> None:
    """A fee/tax row coded to the CASH pseudo-ticker (e.g. a withholding tax an
    exporter couldn't attach to a specific holding) nets into cash_realized — it
    still reduces total realized P&L, but no phantom CASH position or per-security
    realized entry is created."""
    state = derive([Event(date(2024, 1, 1), CASH_TICKER, "fee", quantity=0.0, price=0.0, fee=3.50)])
    assert isclose(state.cash_realized, -3.50)
    assert isclose(state.total_realized(), -3.50)  # the cost still counts in the total
    assert CASH_TICKER not in state.realized  # no per-security attribution
    assert CASH_TICKER not in state.positions  # no phantom position
    assert isclose(state.total_fees(), 3.50)  # still tracked informationally


def test_interest_on_cash_is_cash_account_income() -> None:
    # Broker interest on idle cash is real income: it counts in total_realized via
    # cash_realized, without fabricating a CASH position or a realized["CASH"] entry.
    state = derive(
        [Event(date(2024, 3, 1), CASH_TICKER, "interest", quantity=0.0, price=0.0,
               fee=0.0, cash=12.0)]
    )
    assert isclose(state.cash_realized, 12.0)
    assert isclose(state.total_realized(), 12.0)
    assert CASH_TICKER not in state.realized
    assert CASH_TICKER not in state.positions


def test_trade_of_cash_pseudo_ticker_is_rejected() -> None:
    # CASH is not a security; a buy/sell row on it is an importer error, not a
    # position to be silently created.
    with pytest.raises(ValueError, match="CASH pseudo-ticker"):
        derive([Event(date(2024, 1, 1), CASH_TICKER, "buy", quantity=1.0, price=1.0, fee=0.0)])


def test_deposit_on_security_ticker_is_rejected() -> None:
    # The symmetric guard: an external cash flow mis-tickered to a security would
    # otherwise vanish silently from all accounting.
    with pytest.raises(ValueError, match="must use the CASH pseudo-ticker"):
        derive([Event(date(2024, 1, 1), "VOO", "deposit", quantity=0.0, price=0.0,
                      fee=0.0, cash=500.0)])


def test_fee_listed_on_trade_and_standalone_double_counts_but_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If an importer lists the same cost BOTH embedded in a trade row's Fee column
    AND as a separate fee line, derive counts it twice — once raising cost basis (via
    the buy fee) and again subtracted from realized (the standalone fee). Derive
    cannot tell a duplicate from two genuine same-amount fees, so the importer
    contract is "a cost appears exactly once" and derive WARNS on the suspicious
    pattern (same-day, same-ticker, same amount) instead of silently double-counting."""
    events = [  # the same $5 commission listed twice: on the buy row and as its own line
        Event(date(2024, 1, 1), "VOO", "buy", quantity=1.0, price=100.0, fee=5.0),
        Event(date(2024, 1, 1), "VOO", "fee", quantity=0.0, price=0.0, fee=5.0),
        Event(date(2024, 6, 1), "VOO", "sell", quantity=1.0, price=110.0, fee=0.0),
    ]
    with caplog.at_level(logging.WARNING, logger="app.derive"):
        state = derive(events)
    # The double-count itself is unchanged (derive can't know it's the same $5)…
    assert isclose(state.realized["VOO"], 0.0)
    assert isclose(state.total_fees(), 10.0)
    # …but it is no longer silent.
    assert any("matches a same-day trade-row fee" in r.message for r in caplog.records)


def test_distinct_standalone_fee_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    # A standalone fee that does NOT match any same-day trade-row fee (different
    # amount) is the legitimate case (e.g. a separate tax) — no warning.
    events = [
        Event(date(2024, 1, 1), "VOO", "buy", quantity=1.0, price=100.0, fee=5.0),
        Event(date(2024, 1, 1), "VOO", "fee", quantity=0.0, price=0.0, fee=0.30),
    ]
    with caplog.at_level(logging.WARNING, logger="app.derive"):
        derive(events)
    assert not any("matches a same-day trade-row fee" in r.message for r in caplog.records)


@given(
    dividends=st.lists(
        st.floats(min_value=0.0, max_value=1000, allow_nan=False, allow_infinity=False),
        min_size=0,
        max_size=20,
    )
)
def test_invariant_dividends_only_add_to_realized(dividends: list[float]) -> None:
    """Dividend rows must not change share counts or cost basis."""
    events: list[Event] = [
        Event(date=date(2024, 1, 1), ticker="TST",
              action="buy", quantity=10.0, price=100.0, fee=0.0),
    ]
    for i, cash in enumerate(dividends):
        events.append(
            Event(date=date(2024, 6, (i % 28) + 1), ticker="TST",
                  action="dividend", quantity=0.0, price=0.0, cash=cash, fee=0.0)
        )
    state = derive(events)
    pos = state.positions["TST"]
    assert isclose(pos.shares, 10.0)
    assert isclose(pos.cost_basis, 1000.0)
    assert isclose(state.realized["TST"], sum(dividends), abs_tol=1e-6)


# --- Regression tests for each finding in the slice-1 review ---
def test_sell_with_no_shares_raises() -> None:
    events = [
        Event(date=date(2024, 1, 1), ticker="TST",
              action="sell", quantity=1.0, price=100.0, fee=0.0),
    ]
    with pytest.raises(ValueError, match="no shares held"):
        derive(events)


def test_oversell_raises(tmp_path: Path) -> None:
    """Finding 3: SELL qty > held qty must be rejected, not silently negative."""
    events = [
        Event(date=date(2024, 1, 1), ticker="TST",
              action="buy", quantity=10.0, price=100.0, fee=0.0),
        Event(date=date(2024, 6, 1), ticker="TST",
              action="sell", quantity=12.0, price=110.0, fee=0.0),
    ]
    with pytest.raises(ValueError, match="oversell"):
        derive(events)


def test_load_handles_empty_quantity_and_fee(tmp_path: Path) -> None:
    """Finding 1: empty Quantity/Fee cells must parse as 0.0, not crash."""
    csv = tmp_path / "empty.csv"
    csv.write_text(
        "Date,Code,DataSource,Currency,Price,Quantity,Action,Fee,Note\n"
        "2024-01-01,VOO,YAHOO,USD,400.00,10,buy,,initial\n"  # empty fee
        "2024-06-01,VOO,YAHOO,USD,2.50,,dividend,0.00,quarterly\n"  # empty qty
    )
    events = load_events(csv)
    assert len(events) == 2
    assert events[0].fee == 0.0
    assert events[1].action == "dividend"
    assert events[1].cash == 2.5
    assert events[1].quantity == 0.0


def test_load_missing_required_column(tmp_path: Path) -> None:
    """Finding 2 (a): missing required column raises a helpful ValueError, not bare KeyError."""
    csv = tmp_path / "no_action.csv"
    csv.write_text(
        "Date,Code,DataSource,Currency,Price,Quantity,Fee,Note\n"
        "2024-01-01,VOO,YAHOO,USD,400.00,10,1.00,\n"
    )
    with pytest.raises(ValueError, match="missing required column"):
        load_events(csv)


def test_load_strips_utf8_bom(tmp_path: Path) -> None:
    """Finding 2 (b): an Excel-saved CSV with a UTF-8 BOM must still load."""
    csv = tmp_path / "bom.csv"
    csv.write_bytes(
        b"\xef\xbb\xbfDate,Code,DataSource,Currency,Price,Quantity,Action,Fee,Note\n"
        b"2024-01-01,VOO,YAHOO,USD,400.00,10,buy,1.00,\n"
    )
    events = load_events(csv)
    assert events[0].ticker == "VOO"


def test_load_rejects_non_iso_date(tmp_path: Path) -> None:
    """Finding 2 (c): non-ISO dates raise a clear error mentioning the format."""
    csv = tmp_path / "bad_date.csv"
    csv.write_text(
        "Date,Code,DataSource,Currency,Price,Quantity,Action,Fee,Note\n"
        "01/03/2024,VOO,YAHOO,USD,400.00,10,buy,1.00,\n"
    )
    with pytest.raises(ValueError, match="ISO 8601"):
        load_events(csv)


def test_load_same_day_sell_before_buy_orders_correctly(tmp_path: Path) -> None:
    """Finding 4: same-day SELL listed before its BUY must not crash — buys go first."""
    csv = tmp_path / "sameday.csv"
    csv.write_text(
        "Date,Code,DataSource,Currency,Price,Quantity,Action,Fee,Note\n"
        "2024-03-01,VOO,YAHOO,USD,500.00,5,sell,1.00,\n"
        "2024-03-01,VOO,YAHOO,USD,400.00,10,buy,1.00,\n"
    )
    state = derive(load_events(csv))
    # After the day: 10 bought, 5 sold → 5 shares held, avg cost = (10*400+1)/10 = 400.1
    pos = state.held()["VOO"]
    assert isclose(pos.shares, 5.0)
    assert isclose(pos.avg_cost, 400.1, abs_tol=1e-9)


def test_load_default_source_is_manual_when_missing(tmp_path: Path) -> None:
    """Finding 8: don't fabricate 'YAHOO' for rows without a DataSource."""
    csv = tmp_path / "no_source.csv"
    csv.write_text(
        "Date,Code,Currency,Price,Quantity,Action,Fee,Note\n"
        "2024-01-01,VOO,USD,400.00,10,buy,1.00,\n"
    )
    events = load_events(csv)
    assert events[0].source == "MANUAL"


def test_derive_dust_snap_on_full_sell() -> None:
    """Finding 9: float residue after a full sell must snap to exactly 0."""
    events = [
        Event(date=date(2024, 1, 1), ticker="TST",
              action="buy", quantity=0.1, price=100.0, fee=0.0),
        Event(date=date(2024, 1, 2), ticker="TST",
              action="buy", quantity=0.2, price=100.0, fee=0.0),
        Event(date=date(2024, 6, 1), ticker="TST",
              action="sell", quantity=0.3, price=100.0, fee=0.0),
    ]
    state = derive(events)
    pos = state.positions["TST"]
    # Without the snap, pos.shares would be a tiny non-zero like 5.5e-17.
    assert pos.shares == 0.0
    assert pos.cost_basis == 0.0
    assert state.held() == {}
