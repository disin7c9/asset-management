"""Tests for the event-replay engine.

- `test_golden_*` — reconcile against hand-verified numbers from `examples/07_derive_from_log.py`.
- `test_invariant_*` — property-based tests for invariants that must always hold.
- `test_load_*` / `test_derive_*` — regression tests covering each bug fixed in slice 1's review pass.
"""

from __future__ import annotations

from datetime import date
from math import isclose
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from app.derive import DerivedState, derive
from app.events import Event, load_events

# --- Golden values: hand-verified from examples/07_derive_from_log.py ---
GOLDEN_HELD = {
    # ticker: (shares, cost_basis, realized)
    "VOO": (12.0, 4501.6, 295.80),
    "BND": (50.0, 3613.0, 45.10),
    "IAU": (25.0, 930.625, 220.625),
    "VEA": (15.0, 604.0, 0.0),
}
GOLDEN_TOTAL_FEES = 8.0  # 8 non-dividend rows × $1 (3 VOO + 2 BND + 2 IAU + 1 VEA)


def _abs_close(a: float, b: float, tol: float = 1e-6) -> bool:
    return isclose(a, b, abs_tol=tol)


def _example_csv() -> Path:
    """Resolve the example transactions CSV path relative to repo root."""
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "examples" / "data" / "transactions.csv"


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
    assert _abs_close(
        state.total_realized(),
        sum(r for _, _, r in GOLDEN_HELD.values()),
    )


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
