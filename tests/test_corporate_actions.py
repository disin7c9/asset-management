"""Tests for split adjustment (slice 7). Pure — no I/O."""

from __future__ import annotations

from datetime import date

import pytest

from app.corporate_actions import adjust_for_splits, cumulative_split_factor
from app.events import Event


def _buy(d: date, tk: str, qty: float, px: float) -> Event:
    return Event(d, tk, "buy", quantity=qty, price=px, fee=0.0)


def test_split_after_buy_adjusts_qty_and_price_cost_invariant() -> None:
    # The NVDA case: 1 share @ $1120 pre-split, then 10:1 → 10 shares @ $112.
    ev = _buy(date(2024, 5, 30), "NVDA", 1.0, 1120.0)
    out = adjust_for_splits([ev], {"NVDA": [(date(2024, 6, 10), 10.0)]})
    assert out[0].quantity == pytest.approx(10.0)
    assert out[0].price == pytest.approx(112.0)
    # Total cost is unchanged — only the share basis changes.
    assert out[0].quantity * out[0].price == pytest.approx(ev.quantity * ev.price)


def test_trade_on_or_after_split_not_adjusted() -> None:
    after = _buy(date(2024, 6, 18), "NVDA", 4.0, 130.5)
    on = _buy(date(2024, 6, 10), "NVDA", 4.0, 130.5)  # on the split day = already post-split
    out = adjust_for_splits([after, on], {"NVDA": [(date(2024, 6, 10), 10.0)]})
    assert out[0] == after and out[1] == on


def test_dividend_and_fee_pass_through() -> None:
    div = Event(date(2024, 5, 30), "NVDA", "dividend", quantity=0.0, price=0.0, cash=5.0, fee=0.0)
    out = adjust_for_splits([div], {"NVDA": [(date(2024, 6, 10), 10.0)]})
    assert out[0] == div  # cash isn't changed by a split


def test_no_splits_unchanged() -> None:
    ev = _buy(date(2024, 1, 1), "VOO", 2.0, 400.0)
    assert adjust_for_splits([ev], {}) == [ev]
    assert adjust_for_splits([ev], {"VOO": []}) == [ev]


def test_multiple_splits_compound() -> None:
    ev = _buy(date(2020, 1, 1), "X", 1.0, 100.0)
    out = adjust_for_splits([ev], {"X": [(date(2021, 1, 1), 2.0), (date(2022, 1, 1), 5.0)]})
    assert out[0].quantity == pytest.approx(10.0)  # 2 × 5
    assert out[0].price == pytest.approx(10.0)


def test_reverse_split_adjusts_down() -> None:
    # 1:10 reverse split → ratio 0.1 → 10 pre-split shares become 1.
    ev = _buy(date(2024, 1, 1), "RV", 10.0, 5.0)
    out = adjust_for_splits([ev], {"RV": [(date(2024, 6, 1), 0.1)]})
    assert out[0].quantity == pytest.approx(1.0)
    assert out[0].price == pytest.approx(50.0)
    assert out[0].quantity * out[0].price == pytest.approx(50.0)  # cost invariant


def test_cumulative_factor_boundaries() -> None:
    splits = [(date(2024, 6, 10), 10.0)]
    assert cumulative_split_factor(splits, date(2024, 5, 30)) == 10.0  # before → applies
    assert cumulative_split_factor(splits, date(2024, 6, 10)) == 1.0   # on the day → not
    assert cumulative_split_factor(splits, date(2024, 7, 1)) == 1.0    # after → not
