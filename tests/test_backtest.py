"""Tests for the notional backtest engine. Pure — synthetic price series."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from app.backtest import backtest_compare, simulate


def _series(prices: list[float], dates: list[str]) -> "pd.Series[float]":
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    return pd.Series(prices, index=idx, dtype=float)


def test_simulate_buyhold_round_trip() -> None:
    # A doubles then returns; B flat. Buy-and-hold ends where it started.
    dates = ["2024-01-31", "2024-02-01", "2024-02-02", "2024-03-01"]
    series = {"A": _series([10, 20, 20, 10], dates), "B": _series([10, 10, 10, 10], dates)}
    bh = simulate(series, {"A": 0.5, "B": 0.5}, schedule="never", initial=1000.0)
    assert bh.iloc[0] == pytest.approx(1000.0)
    assert bh.iloc[-1] == pytest.approx(1000.0)


def test_simulate_rebalancing_trims_the_spike() -> None:
    # Monthly rebalance trims A at its peak (2024-02-01) → ends ABOVE buy-and-hold
    # when A reverts. Hand-computed: rebalanced final = 1125.
    dates = ["2024-01-31", "2024-02-01", "2024-02-02", "2024-03-01"]
    series = {"A": _series([10, 20, 20, 10], dates), "B": _series([10, 10, 10, 10], dates)}
    rb = simulate(series, {"A": 0.5, "B": 0.5}, schedule="monthly", initial=1000.0)
    assert rb.iloc[-1] == pytest.approx(1125.0)


def test_simulate_empty_without_priced_target() -> None:
    assert simulate({}, {"A": 1.0}, schedule="never").empty


def test_backtest_compare_two_legs_and_dates() -> None:
    dates = pd.bdate_range("2024-01-01", periods=60)
    a = pd.Series([100.0 + i for i in range(60)], index=dates, dtype=float)
    b = pd.Series([50.0] * 60, index=dates, dtype=float)
    res = backtest_compare({"A": a, "B": b}, {"A": 0.5, "B": 0.5},
                           schedule="monthly", bootstrap_n=50)
    assert res is not None
    assert len(res.legs) == 2
    assert res.legs[0].label == "rebalanced (monthly)"
    assert res.legs[1].label == "buy & hold"
    assert res.missing == ()
    assert res.start == date(2024, 1, 1)
    assert res.end == dates[-1].date()
    assert res.legs[0].final_value > 0 and res.legs[1].final_value > 0


def test_backtest_compare_reports_unpriced_target_ticker() -> None:
    dates = pd.bdate_range("2024-01-01", periods=40)
    a = pd.Series([100.0 + i for i in range(40)], index=dates, dtype=float)
    res = backtest_compare({"A": a}, {"A": 0.6, "ZZZ": 0.4},
                           schedule="never", bootstrap_n=20)
    assert res is not None
    assert res.missing == ("ZZZ",)  # ZZZ has no series → excluded + reported


def test_backtest_compare_none_without_data() -> None:
    assert backtest_compare({}, {"A": 1.0}) is None


def test_zero_weight_priced_subset_is_graceful_not_crash() -> None:
    # The real-weight ticker has no history; only the 0-weight "close" ticker does
    # → priced-subset weight sums to 0 → graceful None, not ZeroDivisionError.
    dates = pd.bdate_range("2024-01-01", periods=40)
    z = pd.Series([10.0] * 40, index=dates, dtype=float)
    res = backtest_compare({"ZZZ": z}, {"AAA": 1.0, "ZZZ": 0.0},
                           schedule="never", bootstrap_n=10)
    assert res is None


def test_simulate_drops_zero_first_price() -> None:
    # A ticker whose first close is 0.0 (bad data) is dropped, not divided by.
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    series = {"A": _series([0.0, 10.0, 11.0], dates), "B": _series([5.0, 5.0, 5.0], dates)}
    curve = simulate(series, {"A": 0.5, "B": 0.5}, schedule="never", initial=1000.0)
    assert not curve.empty
    assert curve.iloc[0] == pytest.approx(1000.0)  # weight renormalized onto B
