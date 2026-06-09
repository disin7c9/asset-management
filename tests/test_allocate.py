"""Tests for the allocation library (slice 9): equal_weight, inverse_vol, caps."""

from __future__ import annotations

from math import isclose

import pandas as pd
import pytest

from app.allocate import allocation_kind, apply_caps, equal_weight, inverse_vol


def _series(values: list[float], start: str = "2024-01-01") -> "pd.Series[float]":
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def test_equal_weight_is_uniform_and_dedups() -> None:
    w = equal_weight(["VOO", "BND", "VEA", "VOO"])  # duplicate VOO collapses
    assert set(w) == {"VOO", "BND", "VEA"}
    assert all(isclose(v, 1 / 3) for v in w.values())
    assert isclose(sum(w.values()), 1.0)


def test_equal_weight_empty() -> None:
    assert equal_weight([]) == {}


def test_inverse_vol_gives_calm_assets_more_weight() -> None:
    calm = _series([100, 100.1, 100.0, 100.1, 100.0, 100.1, 100.0, 100.1])
    bumpy = _series([100, 110, 92, 115, 88, 112, 90, 118])
    w = inverse_vol({"CALM": calm, "BUMPY": bumpy})
    assert isclose(sum(w.values()), 1.0)
    assert w["CALM"] > w["BUMPY"]  # lower vol → larger weight (equal-risk, not equal-$)


def test_inverse_vol_drops_short_series_and_empties() -> None:
    assert inverse_vol({}) == {}
    assert inverse_vol({"X": _series([100.0])}) == {}  # < 2 prices → dropped → empty


def test_inverse_vol_drops_degenerate_series_without_poisoning() -> None:
    # A 2-price series → 1 return → sample std is NaN; a flat series → 0 vol. Both
    # must be DROPPED, not poison the normalization (NaN) or dominate (floored 0).
    good = _series([100, 101, 99, 102, 98, 103])
    two = _series([100.0, 110.0])  # 1 return → std NaN
    flat = _series([100.0] * 6)  # zero volatility
    w = inverse_vol({"GOOD": good, "TWO": two, "FLAT": flat})
    assert set(w) == {"GOOD"} and isclose(w["GOOD"], 1.0)  # only the usable one survives


def test_apply_caps_redistributes_excess() -> None:
    w = apply_caps({"A": 0.7, "B": 0.2, "C": 0.1}, cap=0.5)
    assert max(w.values()) <= 0.5 + 1e-9
    assert isclose(sum(w.values()), 1.0)
    assert isclose(w["A"], 0.5)
    assert isclose(w["B"], 0.2 + 0.2 * (0.2 / 0.3))  # excess 0.2 split B:C = 2:1
    assert isclose(w["C"], 0.1 + 0.2 * (0.1 / 0.3))


def test_apply_caps_iterates_to_fixed_point() -> None:
    # Capping A pushes B over the cap → a second pass is required.
    w = apply_caps({"A": 0.8, "B": 0.15, "C": 0.05}, cap=0.4)
    assert max(w.values()) <= 0.4 + 1e-9
    assert isclose(sum(w.values()), 1.0)
    assert isclose(w["A"], 0.4) and isclose(w["B"], 0.4) and isclose(w["C"], 0.2)


def test_apply_caps_infeasible_or_nonpositive_raises() -> None:
    with pytest.raises(ValueError, match="too small"):
        apply_caps({"A": 0.5, "B": 0.3, "C": 0.2}, cap=0.2)  # 0.2 × 3 < 1
    with pytest.raises(ValueError, match="positive"):
        apply_caps({"A": 1.0}, cap=0.0)


def test_allocation_kind() -> None:
    assert allocation_kind("equal_weight") == "discipline"
    assert allocation_kind("inverse_vol") == "discipline"
    assert allocation_kind("mystery_optimizer") == "edge"  # unknown → fail-safe
