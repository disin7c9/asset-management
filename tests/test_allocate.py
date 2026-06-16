"""Tests for the allocation library: the allocate() dispatcher (gate inside),
equal_weight, inverse_vol, caps."""

from __future__ import annotations

from math import isclose

import pandas as pd
import pytest

import app.allocate as A
from app.allocate import (
    VALID_RULES,
    UnvalidatedEdgeError,
    allocate,
    allocation_kind,
    apply_caps,
    equal_weight,
    inverse_vol,
    needs_series,
)
from app.events import CASH_TICKER


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


# --- The allocate() dispatcher: one entry point, gate inside ---


def test_allocate_dispatches_equal_weight() -> None:
    w = allocate("equal_weight", ["BND", "VOO"])
    assert w == equal_weight(["BND", "VOO"])
    assert isclose(sum(w.values()), 1.0)


def test_allocate_dispatches_inverse_vol_restricted_to_universe() -> None:
    calm = _series([100, 100.1, 100.0, 100.1, 100.0, 100.1])
    bumpy = _series([100, 110, 92, 115, 88, 112])
    series = {"CALM": calm, "BUMPY": bumpy, "EXTRA": bumpy}  # EXTRA not in universe
    w = allocate("inverse_vol", ["CALM", "BUMPY"], series)
    assert set(w) == {"CALM", "BUMPY"}  # the extra series is ignored
    assert w == inverse_vol({"CALM": calm, "BUMPY": bumpy})


def test_allocate_applies_cap() -> None:
    calm = _series([100, 100.1, 100.0, 100.1, 100.0, 100.1])
    bumpy = _series([100, 110, 92, 115, 88, 112])
    w = allocate("inverse_vol", ["CALM", "BUMPY"], {"CALM": calm, "BUMPY": bumpy}, cap=0.60)
    assert max(w.values()) <= 0.60 + 1e-9
    assert isclose(sum(w.values()), 1.0)


def test_allocate_infeasible_cap_raises() -> None:
    with pytest.raises(ValueError, match="too small"):
        allocate("equal_weight", ["A", "B"], cap=0.30)  # 0.3 * 2 < 1


def test_allocate_unknown_rule_raises() -> None:
    with pytest.raises(ValueError, match="unknown allocation rule"):
        allocate("mystery_optimizer", ["VOO"])


def test_allocate_excludes_cash_pseudo_ticker() -> None:
    w = allocate("equal_weight", ["VOO", CASH_TICKER, "BND"])
    assert CASH_TICKER not in w
    assert set(w) == {"VOO", "BND"}


def test_allocate_inverse_vol_without_series_raises() -> None:
    with pytest.raises(ValueError, match="needs price history"):
        allocate("inverse_vol", ["VOO"])


def test_needs_series() -> None:
    assert needs_series("inverse_vol")
    assert not needs_series("equal_weight")


def test_allocate_edge_rule_gated_inside_dispatcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # THE altitude fix: register a fake edge rule and call the dispatcher directly
    # (no CLI involved). The gate must fire inside allocate() itself, and only the
    # walk-forward machinery's validated=True may open it.
    monkeypatch.setitem(A._RULES, "fake_optimizer", lambda universe, series: {"VOO": 1.0})
    assert allocation_kind("fake_optimizer") == "edge"  # unregistered kind → fail-safe
    with pytest.raises(UnvalidatedEdgeError, match="walk-forward"):
        allocate("fake_optimizer", ["VOO"])
    assert allocate("fake_optimizer", ["VOO"], validated=True) == {"VOO": 1.0}


def test_rule_registries_stay_in_sync() -> None:
    # argparse offers VALID_RULES (from the Literal); the dispatcher runs _RULES;
    # _RULE_KIND gates them. A rule added to one but not the others either errors at
    # runtime ("unknown allocation rule") or is silently blocked as edge — pin them.
    assert set(A._RULES) == set(VALID_RULES)
    assert set(A._RULE_KIND) >= set(A._RULES)  # every runnable rule has an explicit kind


def test_rule_kind_is_explicit_classification_not_blanket() -> None:
    # Regression for the fail-safe inversion (v2.0.0 Phase 0). _RULE_KIND is an
    # explicit per-rule map (NOT {r: "discipline" for r in VALID_RULES}), so a future
    # edge rule added to the AllocationRule Literal is not silently waved through as
    # discipline: a name absent from the map reads edge.
    assert A._RULE_KIND["equal_weight"] == "discipline"
    assert A._RULE_KIND["inverse_vol"] == "discipline"
    assert allocation_kind("max_sharpe") == "edge"  # absent → fail-safe edge


def test_allocate_edge_rule_opens_only_on_validated_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The Phase-0 bridge, end to end: an edge allocator opens ONLY when a real role
    # check validates it. validate_from_role(improved) → True opens the gate; any
    # other verdict → False keeps UnvalidatedEdgeError.
    from app.backtest import RoleCheck, validate_from_role

    monkeypatch.setitem(A._RULES, "fake_optimizer", lambda universe, series: {"VOO": 1.0})
    improved = RoleCheck("X", 0.05, "quarterly", (), "improved", "")
    inconclusive = RoleCheck("X", 0.05, "quarterly", (), "inconclusive", "")

    assert allocate(
        "fake_optimizer", ["VOO"], validated=validate_from_role(improved)
    ) == {"VOO": 1.0}
    with pytest.raises(UnvalidatedEdgeError, match="walk-forward"):
        allocate("fake_optimizer", ["VOO"], validated=validate_from_role(inconclusive))
