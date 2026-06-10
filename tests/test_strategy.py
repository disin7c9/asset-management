"""Tests for the strategy/suggestion engine. Pure — no I/O (load_target moved to events.py)."""

from __future__ import annotations

import math

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from app.strategy import (
    Suggestion,
    may_suggest,
    strategy_kind,
    suggest,
)

# A simple, hand-checkable book: $10,000 split 60/40 VOO/BND; IAU is a buy-in.
HELD = {"VOO": 6000.0, "BND": 4000.0}
PRICES = {"VOO": 300.0, "BND": 80.0, "IAU": 40.0}
TARGET = {"VOO": 0.50, "BND": 0.25, "IAU": 0.25}


def _by_ticker(sugs: list[Suggestion]) -> dict[str, Suggestion]:
    return {s.ticker: s for s in sugs}


# ── to_total ───────────────────────────────────────────────────────────────


def test_to_total_trades_to_target_cash_neutral() -> None:
    s = _by_ticker(suggest("to_total", HELD, PRICES, TARGET))
    assert s["VOO"].action == "sell" and s["VOO"].dollars == pytest.approx(1000.0)
    assert s["VOO"].shares == pytest.approx(1000.0 / 300.0)
    assert s["BND"].action == "sell" and s["BND"].dollars == pytest.approx(1500.0)
    assert s["IAU"].action == "buy" and s["IAU"].dollars == pytest.approx(2500.0)
    # Rebalance is cash-neutral: buys == sells.
    buys = sum(x.dollars for x in s.values() if x.action == "buy")
    sells = sum(x.dollars for x in s.values() if x.action == "sell")
    assert buys == pytest.approx(sells)


def test_to_total_resulting_weights_equal_target() -> None:
    s = _by_ticker(suggest("to_total", HELD, PRICES, TARGET))
    base = sum(HELD.values())
    for tk, tgt in TARGET.items():
        signed = s[tk].dollars * (1 if s[tk].action == "buy" else -1 if s[tk].action == "sell" else 0)
        resulting = HELD.get(tk, 0.0) + signed
        assert resulting / base == pytest.approx(tgt)


def test_to_total_sells_out_ticker_absent_from_target() -> None:
    held = {"VOO": 5000.0, "SCHP": 5000.0}
    s = _by_ticker(suggest("to_total", held, {"VOO": 300.0, "SCHP": 25.0}, {"VOO": 1.0}))
    assert s["SCHP"].action == "sell" and s["SCHP"].dollars == pytest.approx(5000.0)
    assert s["SCHP"].target_weight == 0.0
    # The reason makes clear this is a full exit (target 0%).
    assert "full exit" in s["SCHP"].reason


def test_explicit_zero_weight_exits_position() -> None:
    # A held ticker listed at weight 0 → sold to $0, same as omitting it.
    held = {"VOO": 6000.0, "GLDM": 2000.0}
    s = _by_ticker(suggest("to_total", held, {"VOO": 300.0, "GLDM": 100.0},
                           {"VOO": 1.0, "GLDM": 0.0}))
    assert s["GLDM"].action == "sell" and s["GLDM"].dollars == pytest.approx(2000.0)
    assert s["GLDM"].target_weight == 0.0


def test_to_total_deploys_new_cash() -> None:
    # 10k held, +2k cash, single-target VOO → buy the whole 2k of VOO.
    s = _by_ticker(suggest("to_total", {"VOO": 10000.0}, {"VOO": 100.0},
                           {"VOO": 1.0}, new_cash=2000.0))
    assert s["VOO"].action == "buy" and s["VOO"].dollars == pytest.approx(2000.0)


# ── cash_flow_only ─────────────────────────────────────────────────────────


def test_cash_flow_only_buys_underweight_never_sells() -> None:
    s = _by_ticker(suggest("cash_flow_only", HELD, PRICES, TARGET, new_cash=1000.0))
    assert all(x.action in ("buy", "hold") for x in s.values())  # never sells
    # VOO & BND are above their post-cash target → hold; IAU underweight → gets all cash.
    assert s["IAU"].action == "buy" and s["IAU"].dollars == pytest.approx(1000.0)
    assert s["VOO"].action == "hold"
    assert s["BND"].action == "hold"


def test_cash_flow_only_no_cash_is_all_hold() -> None:
    s = _by_ticker(suggest("cash_flow_only", HELD, PRICES, TARGET, new_cash=0.0))
    assert all(x.action == "hold" for x in s.values())


# ── fixed_dca ──────────────────────────────────────────────────────────────


def test_fixed_dca_buys_target_mix_ignoring_drift() -> None:
    s = _by_ticker(suggest("fixed_dca", HELD, PRICES, TARGET, new_cash=1000.0))
    assert all(x.action in ("buy", "hold") for x in s.values())
    assert s["VOO"].dollars == pytest.approx(500.0)   # 50% of 1000
    assert s["BND"].dollars == pytest.approx(250.0)
    assert s["IAU"].dollars == pytest.approx(250.0)


# ── bands ──────────────────────────────────────────────────────────────────


def test_bands_holds_within_band_acts_outside() -> None:
    # VOO 60% vs 55% target = 5pp drift (== band → hold); BND 40% vs 45% (within → hold).
    s = _by_ticker(suggest("bands", HELD, PRICES, {"VOO": 0.55, "BND": 0.45}, band=0.05))
    assert s["VOO"].action == "hold" and "within" in s["VOO"].reason
    assert s["BND"].action == "hold"


def test_bands_acts_when_drift_exceeds_band() -> None:
    # VOO 60% vs 25% target → 35pp drift > 5pp → sell.
    s = _by_ticker(suggest("bands", HELD, PRICES, TARGET, band=0.05))
    assert s["VOO"].action == "sell"
    assert "band" in s["VOO"].reason


# ── edges ──────────────────────────────────────────────────────────────────


def test_unpriced_target_ticker_is_skipped() -> None:
    # IAU has no price → dropped from the universe, not suggested.
    s = _by_ticker(suggest("to_total", HELD, {"VOO": 300.0, "BND": 80.0}, TARGET))
    assert "IAU" not in s


# ── discipline-vs-edge gate ────────────────────────────────────────────────


def test_all_v1_modes_are_discipline_and_may_suggest() -> None:
    for mode in ("to_total", "cash_flow_only", "fixed_dca", "bands"):
        assert strategy_kind(mode) == "discipline"
        assert may_suggest(mode) is True


def test_unknown_mode_is_edge_and_gated() -> None:
    # An unrecognized/future strategy defaults to edge → blocked until validated.
    assert strategy_kind("momentum") == "edge"
    assert may_suggest("momentum") is False
    assert may_suggest("momentum", backtest_validated=True) is True


def test_drift_property() -> None:
    s = _by_ticker(suggest("to_total", HELD, PRICES, TARGET))
    assert s["VOO"].drift == pytest.approx(s["VOO"].current_weight - s["VOO"].target_weight)


def test_zero_price_ticker_skipped_no_crash() -> None:
    # A 0.0 price (bad/stale quote) must be skipped, not divided by → no crash.
    s = _by_ticker(suggest("to_total", {"VOO": 6000.0, "BND": 4000.0},
                           {"VOO": 300.0, "BND": 0.0}, {"VOO": 0.5, "BND": 0.5}))
    assert "BND" not in s
    assert "VOO" in s


def test_bands_boundary_deterministic_hold() -> None:
    # Nominal exactly-5pp drift holds regardless of float rounding of the weight diff.
    s = _by_ticker(suggest("bands", {"VOO": 5500.0, "BND": 4500.0},
                           {"VOO": 100.0, "BND": 100.0}, {"VOO": 0.50, "BND": 0.50},
                           band=0.05))
    assert s["VOO"].action == "hold" and s["BND"].action == "hold"


def test_suggest_refuses_edge_mode_at_the_chokepoint() -> None:
    # The gate is enforced INSIDE suggest(), so any caller is refused an unknown/
    # edge mode (not just the CLI).
    with pytest.raises(ValueError, match="edge strategy"):
        suggest("momentum", HELD, PRICES, TARGET)  # type: ignore[arg-type]


# ── property-based invariants (the v1 product) ──────────────────────────────

_TICKERS = ["A", "B", "C", "D", "E"]
_finite = {"allow_nan": False, "allow_infinity": False}


@st.composite
def _books(draw: st.DrawFn) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """A random (held_value, prices, target) over a shared, fully-priced universe.

    Target weights are normalized to sum to 1 (as `load_target` would), so the
    'resulting weight == target' and cash-neutral identities hold exactly. A
    weight may be 0 (a deliberate exit). Every ticker has a positive price, so
    none are dropped — the invariants quantify over the whole universe."""
    n = draw(st.integers(min_value=1, max_value=5))
    tickers = _TICKERS[:n]
    held = {tk: draw(st.floats(min_value=0.0, max_value=1e6, **_finite)) for tk in tickers}
    prices = {tk: draw(st.floats(min_value=1.0, max_value=1000.0, **_finite)) for tk in tickers}
    weights = [draw(st.floats(min_value=0.0, max_value=1.0, **_finite)) for _ in tickers]
    assume(sum(weights) > 0)  # a target must be normalizable
    total_w = sum(weights)
    target = {tk: w / total_w for tk, w in zip(tickers, weights, strict=True)}
    return held, prices, target


@given(_books())
def test_prop_to_total_is_cash_neutral(book: tuple[dict[str, float], ...]) -> None:
    # With no new cash, Σbuys == Σsells (up to the per-ticker $1 min-trade hold).
    held, prices, target = book
    sugs = suggest("to_total", held, prices, target, new_cash=0.0)
    buys = sum(s.dollars for s in sugs if s.action == "buy")
    sells = sum(s.dollars for s in sugs if s.action == "sell")
    assert buys == pytest.approx(sells, abs=float(len(held)))  # ≤ n dropped sub-$1 trades


@given(_books())
def test_prop_to_total_reaches_target_weights(
    book: tuple[dict[str, float], ...],
) -> None:
    # Pin the OUTCOME (not just "moved toward"): after a cash-neutral to_total
    # rebalance every ticker's resulting weight EQUALS its target — exactly when
    # traded, within one min-trade dollar's worth of weight when the drift was
    # sub-$1 and the ticker held. This catches a wrong base / mis-normalization
    # (which would land resulting_w on tgt_w·base/total ≠ tgt_w); an invariant
    # reconstructed only from the suggestion's own dollars could not.
    held, prices, target = book
    total = sum(held.values())
    assume(total > 0)  # weights are undefined for an empty book
    sugs = {s.ticker: s for s in suggest("to_total", held, prices, target, new_cash=0.0)}
    for tk, s in sugs.items():
        signed = s.dollars * (1 if s.action == "buy" else -1 if s.action == "sell" else 0)
        resulting_w = (held.get(tk, 0.0) + signed) / total  # cash-neutral → base unchanged
        tgt_w = target.get(tk, 0.0)
        assert abs(resulting_w - tgt_w) <= 1.0 / total + 1e-9  # ≤ one min-trade dollar


@given(_books(), st.floats(min_value=0.0, max_value=1e5, **_finite))
def test_prop_cash_modes_within_budget_and_never_sell(
    book: tuple[dict[str, float], ...], new_cash: float
) -> None:
    held, prices, target = book
    for mode in ("cash_flow_only", "fixed_dca"):
        sugs = suggest(mode, held, prices, target, new_cash=new_cash)  # type: ignore[arg-type]
        assert all(s.action in ("buy", "hold") for s in sugs)  # tax-friendly: never sells
        spent = sum(s.dollars for s in sugs if s.action == "buy")
        assert spent <= new_cash + 1e-6 + new_cash * 1e-9  # never deploy more than available


@given(_books(), st.floats(min_value=0.0, max_value=1e5, **_finite))
def test_prop_no_negative_or_nan_trades(
    book: tuple[dict[str, float], ...], new_cash: float
) -> None:
    held, prices, target = book
    for mode in ("to_total", "cash_flow_only", "fixed_dca", "bands"):
        for s in suggest(mode, held, prices, target, new_cash=new_cash):  # type: ignore[arg-type]
            assert math.isfinite(s.shares) and s.shares >= 0.0
            assert math.isfinite(s.dollars) and s.dollars >= 0.0
            assert (s.action == "hold") == (s.dollars == 0.0)  # hold ⟺ zero-dollar trade
