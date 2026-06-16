"""Tests for the notional backtest engine. Pure — synthetic price series."""

from __future__ import annotations

import math
from datetime import date

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

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


# ── property-based invariants ───────────────────────────────────────────────

_TICKERS = ["A", "B", "C"]
_finite = {"allow_nan": False, "allow_infinity": False}


@st.composite
def _price_book(draw: st.DrawFn) -> tuple[dict[str, "pd.Series[float]"], dict[str, float]]:
    """A random set of strictly-positive daily price paths + a normalized target."""
    n_tk = draw(st.integers(min_value=1, max_value=3))
    n_days = draw(st.integers(min_value=35, max_value=80))
    dates = pd.bdate_range("2024-01-03", periods=n_days)
    series: dict[str, pd.Series[float]] = {}
    raw_w: list[float] = []
    for i in range(n_tk):
        start_px = draw(st.floats(min_value=5.0, max_value=400.0, **_finite))
        rets = draw(
            st.lists(
                st.floats(min_value=-0.08, max_value=0.08, **_finite),
                min_size=n_days - 1, max_size=n_days - 1,
            )
        )
        px = [start_px]
        for r in rets:
            px.append(max(0.01, px[-1] * (1.0 + r)))  # stay strictly positive
        series[_TICKERS[i]] = pd.Series(px, index=dates, dtype=float)
        raw_w.append(draw(st.floats(min_value=0.1, max_value=1.0, **_finite)))
    total = sum(raw_w)
    target = {_TICKERS[i]: raw_w[i] / total for i in range(n_tk)}
    return series, target


@settings(max_examples=40, deadline=None)
@given(_price_book())
def test_prop_curves_start_at_initial_and_stay_positive(
    book: tuple[dict[str, "pd.Series[float]"], dict[str, float]],
) -> None:
    series, target = book
    for sched in ("never", "monthly", "quarterly"):
        curve = simulate(series, target, schedule=sched, initial=10_000.0)
        assert not curve.empty
        assert curve.iloc[0] == pytest.approx(10_000.0)  # weights sum to 1 → day-0 = initial
        assert curve.notna().all() and (curve > 0).all()  # no NaN, never wiped out


@settings(max_examples=30, deadline=None)
@given(st.integers(min_value=40, max_value=80), st.floats(min_value=1.0, max_value=500.0, **_finite))
def test_prop_flat_market_stays_at_initial(n_days: int, px: float) -> None:
    # Constant prices → every day is worth `initial`, and rebalancing changes nothing.
    dates = pd.bdate_range("2024-01-03", periods=n_days)
    series = {
        "A": pd.Series([px] * n_days, index=dates, dtype=float),
        "B": pd.Series([px * 2.0] * n_days, index=dates, dtype=float),
    }
    for sched in ("never", "monthly", "quarterly"):
        curve = simulate(series, {"A": 0.5, "B": 0.5}, schedule=sched, initial=10_000.0)
        assert curve.min() == pytest.approx(10_000.0)
        assert curve.max() == pytest.approx(10_000.0)


@settings(max_examples=25, deadline=None)
@given(_price_book())
def test_prop_compare_legs_are_finite_and_well_formed(
    book: tuple[dict[str, "pd.Series[float]"], dict[str, float]],
) -> None:
    series, target = book
    res = backtest_compare(series, target, schedule="monthly", bootstrap_n=20, seed=1)
    if res is None:  # too-short a window to score is an acceptable graceful outcome
        return
    assert len(res.legs) == 2
    assert res.start <= res.end
    for leg in res.legs:
        assert math.isfinite(leg.final_value) and leg.final_value > 0
        # max drawdown ≤ 0 with an ordered bootstrap band. The point estimate may
        # fall OUTSIDE the band: resampling tends to break up the worst consecutive
        # run, so the realized maxDD is often deeper than the resampled 95% — honest,
        # not a bug. So assert the band is ordered + non-positive, not point-within.
        dd = leg.risk.max_drawdown_ci
        assert dd.low <= dd.high <= 1e-9
        assert dd.point <= 1e-9


# --- walk-forward role check (v1.9.0) ---


def _close_path(rets: list[float], start: str = "2024-01-02") -> "pd.Series[float]":
    idx = pd.bdate_range(start, periods=len(rets))
    return (1.0 + pd.Series(rets, index=idx, dtype=float)).cumprod() * 100.0


def test_role_check_hedge_improves_oos() -> None:
    # Target asset rises in-sample then crashes in the held-out window; the
    # candidate rises through the crash. A big sleeve → decisively shallower
    # OOS drawdown → "improved" (seeded bootstrap keeps the verdict stable).
    from app.backtest import role_check

    aaa = _close_path([0.001] * 210 + [-0.005] * 60 + [0.001] * 30)
    hedge = _close_path([0.0002] * 210 + [0.004] * 90)
    rc = role_check({"AAA": aaa, "HEDGE": hedge}, {"AAA": 1.0}, "HEDGE", sleeve=0.5)
    assert rc.verdict == "improved"
    oos = rc.oos
    assert oos is not None and oos.dd_with > oos.dd_without  # shallower drawdown
    assert "OOS" in rc.reason and "overfitting gap" in rc.reason


def test_role_check_amplifier_worsens_oos() -> None:
    # The candidate is a 3x clone of the target asset: the OOS crash gets
    # deeper with the sleeve → "worsened".
    from app.backtest import role_check

    rets = [0.001] * 210 + [-0.005] * 60 + [0.001] * 30
    aaa = _close_path(rets)
    lev = _close_path([3 * r for r in rets])
    rc = role_check({"AAA": aaa, "LEV": lev}, {"AAA": 1.0}, "LEV", sleeve=0.5)
    assert rc.verdict == "worsened"
    oos = rc.oos
    assert oos is not None and oos.dd_with < oos.dd_without  # deeper drawdown


def test_role_check_clone_is_inconclusive() -> None:
    # A candidate identical to the target changes nothing → differences are
    # inside the noise margins → "inconclusive", honestly.
    from app.backtest import role_check

    rets = [0.002, -0.001] * 150
    aaa = _close_path(rets)
    clone = _close_path(rets)
    rc = role_check({"AAA": aaa, "CLONE": clone}, {"AAA": 1.0}, "CLONE", sleeve=0.5)
    assert rc.verdict == "inconclusive"
    assert "no clear out-of-sample improvement" in rc.reason


def test_role_check_insufficient_data_paths() -> None:
    from app.backtest import role_check

    short = _close_path([0.001] * 80)  # 80 common days → OOS window below the floor
    aaa = _close_path([0.001] * 80)
    rc = role_check({"AAA": aaa, "NEW": short}, {"AAA": 1.0}, "NEW")
    assert rc.verdict == "insufficient" and "common days" in rc.reason

    rc2 = role_check({"AAA": aaa}, {"AAA": 1.0}, "GHOST")
    assert rc2.verdict == "insufficient" and "no usable price history" in rc2.reason

    rc3 = role_check({"AAA": aaa}, {"AAA": 1.0}, "AAA")
    assert rc3.verdict == "insufficient" and "already in the target" in rc3.reason

    rc4 = role_check({"AAA": aaa, "NEW": short}, {"AAA": 1.0}, "NEW", sleeve=1.5)
    assert rc4.verdict == "insufficient" and "sleeve" in rc4.reason


def test_role_check_refuses_renormalized_target() -> None:
    # Regression (review 2026-06-12): a target ticker with no usable history must
    # make the verdict "insufficient" — simulate would silently renormalize the
    # rest and the verdict would be about a DIFFERENT target than the user's.
    from app.backtest import role_check

    aaa = _close_path([0.001] * 300)
    cand = _close_path([0.002] * 300)
    rc = role_check({"AAA": aaa, "NEW": cand}, {"AAA": 0.6, "GONE": 0.4}, "NEW")
    assert rc.verdict == "insufficient"
    assert "GONE" in rc.reason and "renormalized" in rc.reason


def test_role_check_aligns_mismatched_calendars() -> None:
    # The candidate trades on a sparser calendar (different exchange): the legs'
    # daily returns are inner-joined BY DATE, so the paired comparison never
    # pairs different days. Must run end-to-end without crashing and report a
    # coherent OOS window.
    from app.backtest import role_check

    rets = [0.002, -0.001] * 150
    aaa = _close_path(rets)
    cand_full = _close_path([0.001] * 300)
    cand_sparse = cand_full[::2]  # every other trading day only
    rc = role_check({"AAA": aaa, "SPARSE": cand_sparse}, {"AAA": 1.0}, "SPARSE", sleeve=0.5)
    assert rc.verdict in ("improved", "worsened", "inconclusive", "insufficient")
    oos = rc.oos
    if oos is not None:
        assert oos.n_days > 0  # aligned length, one number for both legs


# --- validate_from_role: the verdict → edge-gate bridge (v2.0.0 Phase 0) ---


def test_validate_from_role_only_improved_opens_the_gate() -> None:
    # The verdict→bool bridge to the edge gate. ONLY a held-out "improved" (which
    # already required the paired-bootstrap CI to exclude zero) opens the gate;
    # every other verdict keeps it shut.
    from app.backtest import RoleCheck, RoleVerdict, validate_from_role

    def rc(verdict: RoleVerdict) -> RoleCheck:
        return RoleCheck("C", 0.05, "quarterly", (), verdict, "t")

    assert validate_from_role(rc("improved")) is True
    assert validate_from_role(rc("worsened")) is False
    assert validate_from_role(rc("inconclusive")) is False
    assert validate_from_role(rc("insufficient")) is False


def test_validate_from_role_end_to_end() -> None:
    # The real bridge: a decisive OOS hedge → role_check "improved" → gate opens;
    # a clone → "inconclusive" → gate stays shut.
    from app.backtest import role_check, validate_from_role

    aaa = _close_path([0.001] * 210 + [-0.005] * 60 + [0.001] * 30)
    hedge = _close_path([0.0002] * 210 + [0.004] * 90)
    rc = role_check({"AAA": aaa, "HEDGE": hedge}, {"AAA": 1.0}, "HEDGE", sleeve=0.5)
    assert validate_from_role(rc) is True

    rets = [0.002, -0.001] * 150
    clone_book = {"AAA": _close_path(rets), "CLONE": _close_path(rets)}
    rc2 = role_check(clone_book, {"AAA": 1.0}, "CLONE", sleeve=0.5)
    assert validate_from_role(rc2) is False
