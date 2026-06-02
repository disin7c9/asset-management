"""Tests for the risk engine: drawdown family, ratios, bootstrap CIs.

Validation strategy (no hand calculation):
- Synthetic index curves with known peaks/troughs for the drawdown family.
- empyrical-reloaded as the canonical reference for max-drawdown / Sharpe.
- Fixed-seed reproducibility for the bootstrap.
- Property-based invariants (hypothesis) for the relationships between metrics.
"""

from __future__ import annotations

import empyrical
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.risk import (
    MetricCI,
    bootstrap_ci,
    calmar,
    cdar,
    max_drawdown,
    sharpe,
    sortino,
    summarize_risk,
    ulcer_index,
)


def _index(values: list[float]) -> "pd.Series[float]":
    idx = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float)


# ── drawdown family on synthetic curves ───────────────────────────────────


def test_max_drawdown_depth_and_dates() -> None:
    # peak 1.20 (day 2), trough 0.90 (day 3), recover to ≥1.20 at day 5 (1.25)
    index = _index([1.0, 1.1, 1.2, 0.9, 0.95, 1.25])
    info = max_drawdown(index)
    assert info.depth == pytest.approx(0.9 / 1.2 - 1.0)  # -0.25
    assert info.peak_date.isoformat() == "2024-01-03"
    assert info.trough_date.isoformat() == "2024-01-04"
    assert info.recovery_date is not None
    assert info.recovery_date.isoformat() == "2024-01-06"


def test_max_drawdown_unrecovered() -> None:
    index = _index([1.0, 1.2, 0.8, 0.85, 0.9])  # never returns to 1.2
    info = max_drawdown(index)
    assert info.depth == pytest.approx(0.8 / 1.2 - 1.0)
    assert info.recovery_date is None


def test_time_underwater() -> None:
    index = _index([1.0, 0.9, 0.95, 1.0, 1.1])  # days 1,2 below peak; day 3 back at peak
    info = max_drawdown(index)
    # days 1 and 2 are strictly below a prior peak → 2 of 5 = 40%
    assert info.time_underwater_pct == pytest.approx(0.4)


def test_max_drawdown_matches_empyrical() -> None:
    """Our index-based depth must equal empyrical's return-based max_drawdown."""
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.001, 0.02, 300))
    index = (1.0 + returns).cumprod()
    ours = max_drawdown(index).depth
    theirs = float(empyrical.max_drawdown(returns))
    assert ours == pytest.approx(theirs, abs=1e-9)


def test_ulcer_and_cdar_are_positive_magnitudes() -> None:
    index = _index([1.0, 0.9, 0.8, 0.85, 1.0])
    assert ulcer_index(index) > 0
    assert cdar(index) > 0


def test_cdar_not_greater_than_max_drawdown_magnitude() -> None:
    """CDaR (avg of worst tail) must be ≤ the single worst drawdown magnitude."""
    index = _index([1.0, 1.1, 0.7, 0.75, 0.95, 1.2])
    worst = abs(max_drawdown(index).depth)
    assert cdar(index) <= worst + 1e-12


# ── ratios delegate to empyrical ──────────────────────────────────────────


def test_sharpe_independent_formula_golden() -> None:
    """Independent golden: Sharpe = mean/std(ddof=1) * sqrt(252), computed in raw
    numpy — NOT by re-calling the same empyrical function our wrapper calls. This
    catches a wrapper that silently changed period or risk-free defaults."""
    rng = np.random.default_rng(1)
    arr = rng.normal(0.001, 0.02, 252)
    returns = pd.Series(arr)
    expected = (arr.mean() / arr.std(ddof=1)) * np.sqrt(252.0)
    assert sharpe(returns) == pytest.approx(expected, rel=1e-6)


def test_sharpe_sign_tracks_drift() -> None:
    """Positive-drift series → positive Sharpe; negative-drift → negative."""
    up = pd.Series(np.full(252, 0.001))
    down = pd.Series(np.full(252, -0.001))
    # add a little noise so std > 0
    rng = np.random.default_rng(9)
    up = up + rng.normal(0, 0.005, 252)
    down = down + rng.normal(0, 0.005, 252)
    assert sharpe(up) > 0
    assert sharpe(down) < 0


def test_calmar_matches_empyrical_with_explicit_defaults() -> None:
    """Delegation contract: our wrapper must equal empyrical called with the
    period we rely on (catches an accidental period/arg change in the wrapper)."""
    rng = np.random.default_rng(2)
    returns = pd.Series(rng.normal(0.001, 0.02, 500))
    assert calmar(returns) == pytest.approx(float(empyrical.calmar_ratio(returns, period="daily")))


def test_bootstrap_ci_non_finite_point_renders_degenerate() -> None:
    """Regression for review finding 2: an all-positive series gives calmar=nan /
    sortino=inf; bootstrap_ci must NOT crash and must return a degenerate band so
    the report can show n/a instead of '+nan'."""
    pos = pd.Series(np.full(60, 0.01))  # no drawdown, no downside
    ci_cal = bootstrap_ci(calmar, pos, n=100, seed=1)
    ci_sor = bootstrap_ci(sortino, pos, n=100, seed=1)
    assert not np.isfinite(ci_cal.point)  # nan (max drawdown 0 → div by zero)
    assert ci_cal.low == ci_cal.point or np.isnan(ci_cal.low)
    assert np.isinf(ci_sor.point) or np.isnan(ci_sor.point)


# ── bootstrap CI ───────────────────────────────────────────────────────────


def test_bootstrap_ci_is_reproducible() -> None:
    rng = np.random.default_rng(3)
    returns = pd.Series(rng.normal(0.001, 0.02, 252))
    a = bootstrap_ci(sharpe, returns, n=500, seed=123)
    b = bootstrap_ci(sharpe, returns, n=500, seed=123)
    assert a == b


def test_bootstrap_ci_brackets_point() -> None:
    rng = np.random.default_rng(4)
    returns = pd.Series(rng.normal(0.002, 0.015, 500))
    ci = bootstrap_ci(sharpe, returns, n=1000, seed=7)
    assert ci.low <= ci.point <= ci.high
    assert ci.width > 0


def test_bootstrap_ci_degenerate_short_input() -> None:
    returns = pd.Series([0.01], dtype=float)
    ci = bootstrap_ci(sharpe, returns, n=100, seed=1)
    assert ci == MetricCI(ci.point, ci.point, ci.point)


# ── summarize_risk ─────────────────────────────────────────────────────────


def test_summarize_risk_none_on_too_little_data() -> None:
    returns = pd.Series([0.01], dtype=float)
    index = pd.Series([1.01], dtype=float)
    assert summarize_risk(returns, index) is None


def test_summarize_risk_full_panel() -> None:
    rng = np.random.default_rng(5)
    returns = pd.Series(
        rng.normal(0.0008, 0.012, 600),
        index=pd.date_range("2022-01-01", periods=600, freq="D"),
    )
    index = (1.0 + returns).cumprod()
    summary = summarize_risk(returns, index, bootstrap_n=300)
    assert summary is not None
    assert summary.n_days == 600
    assert summary.is_noisy is False  # 600 > 504
    assert summary.drawdown.depth <= 0
    assert summary.sharpe.low <= summary.sharpe.point <= summary.sharpe.high


def test_summarize_risk_flags_noisy_short_history() -> None:
    rng = np.random.default_rng(6)
    returns = pd.Series(
        rng.normal(0.001, 0.02, 100),
        index=pd.date_range("2024-01-01", periods=100, freq="D"),
    )
    index = (1.0 + returns).cumprod()
    summary = summarize_risk(returns, index, bootstrap_n=200)
    assert summary is not None
    assert summary.is_noisy is True  # 100 < 504


# ── property-based invariants ──────────────────────────────────────────────


@given(
    rets=st.lists(
        st.floats(min_value=-0.2, max_value=0.2, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=200,
    )
)
@settings(max_examples=50)
def test_invariant_drawdown_non_positive_and_cdar_bounded(rets: list[float]) -> None:
    returns = pd.Series(rets, index=pd.date_range("2024-01-01", periods=len(rets), freq="D"))
    index = (1.0 + returns).cumprod()
    info = max_drawdown(index)
    assert info.depth <= 1e-12  # drawdown is ≤ 0
    assert ulcer_index(index) >= 0.0
    assert cdar(index) <= abs(info.depth) + 1e-9  # CDaR ≤ worst single drawdown
