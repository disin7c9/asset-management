"""Risk metrics: drawdown family + risk-adjusted ratios with confidence bands.

Pure functions over a daily-return Series (and its growth-of-1 index). No I/O.

Drawdown is computed on the **time-weighted index** (investment drawdown,
contributions removed) — the standard, comparable, library-checkable
definition. `DrawdownInfo.depth` is a negative fraction (e.g. -0.23 = -23%);
`ulcer_index` and `cdar` are reported as positive magnitudes.

Risk-adjusted ratios (Sharpe / Sortino / Calmar) delegate to
`empyrical-reloaded` so they are correct by construction and golden-testable.

Confidence bands come from a **moving-block bootstrap** of the daily returns:
resample contiguous blocks (preserving local ordering, volatility clustering,
and the runs that drawdown depends on), recompute the metric, take percentiles.
An i.i.d. bootstrap would destroy the path that max-drawdown measures and would
understate the band for autocorrelated returns, so blocks are used instead.
With short histories the bands are wide — that honesty is the point.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any, NamedTuple

import empyrical
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Below this many return-days, treat every ratio as statistically noisy.
NOISY_THRESHOLD_DAYS = 504  # ~2 trading years


class MetricCI(NamedTuple):
    """A point estimate with a bootstrap confidence interval."""

    point: float
    low: float
    high: float

    @property
    def width(self) -> float:
        return self.high - self.low


@dataclass(frozen=True)
class DrawdownInfo:
    depth: float                 # negative fraction, e.g. -0.23
    peak_date: date
    trough_date: date
    recovery_date: date | None   # None if not yet recovered
    duration_days: int           # peak → recovery (or → last day if unrecovered)
    time_underwater_pct: float   # fraction of days below a prior peak


@dataclass(frozen=True)
class DollarDrawdown:
    """The 'gains given back' drawdown: the largest dollar decline in cumulative
    market P&L (`returns.pnl_curve`) — how many dollars of profit you watched
    evaporate from a peak.

    Flow-neutral: deposits, withdrawals, and trades all cancel in P&L, so funding
    and broker transfers don't distort it (the raw value/account-value curves dip
    on sells and transfers respectively, fabricating drawdowns). Realized-path
    figure, no bootstrap CI."""

    giveback_dollars: float   # negative: trough_pnl − peak_pnl
    peak_pnl: float           # cumulative market P&L at the peak
    trough_pnl: float         # cumulative market P&L at the trough
    peak_date: date
    trough_date: date
    recovery_date: date | None
    duration_days: int


@dataclass(frozen=True)
class RiskSummary:
    n_days: int
    drawdown: DrawdownInfo
    max_drawdown_ci: MetricCI    # depth as a negative fraction, with band
    ulcer_index: MetricCI        # positive magnitude, with band
    cdar: MetricCI               # positive magnitude (avg of worst-5% drawdowns), with band
    sharpe: MetricCI
    sortino: MetricCI
    calmar: MetricCI

    @property
    def is_noisy(self) -> bool:
        return self.n_days < NOISY_THRESHOLD_DAYS


# ── drawdown family (operate on a growth-of-1 index) ──────────────────────


def _drawdown_curve(index: "pd.Series[float]") -> "pd.Series[float]":
    """Fractional drawdown at each point: index / running_peak - 1 (≤ 0)."""
    running_peak = index.cummax()
    return index / running_peak - 1.0


def _drawdown_walk(
    level: "pd.Series[float]", drop: "pd.Series[float]"
) -> "tuple[Any, Any, Any | None, int]":
    """Peak / trough / recovery index labels + duration (days) of the deepest
    drawdown.

    `drop` (≤ 0) defines the trough via its idxmin — fractional for the growth
    index, dollars for the P&L curve; `level` defines the peak (its running max up
    to the trough) and the recovery threshold. Labels are returned RAW (the index
    may be ints in tests), so callers wrap `pd.Timestamp` for `.date()`. Shared by
    max_drawdown + dollar_drawdown so the recovery / duration walk lives once.
    """
    trough_ts = drop.idxmin()
    peak_ts = level.loc[:trough_ts].idxmax()
    peak_level = float(level.loc[peak_ts])
    post = level.loc[trough_ts:]
    recovered = post[post >= peak_level]
    recovery_ts = recovered.index[0] if len(recovered) > 0 else None
    end_ts = recovery_ts if recovery_ts is not None else level.index[-1]
    duration_days = int((pd.Timestamp(end_ts) - pd.Timestamp(peak_ts)).days)
    return peak_ts, trough_ts, recovery_ts, duration_days


def max_drawdown(index: "pd.Series[float]") -> DrawdownInfo:
    """Largest peak-to-trough decline of the index, with dates + recovery."""
    dd = _drawdown_curve(index)
    peak_ts, trough_ts, recovery_ts, duration_days = _drawdown_walk(index, dd)
    return DrawdownInfo(
        depth=float(dd.loc[trough_ts]),
        peak_date=pd.Timestamp(peak_ts).date(),
        trough_date=pd.Timestamp(trough_ts).date(),
        recovery_date=pd.Timestamp(recovery_ts).date() if recovery_ts is not None else None,
        duration_days=duration_days,
        time_underwater_pct=float((dd < 0).mean()),
    )


def dollar_drawdown(pnl_curve: "pd.Series[float]") -> DollarDrawdown | None:
    """Largest dollar decline in cumulative market P&L — 'gains given back'.

    Operates on `returns.pnl_curve` (flow-neutral), so it's the felt dollar loss
    undistorted by funding or broker transfers. Returns None for < 2 points, or
    when the curve never declines from a peak (nothing given back).
    """
    if len(pnl_curve) < 2:
        return None
    drop = pnl_curve - pnl_curve.cummax()  # dollars below the prior P&L peak (≤ 0)
    if float(drop.min()) >= -1e-9:
        return None  # monotonic / no decline → nothing given back (no degenerate line)
    peak_ts, trough_ts, recovery_ts, duration_days = _drawdown_walk(pnl_curve, drop)
    peak_pnl = float(pnl_curve.loc[peak_ts])
    trough_pnl = float(pnl_curve.loc[trough_ts])
    return DollarDrawdown(
        giveback_dollars=trough_pnl - peak_pnl,
        peak_pnl=peak_pnl,
        trough_pnl=trough_pnl,
        peak_date=pd.Timestamp(peak_ts).date(),
        trough_date=pd.Timestamp(trough_ts).date(),
        recovery_date=pd.Timestamp(recovery_ts).date() if recovery_ts is not None else None,
        duration_days=duration_days,
    )


def ulcer_index(index: "pd.Series[float]") -> float:
    """Root-mean-square drawdown — penalizes deep AND long drawdowns. Positive."""
    dd = _drawdown_curve(index)
    return float(np.sqrt(np.mean(np.square(dd.to_numpy()))))


def cdar(index: "pd.Series[float]", alpha: float = 0.05) -> float:
    """Conditional Drawdown-at-Risk: mean of the worst `alpha` fraction of drawdowns.

    Returned as a positive magnitude. alpha=0.05 → average of the worst 5% of days.
    Takes the worst ``ceil(alpha·N)`` days explicitly (at least one), NOT the days
    at/above the (1−alpha) quantile: when fewer than alpha of the days are underwater
    the quantile is 0.0 and a ``>= threshold`` filter sweeps in every zero-drawdown day,
    collapsing CDaR to the overall mean and understating the true worst tail (~20× on a
    mostly-at-peak leg) — which since v2.9.0 also blinds the verdict's CDaR gate.
    """
    dd_mag = (-_drawdown_curve(index)).to_numpy()  # positive magnitudes
    n = dd_mag.size
    if n == 0:
        return 0.0
    k = max(1, int(np.ceil(alpha * n)))          # worst-alpha count, at least one day
    worst = np.partition(dd_mag, n - k)[n - k:]  # the k deepest magnitudes (O(n), unordered)
    return float(worst.mean())


# ── risk-adjusted ratios (delegate to empyrical) ──────────────────────────


def sharpe(returns: "pd.Series[float]") -> float:
    return float(empyrical.sharpe_ratio(returns))


def sortino(returns: "pd.Series[float]") -> float:
    return float(empyrical.sortino_ratio(returns))


def calmar(returns: "pd.Series[float]") -> float:
    return float(empyrical.calmar_ratio(returns))


def _max_drawdown_depth(returns: "pd.Series[float]") -> float:
    """Max-drawdown depth (negative fraction) straight from a return series."""
    return float(empyrical.max_drawdown(returns))


def ulcer_from_returns(returns: "pd.Series[float]") -> float:
    """Ulcer index from a return series (builds the growth index internally).

    Public: the ONE returns→Ulcer bridge — `summarize_risk`'s CI and `backtest`'s
    held-out verdict both use it, so the definition can't drift."""
    return ulcer_index((1.0 + returns).cumprod())


def cdar_from_returns(returns: "pd.Series[float]") -> float:
    """CDaR from a return series (builds the growth index internally).

    Public for the same reason as `ulcer_from_returns` (backtest's verdict)."""
    return cdar((1.0 + returns).cumprod())


# ── bootstrap confidence intervals ────────────────────────────────────────


def path_block(n: int) -> int:
    """Moving-block size for path-dependent statistics (~√n, at least 2). Shared by
    ``summarize_risk``'s max-DD/Ulcer/CDaR/Calmar CIs and ``backtest``'s paired Ulcer-gain
    CI, so the block rule can't drift between the point panel and the held-out verdict.
    Calmar qualifies through its max-drawdown denominator, not by being an extremum."""
    return max(2, int(round(n**0.5)))


def moving_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Indices for one moving-block resample of length `n` (blocks of size `block`).

    Public: the ONE block resampler — `bootstrap_ci` here and `backtest`'s paired
    drawdown-difference CI both use it, so the resampling rule can't drift."""
    n_blocks = (n + block - 1) // block
    max_start = max(1, n - block + 1)
    starts = rng.integers(0, max_start, size=n_blocks)
    offsets = np.arange(block)
    return (starts[:, None] + offsets[None, :]).ravel()[:n]


def bootstrap_ci(
    metric_fn: Callable[["pd.Series[float]"], float],
    returns: "pd.Series[float]",
    *,
    n: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    block: int | None = None,
) -> MetricCI:
    """Moving-block bootstrap CI for a return-series metric.

    Resamples contiguous blocks of daily returns `n` times (preserving the
    local ordering that drawdown depends on and the autocorrelation that makes
    an i.i.d. CI overconfident), recomputes the metric on each resample, and
    returns the point estimate plus the central `ci` percentile band. Block
    length defaults to ~n**(1/3) (a standard heuristic). NaN/inf resample
    results are skipped; a non-finite point estimate yields a degenerate band
    so the report can render it as n/a.
    """
    point = float(metric_fn(returns))
    arr = returns.to_numpy()
    if arr.size < 2 or not np.isfinite(point):
        return MetricCI(point, point, point)
    blk = block if block is not None else max(2, round(arr.size ** (1.0 / 3.0)))
    blk = min(blk, arr.size)
    rng = np.random.default_rng(seed)
    idx = returns.index
    samples: list[float] = []
    for _ in range(n):
        bidx = moving_block_indices(arr.size, blk, rng)
        resampled = pd.Series(arr[bidx], index=idx[: len(bidx)])
        try:
            val = float(metric_fn(resampled))
        except Exception:  # noqa: BLE001,S112 — a degenerate resample shouldn't kill the CI (or log 1000×)
            continue
        if np.isfinite(val):
            samples.append(val)
    if not samples:
        return MetricCI(point, point, point)
    lo_q = (1.0 - ci) / 2.0
    hi_q = 1.0 - lo_q
    low = float(np.quantile(samples, lo_q))
    high = float(np.quantile(samples, hi_q))
    return MetricCI(point, low, high)


def summarize_risk(
    daily_returns: "pd.Series[float]",
    index: "pd.Series[float]",
    *,
    bootstrap_n: int = 1000,
    seed: int = 42,
) -> RiskSummary | None:
    """Compute the full risk panel. Returns None if there isn't enough data."""
    if daily_returns.empty or len(daily_returns) < 2 or index.empty:
        return None
    # Path-dependent extrema (max-DD, Ulcer, CDaR) need a LARGER bootstrap block
    # than mean-type ratios: an ~n**(1/3) block can't reassemble a multi-month
    # decline, biasing those bands shallow. Use ~√n for them; ratios keep the default.
    # CALMAR joins them (v2.12.2): it is annualized return ÷ max drawdown, so it is not a
    # mean-type ratio — it INHERITS the denominator's path dependence, and was reading the
    # default block only because it sits among the ratios. Measured on cached series the
    # block choice moved its CI width by -19%..+9% and shifted one interval from
    # straddling zero to excluding it, so this is not cosmetic.
    pblock = path_block(len(daily_returns))
    return RiskSummary(
        n_days=len(daily_returns),
        drawdown=max_drawdown(index),
        max_drawdown_ci=bootstrap_ci(
            _max_drawdown_depth, daily_returns, n=bootstrap_n, seed=seed, block=pblock
        ),
        ulcer_index=bootstrap_ci(
            ulcer_from_returns, daily_returns, n=bootstrap_n, seed=seed, block=pblock
        ),
        cdar=bootstrap_ci(
            cdar_from_returns, daily_returns, n=bootstrap_n, seed=seed, block=pblock
        ),
        sharpe=bootstrap_ci(sharpe, daily_returns, n=bootstrap_n, seed=seed),
        sortino=bootstrap_ci(sortino, daily_returns, n=bootstrap_n, seed=seed),
        calmar=bootstrap_ci(calmar, daily_returns, n=bootstrap_n, seed=seed, block=pblock),
    )
