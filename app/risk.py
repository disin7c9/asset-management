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
from typing import NamedTuple

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


def max_drawdown(index: "pd.Series[float]") -> DrawdownInfo:
    """Largest peak-to-trough decline of the index, with dates + recovery."""
    dd = _drawdown_curve(index)
    trough_ts = dd.idxmin()
    depth = float(dd.loc[trough_ts])
    # Peak = the date the running max was set at/just before the trough.
    pre_trough = index.loc[:trough_ts]
    peak_ts = pre_trough.idxmax()
    peak_value = float(index.loc[peak_ts])

    # Recovery = first day after the trough reaching the prior peak again.
    post = index.loc[trough_ts:]
    recovered = post[post >= peak_value]
    recovery_ts = recovered.index[0] if len(recovered) > 0 else None

    last_ts = index.index[-1]
    end_ts = recovery_ts if recovery_ts is not None else last_ts
    duration_days = int((pd.Timestamp(end_ts) - pd.Timestamp(peak_ts)).days)
    time_underwater = float((dd < 0).mean())

    return DrawdownInfo(
        depth=depth,
        peak_date=pd.Timestamp(peak_ts).date(),
        trough_date=pd.Timestamp(trough_ts).date(),
        recovery_date=pd.Timestamp(recovery_ts).date() if recovery_ts is not None else None,
        duration_days=duration_days,
        time_underwater_pct=time_underwater,
    )


def ulcer_index(index: "pd.Series[float]") -> float:
    """Root-mean-square drawdown — penalizes deep AND long drawdowns. Positive."""
    dd = _drawdown_curve(index)
    return float(np.sqrt(np.mean(np.square(dd.to_numpy()))))


def cdar(index: "pd.Series[float]", alpha: float = 0.05) -> float:
    """Conditional Drawdown-at-Risk: mean of the worst `alpha` fraction of drawdowns.

    Returned as a positive magnitude. alpha=0.05 → average of the worst 5%.
    """
    dd_mag = (-_drawdown_curve(index)).to_numpy()  # positive magnitudes
    if dd_mag.size == 0:
        return 0.0
    threshold = float(np.quantile(dd_mag, 1.0 - alpha))
    tail = dd_mag[dd_mag >= threshold]
    return float(tail.mean()) if tail.size > 0 else float(dd_mag.max())


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


def _ulcer_from_returns(returns: "pd.Series[float]") -> float:
    """Ulcer index from a return series (builds the growth index internally)."""
    return ulcer_index((1.0 + returns).cumprod())


def _cdar_from_returns(returns: "pd.Series[float]") -> float:
    """CDaR from a return series (builds the growth index internally)."""
    return cdar((1.0 + returns).cumprod())


# ── bootstrap confidence intervals ────────────────────────────────────────


def _moving_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Indices for one moving-block resample of length `n` (blocks of size `block`)."""
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
        bidx = _moving_block_indices(arr.size, blk, rng)
        resampled = pd.Series(arr[bidx], index=idx[: len(bidx)])
        try:
            val = float(metric_fn(resampled))
        except Exception:  # noqa: BLE001 — a degenerate resample shouldn't kill the CI
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
    path_block = max(2, round(len(daily_returns) ** 0.5))
    return RiskSummary(
        n_days=len(daily_returns),
        drawdown=max_drawdown(index),
        max_drawdown_ci=bootstrap_ci(
            _max_drawdown_depth, daily_returns, n=bootstrap_n, seed=seed, block=path_block
        ),
        ulcer_index=bootstrap_ci(
            _ulcer_from_returns, daily_returns, n=bootstrap_n, seed=seed, block=path_block
        ),
        cdar=bootstrap_ci(
            _cdar_from_returns, daily_returns, n=bootstrap_n, seed=seed, block=path_block
        ),
        sharpe=bootstrap_ci(sharpe, daily_returns, n=bootstrap_n, seed=seed),
        sortino=bootstrap_ci(sortino, daily_returns, n=bootstrap_n, seed=seed),
        calmar=bootstrap_ci(calmar, daily_returns, n=bootstrap_n, seed=seed),
    )
