"""Allocation rules — deterministic ways to *choose* a target allocation.

Where `strategy.py` decides how to **move** toward a target (the rebalance modes),
this module decides **what the target should be**: pure functions over a universe
and its price history. It plugs in one step upstream of the rebalance machinery —
its output is a target-weight dict, the same shape `strategy.load_target` returns.

All current rules are *discipline*: they make no claim to beat the market (no
return forecast), so they need no backtest to be honest and may always produce a
target — mirroring the `strategy.py` discipline-vs-edge gate. An *edge* allocator
(e.g. mean-variance optimization, which overfits out-of-sample — see `examples/`
05–06) would be classified ``edge`` and gated behind a walk-forward backtest.

Rules:
    equal_weight   1/N over the universe — the robust baseline.
    inverse_vol    weight ∝ 1/volatility — each holding contributes ~equal *risk*,
                   not equal dollars; realized vol only, no return forecast.

`apply_caps` enforces a per-asset weight ceiling (re-normalizing the rest) — a
guard-rail against concentrating into one low-vol or past-winning asset.
"""

from __future__ import annotations

import logging
import math
from typing import Literal, get_args

import pandas as pd

log = logging.getLogger(__name__)

AllocationRule = Literal["equal_weight", "inverse_vol"]
VALID_RULES: frozenset[str] = frozenset(get_args(AllocationRule))

# Discipline (no edge claim) → may always produce a target. Unknown rule → edge
# (fail-safe: must be validated). Same axis as `strategy.StrategyKind`, kept local
# so the two L2 modules stay independent.
AllocationKind = Literal["discipline", "edge"]
_RULE_KIND: dict[str, AllocationKind] = {r: "discipline" for r in VALID_RULES}

_TRADING_DAYS = 252


def allocation_kind(rule: str) -> AllocationKind:
    """Discipline or edge. Unknown rules default to edge (must be validated)."""
    return _RULE_KIND.get(rule, "edge")


def equal_weight(universe: list[str]) -> dict[str, float]:
    """1/N over the (de-duplicated) universe. The robust baseline that beat
    optimization out-of-sample in `examples/` 05–06. Empty universe → empty."""
    tickers = sorted(set(universe))
    if not tickers:
        return {}
    w = 1.0 / len(tickers)
    return {tk: w for tk in tickers}


def inverse_vol(
    series_by_ticker: dict[str, "pd.Series[float]"],
    *,
    lookback: int = _TRADING_DAYS,
) -> dict[str, float]:
    """Weight each ticker by 1/volatility so each contributes ~equal *risk*, not
    equal dollars (risk-parity-lite).

    Volatility = std of daily returns over the last `lookback` days (≤ 0 → use all
    history); realized vol only, no return forecast. The 1/vol ratio is scale-free,
    so daily-vs-annualized doesn't matter (it cancels in the normalization). A
    ticker with fewer than 2 returns, or a flat / degenerate (zero- or NaN-
    volatility) series, is dropped — there's no risk signal to weight on, and a
    floored zero-vol would otherwise dominate the whole book. Weights sum to 1;
    empty if nothing is usable.
    """
    inv: dict[str, float] = {}
    for tk in sorted(series_by_ticker):
        prices = series_by_ticker[tk].dropna()
        if len(prices) < 2:
            continue
        rets = prices.pct_change(fill_method=None).dropna()
        if lookback > 0:
            rets = rets.tail(lookback)
        if len(rets) < 2:
            continue  # need ≥2 returns for a sample std (1 return → std is NaN → poisons all)
        vol = float(rets.std())
        if not math.isfinite(vol) or vol <= 0.0:
            continue  # flat / degenerate → no risk signal; dropping beats floor-and-dominate
        inv[tk] = 1.0 / vol
    total = sum(inv.values())
    if total <= 0.0:
        return {}
    return {tk: v / total for tk, v in inv.items()}


def apply_caps(weights: dict[str, float], cap: float) -> dict[str, float]:
    """Enforce a per-asset weight ceiling, redistributing the excess proportionally
    among the under-cap holdings (iterated to a fixed point — capping one can push
    others over).

    Input weights are normalized first. Raises ValueError if `cap` is non-positive
    or infeasible (``cap * n < 1`` can't sum to 1 with every weight ≤ cap).
    """
    if not weights:
        return {}
    if cap <= 0.0:
        raise ValueError("cap must be positive")
    n = len(weights)
    if cap * n < 1.0 - 1e-9:
        raise ValueError(f"cap {cap} too small for {n} assets (cap*n must be ≥ 1)")
    total = sum(weights.values())
    if total <= 0.0:
        return {}
    w = {k: v / total for k, v in weights.items()}
    for _ in range(n):  # ≤ n iterations reach the fixed point
        over = {k: v for k, v in w.items() if v > cap + 1e-12}
        if not over:
            break
        excess = sum(v - cap for v in over.values())
        for k in over:
            w[k] = cap
        under_total = sum(v for v in w.values() if v < cap - 1e-12)
        if under_total <= 0.0:
            break
        for k, v in w.items():
            if v < cap - 1e-12:
                w[k] = v + excess * (v / under_total)
    return w
