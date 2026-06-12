"""Allocation rules — deterministic ways to *choose* a target allocation.

Where `strategy.py` decides how to **move** toward a target (the rebalance modes),
this module decides **what the target should be**: pure functions over a universe
and its price history. It plugs in one step upstream of the rebalance machinery —
its output is a target-weight dict, the same shape `events.load_target` returns.

`allocate(rule, universe, series, cap=…)` is THE entry point: it dispatches to the
registered rule and enforces the **discipline-vs-edge gate** itself, so no caller
(CLI, MCP tool, future searcher) can reach an unvalidated edge rule by accident.
All current rules are *discipline*: they make no claim to beat the market (no
return forecast), so they need no backtest to be honest and may always produce a
target. An *edge* allocator (e.g. mean-variance optimization, which overfits
out-of-sample — see `examples/` 05–06) registers as ``edge`` and raises
`UnvalidatedEdgeError` until walk-forward machinery passes ``validated=True``.

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
from typing import TYPE_CHECKING, Literal, get_args

import pandas as pd

from app.events import CASH_TICKER

if TYPE_CHECKING:
    from collections.abc import Callable

    _RuleFn = Callable[[list[str], dict[str, pd.Series[float]] | None], dict[str, float]]

log = logging.getLogger(__name__)


class UnvalidatedEdgeError(ValueError):
    """An edge allocation rule was invoked without walk-forward validation."""

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


# --- The dispatcher: one entry point, gate inside ---

def _run_equal_weight(
    universe: list[str], series: dict[str, pd.Series[float]] | None
) -> dict[str, float]:
    return equal_weight(universe)


def _run_inverse_vol(
    universe: list[str], series: dict[str, pd.Series[float]] | None
) -> dict[str, float]:
    if series is None:
        msg = "inverse_vol needs price history (series_by_ticker)"
        raise ValueError(msg)
    rows = {tk: s for tk in universe if (s := series.get(tk)) is not None and not s.empty}
    dropped = sorted(set(universe) - set(rows))
    if dropped:
        log.warning("allocate inverse_vol: no price history for %s", ", ".join(dropped))
    return inverse_vol(rows)


# Runtime registry: adding a rule = one entry here + one in _RULE_KIND (edge rules
# MUST register their kind, or the unknown→edge default blocks them — fail-safe).
_RULES: dict[str, _RuleFn] = {
    "equal_weight": _run_equal_weight,
    "inverse_vol": _run_inverse_vol,
}

# Rules that require price history; the CLI uses this to decide whether to fetch.
NEEDS_SERIES: frozenset[str] = frozenset({"inverse_vol"})


def needs_series(rule: str) -> bool:
    """True if the rule weighs on price history (the caller must supply series)."""
    return rule in NEEDS_SERIES


def allocate(
    rule: str,
    universe: list[str],
    series_by_ticker: dict[str, pd.Series[float]] | None = None,
    *,
    cap: float | None = None,
    validated: bool = False,
) -> dict[str, float]:
    """Produce a target allocation: dispatch + gate + universe hygiene + caps.

    The chokepoint for ALL allocation: an unknown rule raises ValueError; an
    ``edge`` rule raises UnvalidatedEdgeError unless ``validated=True`` (which only
    walk-forward machinery may pass); the CASH pseudo-ticker is never allocated;
    ``cap`` applies `apply_caps` (its ValueError on an infeasible cap propagates).
    Returns {} when no usable weights exist (empty universe, no risk signal).
    """
    runner = _RULES.get(rule)
    if runner is None:
        msg = f"unknown allocation rule {rule!r}; valid: {sorted(_RULES)}"
        raise ValueError(msg)
    if allocation_kind(rule) != "discipline" and not validated:
        msg = (
            f"{rule!r} is an edge allocator and may not produce a target until a "
            "walk-forward backtest validates it"
        )
        raise UnvalidatedEdgeError(msg)
    clean = [tk for tk in universe if tk != CASH_TICKER]
    target = runner(clean, series_by_ticker)
    if cap is not None and target:
        target = apply_caps(target, cap)
    return target
