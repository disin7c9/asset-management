"""Allocation rules — deterministic ways to *choose* a target allocation.

Where `strategy.py` decides how to **move** toward a target (the rebalance modes),
this module decides **what the target should be**: pure functions over a universe
and its price history. It plugs in one step upstream of the rebalance machinery —
its output is a target-weight dict, the same shape `events.load_target` returns.

`allocate(rule, universe, series, cap=…)` is the entry point for the *reweight* rules:
it dispatches to the registered rule and enforces the **discipline-vs-edge gate** itself,
so no caller (CLI, MCP tool, future searcher) can reach an unvalidated edge rule by
accident. The *strategic presets* take a different shape — a role→ticker template, not a
reweight of the universe list — so they have their own entry, `preset_target`,
discipline-only by construction (a prior, never a searched edge); `allocate()` rejects a
preset name with a pointer to it.
All current rules are *discipline*: they make no claim to beat the market (no
return forecast), so they need no backtest to be honest and may always produce a
target. An *edge* allocator (e.g. mean-variance optimization, which overfits
out-of-sample — see `examples/` 05–06) registers as ``edge`` and raises
`UnvalidatedEdgeError` until walk-forward machinery passes ``validated=True``.

Rules:
    equal_weight   1/N over the universe — the robust baseline.
    inverse_vol    weight ∝ 1/volatility — each holding contributes ~equal *risk*,
                   not equal dollars; realized vol only, no return forecast.
    conservative / moderate / aggressive
                   strategic risk-posture PRESETS (see `preset_target`): a fixed
                   role-bucket template (a prior, never a backtest-searched weight),
                   not a reweight of held tickers. Discipline — they pass the gate.

`apply_caps` enforces a per-asset weight ceiling (re-normalizing the rest) — a
guard-rail against concentrating into one low-vol or past-winning asset.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Literal, get_args

import pandas as pd

from app.events import CASH_TICKER
from app.universe import ROLES, Candidate

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
# EXPLICIT per-rule classification — deliberately NOT {r: "discipline" for r in
# VALID_RULES}, which would blanket-mark every rule discipline the moment it joins
# the AllocationRule Literal, silently waving a coming edge allocator (mean-variance
# / max-Sharpe) past the gate. A rule absent here falls through to the unknown→edge
# fail-safe in allocation_kind(); an edge rule MUST be listed explicitly as "edge".
_RULE_KIND: dict[str, AllocationKind] = {
    "equal_weight": "discipline",
    "inverse_vol": "discipline",
    # Strategic presets are risk-posture priors, not return-forecasting optimizers →
    # discipline (they dispatch through `preset_target`, not the `_RULES` registry).
    "conservative": "discipline",
    "moderate": "discipline",
    "aggressive": "discipline",
}

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
    if rule in PRESETS:
        msg = (
            f"{rule!r} is a strategic preset — call preset_target(), not allocate() "
            "(a preset is a role→ticker template, not a reweight of the universe list)"
        )
        raise ValueError(msg)
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


# --- Strategic preset allocations (conservative / moderate / aggressive) ---
#
# A risk-posture PRIOR, not an optimization: the weights are hand-designed (like a
# balanced fund's policy), never searched by backtest score (that overfits — see
# examples/ 05–06). Discipline, so they pass the gate freely. Two levels: a bucket
# weight per preset (the equity/bond/diversifier posture) × a fixed core-satellite
# sub-weight per role WITHIN its bucket (so us-large anchors equity — not 1/N across
# VOO and a thematic sleeve). The CLI resolves each role to a ticker — the holder's
# fund in that role, or a universe default — and passes the {role: ticker} map here.

PresetName = Literal["conservative", "moderate", "aggressive"]
PRESETS: frozenset[str] = frozenset(get_args(PresetName))

# The 14 universe roles → 3 strategic buckets. Pinned to `universe.ROLES` by a test.
_ROLE_BUCKET: dict[str, str] = {
    "us-large": "equity", "us-small-mid": "equity", "us-dividend": "equity",
    "intl-developed": "equity", "em-equity": "equity", "sector-equity": "equity",
    "thematic-equity": "equity",
    "bond-aggregate": "bonds", "treasury": "bonds", "tips": "bonds",
    "corporate-bond": "bonds",
    "gold": "diversifiers", "commodity-broad": "diversifiers", "reit": "diversifiers",
}

# Bucket weights per preset (tunable priors; vendor anchors = Vanguard LifeStrategy
# 40/60 · 60/40 · 80/20). Each preset's buckets sum to 1.
_PRESET_BUCKETS: dict[str, dict[str, float]] = {
    "conservative": {"equity": 0.35, "bonds": 0.50, "diversifiers": 0.15},
    "moderate":     {"equity": 0.55, "bonds": 0.30, "diversifiers": 0.15},
    "aggressive":   {"equity": 0.75, "bonds": 0.12, "diversifiers": 0.13},
}

# Core-satellite sub-weight per role WITHIN its bucket — the fix for BOTH inverse_vol's
# bond skew and equal_weight's 1/N nonsense: us-large anchors equity; sector / thematic /
# broad-commodity are 0 (tactical satellites, not the strategic default — add them via
# --discover or by hand). Each bucket's positive weights sum to 1.
_ROLE_WEIGHT: dict[str, float] = {
    "us-large": 0.55, "intl-developed": 0.22, "us-dividend": 0.10,
    "us-small-mid": 0.08, "em-equity": 0.05, "sector-equity": 0.0, "thematic-equity": 0.0,
    "bond-aggregate": 0.50, "treasury": 0.20, "corporate-bond": 0.18, "tips": 0.12,
    "gold": 0.60, "reit": 0.40, "commodity-broad": 0.0,
}


def preset_target(
    preset: str, role_tickers: dict[str, str], *, cap: float | None = None
) -> dict[str, float]:
    """A risk-posture template → ticker:weight. `role_tickers` maps each role to the
    ticker filling it (the holder's fund in that role, or a universe default). Each
    preset's bucket weight is split across the roles PRESENT in that bucket in
    proportion to their core-satellite sub-weight; weights renormalize to 1 (a missing
    role or empty bucket doesn't leak weight). Roles with a zero sub-weight (tactical
    satellites) are excluded. `cap` applies `apply_caps`. Presets are discipline — they
    are priors, not optimizers, so no gate/validation is needed."""
    buckets = _PRESET_BUCKETS.get(preset)
    if buckets is None:
        raise ValueError(f"unknown preset {preset!r}; valid: {sorted(PRESETS)}")
    present: dict[str, list[str]] = {}
    for role in role_tickers:
        bucket = _ROLE_BUCKET.get(role)
        if bucket is not None and _ROLE_WEIGHT.get(role, 0.0) > 0.0:
            present.setdefault(bucket, []).append(role)
    weights: dict[str, float] = {}
    for bucket, bw in buckets.items():
        roles = present.get(bucket, [])
        subtotal = sum(_ROLE_WEIGHT[r] for r in roles)
        if subtotal <= 0.0:
            # No fund resolved for this whole bucket → its weight is dropped and the
            # rest renormalize, which SHIFTS the posture. Never happens with the bundled
            # universe (every role has a default); warn loudly for a custom universe.
            log.warning(
                "preset %s: no fund for the %s bucket (%.0f%% of the target) — the "
                "posture is renormalized over the remaining buckets", preset, bucket, bw * 100,
            )
            continue
        for role in roles:
            tk = role_tickers[role]
            weights[tk] = weights.get(tk, 0.0) + bw * _ROLE_WEIGHT[role] / subtotal
    total = sum(weights.values())
    if total <= 0.0:
        return {}
    target = {tk: w / total for tk, w in weights.items()}
    if cap is not None:
        target = apply_caps(target, cap)
    return target


def _resolve_role_tickers(
    universe: list[Candidate], held_values: dict[str, float]
) -> dict[str, str]:
    """For each universe role, the ticker that fills it in a preset target: the holder's
    LARGEST fund in that role (the dominant holding wins), else the universe default (the
    top-AUM candidate — first in the AUM-ordered file). A held fund whose role the presets
    zero-weight (e.g. thematic) still maps here; `preset_target` drops it, so a to_total
    rebalance surfaces it as a sell."""
    # One pass over the universe: a ticker→role index (to map the holder's funds) and
    # each role's top-AUM default (first seen — the file is AUM-ordered within a role).
    role_of: dict[str, str] = {}
    default_for: dict[str, str] = {}
    for c in universe:
        role_of[c.ticker] = c.role
        default_for.setdefault(c.role, c.ticker)
    # The holder's LARGEST fund wins its role (ticker breaks an exact-value tie, so the
    # result is independent of book/row order); a role with no held fund takes the default.
    held_in_role: dict[str, str] = {}
    for tk in sorted(held_values, key=lambda t: (-held_values[t], t)):
        role = role_of.get(tk)
        if role is not None:
            held_in_role.setdefault(role, tk)
    role_tickers: dict[str, str] = {}
    for role in ROLES:
        ticker = held_in_role.get(role) or default_for.get(role)
        if ticker is not None:
            role_tickers[role] = ticker
    return role_tickers


def build_preset_target(
    preset: str,
    universe: list[Candidate],
    held_values: dict[str, float],
    *,
    cap: float | None = None,
) -> dict[str, float]:
    """Holdings-aware preset target in one call: resolve each role's ticker from the book +
    universe, then build the posture template. The shared entry for the CLI `--allocate` and
    the MCP `propose_allocation` (so both produce identical weights)."""
    return preset_target(preset, _resolve_role_tickers(universe, held_values), cap=cap)
