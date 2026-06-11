"""Strategy layer: turn current holdings + a target allocation into legible,
named buy/sell suggestions. Pure functions — no I/O, no network.

This is *discipline, not prediction*. A rebalance suggestion ("you're 7pp
overweight VOO → trim") makes no claim that it will beat the market, so it needs
no backtest to be honest. Strategies that claim an *edge* (timing, momentum) are
a separate concern and must be walk-forward validated (see app/backtest.py, a
later slice) before they may surface a suggestion.

Every suggestion carries the **named rule** that produced it and a one-line
reason, so the user learns the rule rather than trusting a black box.

Modes (v1):
    to_total        sell + buy to hit target weights exactly (deploys new cash too)
    cash_flow_only  invest new cash into underweights; never sell (tax-friendly)
    fixed_dca       buy the target mix with a fixed cash amount, ignoring drift
    bands           like to_total, but only act on tickers whose drift exceeds a
                    band (prevents churn); rebalances existing holdings only. The
                    band is the SMALLER of an absolute pp (`band`) or a relative
                    fraction of the target (`band_rel` × target — the "5/25 rule"),
                    so a small sleeve isn't handed a band many times its own size
                    (a 1% target with a flat 5pp band could vanish or 6× untouched).

Weights in the target are **relative** — they are normalized to sum to 1, so
``25,25,25,25`` and ``0.25,0.25,0.25,0.25`` mean the same thing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, get_args

log = logging.getLogger(__name__)

Mode = Literal["to_total", "cash_flow_only", "fixed_dca", "bands"]
VALID_MODES: frozenset[str] = frozenset(get_args(Mode))  # tracks the type

StrategyKind = Literal["discipline", "edge"]
# Every v1 mode is *discipline*: rebalancing makes no claim to beat the market, so
# it needs no backtest to be honest and may always suggest. An *edge* strategy
# (timing/momentum, post-v1) MUST pass a walk-forward backtest before it may
# surface a suggestion — enforced by `may_suggest`. The map is the single source
# of truth; an unknown mode is treated as edge (fail-safe → must be validated).
_MODE_KIND: dict[str, StrategyKind] = {m: "discipline" for m in VALID_MODES}

# Trades below this dollar magnitude are treated as "hold" (avoids micro-orders
# and floating-point residue after an exact rebalance).
_MIN_TRADE_USD = 1.0
# A drift exactly at the band counts as within it (hold). The epsilon makes the
# boundary deterministic instead of swinging on float rounding of weight diffs.
_BAND_EPS = 1e-9


@dataclass(frozen=True)
class Suggestion:
    """One per-ticker recommendation, paired to the rule that produced it."""

    ticker: str
    action: Literal["buy", "sell", "hold"]
    shares: float            # magnitude of shares to trade (0 for hold)
    dollars: float           # magnitude of $ to trade (0 for hold)
    current_weight: float    # share of current portfolio value (0..1)
    target_weight: float     # normalized target weight (0..1)
    rule: str                # the mode name
    reason: str              # human-readable trigger

    @property
    def drift(self) -> float:
        return self.current_weight - self.target_weight


def strategy_kind(mode: str) -> StrategyKind:
    """Discipline or edge. Unknown modes default to edge (must be validated)."""
    return _MODE_KIND.get(mode, "edge")


def may_suggest(mode: str, *, backtest_validated: bool = False) -> bool:
    """The discipline-vs-edge gate. A discipline strategy may always surface a
    suggestion; an *edge* strategy may only after a walk-forward backtest has
    validated it. (No edge strategies exist in v1, so this is a dormant guard
    that makes the future safe — and refuses unknown modes by default.)"""
    return strategy_kind(mode) == "discipline" or backtest_validated


def suggest(
    mode: Mode,
    held_value: dict[str, float],
    prices: dict[str, float],
    target: dict[str, float],
    *,
    new_cash: float = 0.0,
    band: float = 0.05,
    band_rel: float = 0.25,
) -> list[Suggestion]:
    """Produce per-ticker suggestions for one rebalancing rule.

    ``held_value`` maps ticker → current market value ($); ``prices`` maps ticker
    → per-share price (to convert dollars to shares). The universe is the union
    of held and target tickers that have a **positive** price; a ticker with no
    price — or a non-positive one (bad/stale quote) — is skipped (the caller
    reports it), which also makes the dollars→shares division safe. Suggestions
    are returned sorted by ticker.
    """
    if not may_suggest(mode):
        # Enforce the gate at the chokepoint, not just in the CLI: ANY caller of
        # suggest() (a future API, a backtest-driven path) is refused an edge
        # strategy until a walk-forward backtest validates it.
        msg = (
            f"{mode!r} is an edge strategy and must pass a walk-forward backtest "
            "before it may suggest"
        )
        raise ValueError(msg)
    priced = {tk for tk, px in prices.items() if px > 0}
    universe = sorted((set(held_value) | set(target)) & priced)
    total_value = sum(held_value.get(tk, 0.0) for tk in universe)

    if mode == "fixed_dca":
        return _fixed_dca(universe, held_value, prices, target, total_value, new_cash)
    if mode == "cash_flow_only":
        return _cash_flow_only(universe, held_value, prices, target, total_value, new_cash)
    if mode in ("to_total", "bands"):
        # Both trade toward target; bands then gates by drift.
        base = total_value + (new_cash if mode == "to_total" else 0.0)
        return _to_target(
            universe, held_value, prices, target, total_value, base,
            rule=mode, band=(band if mode == "bands" else None), band_rel=band_rel,
        )
    msg = f"unhandled rebalance mode: {mode!r}"  # fail loud if a mode is added without a branch
    raise ValueError(msg)


def _cur_weight(held_value: dict[str, float], tk: str, total_value: float) -> float:
    return held_value.get(tk, 0.0) / total_value if total_value > 0 else 0.0


def _to_target(
    universe: list[str],
    held_value: dict[str, float],
    prices: dict[str, float],
    target: dict[str, float],
    total_value: float,
    base: float,
    *,
    rule: str,
    band: float | None,
    band_rel: float,
) -> list[Suggestion]:
    """Trade each ticker toward target_weight × base. If `band` is set, only act on
    tickers whose |current − target| exceeds the effective band — the SMALLER of the
    absolute `band` (pp) or `band_rel × target_weight` (the 5/25 rule), so a small
    sleeve isn't given a band many times its size. A target of 0 → band 0 → always
    exit (above the $ floor). `band=None` is to_total (no band — act on any drift)."""
    out: list[Suggestion] = []
    for tk in universe:
        tgt_w = target.get(tk, 0.0)  # held but not in target → target 0 → sell out
        cur_w = _cur_weight(held_value, tk, total_value)
        trade = tgt_w * base - held_value.get(tk, 0.0)
        # 5/25 band: the no-trade region is the smaller of the absolute pp band and a
        # relative fraction of the target (tgt 0 → 0 → always exit). None → to_total.
        threshold = min(band, band_rel * tgt_w) if band is not None else None
        if threshold is not None and abs(cur_w - tgt_w) <= threshold + _BAND_EPS:
            out.append(Suggestion(
                tk, "hold", 0.0, 0.0, cur_w, tgt_w, rule,
                f"within {threshold * 100:.2f}pp band",
            ))
            continue
        if abs(trade) < _MIN_TRADE_USD:
            out.append(Suggestion(tk, "hold", 0.0, 0.0, cur_w, tgt_w, rule, "on target"))
            continue
        action: Literal["buy", "sell"] = "buy" if trade > 0 else "sell"
        drift_pp = (cur_w - tgt_w) * 100
        if tgt_w == 0.0:
            # Target 0% (explicitly, or by omission) → a full exit of the position.
            reason = "target 0% → full exit (raise its weight to keep it)"
        elif threshold is not None:
            reason = f"drift {drift_pp:+.1f}pp exceeds {threshold * 100:.2f}pp band"
        else:
            reason = f"{cur_w * 100:.1f}% vs {tgt_w * 100:.1f}% target"
        out.append(Suggestion(
            tk, action, abs(trade) / prices[tk], abs(trade), cur_w, tgt_w, rule, reason,
        ))
    return out


def _cash_flow_only(
    universe: list[str],
    held_value: dict[str, float],
    prices: dict[str, float],
    target: dict[str, float],
    total_value: float,
    new_cash: float,
) -> list[Suggestion]:
    """Deploy `new_cash` into underweights, proportional to each shortfall;
    never sell. If nothing is underweight, fall back to buying the target mix."""
    post_total = total_value + new_cash
    shortfall = {
        tk: max(0.0, target.get(tk, 0.0) * post_total - held_value.get(tk, 0.0))
        for tk in universe
    }
    total_short = sum(shortfall.values())
    out: list[Suggestion] = []
    for tk in universe:
        tgt_w = target.get(tk, 0.0)
        cur_w = _cur_weight(held_value, tk, total_value)
        if new_cash <= 0 or tgt_w <= 0:
            out.append(Suggestion(tk, "hold", 0.0, 0.0, cur_w, tgt_w,
                                  "cash_flow_only", "no cash to deploy"))
            continue
        if total_short > 0:
            buy = new_cash * shortfall[tk] / total_short
            reason = "fill underweight with new cash"
        else:
            buy = new_cash * tgt_w  # all at/above target → spread by target mix
            reason = "no underweights; deploy by target mix"
        if buy < _MIN_TRADE_USD:
            out.append(Suggestion(tk, "hold", 0.0, 0.0, cur_w, tgt_w,
                                  "cash_flow_only", "already at/above target"))
            continue
        out.append(Suggestion(tk, "buy", buy / prices[tk], buy, cur_w, tgt_w,
                              "cash_flow_only", reason))
    return out


def _fixed_dca(
    universe: list[str],
    held_value: dict[str, float],
    prices: dict[str, float],
    target: dict[str, float],
    total_value: float,
    new_cash: float,
) -> list[Suggestion]:
    """Buy the target mix with `new_cash`, ignoring current drift entirely."""
    out: list[Suggestion] = []
    for tk in universe:
        tgt_w = target.get(tk, 0.0)
        cur_w = _cur_weight(held_value, tk, total_value)
        buy = new_cash * tgt_w
        if buy < _MIN_TRADE_USD:
            out.append(Suggestion(tk, "hold", 0.0, 0.0, cur_w, tgt_w,
                                  "fixed_dca", "not in target / no cash"))
            continue
        out.append(Suggestion(tk, "buy", buy / prices[tk], buy, cur_w, tgt_w,
                              "fixed_dca", f"DCA {tgt_w * 100:.1f}% of ${new_cash:,.0f}"))
    return out
