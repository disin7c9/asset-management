"""Discovery: turn a book's role GAPS into screenable candidates from the universe (P3a).

Deterministic + pure (no I/O, no LLM). Given the priced holdings and the curated
``universe``:

  1. **role_exposure** — the share of your priced market value in each role (held tickers
     mapped to their role via the universe; a holding *not* in the universe counts toward
     the total but toward no role — it's real exposure we just can't attribute);
  2. **gap_roles** — the roles the universe offers that you hold ≤ a threshold of (coarse on
     purpose: a token sleeve is effectively a gap);
  3. **find_gaps** — package the gaps + a capped list of candidates per gap (excluding what
     you already hold), for ``screen`` to judge.

This core never predicts — it only surfaces *where you have nothing* and *what could fill it*.
The screen + walk-forward role check vet the candidates; the AI edge (ranking/explaining
among them) lands later in P3b.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.derive import DerivedState
from app.prices import PriceRow
from app.universe import Candidate, candidates_for_role, roles_in

# A role is "covered" once it holds more than this share of priced market value; at or below
# it, it's a gap worth surfacing candidates for. Coarse + tunable.
_GAP_THRESHOLD = 0.03
# Surface at most this many candidates per gap role — a handful of options, not the whole shelf.
_PER_ROLE_CAP = 3


@dataclass(frozen=True)
class Discovery:
    """What discovery found: the roles you're light in, each role's current exposure (for
    context), and the universe candidates (capped, not already held) for the screen to judge."""

    gaps: tuple[str, ...]              # gap roles, sorted
    exposure: dict[str, float]         # role -> share of priced market value
    candidates: tuple[Candidate, ...]  # in gap roles, capped per role, excluding held


def role_exposure(
    state: DerivedState, prices: dict[str, PriceRow], universe: list[Candidate]
) -> dict[str, float]:
    """Share of priced market value in each role. A held ticker absent from the universe still
    counts toward the denominator (it's real money) but is credited to no role, so its weight
    correctly makes the *known* roles look smaller rather than vanishing."""
    role_of = {c.ticker: c.role for c in universe}
    by_role: dict[str, float] = {}
    total = 0.0
    for ticker, pos in state.held().items():
        row = prices.get(ticker)
        if row is None:
            continue
        value = pos.shares * row.close
        if value <= 0:
            continue
        total += value
        role = role_of.get(ticker)
        if role is not None:
            by_role[role] = by_role.get(role, 0.0) + value
    if total <= 0:
        return {}
    return {role: value / total for role, value in by_role.items()}


def gap_roles(
    exposure: dict[str, float], universe: list[Candidate], *, threshold: float = _GAP_THRESHOLD
) -> list[str]:
    """The roles the universe offers that the book has ≤ ``threshold`` exposure to (sorted,
    for a deterministic order)."""
    return sorted(r for r in roles_in(universe) if exposure.get(r, 0.0) <= threshold)


def find_gaps(
    state: DerivedState,
    prices: dict[str, PriceRow],
    universe: list[Candidate],
    *,
    threshold: float = _GAP_THRESHOLD,
    per_role_cap: int = _PER_ROLE_CAP,
) -> Discovery:
    """Book + universe → the gap roles and a capped set of candidates (not already held) for
    the screen to judge. Deterministic: candidates are taken in the universe file's order and
    capped per role. (Needs a *priced* book — with no prices every role reads as a gap, so the
    caller must require the price pipeline.)"""
    exposure = role_exposure(state, prices, universe)
    gaps = gap_roles(exposure, universe, threshold=threshold)
    held = set(state.held())
    picked: list[Candidate] = []
    for role in gaps:  # gap roles (sorted); candidates in universe AUM order, capped, held-excluded
        kept = 0
        for c in candidates_for_role(universe, role):
            if c.ticker in held:
                continue
            picked.append(c)
            kept += 1
            if kept >= per_role_cap:
                break
    return Discovery(gaps=tuple(gaps), exposure=exposure, candidates=tuple(picked))


def restrict_to(discovery: Discovery, roles: set[str]) -> Discovery:
    """Narrow a Discovery to the named gap roles (for ``--discover reit,tips``), keeping the
    same exposure context. The Discovery invariants live here, so the CLI doesn't hand-rebuild
    the frozen dataclass."""
    keep = roles & set(discovery.gaps)
    return Discovery(
        gaps=tuple(sorted(keep)),
        exposure=discovery.exposure,
        candidates=tuple(c for c in discovery.candidates if c.role in keep),
    )
