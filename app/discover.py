"""Discovery: turn a book's role GAPS into screenable candidates from the universe (P3a).

Deterministic + pure (no I/O, no LLM). Given the priced holdings and the curated
``universe``:

  1. **role_exposure** — the share of your priced market value in each role (held tickers
     mapped to their role via the universe; a holding *not* in the universe counts toward
     the total but toward no role — it's real exposure we just can't attribute);
  2. **gap_roles** — the roles the universe offers that you hold ≤ a threshold of (coarse on
     purpose: a token sleeve is effectively a gap). Tactical satellites (the sector/thematic
     aisle) are skipped by default — not holding a sector bet is a stance, not a hole — and
     included only when the caller names them (``include_satellites``);
  3. **find_gaps** — package the gaps + a per-role menu for ``screen`` to judge: the role's
     LEAD SHELF (the flavor its first core row carries — the same shelf the presets buy
     from, so no new stance) capped at 3, core first; other shelves are *named with
     counts*, never silently dropped;
  4. **role_menu** — the explicit paths: a full per-shelf menu for a named role, a shelf
     INDEX for a satellite (picking the sector/theme is the user's decision — the tool
     hands over the map and refuses to shortlist), or a single-shelf DRILL
     (``--discover treasury:long``).

The shelf rule: ``flavor`` groups near-substitutes so every menu offers a genuine ≥3-way
choice of comparable funds; shelves appear in output and addresses, NEVER in judgments —
a gap is always role-level ("you hold no treasuries"), never shelf-level ("you lack
long-duration"), because that would be duration/sector advice.

This core never predicts — it only surfaces *where you have nothing* and *what could fill
it*. The screen + held-out role check vet the candidates; the AI edge (ranking /
explaining among them) narrates in P3b.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.derive import DerivedState
from app.prices import PriceRow
from app.universe import SATELLITE_ROLES, Candidate, candidates_for_role, roles_in

# A role is "covered" once it holds more than this share of priced market value; at or below
# it, it's a gap worth surfacing candidates for. Coarse + tunable.
_GAP_THRESHOLD = 0.03
# Surface at most this many candidates per shelf — a genuine choice among near-substitutes,
# not the whole shelf.
_PER_SHELF_CAP = 3
# A drilled shelf may show a little more — the user asked for exactly this exposure.
_DRILL_CAP = 6


@dataclass(frozen=True)
class Discovery:
    """What discovery found: the roles you're light in, each role's current exposure (for
    context), the universe candidates for the screen to judge (capped, not already held),
    and the shelf map — which shelf the candidates came from and which other shelves exist
    (named with counts, so a default never silently hides an exposure)."""

    gaps: tuple[str, ...]              # gap roles, sorted
    exposure: dict[str, float]         # role -> share of priced market value
    candidates: tuple[Candidate, ...]  # in gap roles, capped per shelf, excluding held
    lead_flavor: dict[str, str] = field(default_factory=dict)   # role -> shelf shown by default
    more_shelves: dict[str, tuple[tuple[str, int], ...]] = field(default_factory=dict)
    # role -> ((flavor, unheld fund count), ...) for shelves with no surfaced candidate


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
    exposure: dict[str, float],
    universe: list[Candidate],
    *,
    threshold: float = _GAP_THRESHOLD,
    include_satellites: bool = False,
) -> list[str]:
    """The roles the universe offers that the book has ≤ ``threshold`` exposure to (sorted,
    for a deterministic order). Satellite roles (the sector/thematic aisle) are only gaps
    when ``include_satellites`` — the caller asked about them by name."""
    roles = roles_in(universe)
    if not include_satellites:
        roles -= SATELLITE_ROLES
    return sorted(r for r in roles if exposure.get(r, 0.0) <= threshold)


def _shelf_groups(pool: list[Candidate]) -> list[tuple[str, list[Candidate]]]:
    """Partition a role's (unheld) rows into shelves by ``flavor``, shelves ordered by first
    appearance in the file, rows in file order. Blank flavor = the one unnamed shelf."""
    groups: dict[str, list[Candidate]] = {}
    for c in pool:
        groups.setdefault(c.flavor, []).append(c)
    return list(groups.items())


def _default_menu(pool: list[Candidate], cap: int) -> tuple[list[Candidate], str]:
    """The default-path pick for one role: the LEAD shelf (flavor of the first core row —
    the shelf the presets buy from, so the default carries no new stance), core first,
    filling from the remaining shelves' core rows (then tilts) only when the lead shelf
    can't reach ``cap`` — a surfaced gap must never hide its candidates. Returns the picks
    and the lead flavor."""
    if not pool:
        return [], ""
    lead = next((c.flavor for c in pool if c.core), pool[0].flavor)
    groups = _shelf_groups(pool)
    ordered = [g for g in groups if g[0] == lead] + [g for g in groups if g[0] != lead]
    fill = [c for _, rows in ordered for c in rows if c.core] + [
        c for _, rows in ordered for c in rows if not c.core
    ]
    return fill[:cap], lead


def find_gaps(
    state: DerivedState,
    prices: dict[str, PriceRow],
    universe: list[Candidate],
    *,
    threshold: float = _GAP_THRESHOLD,
    per_role_cap: int = _PER_SHELF_CAP,
    include_satellites: bool = False,
) -> Discovery:
    """Book + universe → the gap roles, each with its lead-shelf menu (capped, core-first,
    not already held) and the named remainder of its shelf map. Deterministic: file order
    everywhere. A single-shelf role behaves exactly like the pre-shelf rule (core-first
    top-N). (Needs a *priced* book — with no prices every role reads as a gap, so the
    caller must require the price pipeline.)"""
    exposure = role_exposure(state, prices, universe)
    gaps = gap_roles(exposure, universe, threshold=threshold, include_satellites=include_satellites)
    held = set(state.held())
    picked: list[Candidate] = []
    lead_flavor: dict[str, str] = {}
    more_shelves: dict[str, tuple[tuple[str, int], ...]] = {}
    for role in gaps:
        pool = [c for c in candidates_for_role(universe, role) if c.ticker not in held]
        menu, lead = _default_menu(pool, per_role_cap)
        picked.extend(menu)
        # "Also here" counts every shelf's UNSHOWN funds (except the lead shelf, which the
        # menu itself represents). Counting by remaining fund — not by flavor — matters
        # when a thin lead shelf borrowed a fund from another shelf: the borrowed shelf's
        # remaining funds must stay named, never silently dropped.
        picked_tickers = {c.ticker for c in menu}
        rest = tuple(
            (f, n)
            for f, rows in _shelf_groups(pool)
            if f != lead
            and (n := sum(1 for c in rows if c.ticker not in picked_tickers)) > 0
        )
        if lead:
            lead_flavor[role] = lead
        if rest:
            more_shelves[role] = rest
    return Discovery(
        gaps=tuple(gaps), exposure=exposure, candidates=tuple(picked),
        lead_flavor=lead_flavor, more_shelves=more_shelves,
    )


def role_menu(
    state: DerivedState,
    prices: dict[str, PriceRow],
    universe: list[Candidate],
    role: str,
    *,
    flavor: str | None = None,
    per_shelf_cap: int = _PER_SHELF_CAP,
    drill_cap: int = _DRILL_CAP,
) -> Discovery:
    """The explicit paths for ONE named role (the caller has already checked it IS a gap).

    - ``flavor`` given (a DRILL, ``--discover treasury:long``): that shelf's unheld rows,
      core first, capped at ``drill_cap`` — naming a shelf is consent to see its funds,
      tilts included (``corporate-bond:high-yield`` shows junk, labeled as junk).
    - satellite role, no flavor: the shelf INDEX — no candidates, every shelf named with
      its count. Picking a sector/theme is the user's bet; the tool maps, never shortlists.
      A single-shelf satellite drills straight through (no pointless second step).
    - non-satellite role, no flavor: the FULL menu — each shelf with core rows gets its
      core top-``per_shelf_cap``; core-less shelves (styles, junk) stay index lines,
      drillable by name.

    Raises ValueError for a flavor the role doesn't have (message lists the valid ones).
    """
    exposure = role_exposure(state, prices, universe)
    held = set(state.held())
    pool = [c for c in candidates_for_role(universe, role) if c.ticker not in held]
    groups = _shelf_groups(pool)

    if flavor is not None:
        by_flavor = dict(groups)
        if flavor not in by_flavor:
            valid = ", ".join(f or "(unnamed)" for f, _ in groups) or "none"
            raise ValueError(f"{role} has no {flavor!r} shelf; it has: {valid}")
        rows = by_flavor[flavor]
        menu = [c for c in rows if c.core][:drill_cap]
        menu += [c for c in rows if not c.core][: drill_cap - len(menu)]
        rest = tuple((f, len(r)) for f, r in groups if f != flavor)
        return Discovery(
            gaps=(role,), exposure=exposure, candidates=tuple(menu),
            lead_flavor={role: flavor}, more_shelves={role: rest} if rest else {},
        )

    if role in SATELLITE_ROLES and len(groups) > 1:
        index = tuple((f, len(rows)) for f, rows in groups)
        return Discovery(
            gaps=(role,), exposure=exposure, candidates=(),
            lead_flavor={}, more_shelves={role: index},
        )
    if role in SATELLITE_ROLES and groups:  # single-shelf satellite: drill straight through
        only = groups[0][0]
        return role_menu(
            state, prices, universe, role,
            flavor=only, per_shelf_cap=per_shelf_cap, drill_cap=drill_cap,
        )

    full_menu: list[Candidate] = []
    unshown: list[tuple[str, int]] = []
    for f, rows in groups:
        core_rows = [c for c in rows if c.core]
        if core_rows:
            full_menu.extend(core_rows[:per_shelf_cap])
        else:
            unshown.append((f, len(rows)))
    if not full_menu:  # a custom universe whose whole role is tilts: never hide a gap
        full_menu = pool[:per_shelf_cap]
        shown = {c.flavor for c in full_menu}
        unshown = [(f, n) for f, n in unshown if f not in shown]
    return Discovery(
        gaps=(role,), exposure=exposure, candidates=tuple(full_menu),
        lead_flavor={}, more_shelves={role: tuple(unshown)} if unshown else {},
    )


def merge_menus(menus: list[Discovery]) -> Discovery:
    """Combine per-role ``role_menu`` results (``--discover treasury,reit:global``) into one
    Discovery. Exposure is identical across them (same book); gaps sort for a deterministic
    panel order. Assumes ONE menu per role (the CLI collapses duplicate role tokens) —
    the ticker dedup below is a belt for any other caller, not a license to pass two
    menus of the same role (their lead/shelf maps would still last-write-wins)."""
    if not menus:
        return Discovery(gaps=(), exposure={}, candidates=())
    lead: dict[str, str] = {}
    more: dict[str, tuple[tuple[str, int], ...]] = {}
    for m in menus:
        lead.update(m.lead_flavor)
        more.update(m.more_shelves)
    seen: set[str] = set()
    candidates: list[Candidate] = []
    for m in menus:
        for c in m.candidates:
            if c.ticker not in seen:
                seen.add(c.ticker)
                candidates.append(c)
    return Discovery(
        gaps=tuple(sorted({r for m in menus for r in m.gaps})),
        exposure=menus[0].exposure,
        candidates=tuple(candidates),
        lead_flavor=lead, more_shelves=more,
    )
