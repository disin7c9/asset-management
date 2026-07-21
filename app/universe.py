"""The curated ETF universe — candidate tickers + their role, for P3 discovery.

A thin, version-controlled list (``ticker, name, role, summary, core, flavor``): the
**reliable, reproducible** source of candidates the discovery loop screens. It carries
only membership + a clean role tag + a short human/LLM-facing summary + a ``core`` flag
(plain representative of the role vs a style/regional/high-yield tilt within it) + a
``flavor`` shelf label (the sub-exposure within a heterogeneous role — display/address
only, never gap-bearing); the *facts* (cost / liquidity / age) are fetched LIVE at
judge-time via ``metadata.py`` / ``prices.py``, so this file stays thin and the numbers
stay fresh.

The role vocabulary is **fixed and coarse by design** — the ``summary`` carries the
specifics. Everything downstream (gap-matching, the screen) keys off these tags, so
they must stay small and stable. The file is built/refreshed *offline* by
``scripts/build_universe.py`` (yfinance seeds name/category/summary; a human reviews
the role) and committed — the app never rebuilds it at runtime. Long-only by scope:
leveraged / inverse funds are excluded (matches the scope lock + ``screen``).
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# The fixed role vocabulary. Coarse on purpose: enough to express "a gap in your book",
# not Morningstar-granular. Adding one is a deliberate, reviewed change (the gap-matching
# and any role labels depend on the set being stable).
ROLES: frozenset[str] = frozenset(
    {
        # equity
        "us-large", "us-small-mid", "us-dividend", "intl-developed", "em-equity",
        "sector-equity",
        # fixed income
        "bond-aggregate", "treasury", "tips", "corporate-bond",
        # real assets
        "gold", "commodity-broad", "reit",
    }
)

# The same vocabulary as a Literal, for surfaces that publish a schema enum (the MCP
# `discover_gaps` role argument — v2.11.3 lesson: constrained args must advertise their
# domain, not a bare string). Pinned equal to ROLES by test.
RoleName = Literal[
    "us-large", "us-small-mid", "us-dividend", "intl-developed", "em-equity",
    "sector-equity", "bond-aggregate", "treasury", "tips", "corporate-bond",
    "gold", "commodity-broad", "reit",
]

# Tactical-satellite roles: deliberate concentrated equity bets (sector slices AND
# cross-sector themes — merged into one aisle in v2.12; a theme is just another shelf).
# Not holding one is a stance, not a hole, so default gap-surfacing skips them (the
# presets already zero-weight them); they stay screenable and surface when a discovery
# names them explicitly. commodity-broad stays a gap role on purpose — preset-zero too,
# but a genuine strategic diversifier.
SATELLITE_ROLES: frozenset[str] = frozenset({"sector-equity"})

# Display glosses for well-known flavor (shelf) tokens — prose keyed off a machine token,
# never machine behavior keyed off prose. Unknown tokens simply print bare.
FLAVOR_NOTES: dict[str, str] = {
    "intermediate": "the standard sleeve (~3-10y)",
    "long": "20y+ — rate-sensitive, equity-scale drawdowns",
    "short": "1-3y — cash-like",
    "high-yield": "junk — equity-like drawdowns",
    "investment-grade": "the standard sleeve",
    "us": "dedicated US exposure",
    "global": "adds non-US property",
    "blend": "the plain market slice",
    "diversified": "the standard sleeve (broad EM)",
    "growth": "style tilt", "value": "style tilt",
    "mid-blend": "plain mid-caps", "small-blend": "plain small-caps",
    "mid-growth": "style tilt", "mid-value": "style tilt",
    "small-growth": "style tilt", "small-value": "style tilt",
    "asia-pacific": "single region", "europe": "single region", "japan": "single region",
    "small-mid": "size tilt",
    "clean-energy": "narrative-defined theme", "innovation": "narrative-defined theme",
}

_COLUMNS = ("ticker", "name", "role", "summary")

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    """One universe member: a ticker the discovery loop may *consider*. ``role`` is our
    taxonomy (∈ ``ROLES``); ``summary`` is a short sourced description (user/LLM context,
    never a filter input). ``core`` marks a plain representative of its role (Blend /
    diversified / investment-grade) vs a tilt within it (Growth/Value style, single
    region, high-yield) — gap candidates surface core first, so a style bet or junk bond
    never fills a role gap by AUM accident. ``flavor`` is the SHELF label — a short
    curated token naming the sub-exposure within a heterogeneous role (treasury duration,
    a sector, REIT geography); display/address only, NEVER gap-bearing: choosing between
    shelves is always the user's decision. Blank = the role's one unnamed shelf. The
    fund's live facts are fetched separately at judge-time."""

    ticker: str
    name: str
    role: str
    summary: str
    core: bool = True
    flavor: str = ""


def load_universe(path: str | Path) -> list[Candidate]:
    """Parse the curated universe CSV → ``Candidate`` list.

    Degrades honestly (the project's I/O ethos): a bad ROW — an unknown ``role`` or a
    duplicate ticker — is logged and SKIPPED, so one stray row can't disable all discovery.
    Only a broken file STRUCTURE (a missing REQUIRED column) raises ``ValueError``; the
    ``core`` column is optional — a custom universe without it warns once and every row
    counts as core (fail-open: candidates surface rather than hide). In a present column,
    only an explicit ``0`` marks non-core. Blank lines skipped.
    """
    rows: list[Candidate] = []
    seen: set[str] = set()
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in _COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: universe CSV missing column(s): {', '.join(missing)}")
        has_core = "core" in (reader.fieldnames or [])
        if not has_core:
            log.warning(
                "%s: no 'core' column — treating every row as core (gap candidates won't "
                "prefer plain funds over style/regional/high-yield tilts)", path,
            )
        for lineno, row in enumerate(reader, start=2):  # line 1 is the header
            ticker = (row.get("ticker") or "").strip().upper()
            if not ticker:
                continue  # tolerate blank spacer rows
            role = (row.get("role") or "").strip()
            if role not in ROLES:
                log.warning("%s:%d: unknown role %r for %s — skipping the row", path, lineno, role, ticker)
                continue
            if ticker in seen:
                log.warning("%s:%d: duplicate ticker %s — skipping the row", path, lineno, ticker)
                continue
            seen.add(ticker)
            rows.append(
                Candidate(
                    ticker=ticker,
                    name=(row.get("name") or "").strip(),
                    role=role,
                    summary=(row.get("summary") or "").strip(),
                    core=not has_core or (row.get("core") or "").strip() != "0",
                    # Optional column; absent/blank = the role's one unnamed shelf —
                    # behavior degrades to plain top-3, so no warning is needed.
                    flavor=(row.get("flavor") or "").strip(),
                )
            )
    return rows


def candidates_for_role(universe: list[Candidate], role: str) -> list[Candidate]:
    """Every universe member tagged with ``role`` (the role pre-filter that keeps a broad
    universe cheap — only this handful gets its facts fetched + screened per run)."""
    return [c for c in universe if c.role == role]


def roles_in(universe: list[Candidate]) -> set[str]:
    """The set of roles actually represented in this universe (coverage check)."""
    return {c.role for c in universe}
