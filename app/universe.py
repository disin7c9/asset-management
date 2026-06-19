"""The curated ETF universe — candidate tickers + their role, for P3 discovery.

A thin, version-controlled list (``ticker, name, role, summary``): the **reliable,
reproducible** source of candidates the discovery loop screens. It carries only
membership + a clean role tag + a short human/LLM-facing summary; the *facts*
(cost / liquidity / age) are fetched LIVE at judge-time via ``metadata.py`` /
``prices.py``, so this file stays thin and the numbers stay fresh.

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

# The fixed role vocabulary. Coarse on purpose: enough to express "a gap in your book",
# not Morningstar-granular. Adding one is a deliberate, reviewed change (the gap-matching
# and any role labels depend on the set being stable).
ROLES: frozenset[str] = frozenset(
    {
        # equity
        "us-large", "us-small-mid", "us-dividend", "intl-developed", "em-equity",
        "sector-equity", "thematic-equity",
        # fixed income
        "bond-aggregate", "treasury", "tips", "corporate-bond",
        # real assets
        "gold", "commodity-broad", "reit",
    }
)

_COLUMNS = ("ticker", "name", "role", "summary")

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    """One universe member: a ticker the discovery loop may *consider*. ``role`` is our
    taxonomy (∈ ``ROLES``); ``summary`` is a short sourced description (user/LLM context,
    never a filter input). The fund's live facts are fetched separately at judge-time."""

    ticker: str
    name: str
    role: str
    summary: str


def load_universe(path: str | Path) -> list[Candidate]:
    """Parse the curated universe CSV → ``Candidate`` list.

    Degrades honestly (the project's I/O ethos): a bad ROW — an unknown ``role`` or a
    duplicate ticker — is logged and SKIPPED, so one stray row can't disable all discovery.
    Only a broken file STRUCTURE (a missing column) raises ``ValueError``. Blank lines skipped.
    """
    rows: list[Candidate] = []
    seen: set[str] = set()
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in _COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: universe CSV missing column(s): {', '.join(missing)}")
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
