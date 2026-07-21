"""Tests for the curated-universe loader (app/universe.py). Offline — fixture CSVs via
tmp_path, plus guards that the *committed* app/data/universe.csv always loads cleanly and
lives inside the package (the wheel ships only app/, so a repo-root path would strand
installed runs).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from app.universe import ROLES, SATELLITE_ROLES, candidates_for_role, load_universe, roles_in

_REPO = Path(__file__).resolve().parent.parent
_HEADER = "ticker,name,role,summary\n"


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "u.csv"
    p.write_text(_HEADER + body, encoding="utf-8")
    return p


def test_loads_and_queries(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        'VOO,Vanguard S&P 500,us-large,"big US"\nBND,Total Bond,bond-aggregate,"US bonds"\n',
    )
    u = load_universe(p)
    assert [c.ticker for c in u] == ["VOO", "BND"]
    assert candidates_for_role(u, "us-large")[0].name == "Vanguard S&P 500"
    assert candidates_for_role(u, "tips") == []
    assert roles_in(u) == {"us-large", "bond-aggregate"}


def test_unknown_role_row_is_skipped(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # A bad-role row is logged + dropped, not fatal — the good rows still load, so one stray
    # row can't disable all of discovery.
    p = _write(tmp_path, 'XYZ,Bad Fund,not-a-role,"x"\nVOO,Good,us-large,"ok"\n')
    with caplog.at_level(logging.WARNING):
        u = load_universe(p)
    assert [c.ticker for c in u] == ["VOO"]
    assert "unknown role" in caplog.text


def test_duplicate_ticker_row_is_skipped(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    p = _write(tmp_path, 'VOO,A,us-large,"x"\nVOO,B,us-large,"y"\n')
    with caplog.at_level(logging.WARNING):
        u = load_universe(p)
    assert [(c.ticker, c.name) for c in u] == [("VOO", "A")]  # first kept, dup skipped
    assert "duplicate ticker" in caplog.text


def test_rejects_missing_column(tmp_path: Path) -> None:
    (tmp_path / "u.csv").write_text("ticker,name,role\nVOO,A,us-large\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing column"):
        load_universe(tmp_path / "u.csv")


def test_blank_rows_skipped_and_ticker_normalized(tmp_path: Path) -> None:
    p = _write(tmp_path, ',,,\n voo ,Vanguard,us-large,"x"\n')
    u = load_universe(p)
    assert [c.ticker for c in u] == ["VOO"]  # blank row skipped, ticker trimmed + uppercased


def test_core_column_parsed(tmp_path: Path) -> None:
    # With a core column, only an explicit 0 marks a tilt; 1 and blank read core
    # (fail-open — candidates surface rather than hide).
    p = tmp_path / "u.csv"
    p.write_text(
        "ticker,name,role,summary,core\n"
        'VOO,Blend,us-large,"plain",1\n'
        'QQQ,Growth,us-large,"tilt",0\n'
        'IVV,Blend 2,us-large,"plain",\n',
        encoding="utf-8",
    )
    u = load_universe(p)
    assert [(c.ticker, c.core) for c in u] == [("VOO", True), ("QQQ", False), ("IVV", True)]


def test_missing_core_column_warns_once_and_treats_all_core(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # A custom universe without the column keeps working: one warning, every row core.
    p = _write(tmp_path, 'VOO,A,us-large,"x"\nQQQ,B,us-large,"y"\n')
    with caplog.at_level(logging.WARNING):
        u = load_universe(p)
    assert [c.core for c in u] == [True, True]
    assert sum("no 'core' column" in r.message for r in caplog.records) == 1


def test_satellite_roles_are_a_subset_of_roles() -> None:
    # Satellites are ordinary roles (screenable, explicitly discoverable) — just not
    # default gaps; a satellite outside ROLES would silently never match anything.
    assert SATELLITE_ROLES < ROLES


def test_committed_universe_every_role_keeps_a_core_row(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Universe integrity: gap candidates surface core-first, so every role must keep at
    # least one core row (else its gap falls back to tilts-only) — and the committed file
    # must actually carry the column.
    with caplog.at_level(logging.WARNING, logger="app.universe"):
        u = load_universe(_REPO / "app" / "data" / "universe.csv")
    assert "no 'core' column" not in caplog.text
    for role in roles_in(u):
        assert any(c.core for c in u if c.role == role), f"{role}: no core row"


def test_committed_universe_is_valid(caplog: pytest.LogCaptureFixture) -> None:
    # The real file must load cleanly: NO skipped rows (no warnings), known roles, name+summary.
    with caplog.at_level(logging.WARNING, logger="app.universe"):
        u = load_universe(_REPO / "app" / "data" / "universe.csv")
    assert len(u) >= 10
    assert roles_in(u) <= ROLES
    assert all(c.name and c.summary for c in u)
    assert "skipping the row" not in caplog.text  # the committed file has no bad rows


def test_bundled_universe_ships_inside_the_package(monkeypatch: pytest.MonkeyPatch) -> None:
    # The default the CLI/MCP resolve must live INSIDE app/ — the wheel packages only the
    # app/ tree, so a repo-root data/ path exists in a checkout but NOT in an installed run
    # (uvx / pip / .mcpb): --allocate presets and --discover would die "universe unavailable"
    # exactly where the README's zero-setup demo sends a stranger.
    import app
    from app.mcp_server import _universe_path

    monkeypatch.delenv("ASSET_UNIVERSE", raising=False)
    default = _universe_path()
    assert default.is_relative_to(Path(app.__file__).resolve().parent)
    assert default.is_file()


def test_flavor_column_parsed_and_optional(tmp_path: Path) -> None:
    # flavor is the SHELF label: parsed when present, blank when absent (one unnamed
    # shelf per role = pre-shelf behavior; silent — nothing misleading can happen).
    p = tmp_path / "u.csv"
    p.write_text(
        "ticker,name,role,summary,core,flavor\n"
        'TLT,Long,treasury,"20y+",1,long\n'
        'IEF,Int,treasury,"7-10y",1,\n',
        encoding="utf-8",
    )
    u = load_universe(p)
    assert [(c.ticker, c.flavor) for c in u] == [("TLT", "long"), ("IEF", "")]
    q = _write(tmp_path, 'VOO,A,us-large,"x"\n')  # no flavor column at all
    assert load_universe(q)[0].flavor == ""


def test_role_name_literal_matches_roles() -> None:
    # The MCP `role` argument publishes RoleName as its schema enum (v2.11.3 lesson);
    # a role added to ROLES without the Literal would be unreachable over MCP.
    from typing import get_args

    from app.universe import RoleName

    assert set(get_args(RoleName)) == ROLES


def test_committed_universe_core_shelves_offer_a_real_choice() -> None:
    # The shelf promise: every menu offers >=3 near-substitutes. For each non-satellite
    # role, every shelf that would render as a menu (has core rows) must hold >=3 of them.
    # (Satellite shelves render as an index and may be thin — the panel says so.)
    u = load_universe(_REPO / "app" / "data" / "universe.csv")
    shelves: dict[tuple[str, str], int] = {}
    for c in u:
        if c.core and c.role not in SATELLITE_ROLES:
            key = (c.role, c.flavor)
            shelves[key] = shelves.get(key, 0) + 1
    thin = {k: n for k, n in shelves.items() if n < 3}
    assert not thin, f"core shelves without a 3-way choice: {thin}"


def test_committed_universe_lead_shelves_are_glossed() -> None:
    # Every default-path shelf heading should explain itself: the lead shelf (first core
    # row's flavor) of each non-satellite role must have a FLAVOR_NOTES gloss when named.
    from app.universe import FLAVOR_NOTES

    u = load_universe(_REPO / "app" / "data" / "universe.csv")
    for role in sorted(roles_in(u) - SATELLITE_ROLES):
        lead = next((c.flavor for c in u if c.role == role and c.core), "")
        if lead:
            assert lead in FLAVOR_NOTES, f"{role}: lead shelf {lead!r} has no gloss"
