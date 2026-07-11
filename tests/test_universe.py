"""Tests for the curated-universe loader (app/universe.py). Offline — fixture CSVs via
tmp_path, plus guards that the *committed* app/data/universe.csv always loads cleanly and
lives inside the package (the wheel ships only app/, so a repo-root path would strand
installed runs).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from app.universe import ROLES, candidates_for_role, load_universe, roles_in

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
