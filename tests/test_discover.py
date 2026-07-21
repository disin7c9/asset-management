"""Tests for the discovery core (app/discover.py). Pure — no I/O, no network."""

from __future__ import annotations

from datetime import date, datetime, timezone

from app.derive import DerivedState, Position
from app.discover import find_gaps, gap_roles, merge_menus, role_exposure, role_menu
from app.prices import PriceRow
from app.universe import Candidate

_UNIVERSE = [
    Candidate("VOO", "Vanguard S&P 500", "us-large", "big US"),
    Candidate("BND", "Total Bond", "bond-aggregate", "US bonds"),
    Candidate("VWO", "EM", "em-equity", "emerging 1"),
    Candidate("SCHE", "EM 2", "em-equity", "emerging 2"),
    Candidate("IEMG", "EM 3", "em-equity", "emerging 3"),
    Candidate("VNQ", "REIT", "reit", "real estate"),
    Candidate("IAU", "Gold", "gold", "gold"),
]


def _price(ticker: str, close: float) -> PriceRow:
    return PriceRow(ticker, date(2026, 6, 18), close, "cache", datetime(2026, 6, 18, tzinfo=timezone.utc))


def _book(holdings: dict[str, float]) -> DerivedState:
    s = DerivedState()
    for ticker, shares in holdings.items():
        s.positions[ticker] = Position(ticker=ticker, shares=shares, cost_basis=shares * 10)
    return s


def test_role_exposure_maps_holdings_to_roles() -> None:
    s = _book({"VOO": 10, "BND": 10})
    prices = {"VOO": _price("VOO", 100.0), "BND": _price("BND", 100.0)}
    assert role_exposure(s, prices, _UNIVERSE) == {"us-large": 0.5, "bond-aggregate": 0.5}


def test_unknown_ticker_counts_toward_total_not_a_role() -> None:
    # AAPL isn't in the universe → it dilutes the known roles, not credited anywhere.
    s = _book({"VOO": 10, "AAPL": 10})
    prices = {"VOO": _price("VOO", 100.0), "AAPL": _price("AAPL", 100.0)}
    assert role_exposure(s, prices, _UNIVERSE) == {"us-large": 0.5}


def test_gap_roles_are_underexposed_universe_roles() -> None:
    gaps = gap_roles({"us-large": 0.5, "bond-aggregate": 0.5}, _UNIVERSE)
    assert set(gaps) == {"em-equity", "reit", "gold"}  # the universe roles you hold nothing in


def test_find_gaps_caps_per_role_and_excludes_held() -> None:
    s = _book({"VOO": 10, "BND": 10})
    prices = {"VOO": _price("VOO", 100.0), "BND": _price("BND", 100.0)}
    d = find_gaps(s, prices, _UNIVERSE, per_role_cap=2)
    assert "em-equity" in d.gaps and "us-large" not in d.gaps
    em = [c for c in d.candidates if c.role == "em-equity"]
    assert [c.ticker for c in em] == ["VWO", "SCHE"]  # 3 in universe, capped to 2, file order
    assert all(c.ticker not in {"VOO", "BND"} for c in d.candidates)  # held never surfaces


def test_held_role_is_not_a_gap() -> None:
    s = _book({"VOO": 10, "VWO": 10})  # 50% emerging → em-equity covered
    prices = {"VOO": _price("VOO", 100.0), "VWO": _price("VWO", 100.0)}
    d = find_gaps(s, prices, _UNIVERSE)
    assert "em-equity" not in d.gaps
    assert all(c.role != "em-equity" for c in d.candidates)


def test_below_threshold_sleeve_is_still_a_gap() -> None:
    # A token 1% emerging sleeve is effectively a gap (<= 3%).
    s = _book({"VOO": 99, "VWO": 1})
    prices = {"VOO": _price("VOO", 100.0), "VWO": _price("VWO", 100.0)}
    d = find_gaps(s, prices, _UNIVERSE)
    assert "em-equity" in d.gaps  # 1% <= 3% threshold


def test_unpriced_book_yields_no_exposure() -> None:
    # No prices → no attributable exposure; the CLI must require the price pipeline.
    s = _book({"VOO": 10})
    assert role_exposure(s, {}, _UNIVERSE) == {}


def test_satellites_are_not_default_gaps() -> None:
    # sector/thematic: not holding one is a stance, not a hole — default surfacing skips
    # them; include_satellites (the explicitly-named path) brings them back.
    uni = _UNIVERSE + [Candidate("VGT", "Tech", "sector-equity", "tech sector")]
    s = _book({"VOO": 10})
    prices = {"VOO": _price("VOO", 100.0)}
    d = find_gaps(s, prices, uni)
    assert "sector-equity" not in d.gaps
    assert all(c.role != "sector-equity" for c in d.candidates)
    d_all = find_gaps(s, prices, uni, include_satellites=True)
    assert "sector-equity" in d_all.gaps
    assert any(c.ticker == "VGT" for c in d_all.candidates)


def test_core_candidates_surface_before_tilts() -> None:
    # A style tilt or junk bond never fills a gap while a core option exists — even when
    # the tilt leads the file (i.e. has the bigger AUM).
    uni = [
        Candidate("JNK", "Junk", "corporate-bond", "high yield", core=False),
        Candidate("VCIT", "IG Corp", "corporate-bond", "investment grade"),
        Candidate("LQD", "IG Corp 2", "corporate-bond", "investment grade"),
    ]
    s = _book({"AAPL": 10})  # not in the universe → corporate-bond reads as a 0% gap
    prices = {"AAPL": _price("AAPL", 100.0)}
    d = find_gaps(s, prices, uni, per_role_cap=2)
    assert [c.ticker for c in d.candidates] == ["VCIT", "LQD"]  # file-first JNK skipped: not core


def test_tilts_fall_back_when_no_core_candidate_remains() -> None:
    # A surfaced gap must never hide its candidates: when every core row is already held
    # (or the role has none), the tilts surface.
    uni = [
        Candidate("VCIT", "IG Corp", "corporate-bond", "investment grade"),
        Candidate("JNK", "Junk", "corporate-bond", "high yield", core=False),
    ]
    s = _book({"VCIT": 1, "AAPL": 99})  # a token 1% IG sleeve: still a gap, core row held
    prices = {"VCIT": _price("VCIT", 100.0), "AAPL": _price("AAPL", 100.0)}
    d = find_gaps(s, prices, uni)
    assert "corporate-bond" in d.gaps
    assert [c.ticker for c in d.candidates] == ["JNK"]  # the only unheld option


# --- the shelf rule (v2.12): flavors group near-substitutes; menus per shelf ---

_TREASURY = [
    Candidate("IEF", "Int 1", "treasury", "7-10y", flavor="intermediate"),
    Candidate("TLT", "Long 1", "treasury", "20y+", flavor="long"),
    Candidate("VGLT", "Long 2", "treasury", "long", flavor="long"),
    Candidate("TLH", "Long 3", "treasury", "10-20y", flavor="long"),
    Candidate("VGIT", "Int 2", "treasury", "int", flavor="intermediate"),
    Candidate("GOVT", "Int 3", "treasury", "whole market", flavor="intermediate"),
    Candidate("VGSH", "Short 1", "treasury", "1-3y", flavor="short"),
]


def _gapbook() -> tuple[DerivedState, dict[str, PriceRow]]:
    s = _book({"AAPL": 10})  # not in any universe → every role reads as a gap
    return s, {"AAPL": _price("AAPL", 100.0)}


def test_default_menu_is_the_lead_shelf_plus_named_others() -> None:
    # The default shows ONE shelf's near-substitutes (the first core row's flavor — the
    # same shelf the presets buy from) and NAMES the others with counts, never hiding them.
    s, prices = _gapbook()
    d = find_gaps(s, prices, _TREASURY)
    assert [c.ticker for c in d.candidates] == ["IEF", "VGIT", "GOVT"]
    assert d.lead_flavor["treasury"] == "intermediate"
    assert dict(d.more_shelves["treasury"]) == {"long": 3, "short": 1}


def test_default_menu_fills_when_the_lead_shelf_is_thin() -> None:
    # A gap never hides its candidates: a 1-fund lead shelf borrows from the next shelves.
    uni = [
        Candidate("GOVT", "Int", "treasury", "x", flavor="intermediate"),
        Candidate("TLT", "Long", "treasury", "x", flavor="long"),
        Candidate("VGSH", "Short", "treasury", "x", flavor="short"),
    ]
    s, prices = _gapbook()
    d = find_gaps(s, prices, uni)
    assert [c.ticker for c in d.candidates] == ["GOVT", "TLT", "VGSH"]


def test_role_menu_full_shows_a_trio_per_core_shelf() -> None:
    # The explicit path (--discover treasury): every shelf with core rows gets its menu.
    s, prices = _gapbook()
    d = role_menu(s, prices, _TREASURY, "treasury")
    assert [c.ticker for c in d.candidates] == [
        "IEF", "VGIT", "GOVT", "TLT", "VGLT", "TLH", "VGSH",
    ]
    assert d.more_shelves == {}


def test_role_menu_core_less_shelf_is_an_index_line_until_drilled() -> None:
    # Junk never becomes a menu by accident: a shelf with no core rows stays a named
    # count; drilling it BY NAME is consent to see the tilts, labeled as tilts.
    uni = [
        Candidate("VCIT", "IG 1", "corporate-bond", "x", flavor="investment-grade"),
        Candidate("LQD", "IG 2", "corporate-bond", "x", flavor="investment-grade"),
        Candidate("IGIB", "IG 3", "corporate-bond", "x", flavor="investment-grade"),
        Candidate("USHY", "Junk 1", "corporate-bond", "x", core=False, flavor="high-yield"),
        Candidate("HYG", "Junk 2", "corporate-bond", "x", core=False, flavor="high-yield"),
    ]
    s, prices = _gapbook()
    d = role_menu(s, prices, uni, "corporate-bond")
    assert [c.ticker for c in d.candidates] == ["VCIT", "LQD", "IGIB"]
    assert dict(d.more_shelves["corporate-bond"]) == {"high-yield": 2}
    drill = role_menu(s, prices, uni, "corporate-bond", flavor="high-yield")
    assert [c.ticker for c in drill.candidates] == ["USHY", "HYG"]


def test_role_menu_satellite_index_and_single_shelf_autodrill() -> None:
    # A multi-shelf satellite hands over the MAP (no candidates — picking the sector is
    # the user's bet); a single-shelf satellite drills straight through.
    multi = [
        Candidate("VGT", "Tech", "sector-equity", "x", flavor="tech"),
        Candidate("XLV", "Health", "sector-equity", "x", flavor="health"),
    ]
    s, prices = _gapbook()
    d = role_menu(s, prices, multi, "sector-equity")
    assert d.candidates == ()
    assert dict(d.more_shelves["sector-equity"]) == {"tech": 1, "health": 1}
    single = [Candidate("ICLN", "Clean", "sector-equity", "x", flavor="clean-energy")]
    d = role_menu(s, prices, single, "sector-equity")
    assert [c.ticker for c in d.candidates] == ["ICLN"]  # no pointless second step


def test_role_menu_unknown_flavor_names_the_valid_shelves() -> None:
    s, prices = _gapbook()
    try:
        role_menu(s, prices, _TREASURY, "treasury", flavor="mortgage")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "intermediate" in str(exc) and "long" in str(exc)


def test_merge_menus_combines_named_roles() -> None:
    s, prices = _gapbook()
    uni = _TREASURY + [Candidate("VNQ", "REIT", "reit", "x", flavor="us")]
    merged = merge_menus([
        role_menu(s, prices, uni, "reit"),
        role_menu(s, prices, uni, "treasury", flavor="short"),
    ])
    assert merged.gaps == ("reit", "treasury")
    assert {c.ticker for c in merged.candidates} == {"VNQ", "VGSH"}
    assert merged.lead_flavor["treasury"] == "short"


def test_borrowed_shelf_keeps_its_remaining_count() -> None:
    # Review fix: when a thin lead shelf borrows a fund from the next shelf, the borrowed
    # shelf's REMAINING funds must stay in "also here" — counted by fund, never dropped
    # because its flavor technically appeared in the menu.
    uni = [
        Candidate("IEF", "Int 1", "treasury", "x", flavor="intermediate"),
        Candidate("VGIT", "Int 2", "treasury", "x", flavor="intermediate"),
        Candidate("TLT", "Long 1", "treasury", "x", flavor="long"),
        Candidate("VGLT", "Long 2", "treasury", "x", flavor="long"),
        Candidate("TLH", "Long 3", "treasury", "x", flavor="long"),
        Candidate("SPTL", "Long 4", "treasury", "x", flavor="long"),
    ]
    s, prices = _gapbook()
    d = find_gaps(s, prices, uni)
    assert [c.ticker for c in d.candidates] == ["IEF", "VGIT", "TLT"]  # 2 int + 1 borrowed
    assert dict(d.more_shelves["treasury"]) == {"long": 3}  # VGLT/TLH/SPTL still named


def test_merge_menus_dedups_repeated_tickers() -> None:
    # Defensive belt: a caller that passes overlapping menus never double-lists a fund.
    s, prices = _gapbook()
    full = role_menu(s, prices, _TREASURY, "treasury")
    drill = role_menu(s, prices, _TREASURY, "treasury", flavor="long")
    merged = merge_menus([full, drill])
    tickers = [c.ticker for c in merged.candidates]
    assert len(tickers) == len(set(tickers))
