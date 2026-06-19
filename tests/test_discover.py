"""Tests for the discovery core (app/discover.py). Pure — no I/O, no network."""

from __future__ import annotations

from datetime import date, datetime, timezone

from app.derive import DerivedState, Position
from app.discover import find_gaps, gap_roles, restrict_to, role_exposure
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


def test_restrict_to_narrows_to_named_gaps() -> None:
    # restrict_to keeps only requested roles that ARE gaps (intersection), preserving exposure —
    # the --discover reit,tips path. Requesting a non-gap (us-large, which is held) drops it.
    s = _book({"VOO": 10, "BND": 10})
    prices = {"VOO": _price("VOO", 100.0), "BND": _price("BND", 100.0)}
    full = find_gaps(s, prices, _UNIVERSE)
    r = restrict_to(full, {"reit", "us-large"})
    assert r.gaps == ("reit",)
    assert all(c.role == "reit" for c in r.candidates)
    assert r.exposure == full.exposure
