"""Tests for the CSV input boundary (events.py): load_events + load_target.

The loader is the highest-leverage spot for a silent data bug (garbage in →
confidently-wrong everything out), so its contract is pinned directly here rather
than only exercised indirectly through derive/cli.
"""

from __future__ import annotations

from datetime import date
from math import isclose
from pathlib import Path

import pytest

from app.events import VALID_ACTIONS, load_events, load_target

_HEADER = "Date,Code,DataSource,Currency,Price,Quantity,Action,Fee,Note\n"


def _csv(tmp_path: Path, text: str, name: str = "book.csv") -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


# ── load_events ──────────────────────────────────────────────────────────────


def test_load_events_parses_a_basic_row(tmp_path: Path) -> None:
    ev = load_events(_csv(tmp_path, _HEADER + "2024-01-02,voo ,YAHOO,USD,400,10,buy,1.5,first\n"))
    assert len(ev) == 1
    e = ev[0]
    assert e.date == date(2024, 1, 2)
    assert e.ticker == "VOO"  # upper-cased + stripped
    assert e.action == "buy"
    assert e.price == 400.0 and e.quantity == 10.0 and e.fee == 1.5
    assert e.cash == 0.0  # a trade carries no cash
    assert e.currency == "USD" and e.source == "YAHOO" and e.note == "first"


def test_load_events_tolerates_bom_and_empty_numeric_cells(tmp_path: Path) -> None:
    text = "﻿" + _HEADER + "2024-01-02,BND,YAHOO,USD,,5,buy,,\n"  # BOM + empty Price/Fee
    e = load_events(_csv(tmp_path, text))[0]
    assert e.price == 0.0 and e.fee == 0.0 and e.quantity == 5.0


def test_load_events_mixed_case_action(tmp_path: Path) -> None:
    e = load_events(_csv(tmp_path, _HEADER + "2024-01-02,VOO,YAHOO,USD,400,1,BUY,0,\n"))[0]
    assert e.action == "buy"


@pytest.mark.parametrize("action", ["dividend", "interest", "deposit", "withdraw"])
def test_load_events_maps_price_to_cash_for_cash_actions(tmp_path: Path, action: str) -> None:
    # For these the Price column carries a cash amount → moved to `cash`, price 0.
    code = "CASH" if action in ("deposit", "withdraw") else "VOO"
    e = load_events(_csv(tmp_path, _HEADER + f"2024-03-01,{code},YAHOO,USD,12.5,0,{action},0,\n"))[0]
    assert e.cash == 12.5 and e.price == 0.0


def test_load_events_unknown_action_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown action"):
        load_events(_csv(tmp_path, _HEADER + "2024-01-02,VOO,YAHOO,USD,1,1,grow,0,\n"))


def test_load_events_non_iso_date_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ISO 8601"):
        load_events(_csv(tmp_path, _HEADER + "01/03/2024,VOO,YAHOO,USD,1,1,buy,0,\n"))


def test_load_events_compact_iso_date_accepted(tmp_path: Path) -> None:
    e = load_events(_csv(tmp_path, _HEADER + "20240301,VOO,YAHOO,USD,1,1,buy,0,\n"))[0]
    assert e.date == date(2024, 3, 1)


def test_load_events_missing_required_column_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing required column"):  # no Price column
        load_events(_csv(tmp_path, "Date,Code,Action,Quantity,Fee\n2024-01-02,VOO,buy,1,0\n"))


def test_load_events_empty_file_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        load_events(_csv(tmp_path, ""))


def test_load_events_same_day_sell_sorted_after_buy(tmp_path: Path) -> None:
    # A SELL listed BEFORE its BUY on the same day must be reordered buy-first.
    text = _HEADER + (
        "2024-01-02,VOO,YAHOO,USD,400,5,sell,0,\n"
        "2024-01-02,VOO,YAHOO,USD,400,10,buy,0,\n"
    )
    assert [e.action for e in load_events(_csv(tmp_path, text))] == ["buy", "sell"]


def test_load_events_defaults_when_optional_columns_absent(tmp_path: Path) -> None:
    # Only the required columns present → Currency/DataSource/Note take defaults.
    text = "Date,Code,Action,Quantity,Price,Fee\n2024-01-02,VOO,buy,1,400,0\n"
    e = load_events(_csv(tmp_path, text))[0]
    assert e.currency == "USD" and e.source == "MANUAL" and e.note == ""


def test_valid_actions_are_the_seven_known_kinds() -> None:
    assert VALID_ACTIONS == {
        "buy", "sell", "dividend", "fee", "interest", "deposit", "withdraw"
    }


# ── load_target (moved here from strategy.py — the CSV input boundary) ───────


def test_load_target_normalizes_percentages(tmp_path: Path) -> None:
    p = _csv(tmp_path, "Ticker,Weight\nVOO,50\nBND,25\nIAU,25\n", "t.csv")
    assert load_target(p) == {"VOO": 0.5, "BND": 0.25, "IAU": 0.25}


def test_load_target_fractions_same_as_percent(tmp_path: Path) -> None:
    p = _csv(tmp_path, "Ticker,Weight\nVOO,0.5\nBND,0.25\nIAU,0.25\n", "t.csv")
    assert load_target(p) == {"VOO": 0.5, "BND": 0.25, "IAU": 0.25}


def test_load_target_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        load_target(_csv(tmp_path, "Ticker,Weight\n", "t.csv"))


def test_load_target_rejects_negative(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=">= 0"):
        load_target(_csv(tmp_path, "Ticker,Weight\nVOO,50\nBND,-1\n", "t.csv"))


def test_load_target_accepts_zero_as_exit(tmp_path: Path) -> None:
    # 0 is a deliberate close; kept so the CLI can tell it from an omission.
    t = load_target(_csv(tmp_path, "Ticker,Weight\nVOO,50\nBND,50\nGLDM,0\n", "t.csv"))
    assert t["GLDM"] == 0.0
    assert isclose(t["VOO"], 0.5) and isclose(t["BND"], 0.5)


def test_load_target_all_zero_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty or sums to zero"):
        load_target(_csv(tmp_path, "Ticker,Weight\nVOO,0\nBND,0\n", "t.csv"))


def test_load_target_nonnumeric_weight_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-numeric"):
        load_target(_csv(tmp_path, "Ticker,Weight\nVOO,0.5x\n", "t.csv"))


# ── Ghostfolio JSON export (load_events accepts a Ghostfolio export directly) ──


def _gf_json(tmp_path: Path, activities: list[dict[str, object]], wrap: bool = True) -> Path:
    """Write a Ghostfolio JSON export — the full object with an `activities` array, or
    (wrap=False) a bare activities list — and return its path."""
    import json

    obj: object = {"activities": activities} if wrap else activities
    p = tmp_path / "export.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def _act(**kw: object) -> dict[str, object]:
    base: dict[str, object] = {
        "symbol": "VOO", "currency": "USD", "unitPrice": 400, "quantity": 10, "type": "BUY",
        "fee": 1, "dataSource": "YAHOO", "date": "2024-01-02T00:00:00.000Z", "comment": None,
    }
    base.update(kw)
    return base


def test_loads_a_ghostfolio_json_export(tmp_path: Path) -> None:
    p = _gf_json(tmp_path, [
        _act(type="BUY", unitPrice=400, quantity=10, fee=1),
        _act(type="DIVIDEND", unitPrice=22.4, quantity=1, fee=0, date="2024-02-01T00:00:00.000Z"),
        _act(type="SELL", unitPrice=450, quantity=4, fee=1, date="2024-03-01T00:00:00.000Z"),
    ])
    by = {(e.action, e.ticker): e for e in load_events(p)}
    assert by[("buy", "VOO")].quantity == 10 and by[("buy", "VOO")].price == 400
    div = by[("dividend", "VOO")]
    assert div.cash == 22.4 and div.price == 0 and div.quantity == 0  # quantity×unitPrice
    assert by[("sell", "VOO")].quantity == 4


def test_ghostfolio_json_accepts_a_bare_activities_list(tmp_path: Path) -> None:
    assert load_events(_gf_json(tmp_path, [_act()], wrap=False))[0].ticker == "VOO"


def test_ghostfolio_dividend_is_quantity_times_unitprice(tmp_path: Path) -> None:
    # shares × per-share AND 1 × total both yield the right cash ($50).
    assert load_events(_gf_json(tmp_path, [_act(type="DIVIDEND", quantity=100, unitPrice=0.5)]))[0].cash == 50.0
    assert load_events(_gf_json(tmp_path, [_act(type="DIVIDEND", quantity=1, unitPrice=50)]))[0].cash == 50.0


def test_ghostfolio_json_date_rounds_to_nearest_day(tmp_path: Path) -> None:
    # Ghostfolio stores UTC; a KST-midnight entry exports as 15:00Z the PREVIOUS day.
    # Round-to-nearest recovers the intended local date.
    assert load_events(_gf_json(tmp_path, [_act(date="2023-01-04T15:00:00.000Z")]))[0].date == date(2023, 1, 5)
    assert load_events(_gf_json(tmp_path, [_act(date="2024-01-02T00:00:00.000Z")]))[0].date == date(2024, 1, 2)


def test_ghostfolio_json_skips_out_of_scope(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    p = _gf_json(tmp_path, [
        _act(symbol="VOO"),
        _act(symbol="HOUSE", type="ITEM"),
        _act(symbol="ASML.AS", currency="EUR"),
        _act(symbol="BTC", dataSource="COINGECKO"),
    ])
    with caplog.at_level("WARNING"):
        events = load_events(p)
    assert {e.ticker for e in events} == {"VOO"}
    assert "ITEM" in caplog.text and "EUR" in caplog.text


def test_ghostfolio_json_end_to_end_derive(tmp_path: Path) -> None:
    from app.derive import derive

    p = _gf_json(tmp_path, [
        _act(type="BUY", unitPrice=400, quantity=10, fee=1),
        _act(type="DIVIDEND", unitPrice=22.4, quantity=1, fee=0, date="2024-02-01T00:00:00.000Z"),
        _act(type="SELL", unitPrice=450, quantity=4, fee=1, date="2024-03-01T00:00:00.000Z"),
    ])
    state = derive(load_events(p))
    assert state.held()["VOO"].shares == 6
    assert isclose(state.realized["VOO"], 22.4 + 4 * (450.0 - 4001.0 / 10.0) - 1.0)


def test_ghostfolio_json_preserves_fractional_share_precision(tmp_path: Path) -> None:
    # _gf_num must not truncate fractional shares — DRIP buys carry 7-8 decimals, and a
    # 6-dp round (the old %.6f) would drift the share count and cost basis vs the broker.
    p = _gf_json(tmp_path, [_act(type="BUY", quantity=0.123456789, unitPrice=400.0)])
    assert load_events(p)[0].quantity == 0.123456789


def test_ghostfolio_json_skips_symbol_less_activity(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # A null/missing symbol must be skipped with a warning, never minted as a "" ticker.
    p = _gf_json(tmp_path, [_act(symbol=None, type="BUY"), _act(symbol="VOO")])
    with caplog.at_level("WARNING"):
        events = load_events(p)
    assert [e.ticker for e in events] == ["VOO"]
    assert "no symbol" in caplog.text


def test_csv_dividend_with_quantity_collapses(tmp_path: Path) -> None:
    # Ghostfolio's *CSV* import uses our columns but per-share × shares dividends; the
    # income rule handles it (Price 0.62 × Quantity 5 → $3.10), while ours stays Quantity 0.
    assert isclose(load_events(_csv(tmp_path, _HEADER + "2024-02-01,VOO,YAHOO,USD,0.62,5,dividend,0,\n"))[0].cash, 3.10)
    assert load_events(_csv(tmp_path, _HEADER + "2024-02-01,VOO,YAHOO,USD,22.40,0,dividend,0,\n"))[0].cash == 22.40
