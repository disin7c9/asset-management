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
