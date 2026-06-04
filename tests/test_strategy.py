"""Tests for the strategy/suggestion engine. Pure — no I/O except load_target."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.strategy import Suggestion, load_target, suggest

# A simple, hand-checkable book: $10,000 split 60/40 VOO/BND; IAU is a buy-in.
HELD = {"VOO": 6000.0, "BND": 4000.0}
PRICES = {"VOO": 300.0, "BND": 80.0, "IAU": 40.0}
TARGET = {"VOO": 0.50, "BND": 0.25, "IAU": 0.25}


def _by_ticker(sugs: list[Suggestion]) -> dict[str, Suggestion]:
    return {s.ticker: s for s in sugs}


# ── load_target ────────────────────────────────────────────────────────────


def test_load_target_normalizes_percentages(tmp_path: Path) -> None:
    p = tmp_path / "t.csv"
    p.write_text("Ticker,Weight\nVOO,50\nBND,25\nIAU,25\n", encoding="utf-8")
    assert load_target(p) == {"VOO": 0.5, "BND": 0.25, "IAU": 0.25}


def test_load_target_fractions_same_as_percent(tmp_path: Path) -> None:
    p = tmp_path / "t.csv"
    p.write_text("Ticker,Weight\nVOO,0.5\nBND,0.25\nIAU,0.25\n", encoding="utf-8")
    assert load_target(p) == {"VOO": 0.5, "BND": 0.25, "IAU": 0.25}


def test_load_target_rejects_empty(tmp_path: Path) -> None:
    p = tmp_path / "t.csv"
    p.write_text("Ticker,Weight\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_target(p)


def test_load_target_rejects_negative(tmp_path: Path) -> None:
    p = tmp_path / "t.csv"
    p.write_text("Ticker,Weight\nVOO,50\nBND,-1\n", encoding="utf-8")
    with pytest.raises(ValueError, match=">= 0"):
        load_target(p)


def test_load_target_accepts_zero_as_exit(tmp_path: Path) -> None:
    # 0 is a deliberate close; it's kept (so the CLI can tell it from an omission).
    p = tmp_path / "t.csv"
    p.write_text("Ticker,Weight\nVOO,50\nBND,50\nGLDM,0\n", encoding="utf-8")
    t = load_target(p)
    assert t["GLDM"] == 0.0
    assert t["VOO"] == pytest.approx(0.5) and t["BND"] == pytest.approx(0.5)


def test_load_target_all_zero_rejected(tmp_path: Path) -> None:
    p = tmp_path / "t.csv"
    p.write_text("Ticker,Weight\nVOO,0\nBND,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty or sums to zero"):
        load_target(p)


# ── to_total ───────────────────────────────────────────────────────────────


def test_to_total_trades_to_target_cash_neutral() -> None:
    s = _by_ticker(suggest("to_total", HELD, PRICES, TARGET))
    assert s["VOO"].action == "sell" and s["VOO"].dollars == pytest.approx(1000.0)
    assert s["VOO"].shares == pytest.approx(1000.0 / 300.0)
    assert s["BND"].action == "sell" and s["BND"].dollars == pytest.approx(1500.0)
    assert s["IAU"].action == "buy" and s["IAU"].dollars == pytest.approx(2500.0)
    # Rebalance is cash-neutral: buys == sells.
    buys = sum(x.dollars for x in s.values() if x.action == "buy")
    sells = sum(x.dollars for x in s.values() if x.action == "sell")
    assert buys == pytest.approx(sells)


def test_to_total_resulting_weights_equal_target() -> None:
    s = _by_ticker(suggest("to_total", HELD, PRICES, TARGET))
    base = sum(HELD.values())
    for tk, tgt in TARGET.items():
        signed = s[tk].dollars * (1 if s[tk].action == "buy" else -1 if s[tk].action == "sell" else 0)
        resulting = HELD.get(tk, 0.0) + signed
        assert resulting / base == pytest.approx(tgt)


def test_to_total_sells_out_ticker_absent_from_target() -> None:
    held = {"VOO": 5000.0, "SCHP": 5000.0}
    s = _by_ticker(suggest("to_total", held, {"VOO": 300.0, "SCHP": 25.0}, {"VOO": 1.0}))
    assert s["SCHP"].action == "sell" and s["SCHP"].dollars == pytest.approx(5000.0)
    assert s["SCHP"].target_weight == 0.0
    # The reason makes clear this is a full exit (target 0%).
    assert "full exit" in s["SCHP"].reason


def test_explicit_zero_weight_exits_position() -> None:
    # A held ticker listed at weight 0 → sold to $0, same as omitting it.
    held = {"VOO": 6000.0, "GLDM": 2000.0}
    s = _by_ticker(suggest("to_total", held, {"VOO": 300.0, "GLDM": 100.0},
                           {"VOO": 1.0, "GLDM": 0.0}))
    assert s["GLDM"].action == "sell" and s["GLDM"].dollars == pytest.approx(2000.0)
    assert s["GLDM"].target_weight == 0.0


def test_to_total_deploys_new_cash() -> None:
    # 10k held, +2k cash, single-target VOO → buy the whole 2k of VOO.
    s = _by_ticker(suggest("to_total", {"VOO": 10000.0}, {"VOO": 100.0},
                           {"VOO": 1.0}, new_cash=2000.0))
    assert s["VOO"].action == "buy" and s["VOO"].dollars == pytest.approx(2000.0)


# ── cash_flow_only ─────────────────────────────────────────────────────────


def test_cash_flow_only_buys_underweight_never_sells() -> None:
    s = _by_ticker(suggest("cash_flow_only", HELD, PRICES, TARGET, new_cash=1000.0))
    assert all(x.action in ("buy", "hold") for x in s.values())  # never sells
    # VOO & BND are above their post-cash target → hold; IAU underweight → gets all cash.
    assert s["IAU"].action == "buy" and s["IAU"].dollars == pytest.approx(1000.0)
    assert s["VOO"].action == "hold"
    assert s["BND"].action == "hold"


def test_cash_flow_only_no_cash_is_all_hold() -> None:
    s = _by_ticker(suggest("cash_flow_only", HELD, PRICES, TARGET, new_cash=0.0))
    assert all(x.action == "hold" for x in s.values())


# ── fixed_dca ──────────────────────────────────────────────────────────────


def test_fixed_dca_buys_target_mix_ignoring_drift() -> None:
    s = _by_ticker(suggest("fixed_dca", HELD, PRICES, TARGET, new_cash=1000.0))
    assert all(x.action in ("buy", "hold") for x in s.values())
    assert s["VOO"].dollars == pytest.approx(500.0)   # 50% of 1000
    assert s["BND"].dollars == pytest.approx(250.0)
    assert s["IAU"].dollars == pytest.approx(250.0)


# ── bands ──────────────────────────────────────────────────────────────────


def test_bands_holds_within_band_acts_outside() -> None:
    # VOO 60% vs 55% target = 5pp drift (== band → hold); BND 40% vs 45% (within → hold).
    s = _by_ticker(suggest("bands", HELD, PRICES, {"VOO": 0.55, "BND": 0.45}, band=0.05))
    assert s["VOO"].action == "hold" and "within" in s["VOO"].reason
    assert s["BND"].action == "hold"


def test_bands_acts_when_drift_exceeds_band() -> None:
    # VOO 60% vs 25% target → 35pp drift > 5pp → sell.
    s = _by_ticker(suggest("bands", HELD, PRICES, TARGET, band=0.05))
    assert s["VOO"].action == "sell"
    assert "band" in s["VOO"].reason


# ── edges ──────────────────────────────────────────────────────────────────


def test_unpriced_target_ticker_is_skipped() -> None:
    # IAU has no price → dropped from the universe, not suggested.
    s = _by_ticker(suggest("to_total", HELD, {"VOO": 300.0, "BND": 80.0}, TARGET))
    assert "IAU" not in s


def test_drift_property() -> None:
    s = _by_ticker(suggest("to_total", HELD, PRICES, TARGET))
    assert s["VOO"].drift == pytest.approx(s["VOO"].current_weight - s["VOO"].target_weight)


def test_zero_price_ticker_skipped_no_crash() -> None:
    # A 0.0 price (bad/stale quote) must be skipped, not divided by → no crash.
    s = _by_ticker(suggest("to_total", {"VOO": 6000.0, "BND": 4000.0},
                           {"VOO": 300.0, "BND": 0.0}, {"VOO": 0.5, "BND": 0.5}))
    assert "BND" not in s
    assert "VOO" in s


def test_bands_boundary_deterministic_hold() -> None:
    # Nominal exactly-5pp drift holds regardless of float rounding of the weight diff.
    s = _by_ticker(suggest("bands", {"VOO": 5500.0, "BND": 4500.0},
                           {"VOO": 100.0, "BND": 100.0}, {"VOO": 0.50, "BND": 0.50},
                           band=0.05))
    assert s["VOO"].action == "hold" and s["BND"].action == "hold"


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="unhandled rebalance mode"):
        suggest("bogus", HELD, PRICES, TARGET)  # type: ignore[arg-type]


def test_load_target_nonnumeric_weight(tmp_path: Path) -> None:
    p = tmp_path / "t.csv"
    p.write_text("Ticker,Weight\nVOO,0.5x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-numeric"):
        load_target(p)
