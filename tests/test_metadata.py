"""Tests for the securities-metadata layer (offline: `_fetch_yf_meta` is monkey-patched)."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from math import isclose
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from app.events import CASH_TICKER
from app.metadata import MetadataResult, SecurityMeta, fetch_metadata

_VOO_INCEPTION_EPOCH = 1_284_595_200  # 2010-09-16 UTC


def _canned_raw(quote_type: str = "ETF") -> dict[str, Any]:
    """A realistic `_fetch_yf_meta` payload (shapes match yfinance's funds_data)."""
    info = {
        "totalAssets": 1_701_513_003_008,
        "averageVolume": 8_809_723,
        "quoteType": quote_type,
        "category": "Large Blend",
        "fundInceptionDate": _VOO_INCEPTION_EPOCH,
    }
    ops = pd.DataFrame(
        {"VOO": [0.0003, 0.022]},
        index=["Annual Report Expense Ratio", "Annual Holdings Turnover"],
    )
    overview = {"categoryName": "Large Blend", "family": "Vanguard",
                "legalType": "Exchange Traded Fund"}
    holdings = pd.DataFrame(
        {"Name": ["NVIDIA", "Apple"], "Holding Percent": [0.066, 0.060]},
        index=["NVDA", "AAPL"],
    )
    return {"info": info, "ops": ops, "overview": overview, "holdings": holdings}


def test_fetch_normalizes_yfinance_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.metadata._fetch_yf_meta", lambda tk: _canned_raw())
    result = fetch_metadata(["VOO"], cache_dir=tmp_path)
    m = result.rows["VOO"]
    assert result.missing == []
    assert m.expense_ratio is not None and isclose(m.expense_ratio, 0.0003)
    assert m.aum == 1_701_513_003_008
    assert m.avg_volume == 8_809_723
    assert m.category == "Large Blend" and m.family == "Vanguard"
    assert m.legal_type == "Exchange Traded Fund" and m.quote_type == "ETF"
    assert m.inception == date(2010, 9, 16)  # epoch seconds → date
    assert m.top_holdings == {"NVDA": 0.066, "AAPL": 0.060}
    assert m.source == "yfinance"
    age = m.age_years(date(2026, 6, 12))
    assert age is not None and 15.5 < age < 16.0


def test_cache_round_trip_serves_second_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def counting(tk: str) -> dict[str, Any]:
        calls.append(tk)
        return _canned_raw()

    monkeypatch.setattr("app.metadata._fetch_yf_meta", counting)
    first = fetch_metadata(["VOO"], cache_dir=tmp_path)
    second = fetch_metadata(["VOO"], cache_dir=tmp_path)
    assert calls == ["VOO"]  # one live fetch; the second call was served from disk
    assert first.rows["VOO"].source == "yfinance"
    assert second.rows["VOO"].source == "cache"
    # The cached row carries the SAME facts (incl. nested holdings) and timestamp.
    assert second.rows["VOO"].top_holdings == first.rows["VOO"].top_holdings
    assert second.rows["VOO"].fetched_at == first.rows["VOO"].fetched_at
    assert second.n_cache == 1


def _age_cache(tmp_path: Path, ticker: str, *, days: float) -> None:
    """Rewrite a cached file's fetched_at to `days` ago (negative = future)."""
    path = tmp_path / f"{ticker}_meta.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    stamp = datetime.now(timezone.utc) - timedelta(days=days)
    payload["fetched_at"] = stamp.isoformat()
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_stale_cache_refetches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "app.metadata._fetch_yf_meta", lambda tk: calls.append(tk) or _canned_raw()
    )
    fetch_metadata(["VOO"], cache_dir=tmp_path)
    _age_cache(tmp_path, "VOO", days=8)  # past the 7-day TTL
    result = fetch_metadata(["VOO"], cache_dir=tmp_path)
    assert calls == ["VOO", "VOO"]  # stale → live refetch
    assert result.rows["VOO"].source == "yfinance"


def test_future_stamped_cache_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.metadata._fetch_yf_meta", lambda tk: _canned_raw())
    fetch_metadata(["VOO"], cache_dir=tmp_path)
    _age_cache(tmp_path, "VOO", days=-2)  # stamped in the future → tampered/skewed
    offline = fetch_metadata(["VOO"], cache_dir=tmp_path, online=False)
    assert offline.rows == {} and offline.missing == ["VOO"]


def test_offline_cache_or_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    boom = lambda tk: pytest.fail("offline run must not call the network wrapper")  # noqa: E731
    monkeypatch.setattr("app.metadata._fetch_yf_meta", boom)
    no_cache = fetch_metadata(["VOO"], cache_dir=tmp_path, online=False)
    assert no_cache.missing == ["VOO"]

    monkeypatch.setattr("app.metadata._fetch_yf_meta", lambda tk: _canned_raw())
    fetch_metadata(["VOO"], cache_dir=tmp_path)  # warm the cache
    monkeypatch.setattr("app.metadata._fetch_yf_meta", boom)
    served = fetch_metadata(["VOO"], cache_dir=tmp_path, online=False)
    assert served.rows["VOO"].source == "cache" and served.missing == []


def test_cash_pseudo_ticker_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[str] = []
    monkeypatch.setattr(
        "app.metadata._fetch_yf_meta", lambda tk: asked.append(tk) or _canned_raw()
    )
    result = fetch_metadata([CASH_TICKER, "VOO"], cache_dir=tmp_path)
    assert asked == ["VOO"]  # cash is not a security
    assert CASH_TICKER not in result.rows and CASH_TICKER not in result.missing


def test_equity_without_funds_data_degrades_per_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A plain stock: info exists, funds_data raised → fund fields are None, the
    # ticker is NOT missing (per-field degradation, recorded design).
    raw = {"info": {"quoteType": "EQUITY", "averageVolume": 1_000_000},
           "ops": None, "overview": None, "holdings": None}
    monkeypatch.setattr("app.metadata._fetch_yf_meta", lambda tk: raw)
    result = fetch_metadata(["NVDA"], cache_dir=tmp_path)
    m = result.rows["NVDA"]
    assert result.missing == []
    assert m.quote_type == "EQUITY" and m.avg_volume == 1_000_000
    assert m.expense_ratio is None and m.aum is None and m.inception is None
    assert m.top_holdings == {}


def test_wrapper_failure_lands_in_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.metadata._fetch_yf_meta", lambda tk: None)
    result = fetch_metadata(["GHOST"], cache_dir=tmp_path)
    assert result.rows == {} and result.missing == ["GHOST"]


def test_corrupt_cache_refetches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "VOO_meta.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr("app.metadata._fetch_yf_meta", lambda tk: _canned_raw())
    result = fetch_metadata(["VOO"], cache_dir=tmp_path)
    assert result.rows["VOO"].source == "yfinance"  # corrupt file ignored, refetched


def test_valid_json_with_corrupt_field_refetches_not_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression (review 2026-06-12): a FRESH, valid-JSON cache with one malformed
    # value (non-numeric holding weight) must degrade to a refetch — never raise.
    # Before the fix this crashed fetch_metadata with an uncaught ValueError.
    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "top_holdings": {"NVDA": "corrupt"},
    }
    (tmp_path / "VOO_meta.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr("app.metadata._fetch_yf_meta", lambda tk: _canned_raw())
    result = fetch_metadata(["VOO"], cache_dir=tmp_path)
    assert result.rows["VOO"].source == "yfinance"  # malformed cache → live refetch
    offline = fetch_metadata(["GHOST"], cache_dir=tmp_path, online=False)
    assert offline.missing == ["GHOST"]  # and offline it's just missing, no crash


def test_infinite_values_become_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ±inf survives a NaN-only check; the brief must never render "$inf".
    raw = _canned_raw()
    raw["info"]["totalAssets"] = float("inf")
    raw["info"]["averageVolume"] = float("-inf")
    monkeypatch.setattr("app.metadata._fetch_yf_meta", lambda tk: raw)
    m = fetch_metadata(["VOO"], cache_dir=tmp_path).rows["VOO"]
    assert m.aum is None and m.avg_volume is None


def test_metadata_result_shapes() -> None:
    # The dataclass contract consumers (report/screen) rely on.
    m = SecurityMeta(
        ticker="X", expense_ratio=None, aum=None, avg_volume=None, category=None,
        family=None, legal_type=None, quote_type=None, inception=None,
    )
    assert m.age_years(date(2026, 1, 1)) is None
    assert MetadataResult().rows == {} and MetadataResult().missing == []
