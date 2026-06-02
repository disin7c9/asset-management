"""Tests for the price fetcher.

Offline by default: each test monkey-patches the two underlying network
wrappers (`_fetch_yf`, `_fetch_stooq_csv`) so no real HTTP happens.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from app import prices as P
from app.prices import PriceRow, PricesResult, fetch_latest, fetch_series


def _fake_yf_df(price: float, on: date) -> pd.DataFrame:
    return pd.DataFrame({"Close": [price]}, index=pd.DatetimeIndex([pd.Timestamp(on)]))


def _fake_stooq_csv(price: float, on: date) -> str:
    return f"Date,Open,High,Low,Close,Volume\n{on.isoformat()},0,0,0,{price},0\n"


def _write_cache_row(
    path: Path, asof: date, close: float, fetched_at: datetime
) -> None:
    pd.DataFrame(
        [
            {
                "asof_date": pd.Timestamp(asof),
                "close": close,
                "fetched_at": pd.Timestamp(fetched_at),
            }
        ]
    ).to_parquet(path, index=False)


# ── happy paths ──────────────────────────────────────────────────────────


def test_fetch_returns_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(P, "_fetch_yf", lambda t, s, e: _fake_yf_df(100.0, date(2024, 1, 5)))
    monkeypatch.setattr(P, "_fetch_stooq_csv", lambda t: None)
    result = fetch_latest(["VOO"], asof_date=date(2024, 1, 6), cache_dir=tmp_path)
    assert isinstance(result, PricesResult)
    assert "VOO" in result.rows
    p = result.rows["VOO"]
    assert p.close == 100.0
    assert p.source == "yfinance"
    assert p.fetched_at.tzinfo is not None  # tz-aware


def test_fetch_writes_cache_after_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(P, "_fetch_yf", lambda t, s, e: _fake_yf_df(123.45, date(2024, 1, 5)))
    fetch_latest(["VOO"], asof_date=date(2024, 1, 6), cache_dir=tmp_path)
    cache_file = tmp_path / "VOO.parquet"
    assert cache_file.exists()
    df = pd.read_parquet(cache_file)
    assert df["close"].iloc[0] == 123.45
    # The cache no longer stores `source` (it would always be a duplicate of
    # what the *reader* reports as "cache" anyway).
    assert "source" not in df.columns


def test_fetch_uses_cache_when_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asof = date(2024, 1, 5)
    _write_cache_row(
        tmp_path / "VOO.parquet",
        asof,
        99.0,
        datetime.now(timezone.utc) - timedelta(minutes=5),
    )
    monkeypatch.setattr(P, "_fetch_yf", lambda *a, **k: pytest.fail("yf should not be called"))
    monkeypatch.setattr(
        P, "_fetch_stooq_csv", lambda *a, **k: pytest.fail("stooq should not be called")
    )
    result = fetch_latest(["VOO"], asof_date=asof, cache_dir=tmp_path)
    assert result.rows["VOO"].source == "cache"
    assert result.rows["VOO"].close == 99.0


def test_fetch_bypasses_stale_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asof = date(2024, 1, 5)
    _write_cache_row(
        tmp_path / "VOO.parquet",
        asof,
        50.0,
        datetime.now(timezone.utc) - timedelta(days=3),
    )
    monkeypatch.setattr(P, "_fetch_yf", lambda t, s, e: _fake_yf_df(75.0, asof))
    result = fetch_latest(["VOO"], asof_date=asof, cache_dir=tmp_path)
    assert result.rows["VOO"].source == "yfinance"
    assert result.rows["VOO"].close == 75.0


# ── fallback chain ──────────────────────────────────────────────────────────


def test_fetch_falls_back_to_stooq(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asof = date(2024, 1, 5)
    monkeypatch.setattr(P, "_fetch_yf", lambda t, s, e: None)
    monkeypatch.setattr(P, "_fetch_stooq_csv", lambda t: _fake_stooq_csv(88.0, asof))
    result = fetch_latest(["VOO"], asof_date=asof, cache_dir=tmp_path)
    assert result.rows["VOO"].source == "stooq"
    assert result.fallbacks_used == 1


def test_fetch_returns_partial_on_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for review finding 2: one bad ticker must not abort the whole report."""
    asof = date(2024, 1, 5)
    monkeypatch.setattr(
        P,
        "_fetch_yf",
        lambda t, s, e: _fake_yf_df(100.0, asof) if t == "VOO" else None,
    )
    monkeypatch.setattr(P, "_fetch_stooq_csv", lambda t: None)
    result = fetch_latest(["VOO", "ZZZBAD"], asof_date=asof, cache_dir=tmp_path)
    assert "VOO" in result.rows
    assert "ZZZBAD" not in result.rows
    assert result.missing == ["ZZZBAD"]


def test_fetch_lowercase_stooq_header_is_tolerated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asof = date(2024, 1, 5)
    monkeypatch.setattr(P, "_fetch_yf", lambda t, s, e: None)
    monkeypatch.setattr(
        P,
        "_fetch_stooq_csv",
        lambda t: f"date,open,high,low,close,volume\n{asof.isoformat()},0,0,0,77.0,0\n",
    )
    result = fetch_latest(["VOO"], asof_date=asof, cache_dir=tmp_path)
    assert result.rows["VOO"].close == 77.0
    assert result.rows["VOO"].source == "stooq"


# ── shape / hygiene ──────────────────────────────────────────────────────


def test_offline_mode_uses_cache_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asof = date(2024, 1, 5)
    _write_cache_row(
        tmp_path / "VOO.parquet",
        asof,
        60.0,
        datetime.now(timezone.utc) - timedelta(minutes=5),
    )
    monkeypatch.setattr(P, "_fetch_yf", lambda *a, **k: pytest.fail("yf must not be called offline"))
    monkeypatch.setattr(
        P, "_fetch_stooq_csv", lambda *a, **k: pytest.fail("stooq must not be called offline")
    )
    result = fetch_latest(["VOO"], asof_date=asof, cache_dir=tmp_path, online=False)
    assert result.rows["VOO"].source == "cache"
    assert result.missing == []


def test_empty_input_returns_empty_result(tmp_path: Path) -> None:
    result = fetch_latest([], cache_dir=tmp_path)
    assert result.rows == {}
    assert result.missing == []


def test_dedup_preserves_first_occurrence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(P, "_fetch_yf", lambda t, s, e: _fake_yf_df(1.0, date(2024, 1, 5)))
    result = fetch_latest(["VOO", "VOO"], asof_date=date(2024, 1, 6), cache_dir=tmp_path)
    assert list(result.rows) == ["VOO"]


def test_price_row_is_immutable() -> None:
    row = PriceRow(
        ticker="VOO",
        asof_date=date(2024, 1, 5),
        close=100.0,
        source="yfinance",
        fetched_at=datetime.now(timezone.utc),
    )
    with pytest.raises(Exception):  # noqa: PT011 — dataclass frozen raises FrozenInstanceError
        row.close = 200.0  # type: ignore[misc]


# ── regression tests for review findings ───────────────────────────────────


def test_cache_refuses_future_fetched_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for review finding 3: a future-stamped cache entry must NOT be served as 'fresh'."""
    asof = date(2024, 1, 5)
    future = datetime.now(timezone.utc) + timedelta(hours=24)
    _write_cache_row(tmp_path / "VOO.parquet", asof, 999.0, future)
    monkeypatch.setattr(P, "_fetch_yf", lambda t, s, e: _fake_yf_df(50.0, asof))
    monkeypatch.setattr(P, "_fetch_stooq_csv", lambda t: None)
    result = fetch_latest(["VOO"], asof_date=asof, cache_dir=tmp_path)
    assert result.rows["VOO"].source == "yfinance"
    assert result.rows["VOO"].close == 50.0


def test_cache_age_clamped_to_zero() -> None:
    """Even if a tz-aware row's fetched_at is in the future, .cache_age must be ≥ 0."""
    future = datetime.now(timezone.utc) + timedelta(hours=12)
    row = PriceRow(
        ticker="VOO",
        asof_date=date(2024, 1, 5),
        close=100.0,
        source="cache",
        fetched_at=future,
    )
    assert row.cache_age == timedelta(0)


def test_prices_result_helpers() -> None:
    row_yf = PriceRow("A", date(2024, 1, 5), 1.0, "yfinance", datetime.now(timezone.utc))
    row_stooq = PriceRow("B", date(2024, 1, 5), 2.0, "stooq", datetime.now(timezone.utc))
    row_cache = PriceRow("C", date(2024, 1, 5), 3.0, "cache", datetime.now(timezone.utc))
    result = PricesResult(
        rows={"A": row_yf, "B": row_stooq, "C": row_cache},
        missing=["D"],
    )
    assert result.n_yfinance == 1
    assert result.n_stooq == 1
    assert result.n_cache == 1
    assert result.fallbacks_used == 1


# ── fetch_series (slice 3) ─────────────────────────────────────────────────


def _fake_yf_history(prices: list[float], start: date) -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp(start), periods=len(prices), freq="D")
    return pd.DataFrame({"Close": prices}, index=idx)


def test_fetch_series_returns_history_with_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    start = date(2024, 1, 1)
    monkeypatch.setattr(
        P, "_fetch_yf", lambda t, s, e: _fake_yf_history([100.0, 101.0, 102.0], start)
    )
    monkeypatch.setattr(P, "_fetch_stooq_csv", lambda t: None)
    result = fetch_series(["VOO"], start, date(2024, 1, 3), cache_dir=tmp_path)
    assert "VOO" in result.rows
    s = result.rows["VOO"]
    assert len(s) == 3
    assert float(s.iloc[0]) == 100.0
    assert isinstance(s.index, pd.DatetimeIndex)


def test_fetch_series_falls_back_to_stooq(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(P, "_fetch_yf", lambda t, s, e: None)
    csv_text = (
        "Date,Open,High,Low,Close,Volume\n"
        "2024-01-01,0,0,0,10.0,0\n"
        "2024-01-02,0,0,0,11.0,0\n"
        "2024-01-03,0,0,0,12.0,0\n"
    )
    monkeypatch.setattr(P, "_fetch_stooq_csv", lambda t: csv_text)
    result = fetch_series(["VOO"], date(2024, 1, 1), date(2024, 1, 3), cache_dir=tmp_path)
    assert list(result.rows["VOO"].round(1)) == [10.0, 11.0, 12.0]


def test_fetch_series_missing_when_all_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(P, "_fetch_yf", lambda t, s, e: None)
    monkeypatch.setattr(P, "_fetch_stooq_csv", lambda t: None)
    result = fetch_series(["NOPE"], date(2024, 1, 1), date(2024, 1, 3), cache_dir=tmp_path)
    assert result.rows == {}
    assert result.missing == ["NOPE"]


def test_fetch_series_writes_and_reuses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    start = date(2024, 1, 1)
    end = date(2024, 1, 3)
    monkeypatch.setattr(
        P, "_fetch_yf", lambda t, s, e: _fake_yf_history([100.0, 101.0, 102.0], start)
    )
    fetch_series(["VOO"], start, end, cache_dir=tmp_path)
    assert (tmp_path / "VOO_series.parquet").exists()
    # Second call must hit cache: network wrappers now fail the test if called.
    monkeypatch.setattr(P, "_fetch_yf", lambda *a, **k: pytest.fail("yf should not be called"))
    monkeypatch.setattr(P, "_fetch_stooq_csv", lambda *a, **k: pytest.fail("stooq should not be called"))
    result = fetch_series(["VOO"], start, end, cache_dir=tmp_path)
    assert len(result.rows["VOO"]) == 3
