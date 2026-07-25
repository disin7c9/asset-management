"""Tests for the price fetcher.

Offline by default: each test monkey-patches the two underlying network
wrappers (`_fetch_yf`, `_fetch_tiingo_json`) so no real HTTP happens.
"""

from __future__ import annotations

import logging
from dataclasses import FrozenInstanceError
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from app import prices as P
from app.prices import (
    PriceRow,
    PricesResult,
    SeriesResult,
    fetch_latest,
    fetch_series,
    fetch_splits,
)


@pytest.fixture(autouse=True)
def _hermetic_prices_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate the live secret and the once-per-process no-key warning.

    `TIINGO_API_KEY` is cleared so a developer's real `.env` (loaded by `cli.main` in
    another test) can never send this suite to api.tiingo.com — the same rule
    `test_email.py` applies to `RESEND_API_KEY`. The no-key warning latch is reset so
    tests can't leak the "already warned" state into each other. (The yfinance retry
    circuit is now batch-scoped, not a module global, so nothing to reset there.)
    """
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    monkeypatch.setattr(P, "_warned_no_tiingo_key", False)


def _fake_yf_df(price: float, on: date) -> pd.DataFrame:
    return pd.DataFrame({"Close": [price]}, index=pd.DatetimeIndex([pd.Timestamp(on)]))


def _fake_tiingo_rows(price: float, on: date) -> list[dict[str, object]]:
    return [{"date": f"{on.isoformat()}T00:00:00.000Z", "close": price}]


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
    monkeypatch.setattr(P, "_fetch_yf", lambda t, s, e, **k: _fake_yf_df(100.0, date(2024, 1, 5)))
    monkeypatch.setattr(P, "_fetch_tiingo_json", lambda t, s, e, **k: None)
    result = fetch_latest(["VOO"], asof_date=date(2024, 1, 6), cache_dir=tmp_path)
    assert isinstance(result, PricesResult)
    assert "VOO" in result.rows
    p = result.rows["VOO"]
    assert p.close == 100.0
    assert p.source == "yfinance"
    assert p.fetched_at.tzinfo is not None  # tz-aware


def test_latest_missing_close_column_degrades_not_crashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: a yfinance frame without a "Close" column (only "Adj Close", a
    # renamed/partial frame) must degrade the ticker to .missing — never raise
    # KeyError and kill the whole batch. The series path already guarded this via
    # _normalize_close; the latest path now shares that one extractor.
    def _yf(t, s, e, **k):  # type: ignore[no-untyped-def]
        if t == "BAD":
            return pd.DataFrame(
                {"Adj Close": [1.0]}, index=pd.DatetimeIndex([pd.Timestamp("2024-01-05")])
            )
        return _fake_yf_df(100.0, date(2024, 1, 5))

    monkeypatch.setattr(P, "_fetch_yf", _yf)
    monkeypatch.setattr(P, "_fetch_tiingo_json", lambda t, s, e, **k: None)
    result = fetch_latest(["GOOD", "BAD"], asof_date=date(2024, 1, 6), cache_dir=tmp_path)
    assert result.rows["GOOD"].close == 100.0  # the bad ticker didn't poison the batch
    assert result.missing == ["BAD"]


def test_fetch_writes_cache_after_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(P, "_fetch_yf", lambda t, s, e, **k: _fake_yf_df(123.45, date(2024, 1, 5)))
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
        P, "_fetch_tiingo_json", lambda *a, **k: pytest.fail("tiingo should not be called")
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
    monkeypatch.setattr(P, "_fetch_yf", lambda t, s, e, **k: _fake_yf_df(75.0, asof))
    result = fetch_latest(["VOO"], asof_date=asof, cache_dir=tmp_path)
    assert result.rows["VOO"].source == "yfinance"
    assert result.rows["VOO"].close == 75.0


# ── fallback chain ──────────────────────────────────────────────────────────


def test_fetch_falls_back_to_tiingo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asof = date(2024, 1, 5)
    monkeypatch.setattr(P, "_fetch_yf", lambda t, s, e, **k: None)
    monkeypatch.setattr(P, "_fetch_tiingo_json", lambda t, s, e, **k: _fake_tiingo_rows(88.0, asof))
    result = fetch_latest(["VOO"], asof_date=asof, cache_dir=tmp_path)
    assert result.rows["VOO"].source == "tiingo"
    assert result.rows["VOO"].close == 88.0
    assert result.fallbacks_used == 1


def test_fetch_returns_partial_on_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for review finding 2: one bad ticker must not abort the whole report."""
    asof = date(2024, 1, 5)
    monkeypatch.setattr(
        P,
        "_fetch_yf",
        lambda t, s, e, **k: _fake_yf_df(100.0, asof) if t == "VOO" else None,
    )
    monkeypatch.setattr(P, "_fetch_tiingo_json", lambda t, s, e, **k: None)
    result = fetch_latest(["VOO", "ZZZBAD"], asof_date=asof, cache_dir=tmp_path)
    assert "VOO" in result.rows
    assert "ZZZBAD" not in result.rows
    assert result.missing == ["ZZZBAD"]


def test_tiingo_malformed_rows_are_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Forgiving parse: rows missing a field, with a bad date, or a non-numeric
    # close are dropped — one good row still yields a price, never a crash.
    asof = date(2024, 1, 5)
    rows: list[dict[str, object]] = [
        {"close": 1.0},                                        # no date
        {"date": "not-a-date", "close": 2.0},                  # bad date
        {"date": f"{asof.isoformat()}T00:00:00.000Z", "close": "nope"},  # bad close
        {"date": f"{asof.isoformat()}T00:00:00.000Z", "close": 77.0},   # good
    ]
    monkeypatch.setattr(P, "_fetch_yf", lambda t, s, e, **k: None)
    monkeypatch.setattr(P, "_fetch_tiingo_json", lambda t, s, e, **k: rows)
    result = fetch_latest(["VOO"], asof_date=asof, cache_dir=tmp_path)
    assert result.rows["VOO"].close == 77.0
    assert result.rows["VOO"].source == "tiingo"


def test_tiingo_skipped_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # The REAL wrapper (not a mock): without TIINGO_API_KEY it must return None
    # before any HTTP is attempted — the no-key path is a silent skip, not an error.
    monkeypatch.setattr(
        P._TIINGO_OPENER, "open", lambda *a, **k: pytest.fail("no HTTP without a key")
    )
    assert P._fetch_tiingo_json("VOO", date(2024, 1, 1), date(2024, 1, 5)) is None


def test_tiingo_key_placeholder_residue_counts_as_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An MCPB host substitutes an optional user_config field the user left blank with the
    # LITERAL "${user_config.x}". That is not a credential: it must take the clean no-key
    # skip, not post a doomed `Authorization: Token ${...}` per ticker. (mcp_server._env_raw
    # applies the same rule to the path vars; this pins it for the secret.)
    monkeypatch.setenv("TIINGO_API_KEY", "${user_config.tiingo_api_key}")
    monkeypatch.setattr(
        P._TIINGO_OPENER, "open", lambda *a, **k: pytest.fail("residue must not be sent as a key")
    )
    assert P._fetch_tiingo_json("VOO", date(2024, 1, 1), date(2024, 1, 5)) is None


def test_no_tiingo_key_is_logged_exactly_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The reason the once-per-process latch exists: a cold fallback warm skips the key for
    # EVERY ticker, and one INFO per ticker would be dozens of lines of noise.
    monkeypatch.setattr(P._TIINGO_OPENER, "open", lambda *a, **k: pytest.fail("no HTTP"))
    with caplog.at_level(logging.INFO, logger="app.prices"):
        for _ in range(3):
            assert P._fetch_tiingo_json("VOO", date(2024, 1, 1), date(2024, 1, 5)) is None
    hits = [r for r in caplog.records if "TIINGO_API_KEY is not set" in r.getMessage()]
    assert len(hits) == 1


def test_every_credentialed_path_refuses_redirects_so_the_key_cannot_follow() -> None:
    # urllib re-sends every header on a 3xx and does NOT strip Authorization when the host
    # changes, so following a redirect would hand the user's API key to the target.
    from app.http_safe import NoRedirect

    req = P.urllib.request.Request("https://api.tiingo.com/x")
    assert NoRedirect().redirect_request(req, None, 302, "Found", {}, "https://evil.example") is None  # type: ignore[arg-type]
    # That the Tiingo path actually GOES through this opener is pinned behaviorally by the
    # tests below, which patch `_TIINGO_OPENER.open` and would never fire if the code used a
    # bare urlopen. The LLM path has its own end-to-end redirect test in test_llm.py.
    # (Asserting `isinstance` over `opener.handlers` was tried here and deleted: it passes
    # even when the caller ignores the opener, so it pinned nothing.)


def test_tiingo_ticker_is_url_quoted(monkeypatch: pytest.MonkeyPatch) -> None:
    # Tickers arrive from the book CSV / --screen with only .strip().upper(), so a stray
    # '?' or '/' must not be able to rewrite the request path or displace the date query.
    monkeypatch.setenv("TIINGO_API_KEY", "sekrit")
    seen: dict[str, str] = {}

    def fake_open(req, timeout=None):  # type: ignore[no-untyped-def]
        seen["url"] = req.full_url
        raise OSError("stop here")

    monkeypatch.setattr(P._TIINGO_OPENER, "open", fake_open)
    P._fetch_tiingo_json("A/B?x=1", date(2024, 1, 1), date(2024, 1, 5))
    assert "/tiingo/daily/a%2Fb%3Fx%3D1/prices" in seen["url"]
    assert seen["url"].count("?") == 1  # the injected '?' did not add a second query


# ── the price BASIS: Tiingo raw close → yfinance's split-adjusted basis ────────


def _tiingo_row(day: str, close: float, split: float = 1.0) -> dict[str, object]:
    return {"date": f"{day}T00:00:00.000Z", "close": close, "splitFactor": split}


def test_tiingo_rebuilds_yfinance_split_adjusted_basis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression lock for the v2.10.0 review's headline bug.

    Tiingo's `close` is RAW; yfinance's `Close` (auto_adjust=False) is SPLIT-ADJUSTED —
    the basis `corporate_actions.adjust_for_splits` assumes. Serving raw closes fabricated
    a ~-90% one-day return across a split (NVDA 10:1: $1208 -> $121). Every close before a
    split must be divided by that split's factor.
    """
    rows = [
        _tiingo_row("2024-06-07", 1208.88),              # pre-split, raw
        _tiingo_row("2024-06-10", 121.79, split=10.0),   # the 10:1 split lands here
        _tiingo_row("2024-06-11", 120.91),               # post-split
    ]
    monkeypatch.setattr(P, "_fetch_tiingo_json", lambda t, s, e, **k: rows)
    pairs = dict(P._tiingo_closes("NVDA", date(2024, 6, 7), date(2024, 6, 11)))
    assert pairs[date(2024, 6, 7)] == pytest.approx(120.888)  # 1208.88 / 10 — matches yfinance
    assert pairs[date(2024, 6, 10)] == pytest.approx(121.79)  # split day: already post-split
    assert pairs[date(2024, 6, 11)] == pytest.approx(120.91)
    # And the phantom crash is gone: no fabricated ~-90% day.
    closes = [pairs[d] for d in sorted(pairs)]
    worst = min(b / a - 1.0 for a, b in zip(closes, closes[1:], strict=False))
    assert worst > -0.5


def test_tiingo_multi_split_uses_the_product_of_all_later_factors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # TWO splits in one window: a close before both must be divided by the PRODUCT (2 * 3 = 6),
    # not just the next factor. A single-split test can't tell "product of all later" from "only
    # the next" — they coincide for one split — so this pins the accumulation across splits.
    rows = [
        _tiingo_row("2024-01-01", 600.0),             # before BOTH splits → ÷6
        _tiingo_row("2024-02-01", 300.0, split=2.0),  # 2:1 here → before the later 3:1 → ÷3
        _tiingo_row("2024-03-01", 100.0, split=3.0),  # 3:1 here → nothing later → ÷1
        _tiingo_row("2024-04-01", 110.0),             # after both → ÷1
    ]
    monkeypatch.setattr(P, "_fetch_tiingo_json", lambda t, s, e, **k: rows)
    pairs = dict(P._tiingo_closes("X", date(2024, 1, 1), date(2024, 4, 1)))
    assert pairs[date(2024, 1, 1)] == pytest.approx(100.0)  # 600 / (2*3)
    assert pairs[date(2024, 2, 1)] == pytest.approx(100.0)  # 300 / 3
    assert pairs[date(2024, 3, 1)] == pytest.approx(100.0)  # 100 / 1
    assert pairs[date(2024, 4, 1)] == pytest.approx(110.0)  # unadjusted


def test_tiingo_reverse_split_scales_earlier_closes_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A reverse split ships splitFactor < 1 (a 1:10 → 0.1). Dividing by 0.1 scales the pre-split
    # raw closes UP to the post-reverse basis — matching yfinance. Pins the factor<1 direction
    # (a "sanity" guard that rejected sub-1 factors would fabricate a +900% jump here).
    rows = [
        _tiingo_row("2024-01-01", 5.0),              # pre-reverse raw $5 → $50 adjusted
        _tiingo_row("2024-02-01", 52.0, split=0.1),  # 1:10 reverse lands here
        _tiingo_row("2024-03-01", 48.0),
    ]
    monkeypatch.setattr(P, "_fetch_tiingo_json", lambda t, s, e, **k: rows)
    pairs = dict(P._tiingo_closes("X", date(2024, 1, 1), date(2024, 3, 1)))
    assert pairs[date(2024, 1, 1)] == pytest.approx(50.0)  # 5.0 / 0.1
    assert pairs[date(2024, 2, 1)] == pytest.approx(52.0)
    assert pairs[date(2024, 3, 1)] == pytest.approx(48.0)


def test_tiingo_applies_a_split_that_happened_after_the_window_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # yfinance's adjusted Close reflects EVERY split up to today, including one after the
    # requested `end`. So the Tiingo fetch must reach through today and adjust, then slice —
    # else a backtest window ending before a split silently returns raw prices.
    today = date.today()
    seen: dict[str, date] = {}

    def fake_fetch(t: str, s: date, e: date) -> list[dict[str, object]]:
        seen["end"] = e
        return [
            _tiingo_row("2024-01-02", 1000.0),
            _tiingo_row("2024-01-03", 1010.0),
            _tiingo_row(today.isoformat(), 101.0, split=10.0),  # split AFTER the window
        ]

    monkeypatch.setattr(P, "_fetch_tiingo_json", fake_fetch)
    pairs = dict(P._tiingo_closes("X", date(2024, 1, 2), date(2024, 1, 3)))
    assert seen["end"] == today  # reached past `end` to see the later split
    assert set(pairs) == {date(2024, 1, 2), date(2024, 1, 3)}  # but returned only the window
    assert pairs[date(2024, 1, 2)] == pytest.approx(100.0)  # 1000 / 10
    assert pairs[date(2024, 1, 3)] == pytest.approx(101.0)  # 1010 / 10


def test_tiingo_missing_or_bad_split_factor_defaults_to_no_adjustment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows: list[dict[str, object]] = [
        {"date": "2024-01-02T00:00:00.000Z", "close": 10.0},                  # no splitFactor
        {"date": "2024-01-03T00:00:00.000Z", "close": 11.0, "splitFactor": 0},  # nonsense
        {"date": "2024-01-04T00:00:00.000Z", "close": 12.0, "splitFactor": "x"},
    ]
    monkeypatch.setattr(P, "_fetch_tiingo_json", lambda t, s, e, **k: rows)
    pairs = dict(P._tiingo_closes("X", date(2024, 1, 2), date(2024, 1, 4)))
    assert [pairs[d] for d in sorted(pairs)] == [10.0, 11.0, 12.0]  # untouched


# ── the parse contract: one bad row must never abort the batch ────────────────


def test_tiingo_non_dict_rows_degrade_to_missing_not_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A valid JSON list whose elements aren't dicts (an error page, a format shift) used to
    # raise AttributeError out of the un-guarded per-ticker loop and blank the whole report.
    monkeypatch.setattr(P, "_fetch_yf", lambda t, s, e, **k: None)
    monkeypatch.setattr(P, "_fetch_tiingo_json", lambda t, s, e, **k: [None, 123, "oops"])
    result = fetch_latest(["VOO"], asof_date=date(2024, 1, 5), cache_dir=tmp_path)
    assert result.missing == ["VOO"] and result.rows == {}


@pytest.mark.parametrize(
    ("close", "why"),
    [
        (float("nan"), "json.loads parses a bare NaN token"),
        (float("inf"), "float('inf') never raises"),
        ("inf", "the string form doesn't either"),
        (True, "bool is a subclass of int — would become $1.00"),
        (10**400, "float() of a huge int raises OverflowError, not ValueError"),
    ],
)
def test_tiingo_unusable_closes_are_skipped_never_priced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, close: object, why: str
) -> None:
    # The yfinance path drops these via _normalize_close's dropna; the Tiingo path must too.
    # A non-finite close would otherwise be cached and re-served as source="cache", poisoning
    # market value, weights, and every drawdown figure downstream.
    monkeypatch.setattr(P, "_fetch_yf", lambda t, s, e, **k: None)
    monkeypatch.setattr(
        P, "_fetch_tiingo_json",
        lambda t, s, e, **k: [{"date": "2024-01-05T00:00:00.000Z", "close": close}],
    )
    result = fetch_latest(["VOO"], asof_date=date(2024, 1, 5), cache_dir=tmp_path)
    assert result.missing == ["VOO"], why
    assert not (tmp_path / "VOO.parquet").exists()  # nothing unusable reached the cache


def test_tiingo_one_bad_row_does_not_drop_the_good_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows: list[object] = [
        {"date": "2024-01-02T00:00:00.000Z", "close": float("nan")},
        "not a dict",
        {"date": "bad-date", "close": 5.0},
        {"date": "2024-01-03T00:00:00.000Z", "close": 11.0},
    ]
    monkeypatch.setattr(P, "_fetch_tiingo_json", lambda t, s, e, **k: rows)
    pairs = P._tiingo_closes("X", date(2024, 1, 1), date(2024, 1, 3))
    assert pairs == [(date(2024, 1, 3), 11.0)]


# ── the one price-validity rule: finite AND > 0, at every ingest point ────────


def test_usable_price_rejects_the_full_bad_set() -> None:
    assert P.usable_price(10.0)
    for bad in (0.0, -1.0, float("nan"), float("inf"), float("-inf")):
        assert not P.usable_price(bad), bad


def test_tiingo_series_drops_zero_and_negative_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A halted / bad-feed row (close 0.0 or negative) must be dropped, not fed to the return
    # series where it fabricates a -100% then a sign-flip infinity into the drawdown/Ulcer maths.
    rows = [
        {"date": "2024-01-02T00:00:00.000Z", "close": 10.0},
        {"date": "2024-01-03T00:00:00.000Z", "close": 0.0},
        {"date": "2024-01-04T00:00:00.000Z", "close": -5.0},
        {"date": "2024-01-05T00:00:00.000Z", "close": 12.0},
    ]
    monkeypatch.setattr(P, "_fetch_tiingo_json", lambda t, s, e, **k: rows)
    pairs = P._tiingo_closes("X", date(2024, 1, 2), date(2024, 1, 5))
    assert pairs == [(date(2024, 1, 2), 10.0), (date(2024, 1, 5), 12.0)]


def test_normalize_close_drops_inf_and_non_positive() -> None:
    # dropna removes NaN but NOT ±inf, 0, or negatives — the yfinance path must drop them too,
    # else pipeline's "providers reject non-finite upstream" claim is false for yfinance itself.
    df = pd.DataFrame(
        {"Close": [10.0, 0.0, -3.0, float("inf"), float("nan"), 12.0]},
        index=pd.date_range("2024-01-02", periods=6),
    )
    out = P._normalize_close(df)
    assert out is not None
    assert list(out) == [10.0, 12.0]


def test_series_cache_read_drops_poisoned_cells(tmp_path: Path) -> None:
    # A corrupt / legacy <T>_series.parquet must not re-serve non-finite or non-positive closes
    # (they bypass the parser guards). This is the most-travelled ingest path on a warm cache.
    pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
        "close": [float("nan"), float("inf"), 0.0, 13.0],
        "fetched_at": pd.Timestamp(datetime.now(timezone.utc)),
    }).to_parquet(tmp_path / "VOO_series.parquet", index=False)
    res = fetch_series(["VOO"], date(2024, 1, 1), date(2024, 1, 5), cache_dir=tmp_path, online=False)
    assert list(res.rows["VOO"]) == [13.0]  # only the one usable cell survives


def test_from_cache_rejects_a_poisoned_latest_cell(tmp_path: Path) -> None:
    asof = date(2024, 1, 5)
    _write_cache_row(tmp_path / "VOO.parquet", asof, float("inf"),
                     datetime.now(timezone.utc) - timedelta(minutes=5))
    # inf > 0 is True, so a bare `> 0` guard would have served it; usable_price rejects it.
    res = fetch_latest(["VOO"], asof_date=asof, cache_dir=tmp_path, online=False)
    assert res.missing == ["VOO"] and res.rows == {}


# ── the yfinance retry circuit breaker (batch-scoped) ─────────────────────────


def test_yf_retry_disarms_after_consecutive_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    # Under a wholesale Yahoo block (the case the retry exists for), retrying EVERY ticker
    # only doubles request volume and burns 1.5s x N of dead sleep before the fallback.
    # After _YF_BLOCKED_AFTER straight failures ON ONE circuit the retry must disarm.
    calls = {"n": 0}
    sleeps: list[float] = []

    def throttled(*a: object, **k: object) -> pd.DataFrame:
        calls["n"] += 1
        return pd.DataFrame()  # Yahoo's throttle signature: empty, not raising

    monkeypatch.setattr(P.yf, "download", throttled)
    monkeypatch.setattr(P.time, "sleep", lambda s: sleeps.append(s))

    circuit = P._RetryCircuit()
    for _ in range(P._YF_BLOCKED_AFTER):  # each: 2 attempts + 1 sleep
        assert P._fetch_yf_retrying("X", date(2024, 1, 1), date(2024, 1, 2), circuit=circuit) is None
    armed_calls, armed_sleeps = calls["n"], len(sleeps)
    assert armed_calls == 2 * P._YF_BLOCKED_AFTER
    assert armed_sleeps == P._YF_BLOCKED_AFTER
    assert not circuit.armed

    assert P._fetch_yf_retrying("Y", date(2024, 1, 1), date(2024, 1, 2), circuit=circuit) is None
    assert calls["n"] == armed_calls + 1  # a single attempt
    assert len(sleeps) == armed_sleeps    # and no sleep at all


def test_yf_success_rearms_the_circuit(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(P.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(P.yf, "download", lambda *a, **k: pd.DataFrame())
    circuit = P._RetryCircuit()
    for _ in range(P._YF_BLOCKED_AFTER):
        P._fetch_yf_retrying("X", date(2024, 1, 1), date(2024, 1, 2), circuit=circuit)
    assert circuit.consecutive_failures == P._YF_BLOCKED_AFTER

    monkeypatch.setattr(P.yf, "download", lambda *a, **k: _fake_yf_df(1.0, date(2024, 1, 5)))
    assert P._fetch_yf_retrying("X", date(2024, 1, 1), date(2024, 1, 2), circuit=circuit) is not None
    assert circuit.consecutive_failures == 0  # one success re-arms the circuit

    monkeypatch.setattr(P.yf, "download", lambda *a, **k: pd.DataFrame())
    before = len(sleeps)
    P._fetch_yf_retrying("X", date(2024, 1, 1), date(2024, 1, 2), circuit=circuit)
    assert len(sleeps) == before + 1  # retrying again


def test_yf_circuit_is_batch_scoped_not_process_global(monkeypatch: pytest.MonkeyPatch) -> None:
    # A fresh batch (a later, unrelated MCP tool call) re-assesses from zero — a prior
    # blocked batch must not leave the retry disabled for the next one.
    monkeypatch.setattr(P.time, "sleep", lambda s: None)
    monkeypatch.setattr(P.yf, "download", lambda *a, **k: pd.DataFrame())
    blocked = P._RetryCircuit()
    for _ in range(P._YF_BLOCKED_AFTER):
        P._fetch_yf_retrying("X", date(2024, 1, 1), date(2024, 1, 2), circuit=blocked)
    assert not blocked.armed

    fresh_batch = P._RetryCircuit()  # what fetch_latest/fetch_series build per call
    assert fresh_batch.armed  # starts armed regardless of the prior batch

    # And a lone call with no circuit is always armed (independent).
    calls = {"n": 0}

    def count(*a: object, **k: object) -> pd.DataFrame:
        calls["n"] += 1
        return pd.DataFrame()

    monkeypatch.setattr(P.yf, "download", count)
    P._fetch_yf_retrying("Z", date(2024, 1, 1), date(2024, 1, 2))  # circuit=None
    assert calls["n"] == 2  # retried


def test_fetch_yf_retries_once_on_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    # A throttled Yahoo call comes back EMPTY, not raising; one spaced retry
    # must clear the transient case (the real Desktop failure mode).
    calls = {"n": 0}

    def fake_download(*a: object, **k: object) -> pd.DataFrame:
        calls["n"] += 1
        if calls["n"] == 1:
            return pd.DataFrame()  # throttled → empty frame
        return _fake_yf_df(100.0, date(2024, 1, 5))

    sleeps: list[float] = []
    monkeypatch.setattr(P.yf, "download", fake_download)
    monkeypatch.setattr(P.time, "sleep", lambda s: sleeps.append(s))
    df = P._fetch_yf_retrying("VOO", date(2024, 1, 1), date(2024, 1, 6))
    assert df is not None and not df.empty
    assert calls["n"] == 2
    assert sleeps == [P._YF_RETRY_WAIT]  # paced, not hammered


def test_fetch_yf_gives_up_after_two_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    # A hard block stays empty on both attempts → None (falls through to Tiingo);
    # never a third call, never an unbounded loop.
    calls = {"n": 0}

    def fake_download(*a: object, **k: object) -> pd.DataFrame:
        calls["n"] += 1
        return pd.DataFrame()

    monkeypatch.setattr(P.yf, "download", fake_download)
    monkeypatch.setattr(P.time, "sleep", lambda s: None)
    assert P._fetch_yf_retrying("VOO", date(2024, 1, 1), date(2024, 1, 6)) is None
    assert calls["n"] == 2


def test_fetch_yf_no_retry_when_first_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    # The happy path must stay a single request — no added latency per ticker.
    calls = {"n": 0}

    def fake_download(*a: object, **k: object) -> pd.DataFrame:
        calls["n"] += 1
        return _fake_yf_df(100.0, date(2024, 1, 5))

    monkeypatch.setattr(P.yf, "download", fake_download)
    monkeypatch.setattr(P.time, "sleep", lambda s: pytest.fail("must not sleep on success"))
    assert P._fetch_yf_retrying("VOO", date(2024, 1, 1), date(2024, 1, 6)) is not None
    assert calls["n"] == 1


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
        P, "_fetch_tiingo_json", lambda *a, **k: pytest.fail("tiingo must not be called offline")
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
    monkeypatch.setattr(P, "_fetch_yf", lambda t, s, e, **k: _fake_yf_df(1.0, date(2024, 1, 5)))
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
    with pytest.raises(FrozenInstanceError):
        row.close = 200.0  # type: ignore[misc]


# ── regression tests for review findings ───────────────────────────────────


def test_cache_refuses_future_fetched_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for review finding 3: a future-stamped cache entry must NOT be served as 'fresh'."""
    asof = date(2024, 1, 5)
    future = datetime.now(timezone.utc) + timedelta(hours=24)
    _write_cache_row(tmp_path / "VOO.parquet", asof, 999.0, future)
    monkeypatch.setattr(P, "_fetch_yf", lambda t, s, e, **k: _fake_yf_df(50.0, asof))
    monkeypatch.setattr(P, "_fetch_tiingo_json", lambda t, s, e, **k: None)
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
    row_tiingo = PriceRow("B", date(2024, 1, 5), 2.0, "tiingo", datetime.now(timezone.utc))
    row_cache = PriceRow("C", date(2024, 1, 5), 3.0, "cache", datetime.now(timezone.utc))
    result = PricesResult(
        rows={"A": row_yf, "B": row_tiingo, "C": row_cache},
        missing=["D"],
    )
    assert result.n_yfinance == 1
    assert result.n_tiingo == 1
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
        P, "_fetch_yf", lambda t, s, e, **k: _fake_yf_history([100.0, 101.0, 102.0], start)
    )
    monkeypatch.setattr(P, "_fetch_tiingo_json", lambda t, s, e, **k: None)
    result = fetch_series(["VOO"], start, date(2024, 1, 3), cache_dir=tmp_path)
    assert "VOO" in result.rows
    s = result.rows["VOO"]
    assert len(s) == 3
    assert float(s.iloc[0]) == 100.0
    assert isinstance(s.index, pd.DatetimeIndex)


def test_fetch_series_falls_back_to_tiingo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(P, "_fetch_yf", lambda t, s, e, **k: None)
    rows: list[dict[str, object]] = [
        {"date": "2024-01-01T00:00:00.000Z", "close": 10.0},
        {"date": "2024-01-02T00:00:00.000Z", "close": 11.0},
        {"date": "2024-01-03T00:00:00.000Z", "close": 12.0},
    ]
    monkeypatch.setattr(P, "_fetch_tiingo_json", lambda t, s, e, **k: rows)
    result = fetch_series(["VOO"], date(2024, 1, 1), date(2024, 1, 3), cache_dir=tmp_path)
    assert list(result.rows["VOO"].round(1)) == [10.0, 11.0, 12.0]
    assert result.provenance["VOO"][0] == "tiingo"
    assert result.fallbacks_used == 1


def test_fetch_series_missing_when_all_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(P, "_fetch_yf", lambda t, s, e, **k: None)
    monkeypatch.setattr(P, "_fetch_tiingo_json", lambda t, s, e, **k: None)
    result = fetch_series(["NOPE"], date(2024, 1, 1), date(2024, 1, 3), cache_dir=tmp_path)
    assert result.rows == {}
    assert result.missing == ["NOPE"]


def test_fetch_series_writes_and_reuses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    start = date(2024, 1, 1)
    end = date(2024, 1, 3)
    monkeypatch.setattr(
        P, "_fetch_yf", lambda t, s, e, **k: _fake_yf_history([100.0, 101.0, 102.0], start)
    )
    fetch_series(["VOO"], start, end, cache_dir=tmp_path)
    assert (tmp_path / "VOO_series.parquet").exists()
    # Second call must hit cache: network wrappers now fail the test if called.
    monkeypatch.setattr(P, "_fetch_yf", lambda *a, **k: pytest.fail("yf should not be called"))
    monkeypatch.setattr(P, "_fetch_tiingo_json", lambda *a, **k: pytest.fail("tiingo should not be called"))
    result = fetch_series(["VOO"], start, end, cache_dir=tmp_path)
    assert len(result.rows["VOO"]) == 3


def test_fetch_series_records_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A yfinance hit must be recorded as ("yfinance", <fetched_at>) so the CLI can
    # stamp honest provenance (P0-1) instead of a fabricated "series" label.
    monkeypatch.setattr(P, "_fetch_yf", lambda tk, s, e, **k: _fake_yf_df(100.0, date(2024, 1, 2)))
    monkeypatch.setattr(P, "_fetch_tiingo_json", lambda tk, s, e, **k: None)
    res = fetch_series(["VOO"], date(2024, 1, 1), date(2024, 1, 3),
                       cache_dir=tmp_path, online=True)
    assert "VOO" in res.rows
    assert res.provenance["VOO"][0] == "yfinance"


def test_series_fallbacks_used_counts_tiingo() -> None:
    now = datetime.now(timezone.utc)
    res = SeriesResult(
        rows={"A": pd.Series(dtype=float), "B": pd.Series(dtype=float)},
        provenance={"A": ("tiingo", now), "B": ("yfinance", now)},
    )
    assert res.fallbacks_used == 1


# ── series-cache resilience (mirror the latest-cache hardening) ──────────────


def _write_series_cache(
    path: Path, dates: list[str], closes: list[float], fetched_at: datetime
) -> None:
    pd.DataFrame(
        {
            "date": [pd.Timestamp(d) for d in dates],
            "close": closes,
            "fetched_at": pd.Timestamp(fetched_at),
        }
    ).to_parquet(path, index=False)


_DAYS = ["2024-01-01", "2024-01-02", "2024-01-03"]


def test_series_cache_stale_is_refetched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_series_cache(
        tmp_path / "VOO_series.parquet", _DAYS, [1.0, 2.0, 3.0],
        datetime.now(timezone.utc) - timedelta(days=3),  # stale
    )
    monkeypatch.setattr(P, "_fetch_yf", lambda t, s, e, **k: _fake_yf_history([10.0, 11.0, 12.0], date(2024, 1, 1)))
    monkeypatch.setattr(P, "_fetch_tiingo_json", lambda t, s, e, **k: None)
    res = fetch_series(["VOO"], date(2024, 1, 1), date(2024, 1, 3), cache_dir=tmp_path)
    assert list(res.rows["VOO"].round(1)) == [10.0, 11.0, 12.0]  # refetched, not 1/2/3
    assert res.provenance["VOO"][0] == "yfinance"


def test_series_cache_future_stamp_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_series_cache(
        tmp_path / "VOO_series.parquet", _DAYS, [1.0, 2.0, 3.0],
        datetime.now(timezone.utc) + timedelta(hours=24),  # future-stamped
    )
    monkeypatch.setattr(P, "_fetch_yf", lambda t, s, e, **k: _fake_yf_history([10.0, 11.0, 12.0], date(2024, 1, 1)))
    monkeypatch.setattr(P, "_fetch_tiingo_json", lambda t, s, e, **k: None)
    res = fetch_series(["VOO"], date(2024, 1, 1), date(2024, 1, 3), cache_dir=tmp_path)
    assert list(res.rows["VOO"].round(1)) == [10.0, 11.0, 12.0]  # refused → refetched


def test_series_cache_corrupt_is_nonfatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "VOO_series.parquet").write_bytes(b"this is not parquet")
    monkeypatch.setattr(P, "_fetch_yf", lambda t, s, e, **k: _fake_yf_history([5.0, 6.0, 7.0], date(2024, 1, 1)))
    monkeypatch.setattr(P, "_fetch_tiingo_json", lambda t, s, e, **k: None)
    res = fetch_series(["VOO"], date(2024, 1, 1), date(2024, 1, 3), cache_dir=tmp_path)
    assert list(res.rows["VOO"].round(1)) == [5.0, 6.0, 7.0]  # unreadable → refetched


# ── latest-from-series-cache fallback (P1#4: unify the two caches) ───────────


def test_latest_falls_back_to_series_cache_offline(tmp_path: Path) -> None:
    # A risk-on run writes only <T>_series.parquet; --offline fetch_latest must
    # still serve the latest close from that series tail (not report it missing).
    _write_series_cache(
        tmp_path / "VOO_series.parquet",
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        [10.0, 11.0, 12.0, 13.0],
        datetime.now(timezone.utc) - timedelta(hours=1),  # fresh
    )
    res = fetch_latest(["VOO"], asof_date=date(2024, 1, 5), cache_dir=tmp_path, online=False)
    assert res.rows["VOO"].source == "cache"
    assert res.rows["VOO"].close == 13.0  # last close ≤ asof
    assert res.missing == []


def test_latest_series_fallback_respects_asof(tmp_path: Path) -> None:
    # The fallback returns the last close at/before asof, not the very last row.
    _write_series_cache(
        tmp_path / "VOO_series.parquet",
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        [10.0, 11.0, 12.0, 13.0],
        datetime.now(timezone.utc) - timedelta(hours=1),
    )
    res = fetch_latest(["VOO"], asof_date=date(2024, 1, 3), cache_dir=tmp_path, online=False)
    assert res.rows["VOO"].close == 11.0


def test_latest_stale_series_cache_served_offline(tmp_path: Path) -> None:
    # Offline, a series cache older than the TTL is still served: the TTL exists to
    # trigger a re-fetch, and offline there is nothing to fetch — the newest cached
    # close (honestly dated) beats reporting the holding missing. (A cache warmed one
    # evening is already >20h old by the next evening; that must not blank the book.)
    _write_series_cache(
        tmp_path / "VOO_series.parquet", _DAYS, [1.0, 2.0, 3.0],
        datetime.now(timezone.utc) - timedelta(days=3),  # stale, but offline serves it
    )
    res = fetch_latest(["VOO"], asof_date=date(2024, 1, 3), cache_dir=tmp_path, online=False)
    assert res.rows["VOO"].source == "cache"
    assert res.rows["VOO"].close == 3.0  # last close ≤ asof
    assert res.missing == []


def test_series_cache_stale_served_offline(tmp_path: Path) -> None:
    # Same age-tolerance for the price *history* (risk/backtest): offline, fetch_series
    # serves a stale-but-covering cache instead of reporting it missing.
    _write_series_cache(
        tmp_path / "VOO_series.parquet", _DAYS, [1.0, 2.0, 3.0],
        datetime.now(timezone.utc) - timedelta(days=3),  # stale
    )
    res = fetch_series(
        ["VOO"], date(2024, 1, 1), date(2024, 1, 3), cache_dir=tmp_path, online=False
    )
    assert res.missing == []
    assert list(res.rows["VOO"].round(1)) == [1.0, 2.0, 3.0]
    assert res.provenance["VOO"][0] == "cache"


# ── fresh-but-short series cache (F6a, fresh-eyes audit 2026-07-11) ──────────


def test_series_cache_fresh_but_short_is_served_not_refetched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A delisted/halted ticker's series can never cover `end` (its data simply stops), so
    # the coverage check sent it back to the network on EVERY run, forever. A cache fetched
    # fresh-within-TTL that still falls short means the provider had no newer rows — serve
    # it; the pipeline's staleness floor decides whether the tail counts as a current price.
    today = date.today()
    dead_days = [(today - timedelta(days=n)).isoformat() for n in (32, 31, 30)]
    _write_series_cache(
        tmp_path / "ATVI_series.parquet", dead_days, [93.0, 94.0, 94.42],
        datetime.now(timezone.utc) - timedelta(hours=1),  # FRESH fetch, short data
    )
    monkeypatch.setattr(P, "_fetch_yf", lambda *a, **k: pytest.fail("yf must not be re-asked"))
    monkeypatch.setattr(
        P, "_fetch_tiingo_json", lambda *a, **k: pytest.fail("tiingo must not be re-asked")
    )
    res = fetch_series(["ATVI"], today - timedelta(days=40), today, cache_dir=tmp_path)
    assert res.missing == []
    assert res.provenance["ATVI"][0] == "cache"
    assert float(res.rows["ATVI"].iloc[-1]) == 94.42


def test_series_cache_stale_and_short_is_refetched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The freshness carve-out is TTL-bounded: once the short cache ages past the TTL the
    # provider IS re-asked (≈ one fetch per day), catching a resumed/late-filled feed.
    today = date.today()
    dead_days = [(today - timedelta(days=n)).isoformat() for n in (32, 31, 30)]
    _write_series_cache(
        tmp_path / "ATVI_series.parquet", dead_days, [93.0, 94.0, 94.42],
        datetime.now(timezone.utc) - timedelta(days=3),  # STALE fetch → re-ask
    )
    monkeypatch.setattr(
        P, "_fetch_yf",
        lambda t, s, e, **k: _fake_yf_history([10.0, 11.0, 12.0], today - timedelta(days=2)),
    )
    monkeypatch.setattr(P, "_fetch_tiingo_json", lambda t, s, e, **k: None)
    res = fetch_series(["ATVI"], today - timedelta(days=40), today, cache_dir=tmp_path)
    assert res.provenance["ATVI"][0] == "yfinance"  # went live, not the old cache


def test_series_cache_fresh_but_entirely_before_start_is_refetched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Guard on the carve-out: a fresh-but-short cache whose data ends BEFORE the requested
    # window must not serve an empty slice — that's a cache miss, go live.
    today = date.today()
    old_days = [(today - timedelta(days=n)).isoformat() for n in (60, 59, 58)]
    _write_series_cache(
        tmp_path / "VOO_series.parquet", old_days, [1.0, 2.0, 3.0],
        datetime.now(timezone.utc) - timedelta(hours=1),  # fresh, but out of window
    )
    monkeypatch.setattr(
        P, "_fetch_yf",
        lambda t, s, e, **k: _fake_yf_history([10.0, 11.0, 12.0], today - timedelta(days=2)),
    )
    monkeypatch.setattr(P, "_fetch_tiingo_json", lambda t, s, e, **k: None)
    res = fetch_series(["VOO"], today - timedelta(days=40), today, cache_dir=tmp_path)
    assert res.provenance["VOO"][0] == "yfinance"
    assert list(res.rows["VOO"].round(1)) == [10.0, 11.0, 12.0]


def test_latest_series_future_stamp_still_refused_offline(tmp_path: Path) -> None:
    # Age-tolerance must NOT weaken the future-stamp guard: a future fetched_at is
    # corruption / clock-skew, not mere staleness, so it is rejected even offline.
    _write_series_cache(
        tmp_path / "VOO_series.parquet", _DAYS, [1.0, 2.0, 3.0],
        datetime.now(timezone.utc) + timedelta(hours=24),  # future-stamped → refused
    )
    res = fetch_latest(["VOO"], asof_date=date(2024, 1, 3), cache_dir=tmp_path, online=False)
    assert res.missing == ["VOO"]


def test_latest_cache_still_preferred_over_series(tmp_path: Path) -> None:
    # When both caches are fresh, the dedicated latest cache wins (exact asof match).
    asof = date(2024, 1, 3)
    _write_cache_row(tmp_path / "VOO.parquet", asof, 99.0, datetime.now(timezone.utc) - timedelta(minutes=5))
    _write_series_cache(
        tmp_path / "VOO_series.parquet", _DAYS, [1.0, 2.0, 3.0],
        datetime.now(timezone.utc) - timedelta(minutes=5),
    )
    res = fetch_latest(["VOO"], asof_date=asof, cache_dir=tmp_path, online=False)
    assert res.rows["VOO"].close == 99.0  # from VOO.parquet, not the series tail (3.0)


def test_online_latest_miss_does_not_serve_stale_series_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Review #2: ONLINE, the series-cache fallback must not shadow the network.
    # A fresh-within-TTL series cache dated days before asof would otherwise be
    # served as "latest" instead of fetching the current price.
    _write_series_cache(
        tmp_path / "VOO_series.parquet", _DAYS, [10.0, 11.0, 12.0],
        datetime.now(timezone.utc) - timedelta(hours=1),  # fresh, but dated 2024-01-03
    )  # no VOO.parquet → dedicated latest cache misses
    monkeypatch.setattr(P, "_fetch_yf", lambda t, s, e, **k: _fake_yf_df(99.0, date(2024, 1, 10)))
    monkeypatch.setattr(P, "_fetch_tiingo_json", lambda t, s, e, **k: None)
    res = fetch_latest(["VOO"], asof_date=date(2024, 1, 10), cache_dir=tmp_path, online=True)
    assert res.rows["VOO"].source == "yfinance"  # fetched fresh, NOT the stale tail
    assert res.rows["VOO"].close == 99.0


def test_series_cache_nat_fetched_at_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Review #4: an all-NaT fetched_at must not pass the freshness gate (NaT
    # comparisons are False → it would otherwise be "fresh forever").
    pd.DataFrame(
        {"date": [pd.Timestamp("2024-01-01")], "close": [5.0], "fetched_at": [pd.NaT]}
    ).to_parquet(tmp_path / "VOO_series.parquet", index=False)
    monkeypatch.setattr(P, "_fetch_yf", lambda t, s, e, **k: _fake_yf_history([10.0, 11.0, 12.0], date(2024, 1, 1)))
    monkeypatch.setattr(P, "_fetch_tiingo_json", lambda t, s, e, **k: None)
    res = fetch_series(["VOO"], date(2024, 1, 1), date(2024, 1, 3), cache_dir=tmp_path)
    assert list(res.rows["VOO"].round(1)) == [10.0, 11.0, 12.0]  # refused → refetched
    assert res.provenance["VOO"][0] == "yfinance"


def test_latest_nat_series_cache_not_served_offline(tmp_path: Path) -> None:
    # The NaT guard also protects the latest-from-series fallback under --offline.
    pd.DataFrame(
        {"date": [pd.Timestamp("2024-01-01")], "close": [5.0], "fetched_at": [pd.NaT]}
    ).to_parquet(tmp_path / "VOO_series.parquet", index=False)
    res = fetch_latest(["VOO"], asof_date=date(2024, 1, 3), cache_dir=tmp_path, online=False)
    assert res.missing == ["VOO"]


def test_latest_cache_future_stamp_still_refused_offline(tmp_path: Path) -> None:
    # Offline age-tolerance (allow_stale) must NOT weaken the future-stamp guard on the DEDICATED
    # latest cache either — a future/clock-skewed row is corruption, not staleness (T1). (The
    # existing series-cache future-stamp test covers _latest_from_series_cache; this covers
    # _from_cache, the function the allow_stale wiring changed.)
    asof = date(2024, 1, 3)
    _write_cache_row(
        tmp_path / "VOO.parquet", asof, 99.0,
        datetime.now(timezone.utc) + timedelta(hours=24),  # future-stamped → refused
    )
    res = fetch_latest(["VOO"], asof_date=asof, cache_dir=tmp_path, online=False)
    assert res.missing == ["VOO"]


def test_latest_series_too_old_not_served_offline(tmp_path: Path) -> None:
    # Offline serves a stale cache, but NOT a series tail older than the floor (P2): don't pass
    # off a months-old close as a "current" price. fetched_at is fresh; the DATA is old.
    _write_series_cache(
        tmp_path / "VOO_series.parquet", _DAYS, [1.0, 2.0, 3.0],
        datetime.now(timezone.utc) - timedelta(hours=1),  # freshly fetched, but tail is 2024-01-03
    )
    res = fetch_latest(["VOO"], asof_date=date(2024, 6, 1), cache_dir=tmp_path, online=False)
    assert res.missing == ["VOO"]  # tail 2024-01-03 is ~5 months before asof → beyond the floor


# ── fetch_splits (slice 7: corporate actions) ──────────────────────────────


def _fake_splits(pairs: list[tuple[str, float]]) -> "pd.Series[float]":
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d, _ in pairs])
    return pd.Series([r for _, r in pairs], index=idx, dtype=float)


def test_fetch_splits_parses_and_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(P, "_fetch_yf_splits", lambda tk: _fake_splits([("2024-06-10", 10.0)]))
    res = fetch_splits(["NVDA"], cache_dir=tmp_path)
    assert res["NVDA"] == [(date(2024, 6, 10), 10.0)]
    assert (tmp_path / "NVDA_splits.parquet").exists()
    # Second call must hit the cache (network wrapper fails the test if called).
    monkeypatch.setattr(P, "_fetch_yf_splits", lambda tk: pytest.fail("should not refetch"))
    assert fetch_splits(["NVDA"], cache_dir=tmp_path)["NVDA"] == [(date(2024, 6, 10), 10.0)]


def test_fetch_splits_no_splits_cached_as_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(P, "_fetch_yf_splits", lambda tk: None)  # no split history
    assert fetch_splits(["VOO"], cache_dir=tmp_path)["VOO"] == []
    # The "no splits" fact is cached (placeholder) → offline returns [] without refetch.
    assert fetch_splits(["VOO"], cache_dir=tmp_path, online=False)["VOO"] == []


def test_fetch_splits_offline_no_cache_is_empty(tmp_path: Path) -> None:
    # Unknown offline → no adjustment (the price-basis-mismatch guard is the net).
    assert fetch_splits(["XYZ"], cache_dir=tmp_path, online=False) == {"XYZ": []}


def _write_splits_row(path: Path, when: str, ratio: float, fetched_at: datetime) -> None:
    pd.DataFrame({
        "date": [pd.Timestamp(when)], "ratio": [ratio],
        "fetched_at": [pd.Timestamp(fetched_at)],
    }).to_parquet(path, index=False)


def test_fetch_splits_stale_cache_served_offline(tmp_path: Path) -> None:
    # Offline, a >7d-stale splits cache is SERVED, not dropped — splits are stable facts. Else a
    # split holding would be valued with pre-split shares at a post-split price (P1): the price
    # path now serves stale offline, so the split path must too, or the two decouple.
    _write_splits_row(
        tmp_path / "NVDA_splits.parquet", "2024-06-10", 10.0,
        datetime.now(timezone.utc) - timedelta(days=30),  # well past the 7d splits TTL
    )
    assert fetch_splits(["NVDA"], cache_dir=tmp_path, online=False)["NVDA"] == [(date(2024, 6, 10), 10.0)]


def test_fetch_splits_stale_cache_refetched_online(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Online, the SAME >7d cache is refetched (the TTL still triggers a refresh, catching a new
    # split) — the age-tolerance is offline-only, the online contract is unchanged.
    _write_splits_row(
        tmp_path / "NVDA_splits.parquet", "2024-06-10", 10.0,
        datetime.now(timezone.utc) - timedelta(days=30),
    )
    monkeypatch.setattr(
        P, "_fetch_yf_splits",
        lambda tk: _fake_splits([("2024-06-10", 10.0), ("2025-01-02", 4.0)]),
    )
    assert fetch_splits(["NVDA"], cache_dir=tmp_path)["NVDA"] == [
        (date(2024, 6, 10), 10.0), (date(2025, 1, 2), 4.0),
    ]


def test_fetch_splits_drops_noise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A 1:1 ratio is a no-op and must be dropped; a real reverse split is kept.
    monkeypatch.setattr(
        P, "_fetch_yf_splits",
        lambda tk: _fake_splits([("2023-01-01", 1.0), ("2024-01-01", 0.1)]),
    )
    assert fetch_splits(["RV"], cache_dir=tmp_path)["RV"] == [(date(2024, 1, 1), 0.1)]


def test_fetch_splits_corrupt_cache_is_nonfatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A cache file with fetched_at but missing ratio/date must refetch, not KeyError
    # (mirrors the series-cache corrupt test; fetch_splits documents "never raises").
    pd.DataFrame(
        {"fetched_at": [pd.Timestamp(datetime.now(timezone.utc))]}
    ).to_parquet(tmp_path / "FOO_splits.parquet", index=False)
    monkeypatch.setattr(P, "_fetch_yf_splits", lambda tk: _fake_splits([("2024-06-10", 10.0)]))
    assert fetch_splits(["FOO"], cache_dir=tmp_path)["FOO"] == [(date(2024, 6, 10), 10.0)]


def test_fetch_splits_skips_nat_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A NaT split date must be dropped, not crash sorted()/the date comparison
    # (pd.Timestamp(NaT).date() returns NaT without raising).
    s = pd.Series([5.0, 2.0], index=pd.DatetimeIndex([pd.NaT, pd.Timestamp("2024-01-01")]))
    monkeypatch.setattr(P, "_fetch_yf_splits", lambda tk: s)
    assert fetch_splits(["X"], cache_dir=tmp_path)["X"] == [(date(2024, 1, 1), 2.0)]


# ── the return basis: raw (portfolio) vs total-return (notional simulation) ────


def test_total_return_basis_asks_yfinance_for_adjusted_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The whole point of the basis: auto_adjust must reach yfinance. A price-only close
    # books every coupon as a loss, which is how BIL (return = 100% coupon) simulated at ~0.
    seen: list[bool] = []

    def _yf(t: str, s: date, e: date, **k: object) -> pd.DataFrame:
        seen.append(bool(k.get("adjusted")))
        return _fake_yf_df(100.0, date(2024, 1, 2))

    monkeypatch.setattr(P, "_fetch_yf", _yf)
    monkeypatch.setattr(P, "_fetch_tiingo_json", lambda t, s, e, **k: None)
    fetch_series(["VOO"], date(2024, 1, 1), date(2024, 1, 3),
                 cache_dir=tmp_path, online=True, basis="total_return")
    fetch_series(["BND"], date(2024, 1, 1), date(2024, 1, 3),
                 cache_dir=tmp_path, online=True)  # default = raw
    assert seen == [True, False]


def test_the_two_bases_never_share_a_cache_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A raw close and a dividend-adjusted close for the same day are different numbers.
    # If one file served both, warming either would silently corrupt the other.
    def _yf(t: str, s: date, e: date, **k: object) -> pd.DataFrame:
        return _fake_yf_df(120.0 if k.get("adjusted") else 100.0, date(2024, 1, 2))

    monkeypatch.setattr(P, "_fetch_yf", _yf)
    monkeypatch.setattr(P, "_fetch_tiingo_json", lambda t, s, e, **k: None)
    raw = fetch_series(["BND"], date(2024, 1, 1), date(2024, 1, 3), cache_dir=tmp_path)
    tr = fetch_series(["BND"], date(2024, 1, 1), date(2024, 1, 3),
                      cache_dir=tmp_path, basis="total_return")
    assert float(raw.rows["BND"].iloc[0]) == 100.0
    assert float(tr.rows["BND"].iloc[0]) == 120.0
    assert (tmp_path / "BND_series.parquet").exists()
    assert (tmp_path / "BND_series_tr.parquet").exists()

    # ...and each basis reads back its OWN file offline, not the other's.
    raw2 = fetch_series(["BND"], date(2024, 1, 1), date(2024, 1, 3),
                        cache_dir=tmp_path, online=False)
    tr2 = fetch_series(["BND"], date(2024, 1, 1), date(2024, 1, 3),
                       cache_dir=tmp_path, online=False, basis="total_return")
    assert float(raw2.rows["BND"].iloc[0]) == 100.0
    assert float(tr2.rows["BND"].iloc[0]) == 120.0


def test_a_warm_raw_cache_does_not_satisfy_a_total_return_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Offline with only the raw cache warmed, the simulation basis must report the ticker
    # MISSING rather than quietly serve price-only closes into a backtest verdict.
    monkeypatch.setattr(P, "_fetch_yf", lambda t, s, e, **k: _fake_yf_df(100.0, date(2024, 1, 2)))
    monkeypatch.setattr(P, "_fetch_tiingo_json", lambda t, s, e, **k: None)
    fetch_series(["BND"], date(2024, 1, 1), date(2024, 1, 3), cache_dir=tmp_path)
    out = fetch_series(["BND"], date(2024, 1, 1), date(2024, 1, 3),
                       cache_dir=tmp_path, online=False, basis="total_return")
    assert out.missing == ["BND"]
    assert not out.rows


def test_tiingo_total_return_uses_adjclose_not_the_split_reconstruction() -> None:
    # Tiingo ships adjClose already split- AND dividend-adjusted — exactly the TR basis —
    # so the splitFactor rebuild (right for the raw basis) must be skipped, not applied twice.
    rows: list[dict[str, object]] = [
        {"date": "2024-01-02T00:00:00.000Z", "close": 100.0, "adjClose": 90.0, "splitFactor": 1.0},
        {"date": "2024-01-03T00:00:00.000Z", "close": 50.0, "adjClose": 45.5, "splitFactor": 2.0},
    ]
    raw = P._parse_tiingo_rows(rows)
    tr = P._parse_tiingo_rows(rows, total_return=True)
    assert [c for _, c in raw] == [50.0, 50.0]        # 100 / the later 2:1 split
    assert [c for _, c in tr] == [90.0, 45.5]         # adjClose served verbatim


def test_the_latest_cache_hits_when_asof_is_not_itself_a_trading_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: reads matched `asof_date == requested`, while the writer stores the
    provider's ACTUAL close date. The README's flagship workflow is an 08:00 Monday cron —
    before the US open — so yfinance hands back Friday's bar and the read missed on every
    run, forever: fresh network call each time, one duplicate parquet row per run, and a 20h
    TTL that never applied."""
    calls: list[str] = []

    def _yf(t: str, s: date, e: date, **k: object) -> pd.DataFrame:
        calls.append(t)
        return _fake_yf_df(100.0, date(2026, 7, 17))       # Friday's close

    monkeypatch.setattr(P, "_fetch_yf", _yf)
    monkeypatch.setattr(P, "_fetch_tiingo_json", lambda t, s, e, **k: None)
    monday = date(2026, 7, 20)
    rows = [
        P.fetch_latest(["VOO"], monday, cache_dir=tmp_path, online=True).rows["VOO"]
        for _ in range(3)
    ]
    assert len(calls) == 1                                  # was 3
    assert [r.source for r in rows] == ["yfinance", "cache", "cache"]
    assert len(pd.read_parquet(tmp_path / "VOO.parquet")) == 1   # was 3 duplicate rows
    # The row must carry the close's OWN date — labelling Friday's price as Monday's would
    # hide the lag from the report's stale-close display.
    assert all(r.asof_date == date(2026, 7, 17) for r in rows)


def test_the_latest_cache_still_refuses_a_close_past_the_staleness_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Matching on `<= asof` must not turn a delisted ticker's last-ever close into a
    permanent 'current' price — the staleness floor still applies to the cached row."""
    monkeypatch.setattr(P, "_fetch_yf", lambda t, s, e, **k: _fake_yf_df(50.0, date(2026, 1, 5)))
    monkeypatch.setattr(P, "_fetch_tiingo_json", lambda t, s, e, **k: None)
    P.fetch_latest(["DEAD"], date(2026, 1, 6), cache_dir=tmp_path, online=True)
    assert P._from_cache("DEAD", date(2026, 6, 1), tmp_path, allow_stale=True) is None


def test_a_small_clock_step_does_not_invalidate_every_cache() -> None:
    # WSL/VM resume and NTP both step the clock; a file written just before a step is
    # legitimately stamped ahead. With zero tolerance that one step failed all four caches
    # at once and the tool blamed a cold cache — the wrong diagnosis, and the user re-fetches
    # everything. Minutes of grace are physical; hours are a tampered file and still refused.
    from app.prices import _CLOCK_SKEW_GRACE, _now_utc, fresh

    now = _now_utc()
    ttl = timedelta(hours=20)
    assert fresh(now + timedelta(minutes=1), ttl, what="x")          # inside the grace
    assert fresh(now + _CLOCK_SKEW_GRACE - timedelta(seconds=5), ttl, what="x")
    assert not fresh(now + _CLOCK_SKEW_GRACE + timedelta(minutes=1), ttl, what="x")
    assert not fresh(now + timedelta(days=1), ttl, what="x")         # nonsense, still refused
