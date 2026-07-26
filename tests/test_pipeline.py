"""Pipeline cache-warming helpers: `cache_is_cold`, `benchmark_ref_tickers`, `warm_cache`,
plus `held_market_value` — the one valuation sink — and the staleness floor on
`compute_prices_returns_risk`'s series-tail pricing.

The warm path is offline-first onboarding glue over the named fetch adapters, so these
tests mock the adapters and assert which tickers each is asked for — never the network.
"""

from __future__ import annotations

import math
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from app.derive import DerivedState, Position, derive
from app.events import Event
from app.metadata import MetadataResult, SecurityMeta
from app.pipeline import (
    _WARM_MARKER,
    DEMO_BOOK_CSV,
    benchmark_ref_tickers,
    cache_is_cold,
    candidate_and_held_facts,
    compute_prices_returns_risk,
    default_cache_dir,
    held_market_value,
    warm_cache,
    write_demo_book,
)
from app.prices import PriceRow, PricesResult, SeriesResult


@pytest.mark.parametrize(
    ("close", "usable"),
    [
        (10.0, True),
        (0.0, False),
        (-1.0, False),
        (float("nan"), False),
        (float("inf"), False),   # `inf > 0` is True — the guard must test isfinite too
        (float("-inf"), False),
    ],
)
def test_held_market_value_drops_every_unusable_close(close: float, usable: bool) -> None:
    """The valuation sink must not depend on providers rejecting bad prices upstream.

    `nan > 0` is False so NaN was already dropped, but `inf > 0` is True: an inf close
    made market value inf and every weight nan, while the book still scored fully priced.
    """
    state = DerivedState(positions={"VOO": Position("VOO", shares=2.0, cost_basis=100.0)})
    prices = {"VOO": PriceRow("VOO", date(2024, 1, 5), close, "tiingo", datetime.now(timezone.utc))}
    out = held_market_value(state, prices)
    assert ("VOO" in out) is usable
    if usable:
        assert out["VOO"] == pytest.approx(20.0)


# ── the staleness floor on series-tail pricing (F1, fresh-eyes audit 2026-07-11) ──


def _wavy_series(base: float, *, tail_age_days: int, n: int = 300) -> "pd.Series[float]":
    """A daily close series whose LAST row is ~tail_age_days before today. The gentle
    oscillation keeps drawdowns/ratios finite; values stay within ±~12% of base so a
    buy priced off the series can't trip the 2× basis-mismatch guard."""
    end = pd.Timestamp.today().normalize() - pd.Timedelta(days=tail_age_days)
    idx = pd.bdate_range(end=end, periods=n)
    vals = [base * (1.0 + 0.03 * math.sin(i / 4.0) + 0.0003 * i) for i in range(n)]
    return pd.Series(vals, index=idx)


def _buy(s: "pd.Series[float]", ticker: str, i: int = 10) -> Event:
    """A buy executed AT the series' own close on its i-th day — basis-consistent."""
    return Event(
        date=s.index[i].date(), ticker=ticker, action="buy",
        quantity=10.0, price=float(s.iloc[i]), fee=0.0,
    )


def _run_pipeline(
    series_by_ticker: dict[str, "pd.Series[float]"],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[dict[str, object], object]:
    """compute_prices_returns_risk over mocked fetch_series → (run dict, result tuple)."""
    now = datetime.now(timezone.utc)
    result = SeriesResult(
        rows=dict(series_by_ticker),
        missing=[],
        provenance={tk: ("cache", now) for tk in series_by_ticker},
    )
    monkeypatch.setattr("app.pipeline.fetch_series", lambda *a, **k: result)
    events = [_buy(s, tk) for tk, s in sorted(series_by_ticker.items())]
    run: dict[str, object] = {
        "n_series_fetched": 0, "n_series_missing": 0, "fallbacks_used": 0, "status": "ok",
    }
    out = compute_prices_returns_risk(
        events, derive(events), no_risk=False, offline=True,
        cache_dir=tmp_path, today=date.today(), run=run,
    )
    return run, out


def test_stale_series_tail_is_refused_as_a_current_price(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # THE audit P0 (F1): a delisted/halted ticker's series ends at its last-ever close, and
    # the series-tail path minted a "current" PriceRow from it with no age check — ATVI was
    # valued at its 2023 close, 33 months on, with status "ok". Past the 10-day floor the
    # holding must flow into the partial-pricing machinery instead: unpriced, status
    # "partial", felt-dollar drawdown suppressed, money-weighted returns suppressed —
    # AND excluded from TWR & risk (review finding #1: value_curve forward-fills the last
    # close across the gap, so leaving the dead series in the curve would let the risk
    # panel quietly value the position at the very quote the pricing loop refuses).
    run, (prices, returns, risk, missing, twr_excluded, dollar_dd, _series, daily) = (
        _run_pipeline(
            {"LIVE": _wavy_series(100.0, tail_age_days=0),
             "DEAD": _wavy_series(90.0, tail_age_days=30)},
            monkeypatch, tmp_path,
        )
    )
    assert prices is not None and "LIVE" in prices and "DEAD" not in prices
    assert missing == ["DEAD"]
    assert twr_excluded == ["DEAD"]     # out of the risk/TWR curve, not flat-carried in it
    assert risk is not None             # the LIVE-only panel still computes
    assert run["status"] == "partial"
    assert dollar_dd is None            # incomplete P&L curve → n/a, not a confident number
    assert returns is not None and returns.money_weighted_annualized is None


def test_sold_position_with_a_dead_history_stays_in_twr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The stale-tail exclusion is for HELD positions only: a ticker sold before its data
    # ended contributes zero shares (hence zero value) after the sale, and its held-period
    # history is real history — excluding it would silently drop a genuine past position
    # from the time-weighted record.
    live = _wavy_series(100.0, tail_age_days=0)
    dead = _wavy_series(90.0, tail_age_days=30)
    now = datetime.now(timezone.utc)
    result = SeriesResult(
        rows={"LIVE": live, "DEAD": dead},
        missing=[],
        provenance={tk: ("cache", now) for tk in ("LIVE", "DEAD")},
    )
    monkeypatch.setattr("app.pipeline.fetch_series", lambda *a, **k: result)
    events = [
        _buy(live, "LIVE"),
        _buy(dead, "DEAD", i=10),
        Event(date=dead.index[50].date(), ticker="DEAD", action="sell",
              quantity=10.0, price=float(dead.iloc[50]), fee=0.0),  # fully closed
    ]
    run: dict[str, object] = {
        "n_series_fetched": 0, "n_series_missing": 0, "fallbacks_used": 0, "status": "ok",
    }
    prices, _returns, risk, missing, twr_excluded, dollar_dd, _series, _daily = (
        compute_prices_returns_risk(
            events, derive(events), no_risk=False, offline=True,
            cache_dir=tmp_path, today=date.today(), run=run,
        )
    )
    assert twr_excluded == []           # sold → exempt from the stale-tail exclusion
    assert missing == [] and run["status"] == "ok"
    assert prices is not None and set(prices) == {"LIVE"}  # DEAD isn't held, needs no price
    assert risk is not None and dollar_dd is not None


def test_fresh_series_tail_still_prices_normally(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Sibling lock: a tail a business-day-or-three old (any normal weekend/holiday gap)
    # is well inside the floor — the F1 refusal must not degrade the everyday case.
    run, (prices, returns, _risk, missing, _twr_excl, dollar_dd, _series, _daily) = (
        _run_pipeline(
            {"AAA": _wavy_series(100.0, tail_age_days=0),
             "BBB": _wavy_series(50.0, tail_age_days=0)},
            monkeypatch, tmp_path,
        )
    )
    assert prices is not None and set(prices) == {"AAA", "BBB"}
    assert missing == []
    assert run["status"] == "ok"
    assert dollar_dd is not None
    assert returns is not None and returns.money_weighted_annualized is not None


def test_left_truncated_series_is_excluded_not_a_phantom_crash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # F3 end-to-end: TRUNC's history starts ~100 business days ago but its buy is ~a year
    # old (a provider serving a shorter history than the holding). Before the left-edge
    # guard, the buy's cash preceded any TRUNC value in the curve → a fabricated ~−100%
    # day into max-DD/Ulcer/TWR. Now the ticker gets the split-mismatch treatment:
    # excluded from the time-weighted series (and dollar-DD suppressed), while it stays
    # PRICED — its tail is fresh, so holdings/market value are unaffected.
    live = _wavy_series(100.0, tail_age_days=0)          # 300 bdays, covers its buy
    trunc = _wavy_series(90.0, tail_age_days=0, n=100)   # starts AFTER the buy below
    now = datetime.now(timezone.utc)
    result = SeriesResult(
        rows={"LIVE": live, "TRUNC": trunc},
        missing=[],
        provenance={tk: ("cache", now) for tk in ("LIVE", "TRUNC")},
    )
    monkeypatch.setattr("app.pipeline.fetch_series", lambda *a, **k: result)
    events = [
        _buy(live, "LIVE"),
        Event(date=live.index[10].date(), ticker="TRUNC", action="buy",
              quantity=10.0, price=90.0, fee=0.0),  # predates trunc.index[0]
    ]
    run: dict[str, object] = {
        "n_series_fetched": 0, "n_series_missing": 0, "fallbacks_used": 0, "status": "ok",
    }
    prices, _returns, risk, missing, twr_excluded, dollar_dd, _series, daily = (
        compute_prices_returns_risk(
            events, derive(events), no_risk=False, offline=True,
            cache_dir=tmp_path, today=date.today(), run=run,
        )
    )
    assert twr_excluded == ["TRUNC"]
    assert prices is not None and set(prices) == {"LIVE", "TRUNC"}  # still valued
    assert missing == [] and run["status"] == "ok"
    assert daily is not None and float(daily.min()) > -0.5  # no fabricated crash day
    assert risk is not None
    assert dollar_dd is None  # P&L curve omits TRUNC → suppressed, not confidently wrong


def test_benchmark_ref_tickers_is_the_union_of_the_references() -> None:
    # The 8 tickers a benchmark validation needs priced (60-40 / all-weather / permanent).
    assert set(benchmark_ref_tickers()) == {
        "VOO", "BND", "VTI", "TLT", "IEI", "GLD", "DBC", "BIL"
    }


def test_cache_is_cold_keys_on_the_warm_marker(tmp_path: Path) -> None:
    # Marker-based, NOT per-ticker existence — so an unfetchable book ticker (no series file
    # ever written) can't keep the cache 'cold' forever and re-warm online on every call.
    assert cache_is_cold(tmp_path) is True                    # no marker → never warmed → cold
    (tmp_path / _WARM_MARKER).touch()
    assert cache_is_cold(tmp_path) is False                   # fresh marker → warm
    stale = time.time() - 7 * 3600                            # older than the 6h TTL
    os.utime(tmp_path / _WARM_MARKER, (stale, stale))
    assert cache_is_cold(tmp_path) is True                    # stale marker → re-warm (self-heal)


def test_warm_cache_set_composition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # core = book ∪ refs for price history; latest/splits = book only (held need a spot price);
    # metadata = book ∪ extras (refs are price-only). 'full' layers the universe in via extras.
    seen: dict[str, set[str]] = {}

    def rec_series(tickers: object, start: object, end: object, **k: object) -> SeriesResult:
        # warm_cache makes TWO passes: the raw basis over everything, then a total-return
        # pass over the simulation set only (book ∪ refs — never the universe extras).
        key = "series_tr" if k.get("basis") == "total_return" else "series"
        seen[key] = set(tickers)  # type: ignore[arg-type]
        return SeriesResult()

    def rec_latest(tickers: object, **k: object) -> PricesResult:
        seen["latest"] = set(tickers)  # type: ignore[arg-type]
        return PricesResult()

    def rec_splits(tickers: object, **k: object) -> dict[str, list[object]]:
        seen["splits"] = set(tickers)  # type: ignore[arg-type]
        return {}

    def rec_meta(tickers: object, **k: object) -> MetadataResult:
        seen["meta"] = set(tickers)  # type: ignore[arg-type]
        return MetadataResult()

    monkeypatch.setattr("app.pipeline.fetch_series", rec_series)
    monkeypatch.setattr("app.pipeline.fetch_latest", rec_latest)
    monkeypatch.setattr("app.pipeline.fetch_splits", rec_splits)
    monkeypatch.setattr("app.pipeline.fetch_metadata", rec_meta)

    counts = warm_cache(["VOO", "FOO", "CASH"], tmp_path, extra_tickers=["QQQM"])
    refs = set(benchmark_ref_tickers())
    assert seen["series"] == {"VOO", "FOO", "QQQM"} | refs   # CASH dropped
    # the simulation basis skips the universe extras: discovery never simulates, so
    # `--warm full` pays for one ~375-ticker round, not two
    assert seen["series_tr"] == {"VOO", "FOO"} | refs
    assert seen["latest"] == {"VOO", "FOO"}                   # book only, no refs
    assert seen["splits"] == {"VOO", "FOO"}
    assert seen["meta"] == {"VOO", "FOO", "QQQM"}             # book ∪ extra, refs excluded
    assert counts["tickers"] == len({"VOO", "FOO", "QQQM"} | refs)
    assert counts["book_total"] == 2 and counts["book_missing"] == 0  # VOO/FOO (CASH dropped), none missing
    assert (tmp_path / _WARM_MARKER).exists()                 # stamps the marker → cache no longer cold
    assert cache_is_cold(tmp_path) is False


def test_demo_book_is_byte_identical_to_the_committed_sample() -> None:
    # --demo ships the book as a package constant (works for a no-checkout install);
    # data/sample_data/transactions.csv is the browsable repo copy. Pin them so the
    # README's example file and the demo can never drift apart.
    sample = Path(__file__).resolve().parents[1] / "data" / "sample_data" / "transactions.csv"
    assert DEMO_BOOK_CSV == sample.read_text(encoding="utf-8")


def test_write_demo_book_materializes_into_the_cache_dir(tmp_path: Path) -> None:
    path = write_demo_book(tmp_path / "fresh")  # the dir need not pre-exist
    assert path == tmp_path / "fresh" / "demo_book.csv"
    assert path.read_text(encoding="utf-8") == DEMO_BOOK_CSV


def test_default_cache_dir_checkout_vs_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A repo checkout (.git present) keeps the historical data/prices; an INSTALLED
    # root (site-packages under uvx, or the unpacked .mcpb dir) must NOT be written
    # into — it's ephemeral (uv cache clean / extension updates) — so the default
    # falls back to a stable per-user dir. The marker is .git, NOT pyproject.toml:
    # the .mcpb bundle ships pyproject.toml and must still route to the user dir.
    checkout = tmp_path / "repo"
    (checkout / ".git").mkdir(parents=True)
    assert default_cache_dir(checkout) == checkout / "data" / "prices"

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for installed_name in ("site-packages", "bundle-with-pyproject"):
        installed = tmp_path / installed_name
        installed.mkdir()
        (installed / "pyproject.toml").touch()  # the bundle case ships pyproject.toml
        assert default_cache_dir(installed) == (
            tmp_path / "home" / ".asset-management" / "prices"
        )


def _meta(tk: str) -> SecurityMeta:
    return SecurityMeta(
        ticker=tk, expense_ratio=None, aum=None, avg_volume=None, category=None,
        family=None, legal_type=None, quote_type=None, inception=None,
    )


def test_candidate_and_held_facts_online_split_and_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    # The shared screen helper (cli._screen_tickers + mcp.screen_candidate): candidate facts are
    # always fetched (gated by online_candidate); held facts reuse a prior MetadataResult when
    # given, else are fetched (gated by online_held). That split is what lets the MCP path fetch
    # ONLY the candidate online while the warmed held set stays cache-only — so exercise it here
    # directly, not just through the CLI/MCP tests.
    calls: list[tuple[tuple[str, ...], object]] = []

    def rec(tickers: object, **k: object) -> MetadataResult:
        tk_tuple = tuple(sorted(tickers))  # type: ignore[arg-type]
        calls.append((tk_tuple, k.get("online")))
        return MetadataResult(rows={t: _meta(t) for t in tk_tuple})

    monkeypatch.setattr("app.pipeline.fetch_metadata", rec)

    # No held_meta → candidate AND held are fetched, each carrying its own online flag.
    cand, held, missing = candidate_and_held_facts(
        ["CCC"], {"AAA", "BBB"}, None, online_candidate=True, online_held=False,
    )
    assert (("CCC",), True) in calls             # candidate fetched online (on demand)
    assert (("AAA", "BBB"), False) in calls       # held fetched offline — the split
    assert set(cand) == {"CCC"} and set(held) == {"AAA", "BBB"} and missing == []

    # held_meta supplied → held is REUSED, not re-fetched (only the candidate hits fetch_metadata).
    calls.clear()
    prior = MetadataResult(rows={"AAA": _meta("AAA"), "BBB": _meta("BBB")})
    _cand2, held2, _missing2 = candidate_and_held_facts(
        ["CCC"], {"AAA", "BBB"}, None, online_candidate=True, online_held=True, held_meta=prior,
    )
    assert calls == [(("CCC",), True)]            # held NOT re-fetched, only the candidate
    assert set(held2) == {"AAA", "BBB"}


def test_a_close_with_no_provenance_is_left_unpriced_not_stamped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    # Provenance is the field that lets a reader distrust a price. The old default invented
    # one — ("cache", now()) — rendering a close of unknown origin as freshly cached and
    # zero hours old: maximum confidence exactly where the code knows least. Refuse instead,
    # which routes the ticker into the machinery that already exists for missing prices.
    import logging

    from app.pipeline import compute_prices_returns_risk
    from app.prices import PricesResult, SeriesResult

    dates = pd.bdate_range("2024-01-01", pd.Timestamp.today().normalize())
    s = pd.Series([100.0] * len(dates), index=dates, dtype=float)
    monkeypatch.setattr(
        "app.pipeline.fetch_series",
        lambda tickers, *a, **k: SeriesResult(
            rows={tk: s for tk in tickers}, missing=[], provenance={}   # rows, no stamps
        ),
    )
    monkeypatch.setattr(
        "app.pipeline.fetch_latest", lambda tickers, *a, **k: PricesResult(missing=list(tickers))
    )
    monkeypatch.setattr("app.pipeline.fetch_splits", lambda tickers, *a, **k: {})

    events = [
        Event(date(2024, 1, 2), "VOO", "buy", quantity=10.0, price=100.0, fee=1.0),
    ]
    state = derive(events)
    with caplog.at_level(logging.WARNING):
        run: dict[str, object] = {
            "status": "ok", "n_prices_fetched": 0, "n_prices_missing": 0,
            "n_series_fetched": 0, "n_series_missing": 0, "fallbacks_used": 0,
        }
        prices = compute_prices_returns_risk(
            events, state, no_risk=False, offline=True, cache_dir=tmp_path,
            today=date.today(), run=run,
        )[0]

    assert prices == {}, "a price with no provenance was published anyway"
    assert "no provenance" in caplog.text


def _falling(depth: float, n: int = 400) -> "pd.Series[float]":
    """A series that rises then falls to `depth` (a negative fraction)."""
    idx = pd.bdate_range("2024-01-01", periods=n)
    peak = [100.0] * (n // 2)
    fall = [100.0 * (1.0 + depth)] * (n - n // 2)
    return pd.Series(peak + fall, index=idx, dtype=float)


def test_deepest_held_picks_the_worst_faller_and_ignores_everything_else() -> None:
    # The peer bar for the own-drawdown screen. It had NO behavioural test: gutting the body
    # to `return None` silently reverted every own-drawdown check to the old blended-book bar
    # and 225 tests still passed, because the only test was a source-text grep.
    from app.pipeline import deepest_held

    rows = {
        "AAA": _falling(-0.10),
        "BBB": _falling(-0.32),   # the deepest of the HELD set
        "CCC": _falling(-0.55),   # deeper still, but NOT held → must be ignored
    }
    res = SeriesResult(rows=rows, missing=[])

    got = deepest_held(res, {"AAA", "BBB"})
    assert got is not None
    assert got[0] == "BBB" and abs(got[1] - (-0.32)) < 0.01

    # A held ticker with no series is skipped, not crashed on.
    assert deepest_held(res, {"AAA", "BBB", "NOSERIES"})[0] == "BBB"

    # Nothing held, or nothing that ever fell → no peer, and the caller falls back to the
    # blended-book wording rather than inventing a bar.
    assert deepest_held(res, set()) is None
    flat = SeriesResult(rows={"FLAT": pd.Series([100.0] * 300,
                                                index=pd.bdate_range("2024-01-01", periods=300),
                                                dtype=float)}, missing=[])
    assert deepest_held(flat, {"FLAT"}) is None
    assert deepest_held(None, {"AAA"}) is None
