"""Pipeline cache-warming helpers: `cache_is_cold`, `benchmark_ref_tickers`, `warm_cache`.

The warm path is offline-first onboarding glue over the named fetch adapters, so these
tests mock the adapters and assert which tickers each is asked for — never the network.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from app.metadata import MetadataResult, SecurityMeta
from app.pipeline import (
    _WARM_MARKER,
    benchmark_ref_tickers,
    cache_is_cold,
    candidate_and_held_facts,
    warm_cache,
)
from app.prices import PricesResult, SeriesResult


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
        seen["series"] = set(tickers)  # type: ignore[arg-type]
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
    assert seen["latest"] == {"VOO", "FOO"}                   # book only, no refs
    assert seen["splits"] == {"VOO", "FOO"}
    assert seen["meta"] == {"VOO", "FOO", "QQQM"}             # book ∪ extra, refs excluded
    assert counts["tickers"] == len({"VOO", "FOO", "QQQM"} | refs)
    assert counts["book_total"] == 2 and counts["book_missing"] == 0  # VOO/FOO (CASH dropped), none missing
    assert (tmp_path / _WARM_MARKER).exists()                 # stamps the marker → cache no longer cold
    assert cache_is_cold(tmp_path) is False


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
