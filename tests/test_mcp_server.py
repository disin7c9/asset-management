"""Tests for the read-only MCP server.

Offline + hermetic: a temp transaction CSV plus a warmed price cache (no network),
driven through the SDK's in-memory client session (no subprocess). Verifies the
tools are read-only, return the validated core's numbers as structured content, and
degrade to clean MCP errors rather than crashing.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import anyio
import pandas as pd
import pytest
from mcp.shared.memory import create_connected_server_and_client_session as client_session
from mcp.types import (
    CallToolResult,
    GetPromptResult,
    ListPromptsResult,
    ListResourcesResult,
    ListToolsResult,
    ReadResourceResult,
)
from pydantic import AnyUrl

from app import prices as prices_mod
from app.mcp_server import _VERSION, mcp
from app.pipeline import _WARM_MARKER


def _warm_series_cache(cache: Path, ticker: str, base: float, n: int = 320) -> None:
    """Write a fresh <T>_series.parquet ending ~today so an OFFLINE read succeeds.

    A gentle oscillation + drift gives real drawdowns and finite risk ratios (a
    monotonic series would make Calmar non-finite). ``fetched_at`` is backdated a
    minute (still fresh within TTL) so a sub-second backward clock jiggle (WSL/CI)
    can't read the just-written cache as 'in the future' and refuse it."""
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    vals = [base * (1.0 + 0.03 * math.sin(i / 4.0) + 0.0003 * i) for i in range(n)]
    df = pd.DataFrame(
        {
            "date": idx,
            "close": vals,
            "fetched_at": pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=1),
        }
    )
    df.to_parquet(cache / f"{ticker}_series.parquet", index=False)


@pytest.fixture(autouse=True)
def _hermetic_mcp_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # The server reads ASSET_BOOK / ASSET_CSV / ASSET_TARGET / ASSET_MCP_OFFLINE from
    # os.environ. Pin book vars to "" (treated as unset) so a developer's real .env can't
    # leak into the suite — the original 14-failure class. ASSET_MCP_OFFLINE is pinned to
    # "1" so the suite stays hermetic (the cold-cache auto-warm — ON in production — would
    # otherwise reach the network); the auto-warm tests opt back in with setenv("").
    monkeypatch.setenv("ASSET_BOOK", "")
    monkeypatch.setenv("ASSET_CSV", "")
    monkeypatch.setenv("ASSET_TARGET", "")
    monkeypatch.setenv("ASSET_UNIVERSE", "")  # propose_allocation/discover read it → bundled default
    monkeypatch.setenv("ASSET_MCP_OFFLINE", "1")


@pytest.fixture
def warm_book(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    csv = tmp_path / "txn.csv"
    csv.write_text(
        "Date,Code,Action,Quantity,Price,Fee\n"
        "2024-01-02,AAA,buy,10,100,0\n"
        "2024-01-02,BBB,buy,5,50,0\n",
        encoding="utf-8",
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    _warm_series_cache(cache, "AAA", 100.0)
    _warm_series_cache(cache, "BBB", 50.0)
    (cache / _WARM_MARKER).touch()  # a genuinely-warmed cache → cache_is_cold False
    monkeypatch.delenv("ASSET_CSV", raising=False)
    monkeypatch.setenv("ASSET_BOOK", str(csv))
    monkeypatch.setenv("ASSET_CACHE_DIR", str(cache))
    monkeypatch.delenv("ASSET_TARGET", raising=False)
    return tmp_path


def _call(name: str, arguments: dict[str, Any] | None = None) -> CallToolResult:
    async def _go() -> CallToolResult:
        async with client_session(mcp) as client:
            return await client.call_tool(name, arguments or {})

    return anyio.run(_go)


def _list_tools() -> ListToolsResult:
    async def _go() -> ListToolsResult:
        async with client_session(mcp) as client:
            return await client.list_tools()

    return anyio.run(_go)


def _error_text(res: CallToolResult) -> str:
    return " ".join(getattr(c, "text", "") for c in res.content)


def _get_prompt(name: str, arguments: dict[str, str] | None = None) -> GetPromptResult:
    async def _go() -> GetPromptResult:
        async with client_session(mcp) as client:
            return await client.get_prompt(name, arguments)

    return anyio.run(_go)


def _list_prompts() -> ListPromptsResult:
    async def _go() -> ListPromptsResult:
        async with client_session(mcp) as client:
            return await client.list_prompts()

    return anyio.run(_go)


def _list_resources() -> ListResourcesResult:
    async def _go() -> ListResourcesResult:
        async with client_session(mcp) as client:
            return await client.list_resources()

    return anyio.run(_go)


def _read_resource(uri: str) -> ReadResourceResult:
    async def _go() -> ReadResourceResult:
        async with client_session(mcp) as client:
            return await client.read_resource(AnyUrl(uri))

    return anyio.run(_go)


def test_prompts_cover_the_tool_surface() -> None:
    # The chat front door: 5 conversation starters in the client's "+" menu, each with a
    # description a non-finance user can pick from (blank-box problem).
    res = _list_prompts()
    names = {p.name for p in res.prompts}
    assert names == {
        "portfolio_checkup",
        "whats_my_drawdown",
        "should_i_rebalance",
        "fill_my_gaps",
        "propose_a_posture",
    }
    assert all(p.description for p in res.prompts)


def test_every_prompt_carries_the_figures_rule() -> None:
    # Each starter injects the honesty framing into the FIRST user turn: tool figures
    # only, unavailable ≠ estimate, describe-don't-advise. One funnel (_FIGURES_RULE),
    # so assert it survived into every rendered prompt, args included.
    cases: list[tuple[str, dict[str, str] | None]] = [
        ("portfolio_checkup", None),
        ("whats_my_drawdown", None),
        ("should_i_rebalance", None),
        ("fill_my_gaps", None),
        ("propose_a_posture", {"posture": "aggressive"}),
    ]
    for name, args in cases:
        res = _get_prompt(name, args)
        text = getattr(res.messages[0].content, "text", "")
        assert "do not estimate" in text, name
        assert "don't advise" in text, name
    # The parameterized starter threads its argument through (checked on its OWN render,
    # not a loop-leaked variable) — and validates it: an off-menu posture must error at
    # prompt time, not open the conversation on a guaranteed tool failure.
    posture_text = getattr(
        _get_prompt("propose_a_posture", {"posture": "aggressive"}).messages[0].content,
        "text", "",
    )
    assert "'aggressive'" in posture_text
    with pytest.raises(BaseException) as ei:
        _get_prompt("propose_a_posture", {"posture": "balanced"})
    err: BaseException = ei.value
    while isinstance(err, BaseExceptionGroup):  # anyio wraps the McpError in a TaskGroup
        err = err.exceptions[0]
    assert "conservative" in str(err)  # the error names the menu


def test_env_path_treats_template_residue_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    # An MCPB host may substitute an UNSET optional user_config as the literal
    # "${user_config.x}" string. That must route to the "not set" guidance, never to
    # "points at a missing file: …/${user_config.x}".
    from app.mcp_server import _env_path

    monkeypatch.setenv("ASSET_TARGET", "${user_config.target}")
    with pytest.raises(ValueError, match="is not set"):
        _env_path("ASSET_TARGET", "point it at your target CSV")


def test_guarantees_resource_is_the_trust_manifest() -> None:
    # The four locks as a fetchable artifact (instructions steer the model; this one the
    # USER can read) — listed, readable, versioned, and stating each guarantee.
    listed = _list_resources()
    assert any(str(r.uri) == "portfolio://guarantees" for r in listed.resources)
    res = _read_resource("portfolio://guarantees")
    text = getattr(res.contents[0], "text", "")
    for must in (
        "Read-only",
        "never uploaded",
        "Descriptions, not advice",
        "financial advice",
        "shallower",
        "confidence interval",
        _VERSION,
    ):
        assert must in text, must


def test_tools_registered_and_read_only() -> None:
    res = _list_tools()
    names = {t.name for t in res.tools}
    assert {
        "portfolio_summary", "risk_report", "rebalance_check",
        "securities_facts", "discover_gaps", "screen_candidate",
    } <= names
    for t in res.tools:
        assert t.annotations is not None, f"{t.name} missing annotations"
        assert t.annotations.readOnlyHint is True


def test_portfolio_summary(warm_book: Path) -> None:
    res = _call("portfolio_summary")
    assert not res.isError, _error_text(res)
    sc = res.structuredContent
    assert sc is not None
    assert {h["ticker"] for h in sc["holdings"]} == {"AAA", "BBB"}
    assert sc["unpriced_tickers"] == []
    # Pin the actual math (catches a field swap like price<->avg_cost): each holding's
    # market value is shares x price, and unrealized is market value minus cost basis.
    for h in sc["holdings"]:
        assert h["market_value"] == pytest.approx(h["shares"] * h["price"])
        assert h["unrealized_pnl"] == pytest.approx(
            h["market_value"] - h["shares"] * h["avg_cost"]
        )
    # Totals are consistent with the per-holding values (same core math).
    mv = sum(h["market_value"] for h in sc["holdings"])
    assert sc["totals"]["market_value_priced"] == pytest.approx(mv)
    assert sc["totals"]["net_pnl"] == pytest.approx(
        sc["totals"]["unrealized_pnl"] + sc["totals"]["realized_pnl"]
    )


def test_risk_report(warm_book: Path) -> None:
    res = _call("risk_report")
    assert not res.isError, _error_text(res)
    sc = res.structuredContent
    assert sc is not None
    assert sc["n_days"] > 0
    assert sc["max_drawdown"]["depth"] <= 0.0
    for metric in ("sharpe", "sortino", "calmar", "ulcer_index", "cdar"):
        assert {"point", "low", "high"} <= set(sc[metric])


def test_rebalance_check(warm_book: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = warm_book / "target.csv"
    target.write_text("Ticker,Weight\nAAA,50\nBBB,50\n", encoding="utf-8")
    monkeypatch.setenv("ASSET_TARGET", str(target))
    res = _call("rebalance_check", {"mode": "to_total"})
    assert not res.isError, _error_text(res)
    sc = res.structuredContent
    assert sc is not None
    assert sc["mode"] == "to_total"
    by_tk = {t["ticker"]: t for t in sc["suggestions"]}
    assert set(by_tk) == {"AAA", "BBB"}
    # Pin real rebalance math (catches current<->target or shares<->dollars swaps):
    # 50/50 target; AAA is overweight (10x100 vs 5x50) so it sells, BBB buys; the
    # to_total rule is cash-neutral.
    assert by_tk["AAA"]["target_weight"] == pytest.approx(0.5)
    assert by_tk["BBB"]["target_weight"] == pytest.approx(0.5)
    assert by_tk["AAA"]["current_weight"] > by_tk["BBB"]["current_weight"]
    assert by_tk["AAA"]["action"] == "sell" and by_tk["BBB"]["action"] == "buy"
    buys = sum(t["dollars"] for t in sc["suggestions"] if t["action"] == "buy")
    sells = sum(t["dollars"] for t in sc["suggestions"] if t["action"] == "sell")
    assert buys == pytest.approx(sells)


def test_unknown_mode_is_clean_error(warm_book: Path) -> None:
    res = _call("rebalance_check", {"mode": "momentum"})
    assert res.isError
    assert "momentum" in _error_text(res)


def test_missing_book_is_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ASSET_BOOK", raising=False)
    monkeypatch.delenv("ASSET_CSV", raising=False)
    monkeypatch.delenv("ASSET_CACHE_DIR", raising=False)
    res = _call("portfolio_summary")
    assert res.isError
    assert "ASSET_BOOK" in _error_text(res)


def test_asset_csv_env_is_back_compat(warm_book: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # ASSET_BOOK is the canonical var, but the legacy ASSET_CSV must still resolve
    # the book when ASSET_BOOK is unset (a pre-rename .env keeps working).
    book = os.environ["ASSET_BOOK"]
    monkeypatch.delenv("ASSET_BOOK", raising=False)
    monkeypatch.setenv("ASSET_CSV", book)
    res = _call("portfolio_summary")
    assert not res.isError, _error_text(res)
    # The legacy var must resolve the SAME book, not just any non-error: pin its holdings.
    sc = res.structuredContent
    assert sc is not None and {h["ticker"] for h in sc["holdings"]} == {"AAA", "BBB"}


def test_rebalance_refuses_partial_book(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A HELD ticker unpriced offline must NOT yield fabricated orders sized over a
    # partial book (the CLI refuses this exact case). Warm only AAA; BBB is unpriced.
    csv = tmp_path / "txn.csv"
    csv.write_text(
        "Date,Code,Action,Quantity,Price,Fee\n"
        "2024-01-02,AAA,buy,10,100,0\n"
        "2024-01-02,BBB,buy,5,50,0\n",
        encoding="utf-8",
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    _warm_series_cache(cache, "AAA", 100.0)  # BBB deliberately left uncached
    target = tmp_path / "target.csv"
    target.write_text("Ticker,Weight\nAAA,50\nBBB,50\n", encoding="utf-8")
    monkeypatch.delenv("ASSET_CSV", raising=False)
    monkeypatch.setenv("ASSET_BOOK", str(csv))
    monkeypatch.setenv("ASSET_CACHE_DIR", str(cache))
    monkeypatch.setenv("ASSET_TARGET", str(target))

    res = _call("rebalance_check", {"mode": "to_total"})
    assert not res.isError, _error_text(res)
    sc = res.structuredContent
    assert sc is not None
    assert sc["suggestions"] == []  # refused — no orders over a partial book
    assert "BBB" in sc["note"]
    # portfolio_summary surfaces BBB as unpriced rather than valuing it.
    ps = _call("portfolio_summary").structuredContent
    assert ps is not None
    assert "BBB" in ps["unpriced_tickers"]


def test_risk_report_undefined_ratio_is_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A strictly-rising holding has ~zero drawdown → Calmar is undefined (the core
    # returns inf/nan). The tool must serialize it as null, not a non-finite number
    # that would break the structured-output schema.
    csv = tmp_path / "txn.csv"
    csv.write_text(
        "Date,Code,Action,Quantity,Price,Fee\n2024-01-02,UP,buy,10,100,0\n",
        encoding="utf-8",
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=320)
    rising = pd.Series([100.0 * (1.0 + 0.001 * i) for i in range(320)], index=idx, dtype=float)
    prices_mod._write_series_cache("UP", rising, cache)
    monkeypatch.delenv("ASSET_CSV", raising=False)
    monkeypatch.setenv("ASSET_BOOK", str(csv))
    monkeypatch.setenv("ASSET_CACHE_DIR", str(cache))

    res = _call("risk_report")
    assert not res.isError, _error_text(res)
    sc = res.structuredContent
    assert sc is not None
    assert sc["max_drawdown"]["depth"] == pytest.approx(0.0)
    assert sc["calmar"] is None  # undefined with no drawdown → null, not inf/nan


def _canned_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import date as _date

    from app.metadata import MetadataResult, SecurityMeta

    def canned(tickers, **k):  # type: ignore[no-untyped-def]
        return MetadataResult(
            rows={
                tk: SecurityMeta(
                    ticker=tk, expense_ratio=0.0003, aum=5e9, avg_volume=2e6,
                    category="Bond", family="Vanguard", legal_type="ETF",
                    quote_type="ETF", inception=_date(2015, 1, 1),
                )
                for tk in tickers
            }
        )

    # securities_facts calls app.mcp_server.fetch_metadata directly; screen_candidate reaches
    # metadata via app.pipeline.candidate_and_held_facts → app.pipeline.fetch_metadata. Patch
    # BOTH so the canned facts actually drive every consumer (and nothing leaks to the network).
    monkeypatch.setattr("app.mcp_server.fetch_metadata", canned)
    monkeypatch.setattr("app.pipeline.fetch_metadata", canned)


def test_securities_facts(warm_book: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _canned_meta(monkeypatch)
    res = _call("securities_facts")
    assert not res.isError, _error_text(res)
    sc = res.structuredContent
    assert sc is not None
    facts = {f["ticker"]: f for f in sc["securities"]}
    assert set(facts) == {"AAA", "BBB"}
    assert facts["AAA"]["quote_type"] == "ETF"               # the surfaced fund type
    assert facts["AAA"]["expense_ratio"] == pytest.approx(0.0003)
    assert sc["missing"] == []


def test_discover_gaps(warm_book: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Hermetic: a temp universe (not the repo's). AAA/BBB aren't in it, so every role reads
    # as a gap; the roles WITH candidates surface them (deterministic, offline).
    uni = warm_book / "universe.csv"
    uni.write_text(
        "ticker,name,role,summary\n"
        "VNQ,Vanguard Real Estate ETF,reit,US REITs\n"
        "VWO,Vanguard FTSE Emerging Markets ETF,em-equity,Emerging markets\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ASSET_UNIVERSE", str(uni))
    res = _call("discover_gaps")
    assert not res.isError, _error_text(res)
    sc = res.structuredContent
    assert sc is not None
    with_cands = [g for g in sc["gaps"] if g["candidates"]]
    assert with_cands, "expected gap roles with candidates"
    g = with_cands[0]
    assert {"role", "current_exposure", "candidates"} <= set(g)
    assert all(c["ticker"] and c["role"] == g["role"] for c in g["candidates"])
    assert {c["ticker"] for gg in with_cands for c in gg["candidates"]} == {"VNQ", "VWO"}
    assert sc["unpriced_holdings"] == []  # AAA/BBB are priced → no partial-book caveat


def test_screen_candidate_cached(warm_book: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _warm_series_cache(warm_book / "cache", "CCC", 40.0)     # candidate is in the cache
    _canned_meta(monkeypatch)
    res = _call("screen_candidate", {"ticker": "ccc"})
    assert not res.isError, _error_text(res)
    sc = res.structuredContent
    assert sc is not None
    assert sc["ticker"] == "CCC"                             # normalized to upper
    assert sc["verdict"] in {"PASS", "WARN", "FAIL"}
    assert sc["checks"]                                      # the screen actually ran
    assert all({"name", "status", "reason", "values"} <= set(c) for c in sc["checks"])


def test_screen_candidate_cache_miss_is_na(warm_book: Path) -> None:
    res = _call("screen_candidate", {"ticker": "ZZZ"})       # not warmed
    assert not res.isError, _error_text(res)
    sc = res.structuredContent
    assert sc is not None
    assert sc["ticker"] == "ZZZ"
    assert sc["verdict"] == "N/A"
    assert "cache" in sc["note"].lower()                     # honest "warm the cache" note


def test_screen_candidate_rejects_path_like_ticker(warm_book: Path) -> None:
    # The one free-text argument can't be used to read outside the bound book.
    res = _call("screen_candidate", {"ticker": "../../etc/passwd"})
    assert res.isError
    assert "valid ticker" in _error_text(res).lower()


# ── cold-cache auto-warm (ASSET_MCP_OFFLINE opts out) ─────────────────────────


def _cold_book(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A bound book over an EMPTY cache dir (cold) — the addon-user-never-ran-the-CLI case."""
    csv = tmp_path / "txn.csv"
    csv.write_text(
        "Date,Code,Action,Quantity,Price,Fee\n2024-01-02,AAA,buy,10,100,0\n",
        encoding="utf-8",
    )
    cache = tmp_path / "cache"
    cache.mkdir()  # empty → cold
    monkeypatch.delenv("ASSET_CSV", raising=False)
    monkeypatch.setenv("ASSET_BOOK", str(csv))
    monkeypatch.setenv("ASSET_CACHE_DIR", str(cache))
    monkeypatch.delenv("ASSET_TARGET", raising=False)
    return cache


def test_cold_cache_auto_warms_core(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _cold_book(tmp_path, monkeypatch)
    monkeypatch.setenv("ASSET_MCP_OFFLINE", "")  # opt into auto-warm (suite default is off)
    calls: list[list[str]] = []

    def fake_warm(tickers: object, cdir: Path, **k: object) -> dict[str, int]:
        tks = sorted(tickers)  # type: ignore[type-var]
        calls.append(tks)
        for tk in tks:
            _warm_series_cache(cdir, tk, 100.0)  # populate so the offline read then succeeds
        return {"tickers": len(tks), "series_missing": 0, "meta_missing": 0}

    monkeypatch.setattr("app.mcp_server.warm_cache", fake_warm)
    res = _call("portfolio_summary")
    assert not res.isError, _error_text(res)
    assert calls == [["AAA"]]  # auto-warmed once, with the book's tickers
    sc = res.structuredContent
    assert sc is not None and {h["ticker"] for h in sc["holdings"]} == {"AAA"}


def test_asset_mcp_offline_disables_auto_warm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cold_book(tmp_path, monkeypatch)
    monkeypatch.setenv("ASSET_MCP_OFFLINE", "1")  # opt out → strictly airtight
    called: list[int] = []
    monkeypatch.setattr("app.mcp_server.warm_cache", lambda *a, **k: called.append(1))
    res = _call("portfolio_summary")
    assert not called  # never reaches the network
    assert not res.isError, _error_text(res)  # cold cache → unpriced holdings, not a crash
    sc = res.structuredContent
    assert sc is not None and sc["unpriced_tickers"] == ["AAA"]


def test_no_rewarm_when_cache_is_warm(
    warm_book: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ASSET_MCP_OFFLINE", "")  # auto-warm ON, but the cache is already warm
    called: list[int] = []
    monkeypatch.setattr("app.mcp_server.warm_cache", lambda *a, **k: called.append(1))
    res = _call("portfolio_summary")
    assert not res.isError, _error_text(res)
    assert not called  # warm cache → cache_is_cold False → no auto-warm


# ── screen_candidate on-demand fetch (gated by ASSET_MCP_OFFLINE) ──────────────


def _synthetic_series(n: int = 320, base: float = 70.0) -> "pd.Series[float]":
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    return pd.Series([base * (1.0 + 0.02 * math.sin(i / 5.0) + 0.0003 * i) for i in range(n)], index=idx)


def test_screen_candidate_fetches_on_demand_when_not_locked(
    warm_book: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.metadata import MetadataResult

    monkeypatch.setenv("ASSET_MCP_OFFLINE", "")  # default: fetch a missing candidate online
    seen: dict[str, bool] = {}

    def fake_series(tickers: object, start: object, end: object, **k: object) -> object:
        seen["online"] = bool(k.get("online"))
        return prices_mod.SeriesResult(rows={t: _synthetic_series() for t in tickers})  # type: ignore[union-attr]

    monkeypatch.setattr("app.mcp_server.fetch_series", fake_series)
    # screen reaches metadata via pipeline, not mcp_server — patch the real lookup point so the
    # candidate's online metadata fetch is intercepted (else it hits the live network).
    monkeypatch.setattr("app.pipeline.fetch_metadata", lambda tickers, **k: MetadataResult())
    res = _call("screen_candidate", {"ticker": "CCC"})  # CCC not in the warm cache
    assert not res.isError, _error_text(res)
    assert seen["online"] is True  # fetched on demand rather than punting to the user
    assert res.structuredContent is not None
    assert res.structuredContent["verdict"] in {"PASS", "WARN", "FAIL"}  # actually screened


def test_screen_candidate_offline_locked_is_na(
    warm_book: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ASSET_MCP_OFFLINE", "1")  # strictly offline → no on-demand fetch
    seen: dict[str, bool] = {}

    def spy(tickers: object, start: object, end: object, **k: object) -> object:
        seen["online"] = bool(k.get("online"))
        return prices_mod.SeriesResult()  # ZZZZ not found → N/A

    monkeypatch.setattr("app.mcp_server.fetch_series", spy)
    res = _call("screen_candidate", {"ticker": "ZZZZ"})  # not cached
    assert not res.isError, _error_text(res)
    sc = res.structuredContent
    assert sc is not None and sc["verdict"] == "N/A"
    assert "offline" in sc["note"].lower()
    assert seen["online"] is False  # proves no network reach, not just the message


# ── propose_allocation (generate offline; validate cache-gated) ────────────────


def test_propose_allocation_generates_weights(warm_book: Path) -> None:
    res = _call("propose_allocation", {"preset": "moderate", "benchmark": "none"})
    assert not res.isError, _error_text(res)
    sc = res.structuredContent
    assert sc is not None
    assert sc["preset"] == "moderate" and len(sc["weights"]) > 0
    assert sum(w["weight"] for w in sc["weights"]) == pytest.approx(1.0)  # normalized
    assert sc["benchmark"] is None and sc["verdict"] is None  # validation skipped
    # The preset arg actually drives the posture: a different preset → different weights
    # (not just "some normalized weights" — catches a preset→bucket mix-up).
    agg = _call("propose_allocation", {"preset": "aggressive", "benchmark": "none"})
    mod_w = {w["ticker"]: w["weight"] for w in sc["weights"]}
    agg_w = {w["ticker"]: w["weight"] for w in agg.structuredContent["weights"]}  # type: ignore[index]
    assert mod_w != agg_w


def test_propose_allocation_verdict_null_when_refs_cold(warm_book: Path) -> None:
    # The book (AAA/BBB) doesn't cache the 60-40 refs, so the verdict degrades to null with
    # a warm note — while the weights (the generate half) are still returned.
    res = _call("propose_allocation", {"preset": "moderate", "benchmark": "60-40"})
    assert not res.isError, _error_text(res)
    sc = res.structuredContent
    assert sc is not None and len(sc["weights"]) > 0
    assert sc["verdict"] is None and "warm" in sc["validation_note"].lower()


def test_propose_allocation_maps_a_benchmark_result(
    warm_book: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import types

    monkeypatch.setattr(
        "app.mcp_server.benchmark_compare",
        lambda *a, **k: types.SimpleNamespace(
            verdict="shallower", reason="held-out: a shallower drawdown",
            dd_diff_ci=(-0.08, -0.01), missing=(),
        ),
    )
    res = _call("propose_allocation", {"preset": "conservative", "benchmark": "60-40"})
    assert not res.isError, _error_text(res)
    v = res.structuredContent["verdict"]  # type: ignore[index]
    assert v is not None and v["verdict"] == "shallower" and v["reference"] == "60-40"
    assert v["oos_dd_diff_low"] == pytest.approx(-0.08)


def test_propose_allocation_unknown_preset_is_clean_error(warm_book: Path) -> None:
    res = _call("propose_allocation", {"preset": "momentum"})
    assert res.isError and "momentum" in _error_text(res)


def test_propose_allocation_nulls_placeholder_ci(
    warm_book: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # benchmark_compare returns dd_diff_ci=(0.0, 0.0) as a placeholder for inconclusive/
    # insufficient verdicts (no paired bootstrap ran) — it must not surface as a real CI.
    import types

    monkeypatch.setattr(
        "app.mcp_server.benchmark_compare",
        lambda *a, **k: types.SimpleNamespace(
            verdict="inconclusive", reason="within the noise margin",
            dd_diff_ci=(0.0, 0.0), missing=(),
        ),
    )
    res = _call("propose_allocation", {"preset": "moderate", "benchmark": "60-40"})
    assert not res.isError, _error_text(res)
    v = res.structuredContent["verdict"]  # type: ignore[index]
    assert v is not None and v["verdict"] == "inconclusive"
    assert v["oos_dd_diff_low"] is None and v["oos_dd_diff_high"] is None  # placeholder nulled


def test_propose_allocation_validate_fetches_on_demand_when_unlocked(
    warm_book: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Unlocked (the default), the validate path fetches its universe-fill tickers ONLINE on
    # demand (same gate as screen_candidate), so it can judge the full target rather than a
    # renormalized subset. Spy on the validate fetch and assert it asked to go online.
    import types

    monkeypatch.setenv("ASSET_MCP_OFFLINE", "")  # opt back into egress (autouse pins "1")
    seen: list[bool] = []

    def _spy(tickers: object, start: object, end: object, **k: object) -> object:
        seen.append(bool(k.get("online")))
        return types.SimpleNamespace(rows={}, missing=(), provenance={})

    monkeypatch.setattr("app.mcp_server.fetch_series", _spy)
    res = _call("propose_allocation", {"preset": "moderate", "benchmark": "60-40"})
    assert not res.isError, _error_text(res)
    assert seen and all(seen)  # the validate fetch asked to go online, not cache-only


def test_propose_allocation_nulls_verdict_when_target_renormalized(
    warm_book: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A still-cold proposed-target ticker means benchmark_compare renormalized that leg — the
    # verdict would describe a different portfolio than the weights shown, so null it (don't
    # report a number about the wrong basket), while the weights themselves still return.
    import types

    monkeypatch.setattr(
        "app.mcp_server.benchmark_compare",
        lambda *a, **k: types.SimpleNamespace(
            verdict="shallower", reason="held-out: shallower",
            dd_diff_ci=(-0.05, -0.01), missing=("VWO",),
        ),
    )
    res = _call("propose_allocation", {"preset": "moderate", "benchmark": "60-40"})
    assert not res.isError, _error_text(res)
    sc = res.structuredContent
    assert sc is not None and len(sc["weights"]) > 0  # generate half still returns weights
    assert sc["verdict"] is None
    assert "VWO" in sc["validation_note"] and "renormal" in sc["validation_note"].lower()
