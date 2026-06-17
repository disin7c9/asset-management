"""Tests for the read-only MCP server.

Offline + hermetic: a temp transaction CSV plus a warmed price cache (no network),
driven through the SDK's in-memory client session (no subprocess). Verifies the
tools are read-only, return the validated core's numbers as structured content, and
degrade to clean MCP errors rather than crashing.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import anyio
import pandas as pd
import pytest
from mcp.shared.memory import create_connected_server_and_client_session as client_session
from mcp.types import CallToolResult, ListToolsResult

from app import prices as prices_mod
from app.mcp_server import mcp


def _warm_series_cache(cache: Path, ticker: str, base: float, n: int = 320) -> None:
    """Write a fresh <T>_series.parquet ending ~today so an OFFLINE read succeeds.

    A gentle oscillation + drift gives real drawdowns and finite risk ratios (a
    monotonic series would make Calmar non-finite)."""
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    vals = [base * (1.0 + 0.03 * math.sin(i / 4.0) + 0.0003 * i) for i in range(n)]
    prices_mod._write_series_cache(ticker, pd.Series(vals, index=idx, dtype=float), cache)


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
    monkeypatch.setenv("ASSET_CSV", str(csv))
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


def test_tools_registered_and_read_only() -> None:
    res = _list_tools()
    names = {t.name for t in res.tools}
    assert {"portfolio_summary", "risk_report", "rebalance_check"} <= names
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


def test_missing_asset_csv_is_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ASSET_CSV", raising=False)
    monkeypatch.delenv("ASSET_CACHE_DIR", raising=False)
    res = _call("portfolio_summary")
    assert res.isError
    assert "ASSET_CSV" in _error_text(res)


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
    monkeypatch.setenv("ASSET_CSV", str(csv))
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
    monkeypatch.setenv("ASSET_CSV", str(csv))
    monkeypatch.setenv("ASSET_CACHE_DIR", str(cache))

    res = _call("risk_report")
    assert not res.isError, _error_text(res)
    sc = res.structuredContent
    assert sc is not None
    assert sc["max_drawdown"]["depth"] == pytest.approx(0.0)
    assert sc["calmar"] is None  # undefined with no drawdown → null, not inf/nan
