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
from app.mcp_server import _VERSION, Returns, mcp
from app.pipeline import _WARM_MARKER


def _warm_series_cache(cache: Path, ticker: str, base: float) -> None:
    """Write a fresh <T>_series.parquet spanning 2024-01-01..today so an OFFLINE read
    succeeds AND the history covers the fixture book's 2024-01-02 trades — a series
    starting after a trade is now flagged left-truncated and excluded from TWR/risk (F3),
    and a stale tail is refused as a current price (F1), so the fixture must look like
    real, complete data.

    A gentle oscillation + drift gives real drawdowns and finite risk ratios (a
    monotonic series would make Calmar non-finite). ``fetched_at`` is backdated a
    minute (still fresh within TTL) so a sub-second backward clock jiggle (WSL/CI)
    can't read the just-written cache as 'in the future' and refuse it."""
    idx = pd.bdate_range("2024-01-01", pd.Timestamp.today().normalize())
    n = len(idx)
    vals = [base * (1.0 + 0.03 * math.sin(i / 4.0) + 0.0003 * i) for i in range(n)]
    df = pd.DataFrame(
        {
            "date": idx,
            "close": vals,
            "fetched_at": pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=1),
        }
    )
    # Both bases: the portfolio path reads the raw file, the screen/simulate paths read the
    # total-return one. The fixture is synthetic, so one curve legitimately serves as both.
    df.to_parquet(cache / f"{ticker}_series.parquet", index=False)
    df.to_parquet(cache / f"{ticker}_series_tr.parquet", index=False)


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
    # The auto-warm tests turn offline back off; without this a developer's real Tiingo key
    # (loaded from .env) would let the yfinance-miss fallback reach api.tiingo.com for real.
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)


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
    # The chat front door: 6 conversation starters in the client's "+" menu, each with a
    # description a non-finance user can pick from (blank-box problem).
    res = _list_prompts()
    names = {p.name for p in res.prompts}
    assert names == {
        "portfolio_checkup",
        "whats_my_drawdown",
        "should_i_rebalance",
        "fill_my_gaps",
        "find_my_starting_allocation",
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
        ("find_my_starting_allocation", None),
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
    with pytest.raises(BaseException) as ei:  # noqa: PT011 — anyio wraps McpError in a BaseExceptionGroup
        _get_prompt("propose_a_posture", {"posture": "balanced"})
    err: BaseException = ei.value
    while isinstance(err, BaseExceptionGroup):  # anyio wraps the McpError in a TaskGroup
        err = err.exceptions[0]
    assert "conservative" in str(err)  # the error names the menu


def test_should_i_rebalance_threads_mode_and_new_cash() -> None:
    # The rebalance starter takes a mode (and new_cash), like propose_a_posture takes a posture:
    # it threads the chosen mode into the framing, tells the model to deploy new_cash for the
    # cash modes only, and validates the mode at prompt time.
    dca = getattr(
        _get_prompt(
            "should_i_rebalance", {"mode": "fixed_dca", "new_cash": "500"}
        ).messages[0].content,
        "text", "",
    )
    assert "mode='fixed_dca'" in dca and "new_cash=500" in dca
    tot = getattr(  # to_total ignores new_cash → no cash clause
        _get_prompt("should_i_rebalance", {"mode": "to_total"}).messages[0].content, "text", "",
    )
    assert "mode='to_total'" in tot and "new_cash" not in tot
    with pytest.raises(BaseException) as ei:  # noqa: PT011 — anyio wraps McpError; off-menu mode errors at prompt time
        _get_prompt("should_i_rebalance", {"mode": "momentum"})
    err: BaseException = ei.value
    while isinstance(err, BaseExceptionGroup):  # anyio wraps the McpError in a TaskGroup
        err = err.exceptions[0]
    assert "to_total" in str(err)  # the error names the 4-mode menu


def test_tool_descriptions_enumerate_their_option_menus() -> None:
    # Claude Desktop only knows an option set if the description says it: pin that
    # rebalance_check names all 4 modes + the new_cash rule, and the allocation tools name the
    # benchmark menu — so a client never has to guess the valid tokens.
    tools = {t.name: (t.description or "") for t in _list_tools().tools}
    for mode in ("to_total", "bands", "fixed_dca", "cash_flow_only"):
        assert mode in tools["rebalance_check"], mode
    assert "new_cash" in tools["rebalance_check"]
    for tool in ("propose_allocation", "starter_allocation"):
        for bench in ("60-40", "all-weather", "permanent"):
            assert bench in tools[tool], f"{tool}:{bench}"


def test_propose_a_posture_threads_posture_and_benchmark() -> None:
    # Like the rebalance starter, "Propose a posture" now takes BOTH inputs (posture +
    # benchmark) and validates each at prompt time.
    txt = getattr(
        _get_prompt(
            "propose_a_posture", {"posture": "aggressive", "benchmark": "all-weather"}
        ).messages[0].content,
        "text", "",
    )
    assert "preset 'aggressive'" in txt and "benchmark 'all-weather'" in txt
    with pytest.raises(BaseException) as ei:  # noqa: PT011 — anyio wraps McpError; off-menu benchmark errors at prompt time
        _get_prompt("propose_a_posture", {"posture": "moderate", "benchmark": "sp500"})
    err: BaseException = ei.value
    while isinstance(err, BaseExceptionGroup):  # anyio wraps the McpError in a TaskGroup
        err = err.exceptions[0]
    assert "60-40" in str(err)  # the error names the benchmark menu


def test_onboarding_answer_tokens_are_in_sync_with_the_rubric() -> None:
    # The find_my_starting_allocation prompt and the starter_allocation tool description
    # re-list the answer tokens (horizon/loss_response/cash_buffer) as prose. If someone
    # renames an Option.key in app.onboard, the model would be told to send a token the
    # rubric rejects — so pin that EVERY live token appears verbatim in both surfaces.
    from app.onboard import QUESTIONS

    prompt_text = getattr(
        _get_prompt("find_my_starting_allocation").messages[0].content, "text", ""
    )
    tool = next(t for t in _list_tools().tools if t.name == "starter_allocation")
    desc = tool.description or ""
    for q in QUESTIONS:
        for opt in q.options:
            assert opt.key in prompt_text, f"{opt.key} missing from the onboarding prompt"
            assert opt.key in desc, f"{opt.key} missing from starter_allocation description"


def test_every_constrained_arg_publishes_its_choices_as_a_schema_enum() -> None:
    # A bare `string` arg leaves the model GUESSING a token the tool will then reject —
    # exactly how starter_allocation failed in Claude Desktop. Each constrained arg must
    # advertise its domain in the JSON schema, not merely in the description prose.
    from app.allocate import PRESETS
    from app.backtest import BENCHMARKS
    from app.onboard import QUESTIONS
    from app.strategy import VALID_MODES

    schemas = {t.name: t.inputSchema["properties"] for t in _list_tools().tools}
    onboard_tokens = {q.key: {o.key for o in q.options} for q in QUESTIONS}
    expected = {
        ("rebalance_check", "mode"): VALID_MODES,
        ("propose_allocation", "preset"): PRESETS,
        ("propose_allocation", "benchmark"): BENCHMARKS | {"none"},
        ("starter_allocation", "benchmark"): BENCHMARKS | {"none"},
        ("starter_allocation", "horizon"): onboard_tokens["horizon"],
        ("starter_allocation", "loss_response"): onboard_tokens["loss_response"],
        ("starter_allocation", "cash_buffer"): onboard_tokens["cash_buffer"],
    }
    for (tool, arg), choices in expected.items():
        spec = schemas[tool][arg]
        enum = spec.get("enum") or next(
            (a["enum"] for a in spec.get("anyOf", []) if "enum" in a), None
        )
        assert enum is not None, f"{tool}.{arg} publishes no enum — the model must guess it"
        assert set(enum) == set(choices), f"{tool}.{arg} enum drifted from its canonical set"


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
        "propose_allocation", "starter_allocation",
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
    # The TWR rides on the daily history this panel already loads — this is the ONE
    # tool that exposes it (portfolio_summary skips the history); a null here on a
    # long-history book would mean it got computed-then-dropped again (v2.11.4).
    assert sc["true_twr_annualized"] is not None
    assert math.isfinite(sc["true_twr_annualized"])


def test_summary_twr_field_points_to_risk_report(warm_book: Path) -> None:
    """portfolio_summary's TWR is ALWAYS null (it never loads the price history); the
    published field description must send the model to risk_report, or the null reads
    as 'unavailable' with no path to the figure (hit live in Claude Desktop, 2026-07-16)."""
    res = _call("portfolio_summary")
    assert not res.isError, _error_text(res)
    sc = res.structuredContent
    assert sc is not None
    assert sc["returns"]["true_twr_annualized"] is None
    desc = Returns.model_fields["true_twr_annualized"].description
    assert desc is not None and "risk_report" in desc


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


def test_rebalance_check_echoes_new_cash_and_target(
    warm_book: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # E1: the plan is self-describing — it echoes new_cash + the target source, and fixed_dca /
    # cash_flow_only actually deploy fresh cash (they were stuck at all-HOLD before, because
    # rebalance_check hard-coded new_cash to 0 and never accepted it).
    target = warm_book / "target.csv"
    target.write_text("Ticker,Weight\nAAA,50\nBBB,50\n", encoding="utf-8")
    monkeypatch.setenv("ASSET_TARGET", str(target))

    # fixed_dca WITH cash → deploys it across the target mix (no longer all-HOLD) + echoes both
    res = _call("rebalance_check", {"mode": "fixed_dca", "new_cash": 1000.0})
    assert not res.isError, _error_text(res)
    sc = res.structuredContent
    assert sc is not None
    assert sc["new_cash"] == pytest.approx(1000.0)
    assert sc["target_source"] == str(target)
    buy_tks = {t["ticker"] for t in sc["suggestions"] if t["action"] == "buy"}
    assert buy_tks == {"AAA", "BBB"}  # fresh cash deployed across the mix, not stuck at all-HOLD

    # fixed_dca WITHOUT cash → nothing deployed, but the note explains why (not a silent puzzle)
    sc0 = _call("rebalance_check", {"mode": "fixed_dca"}).structuredContent
    assert sc0 is not None
    assert not any(t["action"] == "buy" for t in sc0["suggestions"])
    assert "new_cash=0" in sc0["note"]

    # negative cash is refused
    assert _call("rebalance_check", {"mode": "fixed_dca", "new_cash": -5.0}).isError


def test_rebalance_check_rejects_nonfinite_new_cash() -> None:
    # M3: a non-finite new_cash (a client sending 1e400 → inf, or nan) must be refused, not
    # threaded into Trade dollars. nan/inf can't round-trip cleanly through JSON, so exercise
    # the guard on the tool function directly (FastMCP leaves it callable).
    from app.mcp_server import rebalance_check
    for bad in (float("inf"), float("nan"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            rebalance_check(new_cash=bad)


def test_to_total_deploys_new_cash_on_every_surface(warm_book: Path) -> None:
    # M1: to_total DEPLOYS new_cash (suggest: base = total + new_cash). The tool description must
    # not lump it with bands as "ignoring" cash, and the should_i_rebalance starter must tell the
    # model to deploy it for to_total.
    desc = {t.name: t.description for t in _list_tools().tools}["rebalance_check"] or ""
    assert "deploys new_cash" in desc                # to_total deploys it
    assert "these two IGNORE new_cash" not in desc   # the old wrong to_total+bands grouping is gone
    text = getattr(
        _get_prompt("should_i_rebalance", {"mode": "to_total", "new_cash": "5000"}).messages[0].content,
        "text", "",
    )
    assert "5000" in text and "to_total adds it to the target total" in text


def test_missing_book_falls_back_to_demo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    # F1: book is optional in the manifest (required:false), so a fresh install that never saved
    # a book must not dead-end — _env_book falls back to the bundled DEMO portfolio with a LOUD
    # warning, never the user's real book and never a silent swap.
    import logging

    from app.mcp_server import _env_book
    from app.pipeline import DEMO_BOOK_CSV

    monkeypatch.setenv("ASSET_CACHE_DIR", str(tmp_path / "cache"))  # keep the demo write hermetic
    # ASSET_BOOK / ASSET_CSV are "" via the autouse fixture → no book configured
    with caplog.at_level(logging.WARNING):
        path = _env_book()
    assert path.exists() and path.read_text(encoding="utf-8") == DEMO_BOOK_CSV
    assert "DEMO" in caplog.text and "NOT yours" in caplog.text

    # A book that IS set but points nowhere still errors — never silently swapped for demo data.
    monkeypatch.setenv("ASSET_BOOK", str(tmp_path / "nope.csv"))
    with pytest.raises(ValueError, match="missing file"):
        _env_book()


def test_missing_book_prices_demo_through_a_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # T3: beyond _env_book() returning the demo path, a real tool call on the no-book path must
    # answer with demo HOLDINGS (not error) — the end-to-end fallback, not just the resolver.
    monkeypatch.setenv("ASSET_CACHE_DIR", str(tmp_path / "cache"))  # hermetic demo write
    # ASSET_BOOK / ASSET_CSV "" via the autouse fixture; ASSET_MCP_OFFLINE "1" (no auto-warm)
    res = _call("portfolio_summary")
    assert not res.isError, _error_text(res)
    sc = res.structuredContent
    assert sc is not None and sc["holdings"]  # the demo book derives real positions
    # ...and the response must SAY it's demo data. `_env_book` warns on stderr, which the
    # model never reads — without this the assistant reports fake holdings as the user's.
    assert sc["provenance"]["is_demo"] is True
    assert "DEMO DATA" in sc["note"]


def test_a_configured_book_is_never_labelled_demo(warm_book: Path) -> None:
    # The mirror: the disclosure must not cry wolf on a real book, or it stops meaning anything.
    res = _call("portfolio_summary")
    assert not res.isError, _error_text(res)
    sc = res.structuredContent
    assert sc is not None
    assert sc["provenance"]["is_demo"] is False
    assert "DEMO DATA" not in sc["note"]


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


def test_env_book_residue_falls_through_to_asset_csv(
    warm_book: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # M2: an unsubstituted ${...} template residue in ASSET_BOOK must NOT shadow a legacy
    # ASSET_CSV — residue counts as unset, so the fallback var is still consulted (not demo).
    from app.mcp_server import _env_book
    book = os.environ["ASSET_BOOK"]  # warm_book set a real file
    monkeypatch.setenv("ASSET_BOOK", "${user_config.book}")  # host left it unsubstituted
    monkeypatch.setenv("ASSET_CSV", book)
    assert _env_book() == Path(book)  # resolved via ASSET_CSV, not swapped for demo


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
    # Spans the 2024-01-02 buy (a series starting after the trade would be F3-excluded
    # from risk) and ends today (a stale tail would be F1-unpriced).
    idx = pd.bdate_range("2024-01-01", pd.Timestamp.today().normalize())
    rising = pd.Series(
        [100.0 * (1.0 + 0.001 * i) for i in range(len(idx))], index=idx, dtype=float
    )
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
        "VWO,Vanguard FTSE Emerging Markets ETF,em-equity,Emerging markets\n"
        "VGT,Vanguard Information Technology ETF,sector-equity,Tech sector\n",
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
    # The satellite role never reads as a gap over MCP (no include_satellites arg here),
    # so VGT stays out of the candidate set.
    assert all(gg["role"] != "sector-equity" for gg in sc["gaps"])
    assert {c["ticker"] for gg in with_cands for c in gg["candidates"]} == {"VNQ", "VWO"}
    assert sc["unpriced_holdings"] == []  # AAA/BBB are priced → no partial-book caveat


def test_discover_gaps_shelves_and_drill(
    warm_book: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The shelf mirror is additive: the default call names unsurfaced shelves with counts
    # (never picking one); role= + flavor= drills a named shelf; the satellite role maps.
    uni = warm_book / "universe.csv"
    uni.write_text(
        "ticker,name,role,summary,core,flavor\n"
        "IEF,iShares 7-10 Year Treasury,treasury,7-10y US Treasuries,1,intermediate\n"
        "VGIT,Vanguard Interm Treasury,treasury,intermediate,1,intermediate\n"
        "GOVT,iShares US Treasury,treasury,whole market,1,intermediate\n"
        "TLT,iShares 20+ Year Treasury,treasury,20y+,1,long\n"
        "VGLT,Vanguard Long Treasury,treasury,long,1,long\n"
        "VGT,Vanguard Information Tech,sector-equity,tech sector,1,tech\n"
        "XLV,Health Care Select,sector-equity,health sector,1,health\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ASSET_UNIVERSE", str(uni))
    res = _call("discover_gaps")
    assert not res.isError, _error_text(res)
    sc = res.structuredContent
    assert sc is not None
    tre = next(g for g in sc["gaps"] if g["role"] == "treasury")
    assert [c["ticker"] for c in tre["candidates"]] == ["IEF", "VGIT", "GOVT"]
    assert tre["lead_flavor"] == "intermediate"
    assert {(s["flavor"], s["n"]) for s in tre["other_shelves"]} == {("long", 2)}
    assert all(g["role"] != "sector-equity" for g in sc["gaps"])  # satellite: never default

    res = _call("discover_gaps", {"role": "treasury", "flavor": "long"})
    assert not res.isError, _error_text(res)
    sc = res.structuredContent
    assert sc is not None
    assert [c["ticker"] for c in sc["gaps"][0]["candidates"]] == ["TLT", "VGLT"]

    res = _call("discover_gaps", {"role": "sector-equity"})
    assert not res.isError, _error_text(res)
    sc = res.structuredContent
    assert sc is not None
    sec = sc["gaps"][0]
    assert sec["candidates"] == []  # the map, not a shortlist
    assert {(s["flavor"], s["n"]) for s in sec["other_shelves"]} == {("tech", 1), ("health", 1)}

    res = _call("discover_gaps", {"flavor": "long"})  # flavor without role
    assert res.isError


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


# ── provenance receipts (item 7): what source, how fresh ───────────────────────


def test_portfolio_summary_carries_provenance(warm_book: Path) -> None:
    res = _call("portfolio_summary")
    assert not res.isError, _error_text(res)
    p = res.structuredContent["provenance"]  # type: ignore[index]
    assert p["sources"] == ["cache"]  # served offline from the warmed cache
    assert p["price_asof"] is not None
    assert p["stalest_fetch_hours"] is not None and p["stalest_fetch_hours"] >= 0.0


def test_risk_report_carries_provenance(warm_book: Path) -> None:
    # The gap item 7 closes: the risk panel now says what its numbers are sourced from.
    res = _call("risk_report")
    assert not res.isError, _error_text(res)
    p = res.structuredContent["provenance"]  # type: ignore[index]
    assert p["sources"] == ["cache"]
    assert p["price_asof"] is not None


def test_provenance_price_asof_matches_holdings(warm_book: Path) -> None:
    # The rollup's newest close date equals the latest close the fixture cache holds —
    # one honest 'as of' for the whole tool, derived from the same prices. The expected
    # date is derived from the FIXTURE's own construction (bdate_range snaps to the last
    # business day), NOT from the wall clock: on a Saturday the honest answer is Friday's
    # close, and asserting equality with today made this test fail every weekend.
    res = _call("portfolio_summary")
    sc = res.structuredContent
    assert sc is not None
    last_bday = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=1)[-1]
    assert sc["provenance"]["price_asof"] == last_bday.date().isoformat()
    # both holdings share the one fixture cache → the oldest close is the same date
    assert sc["provenance"]["oldest_close"] == sc["provenance"]["price_asof"]


def test_provenance_stalest_is_the_max_age_not_the_min(
    warm_book: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pin the 'stalest' semantics: with two holdings fetched at DIFFERENT times, the rollup
    # must report the OLDER fetch's age (max), not the fresher one — a max→min inversion
    # would silently understate staleness. (warm_book alone fetches both at ~the same instant,
    # so min==max there and can't catch the swap.)
    from datetime import date, datetime, timedelta, timezone

    from app.derive import DerivedState
    from app.mcp_server import DataProvenance, _Build, _data_provenance
    from app.prices import PriceRow

    now = datetime.now(timezone.utc)
    fresh = PriceRow("AAA", date.today(), 100.0, "cache", now - timedelta(hours=2))
    stale = PriceRow("BBB", date.today(), 50.0, "cache", now - timedelta(hours=50))
    b = _Build(
        state=DerivedState(), prices={"AAA": fresh, "BBB": stale},
        returns=None, risk=None, missing=[], dollar_dd=None,
    )
    prov: DataProvenance = _data_provenance(b)
    assert prov.stalest_fetch_hours == pytest.approx(50.0, abs=0.2)  # the older fetch wins


def test_provenance_oldest_close_is_the_min_not_the_max() -> None:
    # The F2 fix (fresh-eyes audit 2026-07-11): price_asof reports the NEWEST close, so a
    # single stale holding among fresh ones was invisible in the receipt. oldest_close must
    # report the other end — the min — or the one field added to expose staleness would
    # itself hide it.
    from datetime import date, datetime, timedelta, timezone

    from app.derive import DerivedState
    from app.mcp_server import _Build, _data_provenance
    from app.prices import PriceRow

    now = datetime.now(timezone.utc)
    today = date.today()
    fresh = PriceRow("AAA", today, 100.0, "cache", now)
    stale = PriceRow("BBB", today - timedelta(days=900), 50.0, "cache", now)
    b = _Build(
        state=DerivedState(), prices={"AAA": fresh, "BBB": stale},
        returns=None, risk=None, missing=[], dollar_dd=None,
    )
    prov = _data_provenance(b)
    assert prov.price_asof == today
    assert prov.oldest_close == today - timedelta(days=900)


def test_provenance_falls_back_to_series_when_no_current_holdings() -> None:
    # A fully-exited book: risk_report computes real numbers from the price HISTORY while
    # b.prices (current holdings) is empty. Provenance must stamp the series, not return an
    # all-null receipt beside real numbers (the honesty gap item 7 exists to prevent).
    from datetime import datetime, timedelta, timezone

    from app.derive import DerivedState
    from app.mcp_server import _Build, _data_provenance
    from app.prices import SeriesResult

    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=5)
    series = SeriesResult(
        rows={"AAA": pd.Series([1.0] * 5, index=idx)},
        missing=[],
        provenance={"AAA": ("cache", datetime.now(timezone.utc) - timedelta(hours=3))},
    )
    b = _Build(
        state=DerivedState(), prices={}, returns=None, risk=None,
        missing=[], dollar_dd=None, series=series,
    )
    prov = _data_provenance(b)
    assert prov.sources == ["cache"]  # stamped from the series, not all-null
    assert prov.price_asof == idx[-1].date()
    assert prov.stalest_fetch_hours == pytest.approx(3.0, abs=0.2)
    # oldest_close means "the stalest quote a HOLDING is valued at"; nothing is held on
    # this branch and the series set includes long-sold tickers, so it must be null —
    # a min over the tails would report a sold (possibly delisted) position as a stale
    # holding and invite a false narration (review finding #5).
    assert prov.oldest_close is None


def test_provenance_empty_when_no_data_at_all() -> None:
    from app.derive import DerivedState
    from app.mcp_server import _Build, _data_provenance

    b = _Build(state=DerivedState(), prices={}, returns=None, risk=None, missing=[], dollar_dd=None)
    prov = _data_provenance(b)
    assert prov.price_asof is None and prov.sources == [] and prov.stalest_fetch_hours is None


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
            verdict="shallower", reason="held-out: less drawdown pain (Ulcer)",
            ulcer_gain_ci=(0.01, 0.08), missing=(), cause="",
        ),
    )
    res = _call("propose_allocation", {"preset": "conservative", "benchmark": "60-40"})
    assert not res.isError, _error_text(res)
    v = res.structuredContent["verdict"]  # type: ignore[index]
    assert v is not None and v["verdict"] == "shallower" and v["reference"] == "60-40"
    assert v["oos_ulcer_gain_low"] == pytest.approx(0.01)
    assert v["inconclusive_cause"] is None  # a resolved verdict carries no cause


def test_propose_allocation_unknown_preset_is_clean_error(warm_book: Path) -> None:
    res = _call("propose_allocation", {"preset": "momentum"})
    assert res.isError and "momentum" in _error_text(res)


def test_propose_allocation_nulls_placeholder_ci(
    warm_book: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # benchmark_compare returns ulcer_gain_ci=None when no paired bootstrap ran (margin/CDaR-
    # gated or too short) — it must surface as null low/high, with the structured cause.
    import types

    monkeypatch.setattr(
        "app.mcp_server.benchmark_compare",
        lambda *a, **k: types.SimpleNamespace(
            verdict="inconclusive", reason="within the noise margin",
            ulcer_gain_ci=None, missing=(), cause="noise_margin",
        ),
    )
    res = _call("propose_allocation", {"preset": "moderate", "benchmark": "60-40"})
    assert not res.isError, _error_text(res)
    v = res.structuredContent["verdict"]  # type: ignore[index]
    assert v is not None and v["verdict"] == "inconclusive"
    assert v["oos_ulcer_gain_low"] is None and v["oos_ulcer_gain_high"] is None  # no interval
    assert v["inconclusive_cause"] == "noise_margin"  # the structured gate, for clients to branch


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
            ulcer_gain_ci=(0.01, 0.05), missing=("VWO",),
        ),
    )
    res = _call("propose_allocation", {"preset": "moderate", "benchmark": "60-40"})
    assert not res.isError, _error_text(res)
    sc = res.structuredContent
    assert sc is not None and len(sc["weights"]) > 0  # generate half still returns weights
    assert sc["verdict"] is None
    assert "VWO" in sc["validation_note"] and "renormal" in sc["validation_note"].lower()


# ── starter_allocation (onboarding: answers → posture → the same proposal) ──────


def test_starter_allocation_maps_answers_and_embeds_the_proposal(warm_book: Path) -> None:
    res = _call("starter_allocation", {
        "horizon": "over_10_years", "loss_response": "buy_more",
        "cash_buffer": "comfortably", "benchmark": "none",
    })
    assert not res.isError, _error_text(res)
    sc = res.structuredContent
    assert sc is not None
    assert sc["posture"] == "aggressive" and sc["score"] == 6
    assert sc["rationale"] and any("→ aggressive" in line for line in sc["rationale"])
    # The embedded proposal is a full ProposedAllocation for that posture, weights normalized.
    prop = sc["proposal"]
    assert prop["preset"] == "aggressive" and len(prop["weights"]) > 0
    assert sum(w["weight"] for w in prop["weights"]) == pytest.approx(1.0)


def test_starter_allocation_matches_propose_allocation_for_the_same_posture(
    warm_book: Path,
) -> None:
    # Onboarding must not be a second allocation code path: cautious answers → conservative,
    # and its weights must equal propose_allocation('conservative') exactly.
    starter = _call("starter_allocation", {
        "horizon": "under_3_years", "loss_response": "sell",
        "cash_buffer": "no", "benchmark": "none",
    }).structuredContent
    direct = _call("propose_allocation", {
        "preset": "conservative", "benchmark": "none",
    }).structuredContent
    assert starter is not None and direct is not None
    assert starter["posture"] == "conservative"
    assert starter["proposal"]["weights"] == direct["weights"]


def test_starter_allocation_short_horizon_caps_conservative(warm_book: Path) -> None:
    # Growth-tolerant answers but a sub-3y horizon → capped at conservative (the safety rail).
    res = _call("starter_allocation", {
        "horizon": "under_3_years", "loss_response": "buy_more",
        "cash_buffer": "comfortably", "benchmark": "none",
    })
    sc = res.structuredContent
    assert sc is not None and sc["posture"] == "conservative"
    assert any("capped at conservative" in line for line in sc["rationale"])


def test_starter_allocation_unknown_answer_is_clean_error(warm_book: Path) -> None:
    res = _call("starter_allocation", {
        "horizon": "someday", "loss_response": "hold", "cash_buffer": "partly",
    })
    assert res.isError and "someday" in _error_text(res)


def test_starter_allocation_empty_book_needs_no_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The step-0 persona: a book with no trades yet over a stone-cold cache. Nothing is
    # held, so nothing needs pricing — every role fills from the curated universe
    # (mirrors the CLI's bookless --onboard). The cold refusal is only for HELD books.
    empty = tmp_path / "empty.csv"
    empty.write_text("Date,Code,Action,Quantity,Price,Fee\n", encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()  # empty → cold
    monkeypatch.setenv("ASSET_BOOK", str(empty))
    monkeypatch.setenv("ASSET_CACHE_DIR", str(cache))
    res = _call("starter_allocation", {
        "horizon": "over_10_years", "loss_response": "hold",
        "cash_buffer": "comfortably", "benchmark": "none",
    })
    assert not res.isError, _error_text(res)
    sc = res.structuredContent
    assert sc is not None
    assert len(sc["proposal"]["weights"]) > 0  # pure universe defaults, no prices needed


def test_propose_allocation_held_book_cold_cache_still_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The empty-book carve-out must NOT loosen the held case: real holdings the cache
    # can't price would be silently ignored by the proposal — refuse with the warm hint.
    _cold_book(tmp_path, monkeypatch)
    res = _call("propose_allocation", {"preset": "moderate", "benchmark": "none"})
    assert res.isError and "warm" in _error_text(res).lower()


def test_every_tool_that_can_answer_on_demo_data_says_so(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Regression: `note` was switched to a demo-aware default_factory, but three response
    paths passed `note=` explicitly (which skips the default) or were never migrated off the
    plain constant — `rebalance_check`, `discover_gaps`, and `screen_candidate`'s
    couldn't-fetch branch. rebalance_check is the worst of the three: it proposes TRADES and
    carries no DataProvenance, so `note` is its only channel for saying the holdings are fake."""
    monkeypatch.setenv("ASSET_CACHE_DIR", str(tmp_path / "cache"))
    # ASSET_BOOK/ASSET_CSV pinned "" by the autouse fixture → the demo-book fallback
    for tool, args in (
        ("rebalance_check", {"mode": "to_total"}),
        ("discover_gaps", {}),
        ("screen_candidate", {"ticker": "ZZZ"}),   # uncached → the couldn't-fetch branch
    ):
        res = _call(tool, args)
        sc = res.structuredContent
        if sc is None or "note" not in sc:
            continue  # the tool errored for an unrelated reason (no target file, etc.)
        assert "DEMO DATA" in sc["note"], f"{tool} did not disclose demo data: {sc['note'][:120]}"


def test_screen_candidate_uses_the_same_peer_bar_as_the_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # own-drawdown compares a candidate against the deepest fund the user HOLDS, not against
    # the blended book (a blend falls less than its parts, so that bar is near-tautological).
    # The CLI got that peer; this surface calls the same screen and must pass the same thing,
    # or the two answer differently for one ticker — the hand-maintained-parity failure mode
    # this codebase already carries five of.
    import inspect

    from app.mcp_server import screen_candidate

    src = inspect.getsource(screen_candidate)
    assert "held_worst=" in src, "MCP screen_candidate no longer passes the peer bar"
    assert "deepest_held(" in src, "MCP must use the shared helper, not its own copy"
