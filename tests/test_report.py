"""Tests for the report layer: ReportData structure + the three renderers.

No I/O — the report layer is pure. Fixtures build the derived state and the
summaries directly so we never touch the network or the price cache.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from app.derive import DerivedState, Position
from app.prices import PriceRow
from app.report import (
    ReportData,
    build_report_data,
    format_summary,
    render_html,
    render_markdown,
    render_text,
)
from app.returns import ReturnsSummary
from app.risk import DrawdownInfo, MetricCI, RiskSummary
from app.strategy import Suggestion


@pytest.fixture
def state() -> DerivedState:
    s = DerivedState()
    s.positions["VOO"] = Position("VOO", shares=10.0, cost_basis=3000.0)
    s.positions["BND"] = Position("BND", shares=50.0, cost_basis=3600.0)
    s.realized["VOO"] = 50.0
    s.fees["VOO"] = 2.0
    return s


@pytest.fixture
def prices() -> dict[str, PriceRow]:
    now = datetime.now(timezone.utc)
    return {
        "VOO": PriceRow("VOO", date(2026, 6, 2), 320.0, "series", now),
        "BND": PriceRow("BND", date(2026, 6, 2), 73.0, "series", now),
    }


@pytest.fixture
def returns() -> ReturnsSummary:
    return ReturnsSummary(
        period_start=date(2023, 1, 5),
        asof_date=date(2026, 6, 2),
        money_weighted_annualized=0.1828,
        modified_dietz_annualized=0.1794,
        true_twr_annualized=0.1937,
    )


@pytest.fixture
def risk() -> RiskSummary:
    dd = DrawdownInfo(
        depth=-0.0984,
        peak_date=date(2025, 2, 19),
        trough_date=date(2025, 4, 8),
        recovery_date=date(2025, 6, 10),
        duration_days=111,
        time_underwater_pct=0.80,
    )
    return RiskSummary(
        n_days=600,  # > NOISY_THRESHOLD_DAYS so no noisy note
        drawdown=dd,
        max_drawdown_ci=MetricCI(-0.0984, -0.1721, -0.0558),
        ulcer_index=0.0238,
        cdar=0.0673,
        sharpe=MetricCI(1.75, 0.68, 2.83),
        sortino=MetricCI(2.66, 0.96, 4.63),
        calmar=MetricCI(1.97, 0.52, 4.83),
    )


def test_section_order_drawdown_first(state, prices, returns, risk) -> None:
    data = build_report_data(state, prices, returns, risk)
    titles = [s.title for s in data.sections]
    assert titles == [
        "DRAWDOWN (investment, time-weighted)",
        "RISK-ADJUSTED (annualized, 252-day basis, risk-free 0%, ± bootstrap CI)",
        "RETURNS (annualized, 252-day basis)",
        "HOLDINGS",
        "",  # untitled footer
    ]
    assert data.asof_date == "2026-06-02"


def test_format_summary_is_text_render(state, prices, returns, risk) -> None:
    data = build_report_data(state, prices, returns, risk)
    assert format_summary(
        state, prices, returns, risk
    ) == render_text(data)


def test_text_render_has_headers_and_numbers(state, prices, returns, risk) -> None:
    out = render_text(build_report_data(state, prices, returns, risk))
    assert "=== DRAWDOWN (investment, time-weighted) ===" in out
    assert "Max drawdown:      -9.84%  (95% CI -17.21% .. -5.58%)" in out
    assert "Sharpe:   +1.75  (95% CI +0.68 .. +2.83)" in out
    assert "Time-weighted (true TWR):                +19.37%" in out
    # Net P&L = unrealized + realized (fees already netted; not subtracted again)
    assert "Net P&L (unrealized + realized):" in out


def test_markdown_has_headings_and_fences(state, prices, returns, risk) -> None:
    md = render_markdown(build_report_data(state, prices, returns, risk))
    assert md.startswith("# Portfolio brief — 2026-06-02")
    assert "## DRAWDOWN (investment, time-weighted)" in md
    assert "```" in md  # bodies are fenced to preserve alignment
    assert md.endswith("\n")


def test_html_escapes_and_wraps(state, prices, returns, risk) -> None:
    state.positions["A&B"] = Position("A&B", shares=1.0, cost_basis=10.0)
    html = render_html(build_report_data(state, prices, returns, risk))
    assert html.startswith("<!doctype html>")
    assert "<h3" in html and "<pre" in html
    assert "A&amp;B" in html       # escaped
    assert "A&B<" not in html       # never emitted raw


def test_disclaimer_in_every_format(state, prices, returns, risk) -> None:
    data = build_report_data(state, prices, returns, risk)
    needle = "not financial advice"
    assert needle in render_text(data)
    assert needle in render_markdown(data)
    assert needle in render_html(data)


def test_na_when_returns_degenerate(state, prices) -> None:
    degenerate = ReturnsSummary(
        period_start=date(2026, 5, 30),
        asof_date=date(2026, 6, 2),
        money_weighted_annualized=None,
        modified_dietz_annualized=None,
        true_twr_annualized=None,
    )
    out = render_text(build_report_data(state, prices, degenerate, None))
    assert "Time-weighted (true TWR):                n/a" in out
    assert "(n/a = period too short to annualize, or no real solution)" in out


def test_holdings_only_when_no_risk_no_returns(state) -> None:
    data = build_report_data(state)
    titles = [s.title for s in data.sections]
    assert titles == ["HOLDINGS", ""]  # no prices, no returns/risk
    # asof falls back to today's UTC date (no returns object to date it)
    assert isinstance(data, ReportData)
    assert "n/a" in render_text(data)  # prices absent → unpriced rows show n/a


def test_backtest_section_renders(state, prices) -> None:
    import pandas as pd

    from app.backtest import backtest_compare

    dates = pd.bdate_range("2024-01-01", periods=80)
    a = pd.Series([100.0 + i for i in range(80)], index=dates, dtype=float)
    b = pd.Series([50.0] * 80, index=dates, dtype=float)
    bt = backtest_compare({"A": a, "B": b}, {"A": 0.5, "B": 0.5},
                          schedule="quarterly", bootstrap_n=30)
    assert bt is not None
    data = build_report_data(state, prices, backtest=bt)
    assert any(s.title.startswith("BACKTEST") for s in data.sections)
    out = render_text(data)
    assert "rebalanced (quarterly)" in out and "buy & hold" in out
    assert "Max drawdown" in out and "Final value" in out
    assert "simulation, not a prediction" in out


def test_suggestions_lead_and_render(state, prices, returns, risk) -> None:
    sugs = [
        Suggestion("VOO", "sell", 1000.0 / 300, 1000.0, 0.60, 0.50, "to_total",
                   "60.0% vs 50.0% target"),
        Suggestion("IAU", "buy", 62.5, 2500.0, 0.0, 0.25, "to_total",
                   "0.0% vs 25.0% target"),
        Suggestion("BND", "hold", 0.0, 0.0, 0.25, 0.25, "to_total", "on target"),
    ]
    data = build_report_data(state, prices, returns, risk, suggestions=sugs)
    # Actionable panel leads the brief.
    assert data.sections[0].title.startswith("SUGGESTED ACTIONS")
    out = render_text(data)
    assert "SELL" in out and "BUY" in out and "HOLD" in out
    assert "Buy $2,500.00" in out and "Sell $1,000.00" in out
    assert "net +$1,500.00 (cash to deploy)" in out
    assert "## SUGGESTED ACTIONS" in render_markdown(data)
    assert "<h3" in render_html(data)


def test_noisy_note_appears_when_short_history(state, prices, returns) -> None:
    dd = DrawdownInfo(-0.05, date(2025, 1, 1), date(2025, 2, 1), None, 60, 0.5)
    noisy = RiskSummary(
        n_days=120,  # < NOISY_THRESHOLD_DAYS
        drawdown=dd,
        max_drawdown_ci=MetricCI(-0.05, -0.10, -0.02),
        ulcer_index=0.01,
        cdar=0.03,
        sharpe=MetricCI(0.5, -0.5, 1.5),
        sortino=MetricCI(0.7, -0.4, 1.8),
        calmar=MetricCI(0.4, -0.6, 1.4),
    )
    out = render_text(build_report_data(state, prices, returns, noisy))
    assert "treat these figures as noisy." in out
    assert "not yet recovered" in out  # recovery_date None
