"""Tests for the report layer: ReportData structure + the three renderers.

No I/O — the report layer is pure. Fixtures build the derived state and the
summaries directly so we never touch the network or the price cache.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from app.derive import DerivedState, Position
from app.prices import PriceRow
from app.report import (
    ReportData,
    Section,
    build_report_data,
    format_summary,
    render_html,
    render_markdown,
    render_text,
)
from app.returns import ReturnsSummary
from app.risk import DollarDrawdown, DrawdownInfo, MetricCI, RiskSummary
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
        ulcer_index=MetricCI(0.0238, 0.018, 0.031),
        cdar=MetricCI(0.0673, 0.045, 0.092),
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


def test_dollar_drawdown_line_rendered(state, prices, returns, risk) -> None:
    ddl = DollarDrawdown(
        giveback_dollars=-1293.0, peak_pnl=2174.0, trough_pnl=881.0,
        peak_date=date(2026, 1, 23), trough_date=date(2026, 3, 30),
        recovery_date=date(2026, 5, 1), duration_days=98,
    )
    out = render_text(build_report_data(state, prices, returns, risk, dollar_dd=ddl))
    assert "Gains given back: -$1,293" in out
    assert "peak profit $2,174 (2026-01-23) → $881 (2026-03-30)" in out
    assert "flow-neutral" in out


def test_dollar_drawdown_absent_when_none(state, prices, returns, risk) -> None:
    out = render_text(build_report_data(state, prices, returns, risk))  # dollar_dd defaults None
    assert "Account $ drawdown" not in out


def test_markdown_has_headings_and_fences(state, prices, returns, risk) -> None:
    md = render_markdown(build_report_data(state, prices, returns, risk))
    assert md.startswith("# Portfolio brief — 2026-06-02")
    assert "## DRAWDOWN (investment, time-weighted)" in md
    assert "```" in md  # bodies are fenced to preserve alignment
    assert md.endswith("\n")


def test_narration_summary_renders_as_prose_not_monospace(state, prices, returns, risk) -> None:
    # A prose Section (the narration SUMMARY) must lead the brief and render as wrapped
    # paragraphs — NOT a ``` code fence (markdown) or a white-space:pre box (HTML), which
    # would show flowing prose as monospace and horizontally scroll in an email client.
    prose = "Your portfolio fell from its February peak and has since recovered."
    prov = "— wording by claude-haiku-4-5 (paid tier); figures verified by the tool."
    summary = Section("SUMMARY", (prose, "", prov), prose=True)
    data = build_report_data(state, prices, returns, risk, summary=summary)
    assert data.sections[0].title == "SUMMARY"  # leads the brief
    assert prose in render_text(data)

    md = render_markdown(data)
    assert f"## SUMMARY\n\n{prose}" in md   # heading then prose paragraph
    assert f"```\n{prose}" not in md        # NOT inside a code fence

    html = render_html(data)
    assert f'margin:0 0 16px">{prose}<br>' in html  # in a wrapping <p>, not a <pre>


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
    now = datetime.now(timezone.utc)
    prov = {"A": ("yfinance", now), "B": ("cache", now)}
    bt = backtest_compare({"A": a, "B": b}, {"A": 0.5, "B": 0.5},
                          schedule="quarterly", bootstrap_n=30, provenance=prov)
    assert bt is not None
    data = build_report_data(state, prices, backtest=bt)
    assert any(s.title.startswith("BACKTEST") for s in data.sections)
    out = render_text(data)
    assert "rebalanced (quarterly)" in out and "buy & hold" in out
    assert "Max drawdown" in out and "Final value" in out
    assert "Ulcer index" in out and "CDaR (worst 5%)" in out  # the verdict trio (v2.9.0)
    assert "simulation, not a prediction" in out
    # P0-3: Sharpe/Sortino now carry a 95% CI row (not bare points).
    assert out.count("95% CI") >= 3
    # P0-1c: the backtest carries its price provenance.
    assert "prices: cache, yfinance" in out


def test_ulcer_cdar_and_returns_carry_bands_and_labels(state, prices, returns, risk) -> None:
    out = render_text(build_report_data(state, prices, returns, risk))
    ulcer_line = next(ln for ln in out.splitlines() if ln.startswith("Ulcer index:"))
    cdar_line = next(ln for ln in out.splitlines() if ln.startswith("CDaR"))
    assert "95% CI" in ulcer_line and "95% CI" in cdar_line  # P0-3b
    assert "point figures" in out  # P0-3c: returns labelled point-only


def test_provenance_footer_real_source_and_deterministic(state) -> None:
    gen = datetime(2026, 6, 4, 12, 0, tzinfo=timezone.utc)
    fetched = gen - timedelta(hours=20)
    px = {"VOO": PriceRow("VOO", date(2026, 6, 3), 600.0, "cache", fetched)}
    out1 = render_text(build_report_data(state, px, generated_at=gen))
    out2 = render_text(build_report_data(state, px, generated_at=gen))
    assert out1 == out2  # P0-4: deterministic given generated_at (no hidden clock)
    assert "1 cache" in out1  # P0-1: real source, not the fabricated "series"
    assert "20.0h" in out1   # P0-1: real age, not "0s"


def test_stale_close_surfaces_in_footer_and_holdings_grid(state) -> None:
    # F2 (fresh-eyes audit 2026-07-11): fetch age alone hid a stale QUOTE — a close from
    # 2023 downloaded a second ago read "(age: 1s)". A close older than the floor must
    # surface both in the footer (close-date range + ⚠) and beside the holding it prices.
    gen = datetime(2026, 6, 4, 12, 0, tzinfo=timezone.utc)
    px = {
        "VOO": PriceRow("VOO", date(2026, 6, 3), 600.0, "cache", gen - timedelta(seconds=1)),
        "BND": PriceRow("BND", date(2023, 10, 13), 94.42, "cache", gen - timedelta(seconds=1)),
    }
    out = render_text(build_report_data(state, px, generated_at=gen))
    assert "closes 2026-06-03 .. 2023-10-13 ⚠" in out
    assert "⚠ BND: price is the close from 2023-10-13" in out
    assert "⚠ VOO" not in out  # the fresh row stays clean


def test_near_floor_close_shows_dates_without_the_alarm(state) -> None:
    # Between the 4-day display threshold and the 10-day floor (a legitimate offline
    # cache tail): the close dates become visible, but no ⚠ — this is disclosure, not
    # an error.
    gen = datetime(2026, 6, 10, 12, 0, tzinfo=timezone.utc)
    old = date(2026, 6, 4)  # 6 days before gen: > 4d display threshold, ≤ 10d floor
    px = {
        "VOO": PriceRow("VOO", old, 600.0, "cache", gen - timedelta(hours=1)),
        "BND": PriceRow("BND", old, 73.0, "cache", gen - timedelta(hours=1)),
    }
    out = render_text(build_report_data(state, px, generated_at=gen))
    assert "closes 2026-06-04" in out  # same date on both → single date, not a range
    assert "2026-06-04 ⚠" not in out   # within the floor → no alarm mark


def test_stale_close_display_uses_the_report_date_not_the_utc_render_day(state) -> None:
    # Review finding #7: the pipeline gates staleness on the LOCAL as-of date while the
    # render instant is UTC — near local midnight they are different days. The display
    # must anchor on `asof` (the pipeline's day), not generated_at.date(): a KST morning
    # run (UTC still on yesterday) would otherwise understate every close's lag by a day.
    gen = datetime(2026, 6, 9, 22, 0, tzinfo=timezone.utc)  # 07:00 KST on 2026-06-10
    asof = date(2026, 6, 10)                                 # the pipeline's local today
    old = date(2026, 6, 5)  # 5 days before asof (shown); only 4 before gen.date() (hidden)
    px = {
        "VOO": PriceRow("VOO", old, 600.0, "cache", gen - timedelta(hours=1)),
        "BND": PriceRow("BND", old, 73.0, "cache", gen - timedelta(hours=1)),
    }
    out = render_text(build_report_data(state, px, generated_at=gen, asof=asof))
    assert "closes 2026-06-05" in out           # judged against asof: 5d > the 4d grace
    assert "5d before this report" in out       # the grid note uses the same anchor


def test_current_closes_keep_the_footer_clean(state) -> None:
    # The everyday case (closes within a weekend/holiday gap of the report date) must
    # not grow a closes suffix — the stale display is for stale prices only.
    gen = datetime(2026, 6, 4, 12, 0, tzinfo=timezone.utc)
    px = {
        "VOO": PriceRow("VOO", date(2026, 6, 3), 600.0, "cache", gen - timedelta(hours=1)),
        "BND": PriceRow("BND", date(2026, 6, 2), 73.0, "cache", gen - timedelta(hours=1)),
    }
    out = render_text(build_report_data(state, px, generated_at=gen))
    assert "closes" not in out
    assert "⚠" not in out


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
        ulcer_index=MetricCI(0.01, 0.005, 0.02),
        cdar=MetricCI(0.03, 0.01, 0.05),
        sharpe=MetricCI(0.5, -0.5, 1.5),
        sortino=MetricCI(0.7, -0.4, 1.8),
        calmar=MetricCI(0.4, -0.6, 1.4),
    )
    out = render_text(build_report_data(state, prices, returns, noisy))
    assert "treat these figures as noisy." in out
    assert "not yet recovered" in out  # recovery_date None


def test_securities_section_uses_one_consistent_asof(state) -> None:
    # Regression (review 2026-06-12): with asof=None the fund-age date must come
    # from the SAME input-derived default the title uses (generated_at's date) —
    # not a second hidden clock read that can disagree across midnight.
    from app.metadata import MetadataResult, SecurityMeta

    gen = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    meta = MetadataResult(
        rows={
            "VOO": SecurityMeta(
                ticker="VOO", expense_ratio=0.0003, aum=1.7e12, avg_volume=8.8e6,
                category="Large Blend", family="Vanguard", legal_type=None,
                quote_type="ETF", inception=date(2016, 1, 1),
            )
        }
    )
    data = build_report_data(state, metadata=meta, asof=None, generated_at=gen)
    assert data.asof_date == "2026-01-01"  # title asof = gen.date()
    sec = next(s for s in data.sections if s.title.startswith("SECURITIES"))
    assert any("10.0y" in line for line in sec.lines)  # age computed from the SAME date
    assert any("ETF" in line for line in sec.lines)    # quote_type surfaced in the row


def test_shelf_index_drill_hint_skips_blank_flavors() -> None:
    # Review fix: the index's example command must be a runnable drill — a blank flavor
    # would suggest `--discover role:` which just re-renders the index.
    from app.discover import Discovery
    from app.report import _section_discoveries

    blank_first = Discovery(
        gaps=("sector-equity",), exposure={}, candidates=(),
        more_shelves={"sector-equity": (("", 2), ("tech", 3))},
    )
    text = "\n".join(_section_discoveries(blank_first, []).lines)
    assert "--discover sector-equity:tech" in text  # first NAMED shelf, not the blank one

    all_blank = Discovery(
        gaps=("sector-equity",), exposure={}, candidates=(),
        more_shelves={"sector-equity": (("", 2),)},
    )
    text = "\n".join(_section_discoveries(all_blank, []).lines)
    assert "e.g. --discover" not in text  # no runnable example exists → no hint line
