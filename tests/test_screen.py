"""Tests for the deterministic candidate screen (pure; synthetic series + metas)."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd

from app.metadata import SecurityMeta
from app.screen import (
    CandidateScreen,
    CheckResult,
    holdings_overlap,
    screen_candidates,
)

_ASOF = date(2026, 6, 12)
_NOW = datetime(2026, 6, 12, tzinfo=timezone.utc)


def _meta(ticker: str = "X", **over: object) -> SecurityMeta:
    base: dict[str, object] = dict(
        ticker=ticker, expense_ratio=0.0003, aum=5e9, avg_volume=2e6,
        category="Large Blend", family="Vanguard", legal_type="Exchange Traded Fund",
        quote_type="ETF", inception=date(2015, 1, 1), top_holdings={},
        source="cache", fetched_at=_NOW,
    )
    base.update(over)
    return SecurityMeta(**base)  # type: ignore[arg-type]


def _close_from_returns(rets: list[float], start: str = "2024-01-02") -> "pd.Series[float]":
    idx = pd.bdate_range(start, periods=len(rets))
    return (1.0 + pd.Series(rets, index=idx, dtype=float)).cumprod() * 100.0


def _returns(rets: list[float], start: str = "2024-01-02") -> "pd.Series[float]":
    idx = pd.bdate_range(start, periods=len(rets))
    return pd.Series(rets, index=idx, dtype=float)


def _check(result: CandidateScreen, name: str) -> CheckResult:
    return next(c for c in result.checks if c.name == name)


def _screen_one(
    ticker: str = "CAND",
    close: "pd.Series[float] | None" = None,
    port: "pd.Series[float] | None" = None,
    meta: SecurityMeta | None = None,
    held_meta: dict[str, SecurityMeta] | None = None,
    held: set[str] | None = None,
) -> CandidateScreen:
    port = port if port is not None else _returns([0.001, -0.002, 0.003] * 30)
    return screen_candidates(
        [ticker],
        {ticker: close} if close is not None else {},
        port,
        {ticker: meta} if meta is not None else {},
        held_meta or {},
        held or set(),
        asof=_ASOF,
    )[0]


# --- overlap math ---


def test_holdings_overlap_is_min_sum_over_shared() -> None:
    a = {"NVDA": 0.50, "AAPL": 0.30, "MSFT": 0.20}
    b = {"NVDA": 0.40, "AAPL": 0.40, "GOOG": 0.20}
    assert holdings_overlap(a, b) == 0.40 + 0.30  # min per shared symbol


def test_holdings_overlap_none_without_lookthrough() -> None:
    assert holdings_overlap({}, {"A": 1.0}) is None  # physical trust → no holdings
    assert holdings_overlap({"A": 1.0}, {}) is None


# --- per-check goldens ---


def test_structure_rejects_leveraged_and_etn() -> None:
    lev = _screen_one(meta=_meta(category="Trading--Leveraged Equity"))
    assert _check(lev, "structure").status == "fail"
    etn = _screen_one(meta=_meta(legal_type="Exchange Traded Note"))
    assert _check(etn, "structure").status == "fail"
    ok = _screen_one(meta=_meta())
    assert _check(ok, "structure").status == "pass"


def test_cost_tiers() -> None:
    assert _check(_screen_one(meta=_meta(expense_ratio=0.0003)), "cost").status == "pass"
    assert _check(_screen_one(meta=_meta(expense_ratio=0.0035)), "cost").status == "warn"
    assert _check(_screen_one(meta=_meta(expense_ratio=0.0070)), "cost").status == "fail"
    assert _check(_screen_one(meta=_meta(expense_ratio=None)), "cost").status == "n/a"


def test_liquidity_floors() -> None:
    # URAN-shaped: tiny AUM and volume → fail; one floor missed → warn.
    both = _screen_one(meta=_meta(aum=31.7e6, avg_volume=9_000))
    assert _check(both, "liquidity").status == "fail"
    one = _screen_one(meta=_meta(aum=101e6, avg_volume=35_000))
    assert _check(one, "liquidity").status == "warn"
    fine = _screen_one(meta=_meta())
    assert _check(fine, "liquidity").status == "pass"


def test_age_tiers() -> None:
    assert _check(_screen_one(meta=_meta(inception=date(2026, 1, 1))), "age").status == "fail"
    assert _check(_screen_one(meta=_meta(inception=date(2024, 6, 1))), "age").status == "warn"
    assert _check(_screen_one(meta=_meta(inception=date(2015, 1, 1))), "age").status == "pass"
    assert _check(_screen_one(meta=_meta(inception=None)), "age").status == "n/a"


def test_concentration() -> None:
    fat = _meta(top_holdings={"A": 0.40, "B": 0.26})
    assert _check(_screen_one(meta=fat), "concentration").status == "warn"
    broad = _meta(top_holdings={"A": 0.10, "B": 0.05})
    assert _check(_screen_one(meta=broad), "concentration").status == "pass"
    assert _check(_screen_one(meta=_meta()), "concentration").status == "n/a"  # no look-through


# --- the diversifier role test ---


def test_diversifier_clone_fails_and_independent_passes() -> None:
    port = _returns([0.004, -0.003, 0.002, -0.001] * 40)
    clone = _close_from_returns(list(port.values))  # moves identically → ρ≈1
    c = _check(_screen_one(close=clone, port=port), "diversifier")
    assert c.status == "fail" and "ρ=+1.00" in c.reason

    # Period-3 pattern vs the portfolio's period-2 → near-zero correlation.
    offbeat = _close_from_returns([0.0001, 0.0001, -0.0001] * 54)
    c2 = _check(_screen_one(close=offbeat, port=port), "diversifier")
    assert c2.status == "pass"


def test_diversifier_short_overlap_is_na() -> None:
    port = _returns([0.001, -0.001] * 40)
    short = _close_from_returns([0.001] * 10)  # only 10 overlapping days
    c = _check(_screen_one(close=short, port=port), "diversifier")
    assert c.status == "n/a" and "overlapping days" in c.reason


def test_diversifier_red_day_escalation() -> None:
    # Candidate mirrors the portfolio EXACTLY on red days (falls together in
    # stress) but moves on its own pattern on green days → low full-period ρ
    # (~0.4, would pass) with red-day ρ = 1.0 → escalated to warn.
    port_r = [0.004, -0.003, 0.005, -0.002] * 30
    cand_r = [-0.006, -0.003, 0.006, -0.002] * 30
    port = _returns(port_r)
    cand = _close_from_returns(cand_r)
    c = _check(_screen_one(close=cand, port=port), "diversifier")
    assert c.status == "warn"
    assert "red days" in c.reason


def test_diversifier_reports_worst_drawdown_window() -> None:
    # Portfolio: up 30 days, down 30 (the drawdown), up 60. Candidate: steady up.
    port = _returns([0.005] * 30 + [-0.01] * 30 + [0.004] * 60)
    cand = _close_from_returns([0.001] * 120)
    c = _check(_screen_one(close=cand, port=port), "diversifier")
    assert "during your worst drawdown" in c.reason and "it returned" in c.reason


# --- overlap vs held ---


def test_overlap_near_duplicate_fails() -> None:
    cand = _meta(top_holdings={"NVDA": 0.40, "AAPL": 0.35, "MSFT": 0.25})
    held = {"VOO": _meta("VOO", top_holdings={"NVDA": 0.40, "AAPL": 0.35, "GOOG": 0.25})}
    c = _check(_screen_one(meta=cand, held_meta=held), "overlap")
    assert c.status == "fail" and "VOO" in c.reason  # 75% overlap → near-duplicate


def test_overlap_moderate_warns_low_passes() -> None:
    held = {"VOO": _meta("VOO", top_holdings={"NVDA": 0.30, "AAPL": 0.30, "GOOG": 0.40})}
    mid = _meta(top_holdings={"NVDA": 0.45, "GOOG": 0.10, "XOM": 0.45})
    # min(0.45,0.30) + min(0.10,0.40) = 0.40 → exactly the warn floor
    assert _check(_screen_one(meta=mid, held_meta=held), "overlap").status == "warn"
    low = _meta(top_holdings={"XOM": 0.50, "CVX": 0.50})
    assert _check(_screen_one(meta=low, held_meta=held), "overlap").status == "pass"


def test_overlap_commodity_falls_back_to_category() -> None:
    # GLDM/IAU case: physical trusts publish no holdings; same category as a held
    # fund → "likely the same exposure" warn, not a silent n/a.
    cand = _meta("IAU", category="Commodities Focused", top_holdings={})
    held = {"GLDM": _meta("GLDM", category="Commodities Focused", top_holdings={})}
    c = _check(_screen_one(ticker="IAU", meta=cand, held_meta=held), "overlap")
    assert c.status == "warn" and "GLDM" in c.reason
    lonely = _meta(category="Utilities", top_holdings={})
    assert _check(_screen_one(meta=lonely, held_meta=held), "overlap").status == "n/a"


# --- composition ---


def test_already_held_candidate_fails_novelty() -> None:
    r = _screen_one(ticker="VOO", meta=_meta("VOO"), held={"VOO"})
    assert _check(r, "novelty").status == "fail"
    assert r.verdict == "fail"


def test_verdict_aggregation_and_degradation() -> None:
    # No meta, no series → every check n/a, nothing raises, verdict n/a.
    r = _screen_one(meta=None, close=None)
    assert r.verdict in ("n/a", "pass", "warn", "fail")
    assert all(c.status == "n/a" for c in r.checks)
    assert r.verdict == "n/a"
    assert "n-a" in r.counts()


def test_screen_never_raises_on_missing_candidate_data() -> None:
    port = _returns([0.001, -0.001] * 40)
    results = screen_candidates(
        ["GHOST"], {}, port, {}, {}, set(), asof=_ASOF
    )
    assert results[0].ticker == "GHOST"
    assert _check(results[0], "diversifier").status == "n/a"


def test_diversifier_flat_candidate_is_na_not_pass() -> None:
    # Regression (review 2026-06-12): a zero-variance candidate (halted/stale feed,
    # constant price) has corr = NaN; every threshold comparison is False, so before
    # the fix the if-chain bottomed out at a silent "pass" with "ρ=+nan" printed.
    port = _returns([0.002, -0.001] * 50)
    flat = _close_from_returns([0.0] * 100)  # constant price → zero-variance returns
    c = _check(_screen_one(close=flat, port=port), "diversifier")
    assert c.status == "n/a"
    assert "nan" not in c.reason
    assert "no return variation" in c.reason


def test_diversifier_flat_portfolio_is_na() -> None:
    # The same guard covers the portfolio side (a degenerate book).
    port = _returns([0.0] * 100)
    cand = _close_from_returns([0.002, -0.001] * 50)
    c = _check(_screen_one(close=cand, port=port), "diversifier")
    assert c.status == "n/a" and "nan" not in c.reason


def test_threshold_boundaries_are_inclusive() -> None:
    # Pin the <=/< choices at the exact tier edges so a future flip can't slip by.
    assert _check(_screen_one(meta=_meta(expense_ratio=0.0020)), "cost").status == "pass"
    assert _check(_screen_one(meta=_meta(expense_ratio=0.0050)), "cost").status == "warn"
    at_floor = _meta(aum=100e6, avg_volume=100_000)  # exactly the floors → not below
    assert _check(_screen_one(meta=at_floor), "liquidity").status == "pass"
    half = _meta(top_holdings={"A": 0.25, "B": 0.25})  # exactly 50% → not above
    assert _check(_screen_one(meta=half), "concentration").status == "pass"
