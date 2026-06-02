"""Presentation layer: render derived state + prices + returns + risk into text.

Pure functions only — no I/O, no logging side-effects. Sections are separate
functions so the order can change without a rewrite. v0 leads with DRAWDOWN
(what the user feels), then risk-adjusted ratios, then returns, then holdings.
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timedelta, timezone

from app.derive import DerivedState
from app.prices import PriceRow
from app.returns import ReturnsSummary
from app.risk import NOISY_THRESHOLD_DAYS, DrawdownInfo, MetricCI, RiskSummary

_NA = "n/a"


def _pct_or_na(x: float | None) -> str:
    return _NA if x is None or not math.isfinite(x) else f"{x * 100:+.2f}%"


def _ci(ci: MetricCI, *, scale: float, suffix: str) -> str:
    """Render a MetricCI; values multiplied by `scale` with `suffix` appended.

    Falls back to `n/a` when the point estimate is non-finite (e.g. Calmar with
    no drawdown → nan, Sortino with no downside → inf)."""
    if not math.isfinite(ci.point):
        return _NA
    s = suffix
    return (
        f"{ci.point * scale:+.2f}{s}  "
        f"(95% CI {ci.low * scale:+.2f}{s} .. {ci.high * scale:+.2f}{s})"
    )


def _ci_pct(ci: MetricCI) -> str:
    """MetricCI whose values are fractions, rendered as percentages with a band."""
    return _ci(ci, scale=100.0, suffix="%")


def _ci_ratio(ci: MetricCI) -> str:
    """MetricCI whose values are unitless ratios (Sharpe etc.)."""
    return _ci(ci, scale=1.0, suffix="")


def format_summary(
    state: DerivedState,
    prices: dict[str, PriceRow] | None = None,
    returns: ReturnsSummary | None = None,
    risk: RiskSummary | None = None,
    *,
    true_twr: float | None = None,
    missing_tickers: list[str] | None = None,
) -> str:
    """Build the deterministic plain-text brief. Drawdown leads."""
    prices = prices or {}
    missing_tickers = missing_tickers or []
    blocks: list[str] = []

    if risk is not None:
        blocks.append(_render_drawdown(risk))
        blocks.append(_render_risk_adjusted(risk))
    if returns is not None and returns.period_days > 0:
        blocks.append(_render_returns(returns, true_twr))
    blocks.append(_render_holdings(state, prices))

    footer: list[str] = []
    if missing_tickers:
        footer.append(f"Prices unavailable for: {', '.join(sorted(missing_tickers))}")
    if prices:
        footer.append(_provenance_footer(prices))
    if footer:
        blocks.append("\n".join(footer))

    return "\n\n".join(blocks)


# ── sections ──────────────────────────────────────────────────────────────


def _render_drawdown(risk: RiskSummary) -> str:
    dd: DrawdownInfo = risk.drawdown
    lines = ["=== DRAWDOWN (investment, time-weighted) ==="]
    lines.append(
        f"Max drawdown:      {_ci_pct(risk.max_drawdown_ci)}"
    )
    rec = dd.recovery_date.isoformat() if dd.recovery_date else "not yet recovered"
    lines.append(
        f"  peak {dd.peak_date} → trough {dd.trough_date} → {rec}  "
        f"({dd.duration_days} days)"
    )
    lines.append(
        f"Ulcer index:       {risk.ulcer_index * 100:.2f}%   "
        f"CDaR (worst 5%):  {risk.cdar * 100:.2f}%"
    )
    underwater = dd.time_underwater_pct * 100
    lines.append(
        f"You've spent {underwater:.0f}% of this period below a previous high."
    )
    lines.extend(_noisy_note(risk))
    return "\n".join(lines)


def _noisy_note(risk: RiskSummary) -> list[str]:
    """The shared 'treat as noisy' warning, shown on every metric block."""
    if not risk.is_noisy:
        return []
    return [
        f"  ⚠ based on {risk.n_days} return-days (< {NOISY_THRESHOLD_DAYS} ≈ 2y); "
        "treat these figures as noisy."
    ]


def _render_risk_adjusted(risk: RiskSummary) -> str:
    lines = ["=== RISK-ADJUSTED (annualized, 252-day basis, risk-free 0%, ± bootstrap CI) ==="]
    lines.append(f"Sharpe:   {_ci_ratio(risk.sharpe)}")
    lines.append(f"Sortino:  {_ci_ratio(risk.sortino)}")
    lines.append(f"Calmar:   {_ci_ratio(risk.calmar)}")
    lines.extend(_noisy_note(risk))
    return "\n".join(lines)


def _render_returns(returns: ReturnsSummary, true_twr: float | None) -> str:
    lines = ["=== RETURNS (annualized, 252-day basis) ==="]
    lines.append(
        f"Period: {returns.period_start} → {returns.asof_date} "
        f"({returns.period_days} days, ~{returns.period_days / 365.25:.2f}y)"
    )
    lines.append(f"Time-weighted (true TWR):                {_pct_or_na(true_twr)}")
    lines.append(
        f"Money-weighted (IRR):                    "
        f"{_pct_or_na(returns.money_weighted_annualized)}"
    )
    lines.append(
        f"Modified Dietz (approx TWR):             "
        f"{_pct_or_na(returns.modified_dietz_annualized)}"
    )
    if any(
        v is None or not math.isfinite(v)
        for v in (true_twr, returns.money_weighted_annualized,
                  returns.modified_dietz_annualized)
    ):
        lines.append("  (n/a = period too short to annualize, or no real solution)")
    return "\n".join(lines)


def _render_holdings(state: DerivedState, prices: dict[str, PriceRow]) -> str:
    held = state.held()
    lines = ["=== HOLDINGS ==="]
    lines.append(
        f"{'ticker':7}{'shares':>10}{'avg cost':>10}{'price':>9}"
        f"{'mkt value':>13}{'unreal':>12}{'realized':>12}"
    )
    lines.append("-" * 73)

    total_mkt = 0.0
    total_unreal = 0.0
    n_priced = 0
    for tk in sorted(held):
        p = held[tk]
        price = prices.get(tk)
        if price is not None:
            mkt = p.shares * price.close
            unreal = mkt - p.cost_basis
            total_mkt += mkt
            total_unreal += unreal
            n_priced += 1
            price_str, mkt_str, unreal_str = (
                f"{price.close:9.2f}", f"{mkt:13.2f}", f"{unreal:+12.2f}"
            )
        else:
            price_str, mkt_str, unreal_str = f"{_NA:>9}", f"{_NA:>13}", f"{_NA:>12}"
        lines.append(
            f"{tk:7}{p.shares:10.3f}{p.avg_cost:10.2f}{price_str}"
            f"{mkt_str}{unreal_str}{state.realized[tk]:+12.2f}"
        )
    lines.append("-" * 73)

    cost = state.total_cost_basis()
    real = state.total_realized()
    fees = state.total_fees()
    lines.append(f"Total cost basis (held): ${cost:,.2f}")
    if prices and n_priced > 0:
        lines.append(f"Market value (priced):   ${total_mkt:,.2f}")
        lines.append(f"Unrealized P&L:          ${total_unreal:+,.2f}")
    lines.append(f"Realized P&L (sells+div): ${real:+,.2f}")
    lines.append(f"Fees paid (informational): ${fees:,.2f}")
    if prices and n_priced > 0:
        # Fees are already netted into cost_basis (buys) and realized (sells);
        # the bottom line must NOT subtract them again.
        lines.append(f"Net P&L (unrealized + realized): ${total_unreal + real:+,.2f}")
    return "\n".join(lines)


def _format_timedelta(td: timedelta) -> str:
    total = int(td.total_seconds())
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m"
    if total < 86400:
        return f"{total / 3600:.1f}h"
    return f"{total / 86400:.1f}d"


def _provenance_footer(prices: dict[str, PriceRow]) -> str:
    """One-line summary of where the prices came from + how fresh they are."""
    by_source: Counter[str] = Counter(p.source for p in prices.values())
    src_parts = ", ".join(f"{n} {s}" for s, n in sorted(by_source.items()))
    ages = [p.cache_age for p in prices.values()]
    oldest = max(ages) if ages else timedelta(0)
    newest = min(ages) if ages else timedelta(0)
    if oldest == newest:
        fresh = _format_timedelta(newest)
    else:
        fresh = f"{_format_timedelta(newest)} .. {_format_timedelta(oldest)} old"
    asof_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"Prices: {src_parts}  (age: {fresh} as of {asof_now})"
