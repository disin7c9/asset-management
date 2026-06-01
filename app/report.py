"""Presentation layer: render derived state + prices + returns into text.

Pure functions only — no I/O, no logging side-effects. The caller (cli.py)
writes the returned string to stdout or hands it off to a delivery layer.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone

from app.derive import DerivedState
from app.prices import PriceRow
from app.returns import ReturnsSummary

_NA = "n/a"


def _pct_or_na(x: float | None) -> str:
    return _NA if x is None else f"{x * 100:+.2f}%"


def format_summary(
    state: DerivedState,
    prices: dict[str, PriceRow] | None = None,
    returns: ReturnsSummary | None = None,
    *,
    missing_tickers: list[str] | None = None,
) -> str:
    """Build a deterministic plain-text holdings + returns summary."""
    held = state.held()
    prices = prices or {}
    missing_tickers = missing_tickers or []
    lines: list[str] = []
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
            mkt_str = f"{mkt:13.2f}"
            unreal_str = f"{unreal:+12.2f}"
            price_str = f"{price.close:9.2f}"
        else:
            mkt_str = f"{_NA:>13}"
            unreal_str = f"{_NA:>12}"
            price_str = f"{_NA:>9}"
        realized = state.realized[tk]
        lines.append(
            f"{tk:7}{p.shares:10.3f}{p.avg_cost:10.2f}{price_str}"
            f"{mkt_str}{unreal_str}{realized:+12.2f}"
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
        # Fees are already netted into cost_basis (raising it, lowering unrealized)
        # and into realized (sell-fees subtracted there); the bottom line must NOT
        # subtract them again.
        net = total_unreal + real
        lines.append(f"Net P&L (unrealized + realized): ${net:+,.2f}")

    if returns is not None and returns.period_days > 0:
        lines.append("")
        lines.append(
            f"Period: {returns.period_start} → {returns.asof_date} "
            f"({returns.period_days} days, ~{returns.period_days / 365.25:.2f}y)"
        )
        lines.append(
            f"Money-weighted return (IRR, annualized):    "
            f"{_pct_or_na(returns.money_weighted_annualized)}"
        )
        lines.append(
            f"Modified Dietz (annualized, approx TWR):    "
            f"{_pct_or_na(returns.modified_dietz_annualized)}"
        )

    if missing_tickers:
        lines.append("")
        lines.append(f"Prices unavailable for: {', '.join(sorted(missing_tickers))}")

    if prices:
        lines.append("")
        lines.append(_provenance_footer(prices))

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
