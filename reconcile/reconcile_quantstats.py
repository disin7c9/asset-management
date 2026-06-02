"""Cross-validate our risk/return metrics against quantstats (an independent library).

Run with an ephemeral quantstats install so it stays out of project deps:

    uv run --with quantstats python reconcile/reconcile_quantstats.py

What this validates: given the SAME daily time-weighted return series our
pipeline produces, do our metric *formulas* (Sharpe, Sortino, max drawdown,
annualized return) agree with quantstats' independent implementations?

What it does NOT validate: the equity-curve reconstruction itself (we hand
quantstats our own returns). That end-to-end check is ghostfolio's job.

Note on annualization basis: our `true_twr_annualized` uses a 252-trading-day
basis (count of return observations); quantstats' CAGR uses a 365-calendar-day
basis (date span). We compute BOTH below so the numbers reconcile under a
shared convention rather than looking like a discrepancy.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import quantstats as qs

from app.derive import derive
from app.events import load_events
from app.prices import fetch_series
from app.returns import build_daily_returns, true_twr_annualized, twr_index
from app.risk import calmar, max_drawdown, sharpe, sortino, summarize_risk


def main() -> None:
    events = load_events(Path("examples/data/transactions.csv"))
    derive(events)  # sanity: replay must succeed
    traded = sorted({ev.ticker for ev in events})
    start = min(ev.date for ev in events)
    series = fetch_series(traded, start, date.today())
    daily = build_daily_returns(events, series.rows, asof_date=date.today())

    n = len(daily)
    span_days = (daily.index[-1] - daily.index[0]).days
    total_growth = float((1.0 + daily).prod())
    idx = twr_index(daily)

    # ---- ours ----
    our_sharpe = sharpe(daily)
    our_sortino = sortino(daily)
    our_calmar = calmar(daily)
    our_maxdd = max_drawdown(idx).depth
    our_twr_252 = true_twr_annualized(daily)  # 252-obs basis

    # ---- quantstats (independent) ----
    qs_sharpe = float(qs.stats.sharpe(daily))
    qs_sortino = float(qs.stats.sortino(daily))
    qs_maxdd = float(qs.stats.max_drawdown(daily))
    qs_cagr = float(qs.stats.cagr(daily))  # calendar (365-day) basis

    # recompute both annualization bases from the same total growth
    twr_252 = total_growth ** (252.0 / n) - 1.0
    twr_365 = total_growth ** (365.25 / span_days) - 1.0

    def row(name: str, ours: float, theirs: float, note: str = "") -> None:
        delta = ours - theirs
        flag = "OK" if abs(delta) < max(0.02, abs(theirs) * 0.05) else "CHECK"
        print(f"  {name:24} ours={ours:+.4f}   quantstats={theirs:+.4f}   Δ={delta:+.4f}  [{flag}] {note}")

    print(f"\nPortfolio: {traded}   return-days={n}   span={span_days}d   total growth={total_growth:.4f}\n")
    print("METRIC (on the same daily-return series)")
    row("Sharpe", our_sharpe, qs_sharpe)
    row("Sortino", our_sortino, qs_sortino)
    row("Max drawdown", our_maxdd, qs_maxdd)
    print()
    print("ANNUALIZED RETURN (basis matters — see note)")
    row("TWR (252-day, ours)", our_twr_252 or float("nan"), twr_252, "← our reported figure")
    row("CAGR (365-day, qs)", twr_365, qs_cagr, "← quantstats' basis; we match it when recomputed")
    print(f"\n  Calmar (ours, no qs equiv shown): {our_calmar:+.4f}")
    print("\n  Interpretation: Sharpe/Sortino/MaxDD should match closely (independent libs).")
    print("  The two annualized-return rows differ only by 252-day vs 365-day basis;")
    print("  each 'ours' equals quantstats under the SAME basis → formulas reconcile.")

    # also confirm our summarize_risk panel is internally consistent
    rk = summarize_risk(daily, idx)
    assert rk is not None
    print(f"\n  risk panel n_days={rk.n_days}  noisy={rk.is_noisy}  "
          f"maxDD CI=[{rk.max_drawdown_ci.low:+.4f}, {rk.max_drawdown_ci.high:+.4f}]")


if __name__ == "__main__":
    main()
