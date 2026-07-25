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

import sys
from datetime import date
from pathlib import Path

import quantstats as qs

from app.backtest import backtest_compare, simulate
from app.derive import derive
from app.events import load_events, load_target
from app.prices import fetch_series
from app.returns import build_daily_returns, true_twr_annualized, twr_index
from app.risk import calmar, max_drawdown, sharpe, sortino, summarize_risk

_SAMPLE_CSV = Path("data/sample_data/transactions.csv")
_SAMPLE_TARGET = Path("data/sample_data/target.csv")

# The README claims agreement "to 4 decimals" — so that is the gate, not a 5% band.
# The one legitimate exception is the CAGR row, where quantstats' calendar-day count
# differs from our 365.25 convention; it carries its own stated tolerance.
_TOL = 1e-4
_FAILURES: list[str] = []


def _check(name: str, ours: float, theirs: float, tol: float = _TOL) -> str:
    """Record a comparison and return its flag. Any CHECK exits non-zero at the end."""
    if abs(ours - theirs) < tol:
        return "OK"
    _FAILURES.append(f"{name}: ours={ours:+.6f} quantstats={theirs:+.6f} (tol {tol:g})")
    return "CHECK"


def main() -> None:
    events = load_events(_SAMPLE_CSV)
    derive(events)  # sanity: replay must succeed
    traded = sorted({ev.ticker for ev in events})
    start = min(ev.date for ev in events)
    # RAW basis for the portfolio half: dividends are rows in the transaction log, so a
    # dividend-adjusted close would count every one twice.
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

    def row(name: str, ours: float, theirs: float, note: str = "", tol: float = _TOL) -> None:
        delta = ours - theirs
        flag = _check(name, ours, theirs, tol)
        print(f"  {name:24} ours={ours:+.4f}   quantstats={theirs:+.4f}   Δ={delta:+.4f}  [{flag}] {note}")

    print(f"\nPortfolio: {traded}   return-days={n}   span={span_days}d   total growth={total_growth:.4f}\n")
    print("METRIC (on the same daily-return series)")
    row("Sharpe", our_sharpe, qs_sharpe)
    row("Sortino", our_sortino, qs_sortino)
    row("Max drawdown", our_maxdd, qs_maxdd)
    print()
    print("ANNUALIZED RETURN (basis matters — see note)")
    row("TWR (252-day, ours)", our_twr_252 or float("nan"), twr_252, "← our reported figure")
    # 2e-3: quantstats counts calendar days differently than our 365.25 convention, so this
    # row reconciles the BASIS, not the last decimal. Every other row is held to 1e-4.
    row("CAGR (365-day, qs)", twr_365, qs_cagr, "← quantstats' basis; we match it when recomputed",
        tol=2e-3)
    print(f"\n  Calmar (ours, no qs equiv shown): {our_calmar:+.4f}")
    print("\n  Interpretation: Sharpe/Sortino/MaxDD should match closely (independent libs).")
    print("  The two annualized-return rows differ only by 252-day vs 365-day basis;")
    print("  each 'ours' equals quantstats under the SAME basis → formulas reconcile.")

    # also confirm our summarize_risk panel is internally consistent
    rk = summarize_risk(daily, idx)
    assert rk is not None
    print(f"\n  risk panel n_days={rk.n_days}  noisy={rk.is_noisy}  "
          f"maxDD CI=[{rk.max_drawdown_ci.low:+.4f}, {rk.max_drawdown_ci.high:+.4f}]")

    # ...but the backtest half must reconcile what production actually computes, and
    # `simulate` runs on TOTAL RETURN (it holds funds with no log, so a raw close books every
    # coupon as a loss). Handing it `series.rows` reconciled a curve the tool never produces —
    # and it passed, because both sides were computed from the same wrong basis.
    tr = fetch_series(traded, start, date.today(), basis="total_return")
    _reconcile_backtest(tr.rows)


def _reconcile_backtest(series: dict) -> None:  # type: ignore[type-arg]
    """Validate the backtest engine's leg metrics against quantstats on the SAME
    simulated equity curve. Previously the entire backtest was unvalidated."""
    target = load_target(_SAMPLE_TARGET)
    res = backtest_compare(series, target, schedule="quarterly", bootstrap_n=200)
    if res is None:
        print("\n  [backtest] no usable history — skipped")
        return
    print(f"\nBACKTEST LEGS vs quantstats (same simulated curve · {res.start}→{res.end})")
    for leg, sched in ((res.legs[0], "quarterly"), (res.legs[1], "never")):
        curve = simulate(series, target, schedule=sched)
        d = curve.pct_change().dropna()
        qs_sharpe = float(qs.stats.sharpe(d))
        qs_maxdd = float(qs.stats.max_drawdown(d))
        for name, ours, theirs in (
            (f"{leg.label} Sharpe", leg.risk.sharpe.point, qs_sharpe),
            (f"{leg.label} MaxDD", leg.risk.max_drawdown_ci.point, qs_maxdd),
        ):
            delta = ours - theirs
            flag = _check(name, ours, theirs)
            print(f"  {name:28} ours={ours:+.4f}   quantstats={theirs:+.4f}   Δ={delta:+.4f}  [{flag}]")


if __name__ == "__main__":
    main()
    if _FAILURES:
        print(f"\nRECONCILE FAILED — {len(_FAILURES)} metric(s) outside tolerance:", file=sys.stderr)
        for line in _FAILURES:
            print(f"  {line}", file=sys.stderr)
        sys.exit(1)
    print("\nRECONCILE OK — every metric within tolerance.")
