"""CLI entry: read a transaction-log CSV and print a holdings + returns summary."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from app.derive import DerivedState, derive
from app.email import send_report
from app.events import Event, load_events
from app.log_config import setup_logging
from app.prices import PriceRow, PricesResult, fetch_latest, fetch_series
from app.report import (
    ReportData,
    build_report_data,
    render_html,
    render_markdown,
    render_text,
)
from app.returns import (
    ReturnsSummary,
    build_daily_returns,
    price_basis_mismatches,
    summarize,
    true_twr_annualized,
    twr_index,
)
from app.risk import RiskSummary, summarize_risk
from app.backtest import BacktestResult, backtest_compare
from app.strategy import VALID_MODES, Suggestion, load_target, may_suggest, suggest

log = logging.getLogger(__name__)

# Default CSV resolved against the repo root (parent of the `app/` package),
# so the CLI works no matter the current working directory.
_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
_DEFAULT_CSV: Path = _REPO_ROOT / "data" / "sample_data" / "transactions.csv"
_DEFAULT_CACHE: Path = _REPO_ROOT / "data" / "prices"
_DEFAULT_REPORTS: Path = _REPO_ROOT / "reports"


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    load_dotenv(_REPO_ROOT / ".env")  # RESEND_API_KEY / REPORT_TO for --send
    parser = argparse.ArgumentParser(prog="asset-management", description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,  # None = not given → fall back to the bundled example (and warn on real intent)
        help=f"path to the ghostfolio-format transaction CSV (default: the bundled example, {_DEFAULT_CSV.name})",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_DEFAULT_CACHE,
        help=f"on-disk price cache directory (default: {_DEFAULT_CACHE})",
    )
    parser.add_argument(
        "--no-prices",
        action="store_true",
        help="skip price fetching (faster; prints holdings + realized P&L only)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="serve from cache only; do not reach the network on a cache miss",
    )
    parser.add_argument(
        "--no-risk",
        action="store_true",
        help="skip the HOLDINGS drawdown/risk panel (needs price history; slower than "
        "--no-prices). Scope is the holdings panel only: an explicitly-requested "
        "--backtest still reports its own risk metrics.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="also write the brief as markdown to <reports-dir>/<asof>.md",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=_DEFAULT_REPORTS,
        help=f"directory for saved markdown briefs (default: {_DEFAULT_REPORTS})",
    )
    parser.add_argument(
        "--send",
        action="store_true",
        help="email the brief as HTML via Resend (needs RESEND_API_KEY + REPORT_TO)",
    )
    parser.add_argument(
        "--rebalance",
        choices=sorted(VALID_MODES),
        default=None,
        help="emit buy/sell suggestions toward --target using this rule",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help="target-allocation CSV (Ticker,Weight); REQUIRED with --rebalance. A target "
        "is a COMPLETE spec: held tickers not listed are sold to $0. Bootstrap one with "
        "--dump-target, or use data/sample_data/target.csv for the bundled example.",
    )
    parser.add_argument(
        "--dump-target",
        type=Path,
        default=None,
        metavar="PATH",
        help="write your CURRENT allocation (held × price weights) to PATH as a target CSV, "
        "then edit it toward your desired mix (a real-universe starting point)",
    )
    parser.add_argument(
        "--new-cash",
        type=float,
        default=0.0,
        help="new cash to deploy (for cash_flow_only / fixed_dca; also fed to to_total)",
    )
    parser.add_argument(
        "--band",
        type=float,
        default=0.05,
        help="drift threshold for --rebalance bands, as a fraction (default: 0.05 = 5pp)",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="notional $10k historical simulation of --target: rebalanced vs buy-and-hold",
    )
    parser.add_argument(
        "--backtest-start",
        type=date.fromisoformat,
        default=None,
        metavar="YYYY-MM-DD",
        help="start date for --backtest (default: earliest date all target tickers have prices)",
    )
    parser.add_argument(
        "--rebalance-every",
        choices=("monthly", "quarterly", "annually"),
        default="quarterly",
        help="rebalance schedule for the --backtest rebalanced leg (default: quarterly)",
    )
    args = parser.parse_args(argv)
    if args.new_cash < 0:
        parser.error("--new-cash must be >= 0")
    if (args.rebalance or args.backtest) and args.target is None:
        parser.error(
            "--rebalance/--backtest require --target (bootstrap one from your holdings "
            "with --dump-target PATH, or pass data/sample_data/target.csv for the example)"
        )
    if args.backtest_start is not None and args.backtest_start > date.today():
        parser.error("--backtest-start must not be in the future")
    # Footgun guard: a real-intent flag with no --csv (args.csv is None) silently
    # uses the bundled example portfolio, so the HOLDINGS panel would show the
    # sample (not your data) next to your real target/backtest. Warn — keyed on
    # "was --csv supplied?", so an explicit sample path doesn't misfire.
    if args.csv is None and (
        args.rebalance or args.backtest or args.save or args.send or args.dump_target
    ):
        log.warning(
            "no --csv given: using the bundled EXAMPLE portfolio (%s) — the HOLDINGS "
            "panel is the example, not your data. Pass --csv your.csv for your own book.",
            _DEFAULT_CSV.name,
        )

    csv_path: Path = args.csv if args.csv is not None else _DEFAULT_CSV
    today = date.today()  # one as-of date for the title, the filename, and the log
    run: dict[str, Any] = {
        "date": today.isoformat(),
        "source": str(csv_path),
        "n_events_replayed": 0,
        "n_prices_fetched": 0,
        "n_prices_missing": 0,
        "n_series_fetched": 0,
        "n_series_missing": 0,
        "fallbacks_used": 0,
        "status": "ok",
        "report_saved": None,
        "email_sent": False,
        "email_detail": None,
        "rebalance": None,
        "backtest": None,
    }

    if not csv_path.exists():
        log.error("transaction CSV not found: %s", csv_path)
        run["status"] = "error"
        run["error"] = "csv_not_found"
        _log_run_summary(run)
        return 2

    try:
        events = load_events(csv_path)
        run["n_events_replayed"] = len(events)
        state = derive(events)
    except (ValueError, KeyError) as exc:
        log.error("failed to process %s: %s", csv_path, exc)
        run["status"] = "error"
        run["error"] = str(exc)
        _log_run_summary(run)
        return 2

    prices: dict[str, PriceRow] | None = None
    returns: ReturnsSummary | None = None
    risk: RiskSummary | None = None
    missing: list[str] = []
    twr_excluded: list[str] = []

    if not args.no_prices and state.held():
        prices, returns, risk, missing, twr_excluded = _compute_prices_returns_risk(
            events, state, args, run
        )

    if args.dump_target:
        _dump_target(state, prices, args.dump_target)

    suggestions: list[Suggestion] | None = None
    if args.rebalance:
        suggestions = _compute_suggestions(state, prices, args, run)

    backtest: BacktestResult | None = None
    if args.backtest:
        backtest = _compute_backtest(args, run)

    data = build_report_data(
        state, prices=prices, returns=returns, risk=risk,
        suggestions=suggestions, backtest=backtest, missing_tickers=missing,
        asof=today, generated_at=datetime.now(timezone.utc), twr_excluded=twr_excluded,
    )

    sys.stdout.write(render_text(data) + "\n")
    save_path = args.reports_dir / f"{data.asof_date}.md" if args.save else None
    delivered = _deliver(data, run, save_path=save_path, send=args.send)
    _log_run_summary(run)
    # A requested sink that failed → non-zero exit so a scheduler (cron) alerts,
    # even though the brief was already printed to stdout.
    return 0 if delivered else 1


def _deliver(
    data: ReportData,
    run: dict[str, Any],
    *,
    save_path: Path | None,
    send: bool,
) -> bool:
    """Route the built report to the optional sinks (markdown file, email).

    stdout already happened in ``main``; this handles only the ``--save`` /
    ``--send`` sinks and records each outcome in ``run`` for the structured log
    line. Delivery failures are **recorded, not raised** — the brief was already
    printed — but the return value is ``False`` if any *requested* sink failed,
    so the caller can exit non-zero for a scheduler to notice.
    """
    ok = True
    if save_path is not None:
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(render_markdown(data), encoding="utf-8")
            run["report_saved"] = str(save_path)
            log.info("saved markdown brief to %s", save_path)
        except OSError as exc:
            ok = False
            log.error("failed to save markdown brief to %s: %s", save_path, exc)
    if send:
        result = send_report(subject=data.title, html=render_html(data))
        run["email_sent"] = result.sent
        run["email_detail"] = result.detail
        if not result.sent:
            ok = False
    if not ok and run["status"] == "ok":
        run["status"] = "partial"
    return ok


def _compute_prices_returns_risk(
    events: list[Event],
    state: DerivedState,
    args: argparse.Namespace,
    run: dict[str, Any],
) -> tuple[
    dict[str, PriceRow] | None,
    ReturnsSummary | None,
    RiskSummary | None,
    list[str],
    list[str],
]:
    """Fetch prices, derive returns + risk. Single price source per mode.

    When the risk panel is on we fetch price *history* once and derive each
    held ticker's latest price from its series tail — so MWR/Modified Dietz
    and the true-TWR share one price universe (no spot-vs-history mismatch)
    and we avoid a second network round-trip. With --no-risk we fetch only
    the latest prices.
    """
    held = state.held()
    today = date.today()
    online = not args.offline
    prices: dict[str, PriceRow] = {}
    risk: RiskSummary | None = None
    true_twr: float | None = None
    twr_excluded: list[str] = []

    series = None
    if not args.no_risk and events:
        traded = sorted({ev.ticker for ev in events})
        start = min(ev.date for ev in events) - timedelta(days=5)
        series = fetch_series(traded, start, today, cache_dir=args.cache_dir, online=online)
        run["n_series_fetched"] = len(series.rows)
        run["n_series_missing"] = len(series.missing)
        if series.missing and run["status"] == "ok":
            run["status"] = "partial"
        if series.rows:
            # Guard against unhandled stock splits: the log holds raw share counts
            # but yfinance prices are split-adjusted, so a ticker that split during
            # its holding period would fabricate a return in the time-weighted
            # series. Exclude such a ticker from TWR + risk (and warn) — the honest
            # stopgap until v1.x adjusts share counts for corporate actions.
            twr_series = series.rows
            twr_excluded = price_basis_mismatches(events, series.rows)
            if twr_excluded:
                log.warning(
                    "excluding %s from TWR & risk: execution price disagrees with the "
                    "split-adjusted price history (likely an unhandled stock split). The "
                    "time-weighted series mixes raw share counts with adjusted prices and "
                    "would fabricate a return. MWR / Modified Dietz are unaffected; full "
                    "corporate-action handling is a v1.x item.",
                    ", ".join(twr_excluded),
                )
                twr_series = {
                    tk: s for tk, s in series.rows.items() if tk not in twr_excluded
                }
            daily = build_daily_returns(events, twr_series, asof_date=today)
            true_twr = true_twr_annualized(daily)
            risk = summarize_risk(daily, twr_index(daily))

    if series is not None and series.rows:
        # Derive each held ticker's latest price from its series tail, carrying the
        # series' REAL provenance (cache/yfinance/stooq + true fetch time) — not a
        # fabricated "series"/now() stamp, so the footer's source + age are honest.
        for tk in held:
            s = series.rows.get(tk)
            if s is not None and not s.empty:
                source, fetched_at = series.provenance.get(
                    tk, ("cache", datetime.now(timezone.utc))
                )
                prices[tk] = PriceRow(
                    ticker=tk,
                    asof_date=s.index[-1].date(),
                    close=float(s.iloc[-1]),
                    source=source,
                    fetched_at=fetched_at,
                )
        run["fallbacks_used"] = series.fallbacks_used
    else:
        # No risk panel (or series unavailable): fetch latest prices for held.
        result: PricesResult = fetch_latest(
            list(held), cache_dir=args.cache_dir, online=online
        )
        prices = result.rows
        run["fallbacks_used"] = result.fallbacks_used

    # One definition of "priced" for the whole report: a held ticker with a
    # positive, usable close — what market value (and so MWR / Modified Dietz)
    # actually need. A ticker present in `prices` but with a non-positive or NaN
    # close is dropped from value, so it must count as unpriced here too; else a
    # partial book would be scored as fully priced and the money-weighted figures
    # would print a confident wrong number. Counters + partial-status are set here
    # once, for both branches (the series counts are recorded by the caller).
    priced_held = _held_market_value(state, prices)
    prices = {tk: prices[tk] for tk in priced_held}  # drop non-positive/NaN quotes
    missing = [tk for tk in held if tk not in priced_held]
    run["n_prices_fetched"] = len(priced_held)
    run["n_prices_missing"] = len(missing)
    if missing and run["status"] == "ok":
        run["status"] = "partial"
    mkt_value = sum(priced_held.values())
    returns = summarize(
        events, mkt_value, asof_date=today, true_twr=true_twr, fully_priced=not missing
    )
    return prices, returns, risk, missing, twr_excluded


def _held_market_value(
    state: DerivedState, prices: dict[str, PriceRow]
) -> dict[str, float]:
    """Per-held-ticker market value (shares × close), priced tickers only.

    The single source of held-value math for the CLI (holdings sizing, returns,
    suggestions, and --dump-target all go through here), so a future valuation
    change lands in one place. A ticker absent from `prices` or with a
    non-positive close is omitted (no usable price)."""
    held = state.held()
    return {
        tk: held[tk].shares * prices[tk].close
        for tk in held
        if tk in prices and prices[tk].close > 0
    }


def _compute_suggestions(
    state: DerivedState,
    prices: dict[str, PriceRow] | None,
    args: argparse.Namespace,
    run: dict[str, Any],
) -> list[Suggestion] | None:
    """Load the target, ensure a usable price for every (held ∪ target) ticker, run the rule.

    Suggestions size trades in shares, so each ticker needs a positive price: held
    tickers are already priced; target buy-ins are fetched on demand (counted into
    the run summary like any other price fetch). With --no-prices there is nothing
    to size against, so we skip. **If any held ticker lacks a usable price we skip
    entirely** rather than compute weights over a partial portfolio (that would
    understate the total and emit confidently-wrong trades). A bad/missing target
    file is non-fatal — the rest of the brief still prints.
    """
    if not may_suggest(args.rebalance):
        # The discipline-vs-edge gate: an edge strategy must pass a walk-forward
        # backtest before it may suggest. No edge strategies exist in v1, so this
        # is a dormant safety guard (and refuses unknown modes).
        log.warning(
            "--rebalance %s is an edge strategy and must pass a walk-forward backtest "
            "before it may suggest (not implemented in v1)", args.rebalance,
        )
        run["rebalance"] = "skipped: unvalidated edge strategy"
        return None
    if args.rebalance == "bands" and args.new_cash > 0:
        log.warning("--rebalance bands ignores --new-cash (it rebalances existing holdings)")
    if args.rebalance in ("fixed_dca", "cash_flow_only") and args.new_cash <= 0:
        log.warning(
            "--rebalance %s deploys new cash but --new-cash is 0 → nothing to invest "
            "(every line will be HOLD); pass --new-cash N", args.rebalance,
        )
    if not args.target.exists():
        log.error("--rebalance: target file not found: %s", args.target)
        run["rebalance"] = "skipped: no target file"
        return None
    try:
        target = load_target(args.target)
    except (ValueError, OSError) as exc:  # bad contents OR an unreadable path/dir
        log.error("--rebalance: %s", exc)
        run["rebalance"] = "skipped: bad target"
        return None

    combined: dict[str, PriceRow] = dict(prices or {})
    held = state.held()
    omitted_held = sorted(tk for tk in held if tk not in target)
    if omitted_held:
        log.warning(
            "--rebalance: target omits held tickers; they are treated as 0%% "
            "(sold to $0; bands sells only past the band) — add them to keep them: %s",
            ", ".join(omitted_held),
        )
    need = sorted((set(target) | set(held)) - set(combined))
    if need:
        if args.no_prices:
            log.warning("--rebalance needs prices to size trades; remove --no-prices")
            run["rebalance"] = "skipped: --no-prices"
            return None
        result = fetch_latest(need, cache_dir=args.cache_dir, online=not args.offline)
        combined.update(result.rows)
        run["n_prices_fetched"] += len(result.rows)
        run["fallbacks_used"] += result.fallbacks_used
        if result.missing:
            run["n_prices_missing"] += len(result.missing)
            if run["status"] == "ok":
                run["status"] = "partial"
            log.warning("--rebalance: no price for: %s", ", ".join(result.missing))

    price_per_share = {tk: row.close for tk, row in combined.items() if row.close > 0}
    unpriced_held = sorted(tk for tk in held if tk not in price_per_share)
    if unpriced_held:
        log.warning(
            "--rebalance skipped: held tickers lack a usable price: %s",
            ", ".join(unpriced_held),
        )
        run["rebalance"] = "skipped: unpriced holdings"
        return None

    held_value = _held_market_value(state, combined)
    sugg = suggest(
        args.rebalance, held_value, price_per_share, target,
        new_cash=args.new_cash, band=args.band,
    )
    run["rebalance"] = args.rebalance
    return sugg


def _dump_target(
    state: DerivedState,
    prices: dict[str, PriceRow] | None,
    path: Path,
) -> None:
    """Write the user's CURRENT allocation as a target CSV they can edit.

    Weights are current market-value shares (held × price), so the file already
    lists the real universe; the user edits the numbers toward their desired mix.
    Needs prices (skips under --no-prices); unpriced holdings are noted, not guessed.
    """
    if not prices:
        log.warning("--dump-target needs prices; remove --no-prices")
        return
    values = _held_market_value(state, prices)
    total = sum(values.values())
    if total <= 0:
        log.warning("--dump-target: no priced holdings to write")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Ticker", "Weight"])
        # Deterministic order (value desc, then ticker). Critically, a held
        # position must NEVER serialize as 0 — on reload a 0 means a deliberate
        # exit, so a no-edit round-trip would sell it. Widen precision for tiny
        # weights and floor to a strictly-positive value.
        for tk, val in sorted(values.items(), key=lambda kv: (-kv[1], kv[0])):
            pct = val / total * 100
            w_str = f"{pct:.2f}"
            if float(w_str) == 0.0:
                w_str = f"{max(pct, 1e-6):.6f}"
            writer.writerow([tk, w_str])
    log.info(
        "wrote current allocation (%d tickers) to %s — edit toward your target",
        len(values), path,
    )
    omitted = sorted(tk for tk in state.held() if tk not in values)
    if omitted:
        log.warning("--dump-target: skipped unpriced holdings: %s", ", ".join(omitted))


def _compute_backtest(
    args: argparse.Namespace, run: dict[str, Any]
) -> BacktestResult | None:
    """Notional backtest of the target: fetch its price history and simulate the
    rebalanced leg vs buy-and-hold. Independent of the user's holdings (notional);
    a bad target or missing history is non-fatal — the rest of the brief prints."""
    try:
        target = load_target(args.target)
    except (ValueError, OSError) as exc:
        log.error("--backtest: %s", exc)
        run["backtest"] = "skipped: bad target"
        return None
    today = date.today()
    lookback = args.backtest_start or (today - timedelta(days=3653))  # ~10y of history
    series = fetch_series(
        sorted(target), lookback, today, cache_dir=args.cache_dir, online=not args.offline
    )
    if not series.rows:
        log.warning("--backtest: no price history for the target tickers")
        run["backtest"] = "skipped: no prices"
        return None
    result = backtest_compare(
        series.rows, target, schedule=args.rebalance_every,
        start=args.backtest_start, end=today, provenance=series.provenance,
    )
    if result is None:
        run["backtest"] = "skipped: insufficient history"
        return None
    run["backtest"] = args.rebalance_every
    return result


def _log_run_summary(run: dict[str, Any]) -> None:
    """Emit one structured JSON line summarizing the run.

    Schema: {date, source, n_events_replayed, n_prices_fetched, n_prices_missing,
    n_series_fetched, n_series_missing, fallbacks_used, status, report_saved,
    email_sent, email_detail?, rebalance, backtest, error?}.
    """
    log.info("run_summary %s", json.dumps(run, separators=(",", ":")))


if __name__ == "__main__":
    raise SystemExit(main())
