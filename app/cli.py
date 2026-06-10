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

from app.corporate_actions import adjust_for_splits
from app.derive import DerivedState, derive
from app.email import send_report
from app.events import CASH_TICKER, Event, load_events, load_target
from app.log_config import setup_logging
from app.prices import (
    PriceRow,
    PricesResult,
    SeriesResult,
    fetch_latest,
    fetch_series,
    fetch_splits,
)
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
    pnl_curve,
    price_basis_mismatches,
    summarize,
    true_twr_annualized,
    twr_index,
    value_curve,
)
from app.risk import DollarDrawdown, RiskSummary, dollar_drawdown, summarize_risk
from app.allocate import VALID_RULES, allocation_kind, apply_caps, equal_weight, inverse_vol
from app.backtest import BacktestResult, backtest_compare
from app.strategy import VALID_MODES, Suggestion, may_suggest, suggest

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
    parser.add_argument(
        "--allocate",
        choices=sorted(VALID_RULES),
        default=None,
        help="propose a target allocation over your CURRENT holdings with this rule "
        "(equal_weight | inverse_vol): prints the weights, and writes them to --allocate-out "
        "if given. Propose-only — review the file, then run --backtest / --rebalance against it "
        "in a SEPARATE command.",
    )
    parser.add_argument(
        "--allocate-out",
        type=Path,
        default=None,
        metavar="PATH",
        help="write the --allocate proposal to PATH as a target CSV (distinct from "
        "--dump-target, which always writes your CURRENT holdings)",
    )
    parser.add_argument(
        "--allocate-cap",
        type=float,
        default=None,
        metavar="FRAC",
        help="per-asset weight ceiling for --allocate (e.g. 0.30), excess redistributed",
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
    if args.allocate_cap is not None and not 0.0 < args.allocate_cap <= 1.0:
        # A fraction, not a percent: "30" (meaning 30%) would make the cap never bind
        # (cap*n ≥ 1 always) and silently do nothing — reject it loudly instead.
        parser.error("--allocate-cap must be a fraction in (0, 1] (e.g. 0.30 for 30%)")
    if args.allocate and (args.rebalance or args.backtest):
        # propose ≠ act/simulate: keep them separate so you never emit orders or a
        # simulation against a target you haven't reviewed. Do it in two commands.
        parser.error(
            "--allocate is propose-only; run it first (optionally with --allocate-out FILE), "
            "then --backtest / --rebalance --target FILE in a SEPARATE command"
        )
    if args.allocate_out is not None and not args.allocate:
        parser.error("--allocate-out has no effect without --allocate")
    # Footgun guard: a real-intent flag with no --csv (args.csv is None) silently
    # uses the bundled example portfolio, so the HOLDINGS panel would show the
    # sample (not your data) next to your real target/backtest. Warn — keyed on
    # "was --csv supplied?", so an explicit sample path doesn't misfire.
    if args.csv is None and (
        args.rebalance or args.backtest or args.save or args.send
        or args.dump_target or args.allocate
    ):
        log.warning(
            "no --csv given: using the bundled EXAMPLE portfolio (%s) — the HOLDINGS "
            "panel is the example, not your data. Pass --csv your.csv for your own book.",
            _DEFAULT_CSV.name,
        )

    csv_path: Path = args.csv if args.csv is not None else _DEFAULT_CSV
    # Read-only invariant: a generated target CSV must never overwrite the transaction
    # log (the irreplaceable source of truth). Both writers take an explicit path, so a
    # mistyped --dump-target / --allocate-out could otherwise clobber the input.
    src_resolved = csv_path.resolve()
    for flag, out in (("--dump-target", args.dump_target), ("--allocate-out", args.allocate_out)):
        if out is not None and out.resolve() == src_resolved:
            parser.error(
                f"{flag} must not point at the transaction CSV ({csv_path}); "
                "choose a different output path"
            )
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
        "allocate": None,
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
        # Split-adjust raw share counts (yfinance prices are split-adjusted) so
        # holdings AND the time-weighted series share one basis — the real fix for
        # the NVDA-style corruption. Cache-only under --offline/--no-prices; the
        # price-basis-mismatch guard stays the net for any split we couldn't fetch.
        splits = fetch_splits(
            sorted({ev.ticker for ev in events}),
            cache_dir=args.cache_dir,
            online=not args.offline and not args.no_prices,
        )
        raw_events = events
        events = adjust_for_splits(raw_events, splits)
        adjusted_tickers = sorted({
            a.ticker for a, b in zip(raw_events, events, strict=True) if a != b
        })
        if adjusted_tickers:
            log.info("split-adjusted share counts for: %s", ", ".join(adjusted_tickers))
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
    dollar_dd: DollarDrawdown | None = None
    series: SeriesResult | None = None

    if not args.no_prices and state.held():
        prices, returns, risk, missing, twr_excluded, dollar_dd, series = (
            _compute_prices_returns_risk(events, state, args, run)
        )

    if args.dump_target:
        _dump_target(state, prices, args.dump_target)
    if args.allocate:
        # Reuse the price history already fetched above (no second round-trip).
        _compute_allocation(state, prices, series, args, run, today)

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
        dollar_dd=dollar_dd,
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
    DollarDrawdown | None,
    SeriesResult | None,
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
    dollar_dd: DollarDrawdown | None = None

    series = None
    if not args.no_risk and events:
        # Real securities only — the CASH pseudo-ticker (deposit/withdraw legs) has
        # no price history; fetching it would land in series.missing and falsely
        # flip the run status to "partial" on every book that holds cash flows.
        traded = sorted({ev.ticker for ev in events} - {CASH_TICKER})
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
            # Build the holdings value curve once; both the TWR series and the
            # dollar P&L curve share it (same priced universe, no double work).
            value = value_curve(events, twr_series, today)
            daily = build_daily_returns(events, twr_series, asof_date=today, value=value)
            true_twr = true_twr_annualized(daily)
            risk = summarize_risk(daily, twr_index(daily))
            # 'Gains given back' — the flow-neutral dollar P&L drawdown over the
            # same priced universe (deposits/withdrawals/trades cancel).
            dollar_dd = dollar_drawdown(pnl_curve(events, twr_series, today, value=value))

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
    if missing or twr_excluded:
        # Incomplete P&L curve → suppress the felt-dollar drawdown (as we do MWR /
        # Modified Dietz). Either a held ticker is unpriced (`missing`), or a split-
        # mismatched ticker was dropped from the curve (`twr_excluded`); in both
        # cases "Gains given back" would silently omit a holding, so print n/a
        # rather than a confident number missing part of the book.
        dollar_dd = None
    mkt_value = sum(priced_held.values())
    returns = summarize(
        events, mkt_value, asof_date=today, true_twr=true_twr, fully_priced=not missing
    )
    return prices, returns, risk, missing, twr_excluded, dollar_dd, series


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


def _write_target_csv(weights: dict[str, float], path: Path) -> bool:
    """Write a ``{ticker: weight-fraction}`` map as a Ticker,Weight target CSV.

    Weights serialize as percentages, deterministic order (weight desc, then
    ticker). A *positive* weight never serializes as 0.00 — on reload a 0 means a
    deliberate exit, so a no-edit round-trip would sell it; tiny weights widen
    precision instead. Returns True on success; an unwritable path is logged and
    returns False (the run survives — a failed sink is reported, not fatal).
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["Ticker", "Weight"])
            for tk, w in sorted(weights.items(), key=lambda kv: (-kv[1], kv[0])):
                pct = w * 100
                w_str = f"{pct:.2f}"
                if float(w_str) == 0.0 and w > 0.0:
                    w_str = f"{max(pct, 1e-6):.6f}"
                writer.writerow([tk, w_str])
    except OSError as exc:
        log.error("could not write target to %s: %s", path, exc)
        return False
    return True


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
    if not _write_target_csv({tk: v / total for tk, v in values.items()}, path):
        return  # write failed (already logged) — non-fatal
    log.info(
        "wrote current allocation (%d tickers) to %s — edit toward your target",
        len(values), path,
    )
    omitted = sorted(tk for tk in state.held() if tk not in values)
    if omitted:
        log.warning("--dump-target: skipped unpriced holdings: %s", ", ".join(omitted))


def _print_proposed_allocation(
    rule: str,
    target: dict[str, float],
    current_values: dict[str, float],
    omitted: list[str],
    wrote_to: Path | None,
) -> None:
    """Print the proposed allocation beside current weights (a review preview)."""
    cur_total = sum(current_values.values())
    lines = [f"=== PROPOSED ALLOCATION: {rule} ({len(target)} holdings) ==="]
    for tk, w in sorted(target.items(), key=lambda kv: (-kv[1], kv[0])):
        cur_w = (current_values.get(tk, 0.0) / cur_total) if cur_total > 0 else 0.0
        lines.append(
            f"  {tk:<6} {w * 100:6.2f}%   (current {cur_w * 100:5.2f}%, "
            f"{(w - cur_w) * 100:+5.1f}pp)"
        )
    if omitted:
        # A held ticker absent from a target reloads as 0 = sell. Surface it here (not
        # just a stderr warning) so the user sees it before acting on the file.
        lines.append(
            "  NOT in target (unpriced / no usable history) — a to_total rebalance "
            f"would SELL these: {', '.join(omitted)}"
        )
    if wrote_to is not None:
        lines.append(f"  wrote -> {wrote_to}")
    lines.append(
        "  review these weights, then act separately: "
        "--backtest --target <file>  or  --rebalance <mode> --target <file>"
    )
    sys.stdout.write("\n".join(lines) + "\n\n")


def _compute_allocation(
    state: DerivedState,
    prices: dict[str, PriceRow] | None,
    series: SeriesResult | None,
    args: argparse.Namespace,
    run: dict[str, Any],
    today: date,
) -> None:
    """Propose a target allocation over the user's CURRENT holdings — a *discipline*
    re-weighting, not new-ticker selection. Prints the weights for review and writes
    them to --allocate-out if given. Backtesting / rebalancing against the result is a
    SEPARATE command the user runs deliberately (propose ≠ simulate ≠ act).

    `series` is the price history already fetched for the brief; inverse_vol reuses it
    (no second round-trip) and only fetches when it's absent (the --no-risk path)."""
    rule = args.allocate
    if allocation_kind(rule) != "discipline":
        # Dormant guard (every CLI choice is discipline); the chokepoint for any
        # future edge allocator (e.g. an optimizer) — it must be validated first.
        log.warning(
            "%r is an edge allocator and must pass a walk-forward backtest before it "
            "may produce a target (not implemented in v1)", rule,
        )
        run["allocate"] = f"refused: {rule} unvalidated edge"
        return
    if not prices:
        log.warning("--allocate needs prices; remove --no-prices")
        run["allocate"] = "skipped: --no-prices"
        return
    values = _held_market_value(state, prices)
    # Never re-weight the CASH pseudo-ticker: it's a cash balance, not a holding to
    # allocate. Today it's excluded incidentally (0 shares → not in held(); unpriced),
    # but make it explicit so a future change (e.g. pricing cash at $1) can't silently
    # pull it into the weights and dilute every real holding.
    priced = sorted(tk for tk in values if tk != CASH_TICKER)
    if not priced:
        log.warning("--allocate: no priced holdings to allocate over")
        run["allocate"] = "skipped: no prices"
        return

    if rule == "equal_weight":
        target = equal_weight(priced)
    else:  # inverse_vol — needs return history to size each holding's volatility
        src = series  # reuse the history already fetched for the brief (no refetch)
        if src is None:  # --no-risk path: nothing fetched yet → fetch now AND record it
            start = today - timedelta(days=400)  # ~13 months → a full year of returns
            src = fetch_series(
                priced, start, today, cache_dir=args.cache_dir, online=not args.offline
            )
            run["n_series_fetched"] += len(src.rows)
            run["n_series_missing"] += len(src.missing)
            run["fallbacks_used"] += src.fallbacks_used
            if src.missing and run["status"] == "ok":
                run["status"] = "partial"
        rows = {
            tk: s for tk in priced
            if (s := src.rows.get(tk)) is not None and not s.empty
        }
        target = inverse_vol(rows)
        dropped = [tk for tk in priced if tk not in rows]
        if dropped:
            log.warning("--allocate inverse_vol: no price history for %s", ", ".join(dropped))

    if not target:
        log.warning("--allocate: could not compute weights")
        run["allocate"] = "skipped: no weights"
        return
    if args.allocate_cap is not None:
        try:
            target = apply_caps(target, args.allocate_cap)
        except ValueError as exc:
            log.error("--allocate-cap: %s", exc)
            run["allocate"] = "skipped: bad cap"
            return

    omitted = sorted(set(state.held()) - set(target))  # held but unpriced / no history
    wrote_to = args.allocate_out
    if wrote_to is not None:
        if _write_target_csv(target, wrote_to):
            log.info("wrote %s target (%d tickers) to %s", rule, len(target), wrote_to)
        else:
            wrote_to = None  # write failed (error already logged) — still show the preview
    _print_proposed_allocation(rule, target, values, omitted, wrote_to)
    run["allocate"] = rule


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
