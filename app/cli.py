"""CLI entry: read a transaction-log CSV and print a holdings + returns summary."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

from dotenv import load_dotenv

from app.derive import DerivedState
from app.email import send_report
from app.events import CASH_TICKER, Event, load_target
from app.log_config import setup_logging
from app.metadata import MetadataResult, fetch_metadata
from app.screen import CandidateScreen, screen_candidates
from app.prices import PriceRow, SeriesResult, fetch_latest, fetch_series
from app.report import (
    ReportData,
    Section,
    build_report_data,
    render_html,
    render_markdown,
    render_text,
)
from app.llm import complete, load_config
from app.narrate import (
    build_claim_set,
    build_discovery_claims,
    build_discovery_prompt,
    build_prompt,
    render_narration,
)
from app.discover import Discovery, find_gaps, restrict_to
from app.universe import ROLES, load_universe
from app.returns import ReturnsSummary
from app.risk import DollarDrawdown, RiskSummary
from app.pipeline import (
    compute_prices_returns_risk,
    held_market_value,
    load_book,
    record_series_fetch,
)
from app.allocate import (
    VALID_RULES,
    UnvalidatedEdgeError,
    allocate,
    allocation_kind,
    needs_series,
)
from app.backtest import BacktestResult, RoleCheck, backtest_compare, role_check
from app.strategy import VALID_MODES, Suggestion, may_suggest, suggest

log = logging.getLogger(__name__)

# Paths resolved against the repo root (parent of the `app/` package), so the CLI
# works no matter the current working directory.
_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
_DEFAULT_CACHE: Path = _REPO_ROOT / "data" / "prices"
_DEFAULT_REPORTS: Path = _REPO_ROOT / "reports"


def _env_path(var: str) -> Path | None:
    """Read a path from an env var (the .env personal defaults), or None if unset.

    `~` is expanded; a relative path is repo-relative (that's where .env lives),
    so the default works from any working directory.
    """
    raw = os.environ.get(var, "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    resolved = p if p.is_absolute() else _REPO_ROOT / p
    log.info("using %s from .env: %s", var, resolved)
    return resolved


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    load_dotenv(_REPO_ROOT / ".env")  # RESEND_API_KEY / REPORT_TO for --send
    parser = argparse.ArgumentParser(prog="asset-management", description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,  # None = not given → print a usage hint (no silent sample); the sample is opt-in
        help="path to YOUR ghostfolio-format transaction CSV. Required for the brief and for "
        "--rebalance/--allocate/--dump-target/--save/--send. The bundled example is opt-in: "
        "pass --csv data/sample_data/transactions.csv. (--backtest --target works without it.) "
        "Set ASSET_CSV in .env to make your book the default for book runs.",
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
        "--metadata",
        action="store_true",
        help="add a SECURITIES panel: expense ratio, AUM, liquidity, age, category per "
        "holding (published fund facts via yfinance, cached for 7 days; works with "
        "--offline from cache)",
    )
    parser.add_argument(
        "--screen",
        metavar="TICKERS",
        default=None,
        help="judge NEW candidate tickers against your book (comma-separated, e.g. "
        "QQQM,SCHD): diversifier/cost/liquidity/age/concentration/structure/overlap "
        "checks, each with a named reason. Add --target (or ASSET_TARGET in .env) for "
        "the walk-forward ROLE check — did a 5%% sleeve improve drawdown/vol on a "
        "held-out window? Needs the price pipeline (not compatible with --no-prices/"
        "--no-risk) and is propose-only (no --rebalance/--backtest/--allocate in the "
        "same run).",
    )
    parser.add_argument(
        "--discover",
        metavar="ROLES",
        nargs="?",
        const="",
        default=None,
        help="suggest NEW tickers for the roles your book is light in, from the curated "
        "universe (data/universe.csv), each judged by the same screen. Bare --discover "
        "covers every gap; --discover reit,tips targets specific roles. Needs the price "
        "pipeline; propose-only (no --rebalance/--backtest/--allocate in the same run).",
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
        "--dump-target, or use data/sample_data/target.csv for the bundled example. "
        "Set ASSET_TARGET in .env to make it the default.",
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
        help="absolute drift band for --rebalance bands, as a fraction (default: 0.05 = 5pp)",
    )
    parser.add_argument(
        "--band-rel",
        type=float,
        default=0.25,
        metavar="FRAC",
        help="relative band for --rebalance bands (the 5/25 rule): also act when drift "
        "exceeds this fraction of a ticker's target weight, so a small sleeve isn't given "
        "a band many times its size. Effective band = min(--band, --band-rel x target). "
        "Default 0.25 = 25%%.",
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
    parser.add_argument(
        "--narrate",
        action="store_true",
        help="add a plain-language SUMMARY block (opt-in; needs an LLM backend in "
        ".env: ASSET_NARRATE_PROVIDER/MODEL/KEY). The model writes only prose with "
        "{{tokens}}; every number is substituted and verified from the validated "
        "core (SymGen/PCN) — wording is the model's, the figures are the tool's.",
    )
    args = parser.parse_args(argv)
    if args.new_cash < 0:
        parser.error("--new-cash must be >= 0")
    if args.band_rel <= 0.0:
        parser.error("--band-rel must be > 0 (a fraction of the target weight, e.g. 0.25 = 25%)")
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
    if (args.screen or args.discover is not None) and (args.rebalance or args.backtest or args.allocate):
        # Same discipline: judge candidates first, decide/simulate in a separate run.
        parser.error(
            "--screen/--discover are propose-only; review the verdicts, then act "
            "(--allocate / --rebalance / --backtest) in a SEPARATE command"
        )
    if args.allocate_out is not None and not args.allocate:
        parser.error("--allocate-out has no effect without --allocate")
    # No silent sample fallback: book-dependent actions operate on YOUR holdings, so
    # they require your transaction log. --backtest is notional (target-only) and is
    # exempt; the bundled example is opt-in via an explicit --csv path.
    needs_book = bool(
        args.rebalance or args.allocate or args.dump_target or args.save or args.send
        or args.metadata or args.screen or args.narrate or args.discover is not None
    )
    # Personal defaults from .env (gitignored; loaded above). ASSET_CSV fills --csv
    # for runs that want a book; a pure `--backtest --target` run stays notional
    # (book-free) by contract — pass --csv explicitly to include your book there.
    # ASSET_TARGET fills --target only when a rule needs one. Explicit flags win.
    if args.csv is None and (needs_book or not args.backtest):
        args.csv = _env_path("ASSET_CSV")
    if args.target is None and (args.rebalance or args.backtest or args.screen):
        args.target = _env_path("ASSET_TARGET")  # screen: enables the role check row
    if (args.rebalance or args.backtest) and args.target is None:
        parser.error(
            "--rebalance/--backtest require --target (bootstrap one from your holdings "
            "with --dump-target PATH, or pass data/sample_data/target.csv for the example; "
            "or set ASSET_TARGET in .env)"
        )
    if args.csv is None and needs_book:
        parser.error(
            "this run needs your transaction log — pass --csv PATH "
            "(or the bundled example: --csv data/sample_data/transactions.csv; "
            "or set ASSET_CSV in .env). "
            "Only '--backtest --target FILE' runs without --csv."
        )
    today = date.today()  # one as-of date for the title, the filename, and the log
    run: dict[str, Any] = {
        "date": today.isoformat(),
        "source": str(args.csv) if args.csv is not None else "(no book; backtest-only)",
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
        "metadata": None,
        "screen": None,
        "narrate": None,
        "discover": None,
        "discover_narrate": None,
    }

    # Nothing to do: no book and no notional backtest → guide, don't fabricate a brief.
    if args.csv is None and not args.backtest:
        sys.stdout.write(
            "No portfolio given. Pass --csv PATH to see your brief, e.g.\n"
            "  uv run python -m app --csv your_transactions.csv\n"
            "or try the bundled example:\n"
            "  uv run python -m app --csv data/sample_data/transactions.csv\n"
            "or set ASSET_CSV=path/to/your.csv in .env to make it the default.\n"
            "(to backtest a target without a book: --backtest --target FILE; "
            "see --help for all options)\n"
        )
        return 0

    state = DerivedState()
    events: list[Event] = []
    if args.csv is not None:  # a real book → derive holdings (else: notional backtest only)
        csv_path = args.csv
        # Read-only invariant: a generated target CSV must never overwrite the
        # transaction log. Both writers take an explicit path, so a mistyped
        # --dump-target / --allocate-out could otherwise clobber the input.
        src_resolved = csv_path.resolve()
        for flag, out in (("--dump-target", args.dump_target), ("--allocate-out", args.allocate_out)):
            if out is not None and out.resolve() == src_resolved:
                parser.error(
                    f"{flag} must not point at the transaction CSV ({csv_path}); "
                    "choose a different output path"
                )
        if not csv_path.exists():
            log.error("transaction CSV not found: %s", csv_path)
            run["status"] = "error"
            run["error"] = "csv_not_found"
            _log_run_summary(run)
            return 2
        try:
            # Shared loader: replay → split-adjust → derive (the price-basis-mismatch
            # guard in pipeline stays the net for any split we couldn't fetch). Splits
            # are cache-only under --offline / --no-prices.
            events, state = load_book(
                csv_path,
                args.cache_dir,
                online=not args.offline and not args.no_prices,
            )
            run["n_events_replayed"] = len(events)
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
    daily: pd.Series[float] | None = None

    if not args.no_prices and state.held():
        prices, returns, risk, missing, twr_excluded, dollar_dd, series, daily = (
            _compute_prices_returns_risk(events, state, args, run, today)
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
        backtest = _compute_backtest(args, run, today)

    meta: MetadataResult | None = None
    if args.metadata:
        # Published fund facts for the held tickers (cost/size/liquidity/age).
        # Independent of prices: works under --no-prices, and --offline serves
        # from the 7-day cache. A per-ticker miss degrades the run to partial.
        meta = fetch_metadata(
            sorted(state.held()), cache_dir=args.cache_dir, online=not args.offline
        )
        run["metadata"] = f"{len(meta.rows)} fetched, {len(meta.missing)} missing"
        if meta.missing and run["status"] == "ok":
            run["status"] = "partial"

    cands: list[CandidateScreen] | None = None
    if args.screen:
        cands = _compute_screen(state, args, run, today, daily)

    discovery: Discovery | None = None
    discovery_results: list[CandidateScreen] | None = None
    if args.discover is not None:
        discovery, discovery_results = _compute_discover(state, prices, args, run, today, daily)

    summary_section: Section | None = None
    discovery_summary: Section | None = None
    if args.narrate:
        summary_section = _compute_narration(state, prices, returns, risk, dollar_dd, run)
        if discovery is not None and discovery_results is not None:
            discovery_summary = _compute_discovery_narration(discovery, discovery_results, run)

    data = build_report_data(
        state, prices=prices, returns=returns, risk=risk,
        suggestions=suggestions, backtest=backtest, missing_tickers=missing,
        asof=today, generated_at=datetime.now(timezone.utc), twr_excluded=twr_excluded,
        dollar_dd=dollar_dd, metadata=meta, candidates=cands,
        discovery=discovery, discovery_results=discovery_results,
        discovery_summary=discovery_summary, summary=summary_section,
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
    today: date,
) -> tuple[
    dict[str, PriceRow] | None,
    ReturnsSummary | None,
    RiskSummary | None,
    list[str],
    list[str],
    DollarDrawdown | None,
    SeriesResult | None,
    "pd.Series[float] | None",
]:
    """CLI adapter over `compute_prices_returns_risk` (unpacks the argparse fields)."""
    return compute_prices_returns_risk(
        events, state, no_risk=args.no_risk, offline=args.offline,
        cache_dir=args.cache_dir, today=today, run=run,
    )


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

    held_value = held_market_value(state, combined)
    sugg = suggest(
        args.rebalance, held_value, price_per_share, target,
        new_cash=args.new_cash, band=args.band, band_rel=args.band_rel,
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
    values = held_market_value(state, prices)
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
        # Fast-fail courtesy check: refuse an edge rule BEFORE the prices check or
        # any series fetch, so the run log says "refused" (not "skipped") and no
        # network round-trip is wasted. The dispatcher's gate inside allocate()
        # stays the authoritative one for every caller.
        log.warning(
            "--allocate: %r is an edge allocator and may not produce a target until "
            "a walk-forward backtest validates it", rule,
        )
        run["allocate"] = f"refused: {rule} unvalidated edge"
        return
    if not prices:
        log.warning("--allocate needs prices; remove --no-prices")
        run["allocate"] = "skipped: --no-prices"
        return
    values = held_market_value(state, prices)
    priced = sorted(values)  # CASH never reaches here (not held); allocate() also guards
    if not priced:
        log.warning("--allocate: no priced holdings to allocate over")
        run["allocate"] = "skipped: no prices"
        return

    rows = None
    if needs_series(rule):  # rule weighs on return history (inverse_vol)
        src = series  # reuse the history already fetched for the brief (no refetch)
        if src is None:  # --no-risk path: nothing fetched yet → fetch now AND record it
            start = today - timedelta(days=400)  # ~13 months → a full year of returns
            src = fetch_series(
                priced, start, today, cache_dir=args.cache_dir, online=not args.offline
            )
            record_series_fetch(run, src)
        rows = src.rows

    try:
        # The dispatcher enforces the discipline-vs-edge gate itself — the CLI no
        # longer pre-checks, so a future edge rule is blocked at the chokepoint
        # for every caller, not just this one.
        target = allocate(rule, priced, rows, cap=args.allocate_cap)
    except UnvalidatedEdgeError as exc:
        log.warning("--allocate: %s", exc)
        run["allocate"] = f"refused: {rule} unvalidated edge"
        return
    except ValueError as exc:  # infeasible --allocate-cap (cap * n < 1)
        log.error("--allocate-cap: %s", exc)
        run["allocate"] = "skipped: bad cap"
        return

    if not target:
        log.warning("--allocate: could not compute weights")
        run["allocate"] = "skipped: no weights"
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


def _compute_screen(
    state: DerivedState,
    args: argparse.Namespace,
    run: dict[str, Any],
    today: date,
    daily: "pd.Series[float] | None",
) -> list[CandidateScreen] | None:
    """Judge NEW candidate tickers (from --screen) against the book (propose-only)."""
    tickers = sorted({t.strip().upper() for t in args.screen.split(",") if t.strip()})
    if CASH_TICKER in tickers:
        # The pseudo-ticker is not a screenable security; dropping it here keeps
        # fetch_series from a doomed network round-trip (metadata already skips it).
        log.warning("--screen: %s is the cash pseudo-ticker, not a security; dropped", CASH_TICKER)
        tickers = [t for t in tickers if t != CASH_TICKER]
    if not tickers:
        log.error("--screen: no tickers given")
        run["screen"] = "skipped: no tickers"
        return None
    return _screen_tickers(
        state, tickers, args, run, today, daily, status_key="screen", with_role=True
    )


def _screen_tickers(
    state: DerivedState,
    tickers: list[str],
    args: argparse.Namespace,
    run: dict[str, Any],
    today: date,
    daily: "pd.Series[float] | None",
    *,
    status_key: str,
    with_role: bool,
) -> list[CandidateScreen] | None:
    """Fetch the candidates' price history + metadata, and the held tickers' metadata (the
    overlap test compares look-through holdings), then hand everything to the pure screen.
    Shared by --screen (user tickers) and --discover (universe gap-fillers). Non-fatal: a
    missing pipeline degrades with a logged reason — the rest of the brief still prints.
    ``with_role`` runs the walk-forward role check when a --target is present.
    """
    if daily is None or daily.empty:
        # The diversifier test correlates against YOUR portfolio's return series,
        # which only the full price pipeline produces.
        log.warning("--%s needs the portfolio return series; remove --no-prices/--no-risk", status_key)
        run[status_key] = "skipped: needs the price pipeline"
        return None
    online = not args.offline
    start: date = daily.index[0].date()
    cand_series = fetch_series(tickers, start, today, cache_dir=args.cache_dir, online=online)
    record_series_fetch(run, cand_series)
    held = set(state.held())
    tickers_set = set(tickers)
    meta = fetch_metadata(sorted(tickers_set | held), cache_dir=args.cache_dir, online=online)
    if meta.missing and run["status"] == "ok":
        run["status"] = "partial"
    held_meta = {tk: m for tk, m in meta.rows.items() if tk in held}
    cand_meta = {tk: m for tk, m in meta.rows.items() if tk in tickers_set}

    role: dict[str, RoleCheck] | None = None
    if with_role and args.target is not None:
        # The walk-forward role check (the edge gate's evidence): simulate the
        # target vs target+sleeve per candidate, judged on the held-out window.
        try:
            target = load_target(args.target)
        except (ValueError, OSError) as exc:
            log.error("--%s role check: %s", status_key, exc)
        else:
            tgt_series = fetch_series(
                sorted(set(target) - set(cand_series.rows)), start, today,
                cache_dir=args.cache_dir, online=online,
            )
            record_series_fetch(run, tgt_series)
            sim_series = {**tgt_series.rows, **cand_series.rows}
            role = {tk: role_check(sim_series, target, tk) for tk in tickers}

    results = screen_candidates(
        tickers, cand_series.rows, daily, cand_meta, held_meta, held, asof=today, role=role
    )
    run[status_key] = " ".join(f"{r.ticker}:{r.verdict}" for r in results)
    return results


def _compute_discover(
    state: DerivedState,
    prices: dict[str, PriceRow] | None,
    args: argparse.Namespace,
    run: dict[str, Any],
    today: date,
    daily: "pd.Series[float] | None",
) -> tuple[Discovery | None, list[CandidateScreen] | None]:
    """Suggest tickers for the book's role GAPS from the curated universe, each judged by the
    same screen (propose-only). Bare ``--discover`` covers every gap; ``--discover reit,tips``
    targets specific roles. Needs current prices; non-fatal at every step.
    """
    if not prices:
        log.warning("--discover needs current prices; remove --no-prices")
        run["discover"] = "skipped: needs the price pipeline"
        return None, None
    # The curated universe: ASSET_UNIVERSE overrides (like ASSET_CSV), else the bundled file.
    universe_path = _env_path("ASSET_UNIVERSE") or Path(__file__).resolve().parent.parent / "data" / "universe.csv"
    try:
        universe = load_universe(universe_path)
    except (OSError, ValueError) as exc:
        log.error("--discover: cannot load the universe: %s", exc)
        run["discover"] = "skipped: universe unavailable"
        return None, None

    discovery = find_gaps(state, prices, universe)
    if args.discover:  # a non-empty value → restrict to the named roles
        wanted = [r.strip() for r in args.discover.split(",") if r.strip()]
        gapset = set(discovery.gaps)
        for r in wanted:
            if r not in ROLES:
                log.warning("--discover: %r is not a known role; ignoring", r)
            elif r not in gapset:
                log.info("--discover: %r is already covered in your book; skipping", r)
        chosen = {r for r in wanted if r in gapset}
        if not chosen:
            run["discover"] = "no matching gaps"
            return None, None
        discovery = restrict_to(discovery, chosen)

    if not discovery.candidates:
        log.info("--discover: your book already covers the universe's roles — no gaps to fill")
        run["discover"] = "no gaps"
        return None, None

    results = _screen_tickers(
        state, [c.ticker for c in discovery.candidates], args, run, today, daily,
        status_key="discover", with_role=False,
    )
    return (discovery, results) if results is not None else (None, None)


def _compute_backtest(
    args: argparse.Namespace, run: dict[str, Any], today: date
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


def _compute_narration(
    state: DerivedState,
    prices: dict[str, PriceRow] | None,
    returns: ReturnsSummary | None,
    risk: RiskSummary | None,
    dollar_dd: DollarDrawdown | None,
    run: dict[str, Any],
) -> Section | None:
    """Build the opt-in SUMMARY narration: claim set → LLM prose → SymGen/PCN fence
    → a source-labeled Section. Fail-closed at every step (no backend, no figures, a
    model failure, or a fence rejection) → None, and the plain brief still prints.
    The numbers are the tool's; only the wording is the model's."""
    config = load_config()
    if config is None:
        log.warning("--narrate: no LLM backend configured (set ASSET_NARRATE_* in .env)")
        run["narrate"] = "skipped: not configured"
        return None
    claim_set = build_claim_set(state, prices, returns, risk, dollar_dd=dollar_dd)
    if not claim_set:
        run["narrate"] = "skipped: no figures to narrate"
        return None
    system, user = build_prompt(claim_set, tier=config.tier)
    prose = complete(config, system, user)
    fenced = render_narration(prose, claim_set) if prose else None
    if fenced is None:
        log.warning(
            "--narrate: narration withheld — the model returned nothing, or its output "
            "failed the number fence (a stray digit / unknown token)"
        )
        run["narrate"] = "withheld: empty or failed the fence"
        return None
    provenance = (
        f"— wording by {config.model} ({config.tier} tier); the figures are computed "
        "and verified by the tool, not the model. Not financial advice."
    )
    run["narrate"] = f"{config.model} ({config.tier})"
    return Section("SUMMARY", (fenced, "", provenance), prose=True)


def _compute_discovery_narration(
    discovery: Discovery,
    results: list[CandidateScreen],
    run: dict[str, Any],
) -> Section | None:
    """Opt-in (`--discover --narrate`): rank/explain the screened gap-fillers through the
    SAME fence as the brief SUMMARY — claims → LLM prose → SymGen/PCN → a source-labeled
    note leading the DISCOVERY panel. Fail-closed at every step → None (the plain panel
    still prints). Role-fit only; the verdicts and figures are the tool's, not the model's."""
    config = load_config()
    if config is None:
        log.warning("--discover --narrate: no LLM backend configured (set ASSET_NARRATE_* in .env)")
        run["discover_narrate"] = "skipped: not configured"
        return None
    claim_set = build_discovery_claims(discovery, results)
    if not claim_set:
        run["discover_narrate"] = "skipped: nothing to narrate"
        return None
    system, user = build_discovery_prompt(claim_set, tier=config.tier)
    prose = complete(config, system, user)
    fenced = render_narration(prose, claim_set) if prose else None
    if fenced is None:
        log.warning(
            "--discover --narrate: note withheld — the model returned nothing, or its "
            "output failed the number fence (a stray digit/ticker or an unknown token)"
        )
        run["discover_narrate"] = "withheld: empty or failed the fence"
        return None
    provenance = (
        f"— wording by {config.model} ({config.tier} tier); ranked on role-fit by the "
        "tool's screen, not the model. Propose-only; not financial advice."
    )
    run["discover_narrate"] = f"{config.model} ({config.tier})"
    return Section("DISCOVERY — worth a closer look", (fenced, "", provenance), prose=True)


def _log_run_summary(run: dict[str, Any]) -> None:
    """Emit one structured JSON line summarizing the run.

    Schema: {date, source, n_events_replayed, n_prices_fetched, n_prices_missing,
    n_series_fetched, n_series_missing, fallbacks_used, status, report_saved,
    email_sent, email_detail?, rebalance, backtest, allocate, metadata, screen,
    narrate, discover, discover_narrate, error?}.
    """
    log.info("run_summary %s", json.dumps(run, separators=(",", ":")))


if __name__ == "__main__":
    raise SystemExit(main())
