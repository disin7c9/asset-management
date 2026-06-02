"""CLI entry: read a transaction-log CSV and print a holdings + returns summary."""

from __future__ import annotations

import argparse
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
    summarize,
    true_twr_annualized,
    twr_index,
)
from app.risk import RiskSummary, summarize_risk

log = logging.getLogger(__name__)

# Default CSV resolved against the repo root (parent of the `app/` package),
# so the CLI works no matter the current working directory.
_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
_DEFAULT_CSV: Path = _REPO_ROOT / "examples" / "data" / "transactions.csv"
_DEFAULT_CACHE: Path = _REPO_ROOT / "data" / "prices"
_DEFAULT_REPORTS: Path = _REPO_ROOT / "reports"


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    load_dotenv(_REPO_ROOT / ".env")  # RESEND_API_KEY / REPORT_TO for --send
    parser = argparse.ArgumentParser(prog="asset-management", description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=_DEFAULT_CSV,
        help=f"path to the ghostfolio-format transaction CSV (default: {_DEFAULT_CSV})",
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
        help="skip the drawdown/risk panel (needs price history; slower than --no-prices)",
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
    args = parser.parse_args(argv)

    csv_path: Path = args.csv
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

    if not args.no_prices and state.held():
        prices, returns, risk, missing = _compute_prices_returns_risk(
            events, state, args, run
        )

    data = build_report_data(
        state, prices=prices, returns=returns, risk=risk,
        missing_tickers=missing, asof=today,
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
            daily = build_daily_returns(events, series.rows, asof_date=today)
            true_twr = true_twr_annualized(daily)
            risk = summarize_risk(daily, twr_index(daily))

    if series is not None and series.rows:
        # Derive each held ticker's latest price from its series tail (one source).
        fetched_at = datetime.now(timezone.utc)
        for tk in held:
            s = series.rows.get(tk)
            if s is not None and not s.empty:
                prices[tk] = PriceRow(
                    ticker=tk,
                    asof_date=s.index[-1].date(),
                    close=float(s.iloc[-1]),
                    source="series",
                    fetched_at=fetched_at,
                )
        missing = [tk for tk in held if tk not in prices]
    else:
        # No risk panel (or series unavailable): fetch latest prices for held.
        result: PricesResult = fetch_latest(
            list(held), cache_dir=args.cache_dir, online=online
        )
        prices = result.rows
        missing = result.missing
        run["n_prices_fetched"] = len(result.rows)
        run["n_prices_missing"] = len(result.missing)
        run["fallbacks_used"] = result.fallbacks_used
        if result.missing and run["status"] == "ok":
            run["status"] = "partial"

    mkt_value = sum(held[tk].shares * prices[tk].close for tk in prices)
    returns = summarize(events, mkt_value, asof_date=today, true_twr=true_twr)
    return prices, returns, risk, missing


def _log_run_summary(run: dict[str, Any]) -> None:
    """Emit one structured JSON line summarizing the run.

    Schema: {date, source, n_events_replayed, n_prices_fetched, n_prices_missing,
    n_series_fetched, n_series_missing, fallbacks_used, status, report_saved,
    email_sent, email_detail?, error?}.
    """
    log.info("run_summary %s", json.dumps(run, separators=(",", ":")))


if __name__ == "__main__":
    raise SystemExit(main())
