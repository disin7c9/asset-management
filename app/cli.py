"""CLI entry: read a transaction-log CSV and print a holdings + returns summary."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any

from app.derive import derive
from app.events import load_events
from app.log_config import setup_logging
from app.prices import PricesResult, fetch_latest
from app.report import format_summary
from app.returns import ReturnsSummary, summarize

log = logging.getLogger(__name__)

# Default CSV resolved against the repo root (parent of the `app/` package),
# so the CLI works no matter the current working directory.
_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
_DEFAULT_CSV: Path = _REPO_ROOT / "examples" / "data" / "transactions.csv"
_DEFAULT_CACHE: Path = _REPO_ROOT / "data" / "prices"


def main(argv: list[str] | None = None) -> int:
    setup_logging()
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
    args = parser.parse_args(argv)

    csv_path: Path = args.csv
    run: dict[str, Any] = {
        "date": date.today().isoformat(),
        "source": str(csv_path),
        "n_events_replayed": 0,
        "n_prices_fetched": 0,
        "n_prices_missing": 0,
        "fallbacks_used": 0,
        "status": "ok",
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

    prices: dict[str, Any] | None = None
    returns: ReturnsSummary | None = None
    missing: list[str] = []
    if not args.no_prices:
        held = state.held()
        if held:
            result: PricesResult = fetch_latest(
                list(held),
                cache_dir=args.cache_dir,
                online=not args.offline,
            )
            prices = result.rows
            missing = result.missing
            run["n_prices_fetched"] = len(result.rows)
            run["n_prices_missing"] = len(result.missing)
            run["fallbacks_used"] = result.fallbacks_used
            if result.missing:
                run["status"] = "partial"
            mkt_value = sum(
                held[tk].shares * result.rows[tk].close for tk in result.rows
            )
            returns = summarize(events, mkt_value, asof_date=date.today())

    sys.stdout.write(
        format_summary(state, prices=prices, returns=returns, missing_tickers=missing)
        + "\n"
    )
    _log_run_summary(run)
    return 0


def _log_run_summary(run: dict[str, Any]) -> None:
    """Emit one structured JSON line summarizing the run.

    Schema follows CLAUDE.md: {date, source, n_events_replayed, n_prices_fetched,
    n_prices_missing, fallbacks_used, status, error?}.
    """
    log.info("run_summary %s", json.dumps(run, separators=(",", ":")))


if __name__ == "__main__":
    raise SystemExit(main())
