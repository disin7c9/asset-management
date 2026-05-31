"""CLI entry: read a transaction-log CSV and print a derived holdings summary."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from app.derive import derive
from app.events import load_events
from app.log_config import setup_logging
from app.report import format_summary

log = logging.getLogger(__name__)

# Default CSV resolved against the repo root (parent of the `app/` package),
# so the CLI works no matter the current working directory.
_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
_DEFAULT_CSV: Path = _REPO_ROOT / "examples" / "data" / "transactions.csv"


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(prog="asset-management", description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=_DEFAULT_CSV,
        help=f"path to the ghostfolio-format transaction CSV (default: {_DEFAULT_CSV})",
    )
    args = parser.parse_args(argv)

    csv_path: Path = args.csv
    if not csv_path.exists():
        log.error("transaction CSV not found: %s", csv_path)
        return 2

    try:
        events = load_events(csv_path)
        state = derive(events)
    except (ValueError, KeyError) as exc:
        log.error("failed to process %s: %s", csv_path, exc)
        return 2

    sys.stdout.write(format_summary(state) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
