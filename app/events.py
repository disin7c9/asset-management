"""Transaction-log events: typed model + ghostfolio-format CSV loader.

The append-only event log is the source of truth. Holdings, cost basis, and
P&L are *derived* from these events — never stored.

Float arithmetic is used throughout v0 to match the hand-verified golden
values in `tests/test_derive.py`. Upgrade to Decimal later if precision bites.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Literal, get_args

Action = Literal["buy", "sell", "dividend", "fee", "interest"]
VALID_ACTIONS: frozenset[str] = frozenset(get_args(Action))

# Same-day order: buys first, dividends/fees/interest next, sells last.
# Prevents a same-day SELL-before-BUY in the CSV from triggering a spurious
# "no shares held" error when the day's net activity is valid.
_ACTION_ORDER: dict[str, int] = {
    "buy": 0,
    "dividend": 1,
    "fee": 1,
    "interest": 1,
    "sell": 2,
}

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    ["Date", "Code", "Action", "Quantity", "Price", "Fee"]
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Event:
    """One row of the transaction log.

    `price` is per-share for trades; for dividends/interest the cash amount
    lives in `cash` instead (the CSV's Price column is mapped there at load
    time, so `price` stays semantically a per-share number).
    """

    date: date
    ticker: str
    action: Action
    quantity: float
    price: float
    fee: float
    cash: float = 0.0
    currency: str = "USD"
    source: str = "MANUAL"
    note: str = ""


def _parse_action(raw: str) -> Action:
    value = raw.strip().lower()
    if value not in VALID_ACTIONS:
        msg = f"unknown action {raw!r}; expected one of {sorted(VALID_ACTIONS)}"
        raise ValueError(msg)
    # mypy needs the cast; VALID_ACTIONS guarantees membership.
    return value  # type: ignore[return-value]


def _to_float(raw: str | None, *, default: float = 0.0) -> float:
    """Lenient float parser: empty / whitespace / None → default."""
    if raw is None:
        return default
    s = raw.strip()
    if not s:
        return default
    return float(s)


def _parse_date(raw: str) -> date:
    """Accept any ISO 8601 date (incl. compact YYYYMMDD); reject everything else clearly."""
    s = raw.strip()
    try:
        return date.fromisoformat(s)
    except ValueError as exc:
        msg = (
            f"could not parse date {raw!r}; expected ISO 8601 "
            "(e.g. '2024-03-01' or '20240301'). Non-ISO formats like "
            "'01/03/2024' or '01.03.2024' are ambiguous and not supported."
        )
        raise ValueError(msg) from exc


def _validate_columns(header: Sequence[str] | None, path: Path) -> None:
    if header is None:
        msg = f"{path}: file appears empty (no header row)"
        raise ValueError(msg)
    found = {col.strip() for col in header}
    missing = REQUIRED_COLUMNS - found
    if missing:
        msg = (
            f"{path}: missing required column(s) {sorted(missing)}. "
            f"Expected at least: {sorted(REQUIRED_COLUMNS)}. Found: {sorted(found)}."
        )
        raise ValueError(msg)


def load_events(path: Path) -> list[Event]:
    """Parse a ghostfolio-format CSV into an ordered event list.

    Tolerates:
    - UTF-8 BOM (Excel-saved CSVs)
    - empty Quantity/Fee/Price cells (treated as 0.0)
    - mixed case Action values

    Rejects (with clear errors):
    - missing required columns
    - non-ISO dates
    - unknown action names

    Ordering: by date, then by action (buy → div/fee/interest → sell) so a
    same-day SELL listed before its BUY in the CSV doesn't crash.
    """
    events: list[Event] = []
    # utf-8-sig strips a leading BOM if present.
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        _validate_columns(reader.fieldnames, path)
        for row in reader:
            action = _parse_action(row["Action"])
            price = _to_float(row.get("Price"))
            quantity = _to_float(row.get("Quantity"))
            # For dividend/interest rows the CSV's Price column carries cash,
            # not per-share price. Move it to `cash` to keep the schema honest.
            cash = price if action in ("dividend", "interest") else 0.0
            if action in ("dividend", "interest"):
                price = 0.0

            events.append(
                Event(
                    date=_parse_date(row["Date"]),
                    ticker=row["Code"].strip().upper(),
                    action=action,
                    quantity=quantity,
                    price=price,
                    fee=_to_float(row.get("Fee")),
                    cash=cash,
                    currency=(row.get("Currency") or "USD").strip() or "USD",
                    source=(row.get("DataSource") or "MANUAL").strip() or "MANUAL",
                    note=(row.get("Note") or "").strip(),
                )
            )
    events.sort(key=lambda e: (e.date, _ACTION_ORDER.get(e.action, 1)))
    log.info("loaded %d events from %s", len(events), path)
    return events
