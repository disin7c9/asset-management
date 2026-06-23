"""Transaction-log events: typed model + loader (our CSV or a Ghostfolio export).

The append-only event log is the source of truth. Holdings, cost basis, and
P&L are *derived* from these events — never stored. `load_events` accepts our native
CSV OR a Ghostfolio JSON activities export (auto-detected; normalized to our schema on
read), so the rest of the pipeline only ever sees one format.

Float arithmetic is used throughout v0 to match the hand-verified golden
values in `tests/test_derive.py`. Upgrade to Decimal later if precision bites.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import dataclass
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal, get_args

Action = Literal["buy", "sell", "dividend", "fee", "interest", "deposit", "withdraw"]
VALID_ACTIONS: frozenset[str] = frozenset(get_args(Action))

# Actions whose cash amount rides in the CSV's Price column (no per-share price):
# income (dividend/interest) and external cash flows (deposit/withdraw).
_CASH_IN_PRICE: frozenset[str] = frozenset(
    {"dividend", "interest", "deposit", "withdraw"}
)

# Pseudo-ticker that carries external cash flows (deposit/withdraw). It is never a
# real security, so it is excluded from price fetches and the priced-value curves.
CASH_TICKER = "CASH"

# Same-day order: fund (deposit) + buys first, income/fees next, sells, then
# withdrawals. Prevents a same-day SELL-before-BUY in the CSV from triggering a
# spurious "no shares held" error when the day's net activity is valid. (Cash
# flows carry no position, so their order only affects readability.)
_ACTION_ORDER: dict[str, int] = {
    "deposit": 0,
    "buy": 0,
    "dividend": 1,
    "fee": 1,
    "interest": 1,
    "sell": 2,
    "withdraw": 3,
}

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    ["Date", "Code", "Action", "Quantity", "Price", "Fee"]
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Event:
    """One row of the transaction log.

    `price` is per-share for trades; for dividend/interest/deposit/withdraw the
    cash amount lives in `cash` instead (the CSV's Price column is mapped there
    at load time, so `price` stays semantically a per-share number). `ticker` is
    a `CASH` pseudo-symbol for deposit/withdraw (external cash flows).
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


# ── Ghostfolio export adapter ────────────────────────────────────────────────
#
# Accept a Ghostfolio activities export directly, so `--book export.json` Just Works.
# Ghostfolio's native export is **JSON** (`{..., "activities": [...]}`); each activity
# uses `symbol`/`type`/`unitPrice`/`quantity`. We map those to our schema BEFORE the
# parser, so the canonical Event model and everything downstream see only one format.
# Income value is `quantity × unitPrice` (collapsed to our cash-in-Price total by the
# loader). Out of scope (warned + skipped): ITEM/LIABILITY, non-USD, non-equity data
# sources. Ghostfolio has no deposit/withdraw (cash is account-level), unused by the brief.

_GF_ACTION_MAP: dict[str, str] = {  # Ghostfolio type → our action
    "BUY": "buy", "SELL": "sell", "DIVIDEND": "dividend", "FEE": "fee", "INTEREST": "interest",
}
_GF_SKIP_TYPES: frozenset[str] = frozenset({"ITEM", "LIABILITY"})  # not securities
_GF_SKIP_SOURCES: frozenset[str] = frozenset({"COINGECKO"})        # crypto / non-equity


def _gf_date(raw: str) -> str:
    """A Ghostfolio activity date → our ISO `YYYY-MM-DD`. Ghostfolio stores UTC timestamps,
    so a date-only entry becomes local-midnight in UTC (e.g. KST midnight → 15:00Z the
    PREVIOUS day). Round to the NEAREST day (+12h then truncate) to recover the intended
    local calendar date; a plain date passes through.

    The rounding is done in UTC with no tz knowledge, so it is exact only for UTC offsets
    within ±12h (every common zone, incl. the user's KST). Offsets beyond that (UTC+13/+14:
    NZ daylight time, Samoa, Tonga, Kiribati) and a genuine intraday timestamp at exactly
    12:00:00Z can land one day off — acceptable for a weekly, drawdown-first brief."""
    s = raw.strip()
    if "T" not in s:
        return s[:10]
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return (dt + timedelta(hours=12)).date().isoformat()


def _gf_num(x: float) -> str:
    """A clean numeric string for the shared parse loop: an int when integer-valued, else
    the shortest round-tripping float repr. `repr` (not a fixed `%.6f`) so a fractional
    share / sub-cent price keeps full precision through the str→`_to_float` hop."""
    return str(int(x)) if x == int(x) else repr(x)


def _rows_from_ghostfolio_json(raw: str, path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Map a Ghostfolio JSON export's `activities` → our-schema row dicts (+ skip warnings).
    Accepts the full export object (`{..., "activities": [...]}`) or a bare activities list.
    Income keeps `quantity`+`unitPrice` (the loader collapses it to a cash total). Never
    raises on a single bad activity — that one is skipped with a warning."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON ({exc})") from exc
    activities = data.get("activities", []) if isinstance(data, dict) else data
    if not isinstance(activities, list):
        raise ValueError(f"{path}: Ghostfolio JSON has no 'activities' list")
    out: list[dict[str, str]] = []
    warnings: list[str] = []
    for i, a in enumerate(activities, start=1):
        if not isinstance(a, dict):
            warnings.append(f"activity {i}: skipped (not an object)")
            continue
        gtype = str(a.get("type", "")).upper()
        if gtype in _GF_SKIP_TYPES:
            warnings.append(f"activity {i}: skipped {gtype} (not a security)")
            continue
        action = _GF_ACTION_MAP.get(gtype)
        if action is None:
            warnings.append(f"activity {i}: skipped unrecognized type {gtype!r}")
            continue
        currency = str(a.get("currency") or "USD").upper()
        if currency != "USD":
            warnings.append(f"activity {i}: skipped {currency} {a.get('symbol')} (USD-only for now)")
            continue
        if str(a.get("dataSource") or "MANUAL").upper() in _GF_SKIP_SOURCES:
            warnings.append(f"activity {i}: skipped {a.get('symbol')} (non-equity source)")
            continue
        symbol = str(a.get("symbol") or "").upper()
        if not symbol:  # no security to attach to — don't mint a "" ticker
            warnings.append(f"activity {i}: skipped {gtype} (no symbol)")
            continue
        try:
            row = {
                "Date": _gf_date(str(a.get("date") or "")),
                "Code": symbol,
                "Action": action,
                "Quantity": _gf_num(float(a.get("quantity") or 0.0)),
                "Price": _gf_num(float(a.get("unitPrice") or 0.0)),
                "Fee": _gf_num(float(a.get("fee") or 0.0)),
                "Currency": currency,
                "DataSource": str(a.get("dataSource") or "MANUAL"),
                "Note": str(a.get("comment") or ""),
            }
        except (ValueError, TypeError) as exc:
            warnings.append(f"activity {i}: skipped (bad field: {exc})")
            continue
        out.append(row)
    return out, warnings


def load_events(path: Path) -> list[Event]:
    """Parse a transaction file into an ordered event list.

    Accepts EITHER our native CSV OR a **Ghostfolio JSON activities export** — the file is
    auto-detected (JSON if it starts with `{`/`[`) and a Ghostfolio export is normalized to
    our schema on read (income from quantity × unitPrice; non-USD / crypto / ITEM / LIABILITY
    skipped with a warning), so the rest of the pipeline sees one format.

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
    raw = path.read_text(encoding="utf-8-sig")  # utf-8-sig strips a leading BOM
    if raw.lstrip()[:1] in ("{", "["):  # a Ghostfolio JSON activities export
        rows, skipped = _rows_from_ghostfolio_json(raw, path)
        for msg in skipped:
            log.warning("%s: ghostfolio import — %s", path, msg)
    else:
        reader = csv.DictReader(io.StringIO(raw))
        _validate_columns(reader.fieldnames, path)
        rows = list(reader)

    events: list[Event] = []
    for row in rows:
        action = _parse_action(row["Action"])
        price = _to_float(row.get("Price"))
        quantity = _to_float(row.get("Quantity"))
        # Income (dividend/interest) and cash flows carry a cash TOTAL, not a per-share
        # price. Our format puts the total in Price with Quantity 0; a Ghostfolio activity
        # puts a per-unit Price × a Quantity. quantity×price unifies both (Quantity 0 → just
        # Price); the shares aren't a holding change, so zero them out.
        # NOTE: this widened the native-CSV contract — a hand-written income row that left a
        # stray non-zero Quantity now reads as quantity×price, not bare price. Every shipped
        # book uses Quantity 0 on income/cash rows (per the README schema), so none is
        # affected; the per-share×shares reading is the correct one for a Ghostfolio CSV.
        if action in _CASH_IN_PRICE:
            cash = quantity * price if quantity else price
            price = 0.0
            quantity = 0.0
        else:
            cash = 0.0

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


def load_target(path: Path) -> dict[str, float]:
    """Load a target-allocation CSV (columns: Ticker, Weight) → normalized weights.

    Weights are relative and normalized to sum to 1.0. A weight of **0 is allowed**
    and means "close this position" — a deliberate, explicit sell-to-$0. It is kept
    in the returned dict so the caller can tell an intentional 0 apart from a ticker
    that was simply omitted (omission is the ambiguous case the CLI warns about).
    Rejects an empty file, a target that sums to zero, a negative weight, or a
    non-numeric weight (a clear error beats a silently skewed target).

    Lives here (the CSV input boundary, beside `load_events`) so `strategy.py` stays
    pure — it consumes the returned dict and never touches the filesystem.
    """
    raw: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fields = {f.strip() for f in (reader.fieldnames or [])}
        if not {"Ticker", "Weight"} <= fields:
            msg = f"{path}: target CSV needs columns Ticker, Weight (found {sorted(fields)})"
            raise ValueError(msg)
        for row in reader:
            ticker = (row.get("Ticker") or "").strip().upper()
            if not ticker:
                continue
            raw_w = (row.get("Weight") or "0").strip() or "0"
            try:
                weight = float(raw_w)
            except ValueError:
                msg = f"{path}: non-numeric weight {raw_w!r} for {ticker}"
                raise ValueError(msg) from None
            if weight < 0:
                msg = (
                    f"{path}: target weight for {ticker} must be >= 0 "
                    f"(use 0 to close the position; got {weight})"
                )
                raise ValueError(msg)
            raw[ticker] = raw.get(ticker, 0.0) + weight
    total = sum(raw.values())
    if not raw or total <= 0:
        msg = f"{path}: target allocation is empty or sums to zero"
        raise ValueError(msg)
    return {tk: w / total for tk, w in raw.items()}
