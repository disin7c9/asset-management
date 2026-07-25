"""Convert our transaction CSV to ghostfolio's import format.

Differences ghostfolio's importer needs:
- `Action` uppercased (BUY/SELL/DIVIDEND/...).
- DIVIDEND rows: ghostfolio computes the cash as quantity × unitPrice. Our rows
  store the cash in Price with Quantity 0 (→ 0 in ghostfolio). Re-express as
  quantity=1, unitPrice=cash so the dividend is recorded.

Run:  python reconcile/ghostfolio/to_ghostfolio_csv.py
Writes: reconcile/ghostfolio/ghostfolio_import.csv

Everything runs under `main()` behind a `__main__` guard. It used to execute at module
scope, so merely importing this file — a doc tool, a test collector, an editor's symbol
indexer — read the source CSV and OVERWROTE the output. Import must be free of effects.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# One formula-injection rule for the whole repo. Defining a second copy here is how
# the five hand-maintained CLI/MCP parity checks got that way — import it instead.
from app.events import csv_safe

SRC = Path("data/sample_data/transactions.csv")
OUT = Path("reconcile/ghostfolio/ghostfolio_import.csv")

_FIELDS = ["date", "symbol", "currency", "price", "quantity", "type", "fee", "dataSource"]


def convert(src: Path = SRC, out: Path = OUT) -> int:
    rows_out: list[dict[str, str]] = []
    with src.open(newline="") as fh:
        for r in csv.DictReader(fh):
            action = r["Action"].strip().upper()
            price = r["Price"]
            quantity = r["Quantity"]
            if action in ("DIVIDEND", "INTEREST"):
                # cash lived in Price with quantity 0 → express as 1 × cash.
                price = r["Price"]
                quantity = "1"
            rows_out.append(
                {
                    "date": r["Date"],
                    "symbol": csv_safe(r["Code"]),
                    "currency": r["Currency"],
                    "price": price,
                    "quantity": quantity,
                    "type": action,
                    "fee": r["Fee"],
                    "dataSource": csv_safe(r["DataSource"]),
                }
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(rows_out)
    return len(rows_out)


def main() -> None:
    n = convert()
    print(f"wrote {n} rows → {OUT}")


if __name__ == "__main__":
    main()
