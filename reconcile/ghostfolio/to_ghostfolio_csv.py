"""Convert our transaction CSV to ghostfolio's import format.

Differences ghostfolio's importer needs:
- `Action` uppercased (BUY/SELL/DIVIDEND/...).
- DIVIDEND rows: ghostfolio computes the cash as quantity × unitPrice. Our rows
  store the cash in Price with Quantity 0 (→ 0 in ghostfolio). Re-express as
  quantity=1, unitPrice=cash so the dividend is recorded.

Run:  python reconcile/ghostfolio/to_ghostfolio_csv.py
Writes: reconcile/ghostfolio/ghostfolio_import.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

SRC = Path("data/sample_data/transactions.csv")
OUT = Path("reconcile/ghostfolio/ghostfolio_import.csv")

rows_out: list[dict[str, str]] = []
with SRC.open(newline="") as fh:
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
                "symbol": r["Code"],
                "currency": r["Currency"],
                "price": price,
                "quantity": quantity,
                "type": action,
                "fee": r["Fee"],
                "dataSource": r["DataSource"],
            }
        )

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="") as fh:
    writer = csv.DictWriter(
        fh,
        fieldnames=["date", "symbol", "currency", "price", "quantity", "type", "fee", "dataSource"],
    )
    writer.writeheader()
    writer.writerows(rows_out)

print(f"wrote {len(rows_out)} rows → {OUT}")
