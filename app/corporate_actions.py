"""Corporate-action adjustments to the raw transaction log (v1.x — slice 7).

The log records **raw** (unadjusted) share counts and execution prices; the
price history from yfinance is **split-adjusted**. For a ticker that split during
its holding period the two bases disagree — a 10:1 split shows the buy at ~10×
the adjusted close — so `shares × price` is inconsistent across the split and the
time-weighted series fabricates a return (NVDA's 10:1 made the real book's TWR
read 64% instead of ~20%).

`adjust_for_splits` re-expresses each pre-split buy/sell in **post-split share
terms**: multiply the quantity by, and divide the price by, the cumulative split
ratio for splits effective *after* the trade date — leaving total cost unchanged.
Afterwards share counts and adjusted prices share one basis, and every
downstream figure (holdings value, TWR, risk) is consistent.

Pure functions — no I/O. The split *data* is fetched by `prices.fetch_splits`;
the CLI applies this once before deriving holdings and the return series.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

from app.events import Event

# ticker -> [(effective_date, ratio)]; ratio = new shares per old (10:1 → 10.0,
# a 1:10 reverse split → 0.1).
Splits = dict[str, list[tuple[date, float]]]


def cumulative_split_factor(splits: list[tuple[date, float]], on: date) -> float:
    """Product of split ratios effective strictly AFTER `on` (1.0 if none).

    A split *after* a trade multiplies the shares that trade represents; a split
    on or before the trade is already reflected in the recorded quantity.
    """
    factor = 1.0
    for sdate, ratio in splits:
        if sdate > on and ratio > 0:
            factor *= ratio
    return factor


def adjust_for_splits(events: list[Event], splits_by_ticker: Splits) -> list[Event]:
    """Re-express raw buy/sell events in split-adjusted share terms.

    Only buy/sell rows carry a per-share quantity+price to adjust; dividend /
    interest / fee rows store cash (a split doesn't change the cash) and pass
    through untouched, as do tickers with no splits. Order is preserved.
    """
    out: list[Event] = []
    for ev in events:
        factor = cumulative_split_factor(splits_by_ticker.get(ev.ticker, []), ev.date)
        # Tolerance, not ==: split pairs that net to ~1.0 (e.g. a 3:1 then 1:3)
        # shouldn't perturb qty/price by a float ULP or be reported as adjusted.
        if ev.action in ("buy", "sell") and abs(factor - 1.0) > 1e-9:
            out.append(replace(ev, quantity=ev.quantity * factor, price=ev.price / factor))
        else:
            out.append(ev)
    return out
