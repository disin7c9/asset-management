"""Derive holdings, cost basis, and realized P&L from a transaction-log replay.

The replay uses **average cost** (simplest, defendable for v0). Lot/FIFO
upgrade is a future change to the engine, not the schema — the event log
already stores price+date per buy.

Realized P&L = locked-in gains (dividends + profit-on-sells).
Cost basis = total $ paid for shares still held (includes buy fees).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from app.events import Event

log = logging.getLogger(__name__)

# Below this many shares a position is considered "closed"; residue from
# floating-point arithmetic is snapped to exactly zero after a sell.
_SHARE_DUST: float = 1e-9


@dataclass
class Position:
    ticker: str
    shares: float = 0.0
    cost_basis: float = 0.0

    @property
    def avg_cost(self) -> float:
        return self.cost_basis / self.shares if self.shares > 0 else 0.0


@dataclass
class DerivedState:
    positions: dict[str, Position] = field(default_factory=dict)
    realized: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))
    fees: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))

    def held(self) -> dict[str, Position]:
        return {tk: p for tk, p in self.positions.items() if p.shares > _SHARE_DUST}

    def total_realized(self) -> float:
        return sum(self.realized.values())

    def total_cost_basis(self) -> float:
        return sum(p.cost_basis for p in self.held().values())

    def total_fees(self) -> float:
        return sum(self.fees.values())


def derive(events: list[Event]) -> DerivedState:
    """Replay an ordered event list into a derived state.

    Buys add (price*qty + fee) to cost basis; sells use the current avg cost
    to compute realized gain and reduce cost basis proportionally; dividends
    add `cash` to realized; fee/interest rows are tracked separately.

    Raises ValueError on sells with no held shares, or sells whose quantity
    exceeds the held shares (silent negative positions are never produced).
    """
    state = DerivedState()

    for ev in events:
        pos = state.positions.setdefault(ev.ticker, Position(ticker=ev.ticker))
        state.fees[ev.ticker] += ev.fee

        if ev.action == "buy":
            pos.shares += ev.quantity
            pos.cost_basis += ev.quantity * ev.price + ev.fee

        elif ev.action == "sell":
            if pos.shares <= 0:
                msg = f"sell of {ev.ticker} with no shares held on {ev.date}"
                raise ValueError(msg)
            # Tolerate a tiny float overshoot but reject real over-selling.
            if ev.quantity > pos.shares + _SHARE_DUST:
                msg = (
                    f"oversell of {ev.ticker} on {ev.date}: "
                    f"trying to sell {ev.quantity} but only {pos.shares} held"
                )
                raise ValueError(msg)
            avg = pos.cost_basis / pos.shares
            state.realized[ev.ticker] += ev.quantity * ev.price - ev.quantity * avg - ev.fee
            pos.cost_basis -= ev.quantity * avg
            pos.shares -= ev.quantity
            # Snap residue from a "full" sell so the position truly closes.
            if pos.shares < _SHARE_DUST:
                pos.shares = 0.0
                pos.cost_basis = 0.0

        elif ev.action == "dividend":
            # Net any fee on the row (e.g. dividend withholding tax) out of income,
            # consistent with how buy/sell fees reduce P&L. (Fee is also tracked
            # in state.fees informationally.)
            state.realized[ev.ticker] += ev.cash - ev.fee

        elif ev.action == "interest":
            state.realized[ev.ticker] += ev.cash - ev.fee

        elif ev.action == "fee":
            # Tracked in state.fees only; no position change.
            pass

    held_count = len(state.held())
    log.info(
        "derived %d held position(s) from %d events; "
        "total_cost_basis=%.2f total_realized=%.2f total_fees=%.2f",
        held_count,
        len(events),
        state.total_cost_basis(),
        state.total_realized(),
        state.total_fees(),
    )
    return state
