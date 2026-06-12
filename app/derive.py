"""Derive holdings, cost basis, and realized P&L from a transaction-log replay.

The replay uses **average cost** (simplest, defendable for v0). Lot/FIFO
upgrade is a future change to the engine, not the schema — the event log
already stores price+date per buy.

Realized P&L = locked-in gains (dividends + profit-on-sells).
Cost basis = total $ paid for shares still held (includes buy fees).

Cash events are routed by kind: rows on the CASH pseudo-ticker (broker interest
on idle cash, a tax an exporter couldn't attach to a security) net into
`cash_realized` — they never create a Position or per-security realized P&L.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

from app.events import CASH_TICKER, Event

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
    # Income/costs on the cash account (CASH pseudo-ticker): broker interest on
    # idle cash, an unmatched tax. Counted in total_realized, but kept out of the
    # per-security `realized` map and `positions` (CASH is not a security).
    cash_realized: float = 0.0

    def held(self) -> dict[str, Position]:
        return {tk: p for tk, p in self.positions.items() if p.shares > _SHARE_DUST}

    def total_realized(self) -> float:
        return sum(self.realized.values()) + self.cash_realized

    def total_cost_basis(self) -> float:
        return sum(p.cost_basis for p in self.held().values())

    def total_fees(self) -> float:
        return sum(self.fees.values())


def derive(events: list[Event]) -> DerivedState:
    """Replay an ordered event list into a derived state.

    Buys add (price*qty + fee) to cost basis; sells use the current avg cost
    to compute realized gain and reduce cost basis proportionally; dividends
    add `cash` to realized; fee/interest rows are tracked separately.

    Raises ValueError on sells with no held shares, sells whose quantity
    exceeds the held shares (silent negative positions are never produced),
    a buy/sell of the CASH pseudo-ticker, or a deposit/withdraw on anything
    but CASH (both importer errors).

    Importer contract: a cost must appear exactly ONCE — either in a trade
    row's Fee column or as a standalone fee line, never both. Derive cannot
    tell a duplicate from two genuine same-amount fees, so it warns when a
    standalone fee matches a same-day trade-row fee on the same ticker.
    (The heuristic is cost-only: negative fees — rebates — are not matched.)
    """
    state = DerivedState()

    # Trade-row fees by (date, ticker), for the duplicated-cost warning below.
    trade_fees: defaultdict[tuple[date, str], list[float]] = defaultdict(list)
    for ev in events:
        if ev.action in ("buy", "sell") and ev.fee > 0.0:
            trade_fees[(ev.date, ev.ticker)].append(ev.fee)

    for ev in events:
        if ev.action in ("deposit", "withdraw"):
            # External cash flows (CASH pseudo-ticker) carry no position and no
            # realized P&L; they feed the cash balance / account-value curve only.
            # A deposit/withdraw mis-tickered to a security would otherwise be
            # silently dropped — reject it, symmetric with buy/sell-of-CASH below.
            if ev.ticker != CASH_TICKER:
                msg = (
                    f"{ev.action} of {ev.ticker} on {ev.date}: external cash flows "
                    f"must use the {CASH_TICKER} pseudo-ticker (importer error)"
                )
                raise ValueError(msg)
            continue
        if ev.ticker == CASH_TICKER:
            # Cash-account rows routed by kind: income (interest/dividend) and
            # costs (an unmatched tax) net into cash_realized — no Position, no
            # per-security realized entry for a pseudo-ticker.
            if ev.action in ("dividend", "interest"):
                state.cash_realized += ev.cash - ev.fee
            elif ev.action == "fee":
                state.cash_realized -= ev.fee
            else:
                msg = f"{ev.action} of the CASH pseudo-ticker on {ev.date} (importer error)"
                raise ValueError(msg)
            state.fees[ev.ticker] += ev.fee
            continue
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
            # A standalone fee/tax row (e.g. foreign dividend withholding, exported
            # as its own line) is a realized cost: net it into realized P&L for the
            # ticker — consistent with how buy/sell/dividend-row fees reduce P&L.
            # (Still added to total_fees above, informationally.)
            if ev.fee > 0.0 and any(
                math.isclose(ev.fee, f) for f in trade_fees.get((ev.date, ev.ticker), [])
            ):
                log.warning(
                    "standalone fee of %.2f on %s (%s) matches a same-day trade-row fee "
                    "— if it is the same cost listed twice, it is being double-counted "
                    "(a cost must appear exactly once: on the trade row OR standalone)",
                    ev.fee,
                    ev.ticker,
                    ev.date,
                )
            state.realized[ev.ticker] -= ev.fee

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
