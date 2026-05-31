"""Presentation layer: render derived state into human-readable text.

Pure functions only — no I/O, no logging side-effects. `cli.py` writes the
returned string to stdout (or future delivery layers will pass it to email).
"""

from __future__ import annotations

from app.derive import DerivedState


def format_summary(state: DerivedState) -> str:
    """Build a deterministic plain-text summary table."""
    held = state.held()
    lines: list[str] = []
    lines.append(
        f"{'ticker':7}{'shares':>10}{'avg cost':>10}{'cost basis':>13}{'realized':>13}"
    )
    lines.append("-" * 53)
    for tk in sorted(held):
        p = held[tk]
        realized = state.realized[tk]  # defaultdict → 0.0 if absent
        lines.append(
            f"{tk:7}{p.shares:10.3f}{p.avg_cost:10.2f}"
            f"{p.cost_basis:13.2f}{realized:+13.2f}"
        )
    lines.append("-" * 53)
    lines.append(f"Total cost basis (held): ${state.total_cost_basis():,.2f}")
    lines.append(f"Total realized P&L:      ${state.total_realized():+,.2f}")
    lines.append(f"Total fees paid:         ${state.total_fees():,.2f}")
    return "\n".join(lines)
