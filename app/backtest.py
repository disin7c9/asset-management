"""Backtest harness (v1): notional historical simulation of a target allocation,
rebalanced on a schedule vs. buy-and-hold.

**Notional.** Starts a round `initial` ($10k) at the target weights on the first
date every target ticker has a price, then walks the daily closes forward. This
isolates the *strategy* (target + rebalance policy) from the user's actual
contribution timing — a clean lab for "is this allocation + rebalance rule worth
following?".

**Honesty / walk-forward.** A *fixed* rebalance policy fits no parameters, so the
whole history is an out-of-sample-clean simulation (nothing was optimized → there
is nothing to overfit). The walk-forward train/test *selection* machinery is only
needed once a strategy **searches** — tunes parameters or picks among candidates
(an optimizer, or an *edge* strategy). That is deferred; the
`strategy.may_suggest` gate keeps any future edge strategy from suggesting until a
walk-forward backtest validates it.

Pure compute over price Series. Composes `risk` + `returns` (a Layer-2 module that
uses its Layer-2 siblings — acyclic: neither imports back).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal

import numpy as np
import pandas as pd

from app.returns import true_twr_annualized, twr_index
from app.risk import RiskSummary, max_drawdown, moving_block_indices, summarize_risk

INITIAL_CAPITAL = 10_000.0
SCHEDULES: tuple[str, ...] = ("never", "monthly", "quarterly", "annually")

# Role check (v1.9.0): how much of the target a candidate sleeve displaces, the
# share of the common window held out, and the floors below which the verdict is
# honestly "insufficient" rather than a number dressed up as evidence.
CANDIDATE_SLEEVE = 0.05
_OOS_FRACTION = 0.30
_MIN_WINDOW_DAYS = 60
# Point-estimate margins: differences smaller than these are noise, not a verdict.
_DD_MARGIN = 0.005   # 0.5pp of drawdown depth
_VOL_MARGIN = 0.005  # 0.5pp of annualized volatility


@dataclass(frozen=True)
class BacktestLeg:
    """One simulated path (rebalanced, or buy-and-hold)."""

    label: str
    annualized_return: float | None  # true TWR, 252-basis; None if window too short
    final_value: float
    risk: RiskSummary


@dataclass(frozen=True)
class BacktestResult:
    start: date
    end: date
    initial: float
    schedule: str
    legs: tuple[BacktestLeg, ...]  # (rebalanced, buy_and_hold)
    missing: tuple[str, ...]       # target tickers with no usable price history
    provenance: dict[str, tuple[str, datetime]] = field(default_factory=dict)  # ticker → (source, fetched_at)


def _period_key(ts: pd.Timestamp, schedule: str) -> object:
    if schedule == "monthly":
        return (ts.year, ts.month)
    if schedule == "quarterly":
        return (ts.year, (ts.month - 1) // 3)
    return ts.year  # annually


def _rebalance_dates(index: pd.DatetimeIndex, schedule: str) -> set[pd.Timestamp]:
    """First trading day of each period (never the very first day — already allocated)."""
    if schedule == "never" or len(index) == 0:
        return set()
    out: set[pd.Timestamp] = set()
    seen: set[object] = set()
    for ts in index:
        k = _period_key(ts, schedule)
        if k not in seen:
            seen.add(k)
            out.add(ts)
    out.discard(index[0])
    return out


def _priced_tickers(series: dict[str, "pd.Series[float]"], target: dict[str, float]) -> list[str]:
    return [
        tk for tk in target
        if tk in series and series[tk].first_valid_index() is not None
    ]


def simulate(
    series: dict[str, "pd.Series[float]"],
    target: dict[str, float],
    *,
    schedule: str,
    initial: float = INITIAL_CAPITAL,
    start: date | None = None,
    end: date | None = None,
) -> "pd.Series[float]":
    """Daily equity curve of `initial` allocated to `target` and rebalanced on
    `schedule` ('never' = buy-and-hold). Empty Series if no priced target ticker.

    Rebalancing is value-preserving (no cash in/out): on a rebalance day the
    holdings are reset to the target weights at that day's prices.
    """
    tickers = _priced_tickers(series, target)
    if not tickers:
        return pd.Series(dtype=float)

    idx = pd.DatetimeIndex([])
    for tk in tickers:
        idx = idx.union(series[tk].index)
    lo = max(series[tk].first_valid_index() for tk in tickers)
    if start is not None:
        lo = max(lo, pd.Timestamp(start))
    hi = pd.Timestamp(end) if end is not None else idx.max()
    idx = idx[(idx >= lo) & (idx <= hi)].sort_values()
    if len(idx) == 0:
        return pd.Series(dtype=float)

    px = {tk: series[tk].reindex(idx, method="ffill") for tk in tickers}
    p0 = {tk: float(px[tk].iloc[0]) for tk in tickers}
    # Drop tickers without a positive starting price (bad/placeholder data); a 0
    # would divide by zero when sizing shares. Renormalize over what's left, and
    # bail if nothing usable remains (degenerate target → caller skips gracefully).
    tickers = [tk for tk in tickers if p0[tk] > 0]
    wsum = sum(target[tk] for tk in tickers)
    if not tickers or wsum <= 0:
        return pd.Series(dtype=float)
    weight = {tk: target[tk] / wsum for tk in tickers}
    reb = _rebalance_dates(idx, schedule)

    shares = {tk: weight[tk] * initial / p0[tk] for tk in tickers}
    values: list[float] = []
    for ts in idx:
        price = {tk: float(px[tk].loc[ts]) for tk in tickers}
        value = sum(shares[tk] * price[tk] for tk in tickers)
        values.append(value)
        if ts in reb and value > 0:
            # Guard a (rare) zero price mid-history: keep that ticker's shares.
            shares = {
                tk: (weight[tk] * value / price[tk] if price[tk] > 0 else shares[tk])
                for tk in tickers
            }
    return pd.Series(values, index=idx, dtype=float)


def _leg(label: str, curve: "pd.Series[float]", *, bootstrap_n: int, seed: int) -> BacktestLeg | None:
    daily = curve.pct_change().dropna()
    risk = summarize_risk(daily, twr_index(daily), bootstrap_n=bootstrap_n, seed=seed)
    if risk is None:  # too few return-days to compute a risk panel
        return None
    return BacktestLeg(label, true_twr_annualized(daily), float(curve.iloc[-1]), risk)


def backtest_compare(
    series: dict[str, "pd.Series[float]"],
    target: dict[str, float],
    *,
    schedule: str = "quarterly",
    initial: float = INITIAL_CAPITAL,
    start: date | None = None,
    end: date | None = None,
    bootstrap_n: int = 1000,
    seed: int = 42,
    provenance: dict[str, tuple[str, datetime]] | None = None,
) -> BacktestResult | None:
    """Simulate rebalanced-to-target vs buy-and-hold and bundle both legs with
    their drawdown-first risk + return. Returns None if there is no usable price
    history for any target ticker. `provenance` (ticker → source, fetched_at) is
    carried through for the report's freshness line."""
    priced = _priced_tickers(series, target)
    missing = tuple(sorted(tk for tk in target if tk not in priced))
    if not priced:
        return None

    rebalanced = simulate(series, target, schedule=schedule, initial=initial, start=start, end=end)
    buyhold = simulate(series, target, schedule="never", initial=initial, start=start, end=end)
    if rebalanced.empty or buyhold.empty:
        return None

    reb = _leg(f"rebalanced ({schedule})", rebalanced, bootstrap_n=bootstrap_n, seed=seed)
    bh = _leg("buy & hold", buyhold, bootstrap_n=bootstrap_n, seed=seed)
    if reb is None or bh is None:  # window too short to score either leg
        return None

    return BacktestResult(
        start=rebalanced.index[0].date(),
        end=rebalanced.index[-1].date(),
        initial=initial,
        schedule=schedule,
        legs=(reb, bh),
        missing=missing,
        provenance={tk: prov for tk, prov in (provenance or {}).items() if tk in priced},
    )


# ── walk-forward role check (v1.9.0) ────────────────────────────────────────

RoleVerdict = Literal["improved", "worsened", "inconclusive", "insufficient"]


@dataclass(frozen=True)
class RoleWindow:
    """Both portfolios' drawdown-first stats over one window (fresh capital at
    the window start — 'what if you had adopted this mix here?')."""

    label: str            # "in-sample" | "out-of-sample"
    start: date
    end: date
    n_days: int
    dd_without: float     # max drawdown depth, negative fraction
    dd_with: float
    vol_without: float    # annualized daily-return volatility
    vol_with: float
    ret_without: float | None  # annualized TWR; None if window too short
    ret_with: float | None


@dataclass(frozen=True)
class RoleCheck:
    """Did giving the candidate a sleeve actually help, out of sample?

    The screen's PASS says "sane, cheap, liquid, different"; this is the
    *evidence* step behind the edge gate: simulate the user's target vs the
    same target with a `sleeve` carved out for the candidate, split the common
    priced window into in-sample (where the candidate was effectively chosen)
    and held-out out-of-sample, and judge ONLY on the out-of-sample window —
    drawdown-first, with margins so noise can't masquerade as a verdict, and a
    CI-containment downgrade so overlapping uncertainty reads "inconclusive".
    """

    candidate: str
    sleeve: float
    schedule: str
    windows: tuple[RoleWindow, ...]   # (in-sample, out-of-sample) when computable
    verdict: RoleVerdict
    reason: str

    @property
    def oos(self) -> RoleWindow | None:
        return next((w for w in self.windows if w.label == "out-of-sample"), None)


def _aligned_leg_returns(
    series: dict[str, "pd.Series[float]"],
    without: dict[str, float],
    with_cand: dict[str, float],
    *,
    schedule: str,
    start: date,
    end: date,
) -> "tuple[pd.Series[float], pd.Series[float]] | None":
    """Simulate both portfolios ONCE over [start, end] (fresh capital) and return
    their daily returns inner-joined BY DATE — the legs can trade on different
    calendars (the candidate's exchange), and both the point stats and the paired
    bootstrap need day-for-day pairs, never positional truncation. None if either
    curve is unusable."""
    wo = simulate(series, without, schedule=schedule, start=start, end=end)
    w = simulate(series, with_cand, schedule=schedule, start=start, end=end)
    if wo.empty or w.empty:
        return None
    aligned = pd.concat([wo, w], axis=1, join="inner").pct_change().dropna()
    if len(aligned) < 2:
        return None
    return aligned.iloc[:, 0], aligned.iloc[:, 1]


def _window_stats(
    label: str,
    r_wo: "pd.Series[float]",
    r_w: "pd.Series[float]",
    *,
    start: date,
    end: date,
) -> RoleWindow:
    """Drawdown-first stats for both legs from their (aligned) daily returns."""
    def stats(r: "pd.Series[float]") -> tuple[float, float, float | None]:
        depth = max_drawdown(twr_index(r)).depth
        vol = float(r.std()) * math.sqrt(252.0)
        return depth, vol, true_twr_annualized(r)

    dd_wo, vol_wo, ret_wo = stats(r_wo)
    dd_w, vol_w, ret_w = stats(r_w)
    return RoleWindow(
        label=label, start=start, end=end, n_days=len(r_wo),
        dd_without=dd_wo, dd_with=dd_w,
        vol_without=vol_wo, vol_with=vol_w,
        ret_without=ret_wo, ret_with=ret_w,
    )


def _dd_depth(returns: np.ndarray) -> float:
    """Max drawdown depth of a return array (negative fraction)."""
    curve = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(curve)
    return float((curve / peaks - 1.0).min())


def _paired_dd_diff_ci(
    r_wo: np.ndarray,
    r_w: np.ndarray,
    *,
    bootstrap_n: int,
    seed: int,
) -> tuple[float, float]:
    """95% CI of (dd_with − dd_without) via a PAIRED moving-block bootstrap:
    the same resampled blocks index both legs' (date-aligned, equal-length)
    daily returns, so shared market moves cancel and only the candidate's
    effect remains. Returns (0.0, 0.0) — always 'contains zero' → inconclusive
    — when the window is too short to resample."""
    n = len(r_wo)
    if n < 10 or n != len(r_w):
        return (0.0, 0.0)
    block = min(max(int(math.sqrt(n)), 2), n)  # √n — same as risk's path-dependent CIs
    rng = np.random.default_rng(seed)
    diffs = np.empty(bootstrap_n)
    for i in range(bootstrap_n):
        idx = moving_block_indices(n, block, rng)  # the ONE resampler (see risk.py)
        diffs[i] = _dd_depth(r_w[idx]) - _dd_depth(r_wo[idx])
    return (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def role_check(
    series: dict[str, "pd.Series[float]"],
    target: dict[str, float],
    candidate: str,
    *,
    sleeve: float = CANDIDATE_SLEEVE,
    schedule: str = "quarterly",
    bootstrap_n: int = 1000,
    seed: int = 42,
) -> RoleCheck:
    """Walk-forward role check for one candidate. Never raises; an unusable
    setup returns verdict='insufficient' with the reason."""
    def insufficient(reason: str) -> RoleCheck:
        return RoleCheck(candidate, sleeve, schedule, (), "insufficient", reason)

    if candidate in target:
        return insufficient("already in the target — nothing to add")
    if not 0.0 < sleeve < 1.0:
        return insufficient(f"sleeve {sleeve} must be a fraction in (0, 1)")
    cand_series = series.get(candidate)
    if cand_series is None or cand_series.first_valid_index() is None:
        return insufficient("no usable price history for the candidate")
    priced = _priced_tickers(series, target)
    # Refuse to judge a DIFFERENT target: `simulate` renormalizes over the priced
    # tickers, so an unpriced target ticker would silently change the mix the
    # verdict claims to be about (backtest_compare reports `missing`; here the
    # only honest verdict is "couldn't test what you asked").
    unpriced = sorted(set(target) - set(priced))
    if unpriced:
        return insufficient(
            f"target ticker(s) without usable history: {', '.join(unpriced)} — "
            "refusing to judge a renormalized target"
        )

    with_cand = {tk: w * (1.0 - sleeve) for tk, w in target.items()}
    with_cand[candidate] = sleeve

    # The common window: where BOTH portfolios are simulable (the candidate's
    # history is usually the binding constraint). Known limitation: `hi` is the
    # latest of the target tickers' end dates, so a ticker whose data ENDS early
    # (delisted/halted) is forward-filled by `simulate` through the held-out
    # window — same ffill semantics as --backtest; flagged for a staleness warn
    # if it ever bites real data.
    lo = max(
        max(series[tk].first_valid_index() for tk in priced),
        cand_series.first_valid_index(),
    )
    hi = min(max(series[tk].index.max() for tk in priced), cand_series.index.max())
    common = cand_series.index[(cand_series.index >= lo) & (cand_series.index <= hi)]
    full_days = len(common)
    n_oos = int(full_days * _OOS_FRACTION)
    n_is = full_days - n_oos
    if n_is < _MIN_WINDOW_DAYS or n_oos < _MIN_WINDOW_DAYS:
        return insufficient(
            f"only {full_days} common days — needs ≥ {_MIN_WINDOW_DAYS} in BOTH the "
            "in-sample and held-out windows to judge honestly"
        )
    split_ts = common[n_is]

    # Simulate each leg ONCE per window; both consumers (point stats + the
    # paired bootstrap) share the same date-aligned daily returns.
    is_pair = _aligned_leg_returns(
        series, target, with_cand,
        schedule=schedule, start=lo.date(), end=common[n_is - 1].date(),
    )
    is_win = (
        _window_stats("in-sample", *is_pair, start=lo.date(), end=common[n_is - 1].date())
        if is_pair is not None
        else None
    )
    oos_pair = _aligned_leg_returns(
        series, target, with_cand,
        schedule=schedule, start=split_ts.date(), end=hi.date(),
    )
    if oos_pair is None:
        return insufficient("held-out window could not be simulated")
    oos_win = _window_stats("out-of-sample", *oos_pair, start=split_ts.date(), end=hi.date())

    # Verdict on the HELD-OUT window only, drawdown-first with noise margins.
    dd_gain = oos_win.dd_with - oos_win.dd_without   # >0 = shallower drawdown
    vol_gain = oos_win.vol_without - oos_win.vol_with  # >0 = calmer
    if dd_gain >= _DD_MARGIN and vol_gain >= -_VOL_MARGIN:
        verdict: RoleVerdict = "improved"
    elif dd_gain <= -_DD_MARGIN and vol_gain <= _VOL_MARGIN:
        verdict = "worsened"
    else:
        verdict = "inconclusive"

    # Honesty gate: a PAIRED moving-block bootstrap of the drawdown DIFFERENCE
    # (same resampled day-blocks applied to both legs, so common market moves
    # cancel). The verdict stands only if the 95% CI of (dd_with − dd_without)
    # excludes zero; otherwise the difference is inside the uncertainty band.
    if verdict in ("improved", "worsened"):
        lo_ci, hi_ci = _paired_dd_diff_ci(
            oos_pair[0].to_numpy(), oos_pair[1].to_numpy(),
            bootstrap_n=bootstrap_n, seed=seed,
        )
        if lo_ci <= 0.0 <= hi_ci:
            verdict = "inconclusive"

    parts = [
        f"OOS ({oos_win.start}→{oos_win.end}, {oos_win.n_days}d) with a "
        f"{sleeve * 100:.0f}% sleeve: max DD {oos_win.dd_with * 100:.1f}% vs "
        f"{oos_win.dd_without * 100:.1f}% without; vol {oos_win.vol_with * 100:.1f}% vs "
        f"{oos_win.vol_without * 100:.1f}%",
    ]
    if is_win is not None:
        is_gain = is_win.dd_with - is_win.dd_without
        parts.append(
            f"in-sample DD gain was {is_gain * 100:+.1f}pp vs OOS {dd_gain * 100:+.1f}pp"
            " (the overfitting gap to watch)"
        )
    if verdict == "inconclusive":
        parts.append("no clear out-of-sample improvement (inside the noise/CI band)")
    windows = tuple(w for w in (is_win, oos_win) if w is not None)
    return RoleCheck(candidate, sleeve, schedule, windows, verdict, "; ".join(parts))


def validate_from_role(rc: RoleCheck) -> bool:
    """The walk-forward evidence → the edge gate's ``validated`` flag.

    Returns True ONLY when the held-out role check came back ``improved`` — which
    already requires the out-of-sample drawdown to be shallower beyond the noise
    margin AND the paired moving-block bootstrap CI of the difference to exclude
    zero (see ``role_check``). ``worsened`` / ``inconclusive`` / ``insufficient``
    all → False: the gate stays shut unless the evidence is unambiguous.

    This is the ONLY sanctioned producer of ``validated=True`` for
    ``allocate(..., validated=)`` and ``strategy.may_suggest(..., backtest_validated=)``
    — an edge rule or AI-surfaced suggestion opens the gate by passing a real
    walk-forward role check, never by a caller asserting it.
    """
    return rc.verdict == "improved"
