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
from app.risk import (
    RiskSummary,
    cdar_from_returns,
    max_drawdown,
    moving_block_indices,
    path_block,
    summarize_risk,
    ulcer_from_returns,
)

INITIAL_CAPITAL = 10_000.0
SCHEDULES: tuple[str, ...] = ("never", "monthly", "quarterly", "annually")

# Role check (v1.9.0): how much of the target a candidate sleeve displaces, the
# share of the common window held out, and the floors below which the verdict is
# honestly "insufficient" rather than a number dressed up as evidence.
CANDIDATE_SLEEVE = 0.05
_OOS_FRACTION = 0.30
_MIN_WINDOW_DAYS = 60
# Point-estimate margin (v2.9.0): a smaller Ulcer gain than this is noise, not a
# verdict. Ulcer (RMS drawdown) runs at roughly half the scale of max-DD depth, so
# 0.25pp mirrors the 0.5pp max-DD margin the verdict used before it.
_ULCER_MARGIN = 0.0025
# CDaR contradiction slack (its OWN margin, NOT _ULCER_MARGIN): the worst-tail average
# may tie, but must not CONTRADICT the Ulcer direction by more than this to keep an
# improved/worsened verdict. CDaR is a worst-tail MEAN, running near max-DD scale (~2×
# Ulcer), so it is judged at that scale — sharing the tighter Ulcer margin would veto
# genuine Ulcer wins on ordinary tail noise.
_CDAR_SLACK = 0.005


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


def _latest_start(series: dict[str, "pd.Series[float]"], tickers: list[str]) -> "pd.Timestamp":
    """The common window's left edge: the latest first-valid index across `tickers` — the
    earliest date they ALL have data. Callers pass already-priced tickers (each has a
    non-None first_valid_index). Shared by simulate / role_check / benchmark_compare."""
    return max(series[tk].first_valid_index() for tk in tickers)


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

    **Basis contract — pass a TOTAL-RETURN series** (`prices.fetch_series(...,
    basis="total_return")`). This function holds funds with no transaction log, so there is
    nowhere for income to enter: on a raw close every coupon and dividend is booked as a
    permanent loss, which understates return AND deepens drawdown for exactly the sleeves the
    references lean on (60-40 is 40% BND; `permanent` is 25% BIL, whose return is entirely
    coupon). Nothing here can detect the wrong basis — the callers own it.
    """
    tickers = _priced_tickers(series, target)
    if not tickers:
        return pd.Series(dtype=float)

    idx = pd.DatetimeIndex([])
    for tk in tickers:
        idx = idx.union(series[tk].index)
    lo = _latest_start(series, tickers)
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


# ── held-out recent-window role check (v1.9.0) ────────────────────────────────────────

RoleVerdict = Literal["improved", "worsened", "inconclusive", "insufficient"]


@dataclass(frozen=True)
class RoleWindow:
    """Both portfolios' drawdown-first stats over one window (fresh capital at
    the window start — 'what if you had adopted this mix here?')."""

    label: str            # "in-sample" | "out-of-sample"
    start: date
    end: date
    n_days: int
    dd_without: float     # max drawdown depth, negative fraction — DESCRIPTIVE (see _oos_verdict)
    dd_with: float
    ulcer_without: float  # Ulcer index (RMS drawdown), positive — the verdict statistic
    ulcer_with: float
    cdar_without: float   # CDaR (mean of the worst-5% drawdowns), positive — the agreement check
    cdar_with: float
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
    def stats(r: "pd.Series[float]") -> tuple[float, float, float, float, float | None]:
        depth = max_drawdown(twr_index(r)).depth
        vol = float(r.std()) * math.sqrt(252.0)
        return depth, ulcer_from_returns(r), cdar_from_returns(r), vol, true_twr_annualized(r)

    dd_wo, ulcer_wo, cdar_wo, vol_wo, ret_wo = stats(r_wo)
    dd_w, ulcer_w, cdar_w, vol_w, ret_w = stats(r_w)
    return RoleWindow(
        label=label, start=start, end=end, n_days=len(r_wo),
        dd_without=dd_wo, dd_with=dd_w,
        ulcer_without=ulcer_wo, ulcer_with=ulcer_w,
        cdar_without=cdar_wo, cdar_with=cdar_w,
        vol_without=vol_wo, vol_with=vol_w,
        ret_without=ret_wo, ret_with=ret_w,
    )


def _ulcer_np(returns: np.ndarray) -> float:
    """Ulcer index straight from a return array — numpy-native, no pandas round-trip.
    MUST equal ``risk.ulcer_index`` (pinned to ~1e-12 by test); used only inside the
    paired-bootstrap hot loop, where wrapping each of ~1000 resamples in a pd.Series
    (the old form) was measured ~7-10× slower for byte-identical output."""
    curve = np.cumprod(1.0 + returns)
    dd = curve / np.maximum.accumulate(curve) - 1.0
    return float(np.sqrt(np.mean(np.square(dd))))


def _paired_ulcer_gain_ci(
    r_wo: np.ndarray,
    r_w: np.ndarray,
    *,
    bootstrap_n: int,
    seed: int,
) -> tuple[float, float] | None:
    """95% CI of the Ulcer GAIN (ulcer_without − ulcer_with; positive = the tested
    leg carried less drawdown pain) via a PAIRED moving-block bootstrap: the same
    resampled blocks index both legs' (date-aligned, equal-length) daily returns,
    so shared market moves cancel and only the tested leg's effect remains.
    Returns None — no measured interval — when the window is too short to resample
    (< 10 aligned days) or every resample was non-finite."""
    n = len(r_wo)
    if n < 10 or n != len(r_w):
        return None
    block = path_block(n)  # the ONE √n block rule (shared with risk.summarize_risk)
    rng = np.random.default_rng(seed)
    gains = np.empty(bootstrap_n)
    for i in range(bootstrap_n):
        idx = moving_block_indices(n, block, rng)  # the ONE resampler (see risk.py)
        gains[i] = _ulcer_np(r_wo[idx]) - _ulcer_np(r_w[idx])
    gains = gains[np.isfinite(gains)]  # drop non-finite resamples, as risk.bootstrap_ci does
    if gains.size == 0:
        return None
    return (float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5)))


# Why an OOS verdict came back 'inconclusive' — a structured, branchable fact (not prose a
# consumer must regex). "" = the verdict resolved (improved/worsened). Rendered for humans via
# _CAUSE_PHRASE; surfaced on BenchmarkResult.cause so MCP clients can steer (e.g. 'window_too_short'
# → warm a longer history, vs 'noise_margin' → they're genuinely equivalent).
OosCause = Literal[
    "", "noise_margin", "cdar_contradicts", "window_too_short", "bootstrap_unconfirmed"
]
_CAUSE_PHRASE: dict[OosCause, str] = {
    "noise_margin": "the Ulcer gap is within the noise margin",
    "cdar_contradicts": "the worst-tail average (CDaR) contradicts the Ulcer direction",
    "window_too_short": "the held-out window is too short to resample",
    "bootstrap_unconfirmed": "the paired bootstrap does not confirm the gap",
}


def _oos_verdict(
    oos: RoleWindow,
    r_without: "pd.Series[float]",
    r_with: "pd.Series[float]",
    *,
    bootstrap_n: int,
    seed: int,
) -> tuple[RoleVerdict, tuple[float, float] | None, OosCause]:
    """Drawdown-first verdict on a HELD-OUT window — shared by `role_check` (target vs
    +candidate) and `benchmark_compare` (preset vs reference). ``_with`` is the leg under
    test (candidate-added, or the preset), ``_without`` the baseline.

    The verdict statistic is the ULCER INDEX (v2.9.0). Max-DD depth is one worst event —
    an extreme-value statistic so noisy on short windows that its bootstrap CI almost
    always straddled zero (the chronic "inconclusive"); Ulcer is the RMS of EVERY day's
    drawdown, so the same paired bootstrap can actually resolve. Three gates, in order:
      1. the Ulcer gain must clear the noise margin (``_ULCER_MARGIN``);
      2. CDaR (the worst-tail average) may tie but must not CONTRADICT the direction by
         more than its own ``_CDAR_SLACK`` — catches an Ulcer win bought with a deeper tail;
      3. the paired-bootstrap 95% CI of the Ulcer gain must confirm the direction.
    Max drawdown and volatility stay DESCRIPTIVE: reported everywhere, voting nowhere.
    Note vol is NOT a hidden veto: Ulcer already captures the *drawdown-producing* (downside)
    volatility that a drawdown-first verdict cares about; upside volatility is deliberately
    not penalized (it is not, in itself, a defect). Returns (verdict, ulcer_gain_ci, cause):
    ci is a measured interval only when a bootstrap ran (else None), and cause names the gate
    that blocked an unresolved verdict ("" when it resolved)."""
    ulcer_gain = oos.ulcer_without - oos.ulcer_with  # >0 = less pain for the tested leg
    cdar_gain = oos.cdar_without - oos.cdar_with     # >0 = a milder worst tail
    if abs(ulcer_gain) < _ULCER_MARGIN:
        return "inconclusive", None, "noise_margin"
    if (ulcer_gain > 0 and cdar_gain < -_CDAR_SLACK) or (
        ulcer_gain < 0 and cdar_gain > _CDAR_SLACK
    ):
        return "inconclusive", None, "cdar_contradicts"
    ci = _paired_ulcer_gain_ci(r_without.to_numpy(), r_with.to_numpy(),
                               bootstrap_n=bootstrap_n, seed=seed)
    if ci is None:  # too short to resample — the honest cause, not "bootstrap didn't confirm"
        return "inconclusive", None, "window_too_short"
    if ulcer_gain > 0 and ci[0] > 0.0:
        return "improved", ci, ""
    if ulcer_gain < 0 and ci[1] < 0.0:
        return "worsened", ci, ""
    return "inconclusive", ci, "bootstrap_unconfirmed"


def in_sample_end(index: "pd.DatetimeIndex") -> "pd.Timestamp | None":
    """The last date a held-out check may TRAIN on, given a daily calendar.

    `role_check` splits its own common window `_OOS_FRACTION` from the end and judges only
    the tail. Anything that *selects which candidates reach that check* must not read past
    this date: choosing on the held-out window makes it held out from the final comparison
    but not from the choosing, and the verdict stops being independent of the data it is
    measured on.

    This is `screen.py`'s FALLBACK boundary, not the authoritative one. `role_check` splits
    its own `common` window (candidate ∩ target price history), which is a different index
    from the book's return series, so the same formula over the two gives different dates
    whenever a candidate's history ends early. `screen.py` therefore prefers that
    candidate's actual `RoleCheck` window end and uses this only when none exists.

    `None` when the calendar cannot support both windows — `role_check` returns
    "insufficient" there, so no verdict exists to protect and nothing needs holding back.
    """
    n = len(index)
    n_is = n - int(n * _OOS_FRACTION)
    if n_is < _MIN_WINDOW_DAYS or (n - n_is) < _MIN_WINDOW_DAYS:
        return None
    return index[n_is - 1]


def _oos_figure_clause(oos: RoleWindow, baseline: str) -> str:
    """The Ulcer-first OOS figure fragment shared by role_check and benchmark_compare, so
    the two reasons render the same statistics one way: 'Ulcer W vs WO <baseline>; CDaR W
    vs WO; max DD W vs WO (context)'. Ulcer is the verdict statistic, CDaR the tail check;
    max DD trails as descriptive context. Ulcer/CDaR print at 2dp so a gap near the 0.25pp
    margin is legible; ``baseline`` labels the without-leg ('without', 'the reference', …).

    The RETURN cost is printed too. The verdict is deliberately drawdown-first, but a
    less-painful path is usually bought with return, and that price was already computed
    here (`ret_with`/`ret_without`) and previously discarded — so a candidate could earn a
    passing role on a favourable Ulcer while quietly costing percent-a-year. Stating it is
    not a change of verdict; it is refusing to hide the trade-off behind the one it makes."""
    clause = (
        f"Ulcer {oos.ulcer_with * 100:.2f}% vs {oos.ulcer_without * 100:.2f}% {baseline}; "
        f"CDaR {oos.cdar_with * 100:.2f}% vs {oos.cdar_without * 100:.2f}%; "
        f"max DD {oos.dd_with * 100:.1f}% vs {oos.dd_without * 100:.1f}% (context)"
    )
    if oos.ret_with is not None and oos.ret_without is not None:
        gap = (oos.ret_with - oos.ret_without) * 100
        clause += (
            f"; return {oos.ret_with * 100:+.1f}% vs {oos.ret_without * 100:+.1f}%/yr "
            f"({gap:+.1f}pp, context)"
        )
    return clause


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
        _latest_start(series, priced),
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

    # Verdict on the HELD-OUT window only (Ulcer-first: noise margin, CDaR agreement,
    # paired-bootstrap CI honesty gate), shared with benchmark_compare via _oos_verdict.
    verdict, _ci, cause = _oos_verdict(
        oos_win, oos_pair[0], oos_pair[1], bootstrap_n=bootstrap_n, seed=seed
    )
    ulcer_gain = oos_win.ulcer_without - oos_win.ulcer_with  # the verdict statistic's gain

    parts = [
        f"OOS ({oos_win.start}→{oos_win.end}, {oos_win.n_days}d) with a "
        f"{sleeve * 100:.0f}% sleeve: " + _oos_figure_clause(oos_win, "without")
        + f"; vol {oos_win.vol_with * 100:.1f}% vs {oos_win.vol_without * 100:.1f}%",
    ]
    if is_win is not None:
        is_gain = is_win.ulcer_without - is_win.ulcer_with
        parts.append(
            f"in-sample Ulcer gain was {is_gain * 100:+.2f}pp vs OOS {ulcer_gain * 100:+.2f}pp"
            " (the overfitting gap to watch)"
        )
    if verdict == "inconclusive":
        parts.append(f"no clear out-of-sample improvement ({_CAUSE_PHRASE[cause]})")
    windows = tuple(w for w in (is_win, oos_win) if w is not None)
    return RoleCheck(candidate, sleeve, schedule, windows, verdict, "; ".join(parts))


def validate_from_role(rc: RoleCheck) -> bool:
    """The held-out recent-window evidence → the edge gate's ``validated`` flag.

    NOTE the gap this bridges: the gate's contract is a WALK-FORWARD validation, and what
    it is handed here is ONE held-out split. That is weaker evidence, and the first edge
    strategy to call this must either strengthen the check or narrow its own claim.

    Returns True ONLY when the held-out role check came back ``improved`` — which
    already requires the out-of-sample Ulcer (RMS drawdown) to be lower beyond the
    noise margin, the worst-tail CDaR not to contradict it, AND the paired
    moving-block bootstrap CI of the Ulcer gain to confirm the direction (see
    ``_oos_verdict``). ``worsened`` / ``inconclusive`` / ``insufficient``
    all → False: the gate stays shut unless the evidence is unambiguous.

    This is the ONLY sanctioned producer of ``validated=True`` for
    ``allocate(..., validated=)`` and ``strategy.may_suggest(..., backtest_validated=)``
    — an edge rule or AI-surfaced suggestion opens the gate by passing a real
    held-out role check, never by a caller asserting it.

    **WHAT "improved" DOES NOT MEAN — read before writing the first caller.** It means the
    held-out window's drawdown pain was lower, and nothing else. It is deliberately silent
    about return, and that silence is not an oversight in the verdict — it is what
    drawdown-first means. Measured on this exact function (200 trials, 5% sleeve, ~270
    held-out days, `docs/reports/v2.12-AI-agent-reviews/verdict-calibration-2026-07-24.md`):
    a candidate UNCORRELATED with the book genuinely lowers Ulcer even when it earns
    nothing (true gain +0.32pp, positive in 94% of draws) — diversification alone buys
    ~4.9% less volatility — and it *still* lowers Ulcer while destroying -15%/yr (true gain
    +0.17pp, positive in 80%). So a high-fee, low-correlation product earns an honest
    "improved" 8% of the time.

    The verdict is right; using it alone as authorization would not be. **A caller must
    check the return cost itself** — `rc.oos.ret_with` / `rc.oos.ret_without` carry it, and
    `_oos_figure_clause` already renders it. This function is intentionally NOT given a
    return veto: that would overload one word with two meanings and make every existing
    "improved" ambiguous. Keep the verdict single-meaning; put the policy in the caller.

    (There are no production callers today — every shipped rule is `discipline` and passes
    the gate freely. This is a live contract for the first edge strategy, not dead code.)
    """
    return rc.verdict == "improved"


# ── benchmark comparison: a preset vs a canonical reference (slice 2a) ───────

# Well-known reference portfolios (canonical tickers, each summing to 1). 60-40 anchors
# on the S&P 500 (the institutional benchmark); All-Weather (Dalio, retail) and Permanent
# (Browne) anchor on the TOTAL US market (VTI). The verdict vs these is "where the preset
# lands", never "beats" — on a short personal history it is usually inconclusive.
_BENCHMARKS: dict[str, dict[str, float]] = {
    "60-40": {"VOO": 0.60, "BND": 0.40},
    "all-weather": {"VTI": 0.30, "TLT": 0.40, "IEI": 0.15, "GLD": 0.075, "DBC": 0.075},
    "permanent": {"VTI": 0.25, "TLT": 0.25, "BIL": 0.25, "GLD": 0.25},
}
BENCHMARKS: frozenset[str] = frozenset(_BENCHMARKS)


def benchmark_weights(reference: str) -> dict[str, float]:
    """The reference portfolio's ticker:weight (a copy; empty for an unknown name)."""
    return dict(_BENCHMARKS.get(reference, {}))


@dataclass(frozen=True)
class BenchmarkResult:
    """A preset target vs a canonical reference over their COMMON priced window:
    full-history legs (drawdown-first) + a held-out recent-window verdict. The verdict is
    "where the preset's drawdown lands vs the reference", NOT "beats it" — usually
    'inconclusive' on a short history. `ulcer_gain_ci` is the 95% CI of the out-of-sample
    Ulcer gain (reference − preset; positive = the preset carried less drawdown pain), or
    None when no bootstrap ran (margin/CDaR-gated or too short). `cause` names the gate that
    left an inconclusive verdict unresolved ("" when it resolved) — see ``OosCause``."""

    reference: str
    start: date                       # full-history window (the legs' span)
    end: date
    legs: tuple[BacktestLeg, ...]     # (preset, reference)
    oos: RoleWindow | None
    verdict: BenchmarkVerdict
    ulcer_gain_ci: tuple[float, float] | None
    reason: str
    missing: tuple[str, ...]
    cause: OosCause = ""
    provenance: dict[str, tuple[str, datetime]] = field(default_factory=dict)


BenchmarkVerdict = Literal["shallower", "deeper", "inconclusive", "insufficient"]

# role_check's words ("improved"/"worsened") mean a candidate *helped*; for a preset-vs-
# reference comparison the honest word is "shallower"/"deeper" drawdown, NEVER "improved" /
# "beats" — a stored "improved" could be misread (e.g. by a future validate_from_* bridge)
# as "this preset is validated", the framing the report works hard to forbid.
_BENCHMARK_VERDICT: dict[RoleVerdict, BenchmarkVerdict] = {
    "improved": "shallower", "worsened": "deeper",
    "inconclusive": "inconclusive", "insufficient": "insufficient",
}
_BENCHMARK_PHRASE: dict[BenchmarkVerdict, str] = {
    "shallower": "a SHALLOWER drawdown than",
    "deeper": "a DEEPER drawdown than",
    "inconclusive": "no clear drawdown difference from",
}


def benchmark_compare(
    series: dict[str, "pd.Series[float]"],
    target: dict[str, float],
    reference_weights: dict[str, float],
    *,
    reference: str,
    schedule: str = "quarterly",
    initial: float = INITIAL_CAPITAL,
    bootstrap_n: int = 1000,
    seed: int = 42,
    start: date | None = None,
    provenance: dict[str, tuple[str, datetime]] | None = None,
) -> BenchmarkResult | None:
    """Compare a preset `target` against a `reference` portfolio over their common priced
    window: full-history legs (`simulate` + `_leg`), then a held-out recent-window verdict
    (70/30 split, OOS-only, paired-bootstrap CI) reusing `_oos_verdict` with the preset as
    the leg under test and the reference as the baseline. None if either side lacks usable
    history; never raises."""
    priced_t = _priced_tickers(series, target)
    priced_r = _priced_tickers(series, reference_weights)
    missing = tuple(sorted(
        (set(target) - set(priced_t)) | (set(reference_weights) - set(priced_r))
    ))
    if not priced_t or not priced_r:
        return None

    all_priced = sorted(set(priced_t) | set(priced_r))
    lo = _latest_start(series, all_priced)
    if start is not None:  # honor --backtest-start (bound the sim, like backtest_compare)
        lo = max(lo, pd.Timestamp(start))
    hi = min(series[tk].index.max() for tk in all_priced)
    preset_curve = simulate(series, target, schedule=schedule, initial=initial,
                            start=lo.date(), end=hi.date())
    ref_curve = simulate(series, reference_weights, schedule=schedule, initial=initial,
                         start=lo.date(), end=hi.date())
    if preset_curve.empty or ref_curve.empty:
        return None
    preset_leg = _leg("preset", preset_curve, bootstrap_n=bootstrap_n, seed=seed)
    ref_leg = _leg(reference, ref_curve, bootstrap_n=bootstrap_n, seed=seed)
    if preset_leg is None or ref_leg is None:
        return None

    # Walk-forward: split the common calendar 70/30; judge the held-out window only.
    idx = pd.DatetimeIndex([])
    for tk in all_priced:
        idx = idx.union(series[tk].index)
    common = idx[(idx >= lo) & (idx <= hi)].sort_values()
    n_oos = int(len(common) * _OOS_FRACTION)
    n_is = len(common) - n_oos
    oos_win: RoleWindow | None = None
    verdict: RoleVerdict = "insufficient"
    ci: tuple[float, float] | None = None
    cause: OosCause = ""
    if n_is >= _MIN_WINDOW_DAYS and n_oos >= _MIN_WINDOW_DAYS:
        split = common[n_is]
        # baseline = reference, alt = preset → *_with/_without read "preset vs reference"
        oos_pair = _aligned_leg_returns(
            series, reference_weights, target,
            schedule=schedule, start=split.date(), end=hi.date(),
        )
        if oos_pair is not None:
            oos_win = _window_stats("out-of-sample", *oos_pair, start=split.date(), end=hi.date())
            verdict, ci, cause = _oos_verdict(oos_win, *oos_pair, bootstrap_n=bootstrap_n, seed=seed)

    bverdict = _BENCHMARK_VERDICT[verdict]  # "improved"→"shallower" etc. (never "beats")
    if oos_win is None:
        reason = (f"only {len(common)} common days — needs ≥ {_MIN_WINDOW_DAYS} in BOTH "
                  f"windows to judge vs {reference} honestly")
    else:
        reason = (
            f"OOS ({oos_win.start}→{oos_win.end}, {oos_win.n_days}d): "
            + _oos_figure_clause(oos_win, reference)
            + f" — {_BENCHMARK_PHRASE[bverdict]} {reference}"
        )
        if ci is not None:  # the paired bootstrap ran → show its width
            reason += f"; 95% CI of the Ulcer gain [{ci[0] * 100:+.2f}pp, {ci[1] * 100:+.2f}pp]"
        if cause:  # the gate that kept it inconclusive, by name
            reason += f" ({_CAUSE_PHRASE[cause]})"

    return BenchmarkResult(
        reference=reference,
        start=preset_curve.index[0].date(), end=preset_curve.index[-1].date(),
        legs=(preset_leg, ref_leg), oos=oos_win, verdict=bverdict,
        ulcer_gain_ci=ci, reason=reason, missing=missing, cause=cause,
        provenance={tk: p for tk, p in (provenance or {}).items() if tk in all_priced},
    )
