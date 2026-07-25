"""Deterministic candidate screen — judge a NEW ticker by the role it would fill.

The v2 "suggesting" loop's judge: given candidate price history + published fund
facts (`metadata.SecurityMeta`) + the user's portfolio return series, run named,
checkable tests and emit a verdict with a reason per test. **"Good ticker" ≠
highest past return** (the overfitting trap, examples 05–06); good = fills a
role cheaply and reliably:

    structure       leveraged / inverse / ETN → out of scope, hard reject
    cost            expense ratio tiers (cheap / not cheap / expensive)
    liquidity       AUM + average volume floors
    age             young funds get closed
    concentration   a "fund" that is mostly its top 10 is a stock bet in costume
    diversifier     Pearson ρ vs YOUR book, ρ on your red days, and what the
                    candidate did during your worst drawdown window
    overlap         top-10 holdings overlap vs what you already hold
                    (near-equivalents like QQQ/QQQM collapse; physical-commodity
                    trusts have no look-through → category match is the fallback)

Pure Layer 2: no I/O — the CLI fetches series + metadata and passes them in.
Per-field degradation carries through: a missing fact makes that ONE check
"n/a", never a crash (the brief stays honest about what it couldn't test).

A "pass" here is necessary, not sufficient: it means the candidate is a sane,
cheap, liquid, genuinely-different instrument — NOT a prediction. The
held-out **role check** (did adding it actually improve drawdown/vol on a
held-out window?) is the edge gate's evidence: `backtest.role_check` computes
it, and when the caller supplies its results a `role` row joins the checks.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import pandas as pd

from app.backtest import RoleCheck, in_sample_end
from app.metadata import SecurityMeta
from app.returns import twr_index
from app.risk import DrawdownInfo, max_drawdown

log = logging.getLogger(__name__)

CheckStatus = Literal["pass", "warn", "fail", "n/a"]

# Thresholds — module constants, documented here; promote to CLI flags only when
# real use demands tuning (don't over-build).
_COST_PASS = 0.0020       # ≤ 20 bp: cheap
_COST_WARN = 0.0050       # ≤ 50 bp: not cheap; above: expensive
_AUM_MIN = 100e6          # institutional-size floor
_VOLUME_MIN = 100_000     # shares/day
_AGE_FAIL_Y = 1.0         # younger: closure risk is real
_AGE_WARN_Y = 3.0
_CONCENTRATION_WARN = 0.50  # top-10 weight above this → a concentrated bet
_CORR_WARN = 0.60
_CORR_FAIL = 0.85
_DOWNSIDE_ESCALATE = 0.15  # red-day ρ exceeding full ρ by this much → escalate
_OVERLAP_FAIL = 0.70       # near-duplicate of a held fund
_OVERLAP_WARN = 0.40
_MIN_OVERLAP_DAYS = 60     # fewer aligned return days → diversifier test is n/a
_MIN_RED_DAYS = 20         # fewer portfolio red days → downside ρ not computed


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: CheckStatus
    reason: str
    # Machine-readable evidence behind the prose (the v1.9.0 trigger): the role
    # check / edge gate / future MCP tool read these, never parse the reason.
    # Keys are per-check (diversifier: rho, downside_rho, dd_window_return;
    # role: oos_dd_with, oos_dd_without, oos_vol_with, oos_vol_without, ...).
    values: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateScreen:
    ticker: str
    checks: tuple[CheckResult, ...]

    @property
    def verdict(self) -> CheckStatus:
        statuses = {c.status for c in self.checks}
        if "fail" in statuses:
            return "fail"
        if "warn" in statuses:
            return "warn"
        if "pass" in statuses:
            return "pass"
        return "n/a"

    def counts(self) -> str:
        n = {s: 0 for s in ("fail", "warn", "pass", "n/a")}
        for c in self.checks:
            n[c.status] += 1
        return f"{n['fail']} fail / {n['warn']} warn / {n['pass']} pass / {n['n/a']} n-a"


def holdings_overlap(a: dict[str, float], b: dict[str, float]) -> float | None:
    """Top-holdings overlap: sum of min(weight) over shared symbols, or None when
    either side has no look-through (physical-commodity trusts). Top-10 only, so
    it UNDERSTATES true overlap — useful as a floor, decisive when high."""
    if not a or not b:
        return None
    return sum(min(a[s], b[s]) for s in set(a) & set(b))


def _pct_returns(close: "pd.Series[float]") -> "pd.Series[float]":
    return close.pct_change(fill_method=None).dropna()


def _check_structure(m: SecurityMeta | None) -> CheckResult:
    if m is None:
        return CheckResult("structure", "n/a", "no metadata")
    cat = (m.category or "").lower()
    legal = (m.legal_type or "").lower()
    if "trading--" in cat or "leveraged" in cat or "inverse" in cat:
        return CheckResult(
            "structure", "fail",
            f"category {m.category!r} — leveraged/inverse funds are out of scope "
            "(daily reset → path-dependent; breaks the calm-and-legible thesis)",
        )
    if "note" in legal:
        return CheckResult(
            "structure", "fail",
            f"{m.legal_type} — an ETN is issuer credit, not a fund holding assets",
        )
    if not cat and not legal:
        return CheckResult("structure", "n/a", "no category/type published")
    return CheckResult("structure", "pass", m.legal_type or m.category or "fund")


def _check_cost(m: SecurityMeta | None) -> CheckResult:
    if m is None or m.expense_ratio is None:
        return CheckResult("cost", "n/a", "expense ratio not published")
    er = m.expense_ratio
    pct = f"{er * 100:.2f}%"
    if er <= _COST_PASS:
        return CheckResult("cost", "pass", f"{pct} — cheap")
    if er <= _COST_WARN:
        return CheckResult("cost", "warn", f"{pct} — not cheap; demand a role no cheap fund fills")
    return CheckResult("cost", "fail", f"{pct} — expensive (> {_COST_WARN * 100:.2f}%)")


def _check_liquidity(m: SecurityMeta | None) -> CheckResult:
    if m is None or (m.aum is None and m.avg_volume is None):
        return CheckResult("liquidity", "n/a", "AUM/volume not published")
    low_aum = m.aum is not None and m.aum < _AUM_MIN
    low_vol = m.avg_volume is not None and m.avg_volume < _VOLUME_MIN
    aum_s = f"AUM ${m.aum / 1e6:,.0f}M" if m.aum is not None else "AUM n/a"
    vol_s = f"{m.avg_volume:,.0f} sh/day" if m.avg_volume is not None else "volume n/a"
    desc = f"{aum_s}, {vol_s}"
    if low_aum and low_vol:
        return CheckResult("liquidity", "fail", f"{desc} — both below the floors "
                           f"(${_AUM_MIN / 1e6:.0f}M / {_VOLUME_MIN:,.0f})")
    if low_aum or low_vol:
        return CheckResult("liquidity", "warn", f"{desc} — one floor missed")
    return CheckResult("liquidity", "pass", desc)


def _check_age(m: SecurityMeta | None, asof: date) -> CheckResult:
    age = m.age_years(asof) if m is not None else None
    if age is None:
        return CheckResult("age", "n/a", "inception not published")
    if age < _AGE_FAIL_Y:
        return CheckResult("age", "fail", f"{age:.1f}y — too young (closure risk)")
    if age < _AGE_WARN_Y:
        return CheckResult("age", "warn", f"{age:.1f}y — young fund")
    return CheckResult("age", "pass", f"{age:.1f}y")


def _check_concentration(m: SecurityMeta | None) -> CheckResult:
    if m is None or not m.top_holdings:
        return CheckResult("concentration", "n/a", "no look-through holdings published")
    top = sum(m.top_holdings.values())
    if top > _CONCENTRATION_WARN:
        return CheckResult(
            "concentration", "warn",
            f"top 10 = {top * 100:.0f}% of the fund — a concentrated bet, not broad exposure",
        )
    return CheckResult("concentration", "pass", f"top 10 = {top * 100:.0f}%")


def _check_diversifier(
    cand_close: "pd.Series[float] | None",
    portfolio_returns: "pd.Series[float]",
    portfolio_dd: DrawdownInfo | None,
    window_note: str = "",
) -> CheckResult:
    """The role test: does it move differently from what you already own?

    Three views, one verdict: Pearson ρ over the judged window (the gate), ρ on your red
    days (diversification that vanishes in stress is the classic trap), and the
    candidate's return during your worst drawdown window (the bluntest, most
    legible fact). `portfolio_dd` is candidate-independent, so the caller
    computes it once and passes it in.

    `window_note` labels the span these figures were computed over. It is non-empty exactly
    when a role check follows and the series were cut to the in-sample side (see
    `screen_candidates`) — the number that gates must be the number shown.
    """
    if cand_close is None or cand_close.empty or portfolio_returns.empty:
        return CheckResult("diversifier", "n/a", "no usable price history")
    cand = _pct_returns(cand_close)
    joined = pd.concat([cand, portfolio_returns], axis=1, join="inner").dropna()
    if len(joined) < _MIN_OVERLAP_DAYS:
        return CheckResult(
            "diversifier", "n/a",
            f"only {len(joined)} overlapping days (< {_MIN_OVERLAP_DAYS}) — too short to judge",
        )
    c, p = joined.iloc[:, 0], joined.iloc[:, 1]
    # Zero-variance series (a flat/halted price feed) have no correlation to
    # measure; pre-check so corr never emits NaN — a fund with NO return signal
    # must be n/a, never a silent "pass".
    if float(c.std()) == 0.0 or float(p.std()) == 0.0:
        return CheckResult(
            "diversifier", "n/a", "no return variation to correlate (flat price series)"
        )
    rho = float(c.corr(p))
    if math.isnan(rho):
        return CheckResult("diversifier", "n/a", "correlation undefined for this history")
    values: dict[str, float] = {"rho": rho}
    parts = [f"ρ={rho:+.2f} vs your book"]

    red = joined[p < 0]
    downside: float | None = None
    if len(red) >= _MIN_RED_DAYS:
        rc, rp = red.iloc[:, 0], red.iloc[:, 1]
        # Variance pre-check like the full-period guard above: a candidate flat on
        # your red days has no red-day correlation to measure — skip it before corr
        # so numpy never warns (the isnan below stays as a belt).
        if float(rc.std()) > 0.0 and float(rp.std()) > 0.0:
            red_rho = float(rc.corr(rp))
            if not math.isnan(red_rho):
                downside = red_rho
                values["downside_rho"] = downside
                parts.append(f"red-day ρ={downside:+.2f}")

    if portfolio_dd is not None and portfolio_dd.depth < 0:
        window = c[
            (c.index >= pd.Timestamp(portfolio_dd.peak_date))
            & (c.index <= pd.Timestamp(portfolio_dd.trough_date))
        ]
        if not window.empty:
            through = float((1.0 + window).prod() - 1.0)
            values["dd_window_return"] = through
            parts.append(
                f"during your worst drawdown ({portfolio_dd.peak_date}→"
                f"{portfolio_dd.trough_date}, {portfolio_dd.depth * 100:.1f}%) "
                f"it returned {through * 100:+.1f}%"
            )

    if rho > _CORR_FAIL:
        status: CheckStatus = "fail"
        parts.append("moves with what you already own")
    elif rho > _CORR_WARN:
        status = "warn"
    else:
        status = "pass"
    if status == "pass" and downside is not None and downside - rho > _DOWNSIDE_ESCALATE:
        status = "warn"
        parts.append("diversification weakens exactly on your red days")
    return CheckResult("diversifier", status, "; ".join(parts) + window_note, values)


# The candidate's own worst fall: history shorter than this is n/a (a young fund's shallow
# drawdown is false comfort); deeper than this warns even without book context.
_OWN_DD_MIN_YEARS = 2.0
_OWN_DD_WARN = -0.30


def _check_own_drawdown(
    cand_close: "pd.Series[float] | None",
    portfolio_dd: DrawdownInfo | None,
    window_note: str = "",
    held_worst: tuple[str, float] | None = None,
) -> CheckResult:
    """The candidate's OWN worst price fall over the judged window, vs the book's worst.

    The drawdown-first disclosure the diversifier check can't make: long treasuries or
    junk bonds PASS on correlation (they do move differently) while carrying equity-scale
    drawdowns of their own — the computed number says so; the tool never editorializes.

    `window_note` is non-empty when a role check follows and the series was cut to the
    in-sample side. That cut costs history — usually ~30%, but MORE when the candidate's own
    span is shorter than the book's, and in the limit the truncated series is empty — so a
    fund can drop below `_OWN_DD_MIN_YEARS` (or to "no usable price history") here that would
    have been judged on full history. An honest n/a is the right answer; the alternative is
    choosing on the held-out window."""
    if cand_close is None or cand_close.empty:
        return CheckResult("own-drawdown", "n/a", "no usable price history")
    try:
        # min/max, not positional ends: robust to a non-monotonic index; the except keeps
        # the module's "degrades per-check, never raises" contract on a non-date index.
        years = (cand_close.index.max() - cand_close.index.min()).days / 365.25
    except (AttributeError, TypeError):
        return CheckResult("own-drawdown", "n/a", "price history has no usable date index")
    if years < _OWN_DD_MIN_YEARS:
        return CheckResult(
            "own-drawdown", "n/a",
            f"only {years:.1f}y of history (< {_OWN_DD_MIN_YEARS:.0f}y) — too short to judge; "
            "a young fund's shallow drawdown is false comfort" + window_note,
        )
    dd = max_drawdown(cand_close)
    if dd is None or dd.depth >= 0:
        return CheckResult(
            "own-drawdown", "pass", f"no drawdown in {years:.1f}y of history" + window_note
        )
    values: dict[str, float] = {"depth": dd.depth, "history_years": years}
    desc = f"worst fall {dd.depth * 100:.1f}% ({dd.peak_date}→{dd.trough_date}) in {years:.1f}y"
    # Compare LIKE WITH LIKE. The old bar was the blended book's worst fall, but a single
    # fund almost always falls harder than a diversified mix — that is what diversification
    # IS — so the trigger was near-tautological: on the bundled example it would have flagged
    # 3 of the 4 funds the book already holds (VOO -19.0%, IAU -26.4%, VEA -14.4% against a
    # blend of -9.8%). The honest peer is the deepest-falling fund you ALREADY own: clearing
    # that says "no worse than what you live with", and failing it is a real fact about this
    # candidate. The blend is still reported, as context, because it is what you actually felt.
    if held_worst is not None and held_worst[1] < 0:
        peer_tk, peer_depth = held_worst
        values["held_worst_depth"] = peer_depth
        if dd.depth < peer_depth:
            return CheckResult(
                "own-drawdown", "warn",
                f"{desc} — deeper than anything you hold "
                f"(worst: {peer_tk} {peer_depth * 100:.1f}%)" + window_note,
                values,
            )
        desc += f"; shallower than {peer_tk}, which you already hold ({peer_depth * 100:.1f}%)"
    if portfolio_dd is not None and portfolio_dd.depth < 0:
        values["book_depth"] = portfolio_dd.depth
        if held_worst is None:
            desc += f"; your book's worst is {portfolio_dd.depth * 100:.1f}%"
    if dd.depth <= _OWN_DD_WARN:
        return CheckResult(
            "own-drawdown", "warn",
            f"{desc} — equity-scale drawdowns of its own" + window_note, values,
        )
    return CheckResult("own-drawdown", "pass", desc + window_note, values)


def _check_overlap(
    m: SecurityMeta | None, held_meta: dict[str, SecurityMeta]
) -> CheckResult:
    """Near-equivalent collapse vs what you already hold."""
    if m is None:
        return CheckResult("overlap", "n/a", "no metadata")
    best: tuple[str, float] | None = None
    for held_tk, held_m in held_meta.items():
        ov = holdings_overlap(m.top_holdings, held_m.top_holdings)
        if ov is not None and (best is None or ov > best[1]):
            best = (held_tk, ov)
    if best is not None:
        tk, ov = best
        if ov >= _OVERLAP_FAIL:
            return CheckResult(
                "overlap", "fail",
                f"{ov * 100:.0f}% top-10 overlap with held {tk} — a near-duplicate; "
                "keep the cheaper one",
            )
        if ov >= _OVERLAP_WARN:
            return CheckResult(
                "overlap", "warn",
                f"{ov * 100:.0f}% top-10 overlap with held {tk} — substantially the same exposure",
            )
        if ov == 0.0:
            return CheckResult("overlap", "pass", "no top-10 overlap with any held fund")
        return CheckResult("overlap", "pass", f"{ov * 100:.0f}% top-10 overlap with held {tk}")
    # No look-through on one side (physical-commodity trusts): fall back to the
    # category — GLDM vs IAU are the same exposure with zero shared "holdings".
    if m.category:
        same_cat = sorted(
            tk for tk, hm in held_meta.items() if hm.category and hm.category == m.category
        )
        if same_cat:
            return CheckResult(
                "overlap", "warn",
                f"no look-through holdings, but same category ({m.category}) as held "
                f"{', '.join(same_cat)} — likely the same exposure",
            )
    return CheckResult("overlap", "n/a", "no look-through holdings to compare")


# Verdict → screen status. "inconclusive" is n/a, NOT warn: the test ran and reached no
# conclusion, which is an absence of evidence exactly like its sibling "insufficient" (the
# test could not run at all) — not a finding against the candidate. Measured 2026-07-24, the
# held-out check returns inconclusive in 77–100% of trials on ~3y of personal history, so
# mapping it to warn fired on 24/24 candidates and made PASS unreachable on the
# `--discover --target` path. The full reason text still prints under `--screen TICKER`;
# the DISCOVERY panel shows only warn/fail rows, so browsing stays uncluttered.
_ROLE_STATUS: dict[str, CheckStatus] = {
    "improved": "pass",
    "worsened": "fail",
    "inconclusive": "n/a",
    "insufficient": "n/a",
}


def _check_role(rc: RoleCheck | None) -> CheckResult:
    """The held-out role check's verdict as a screen row (evidence behind
    the edge gate: judged on the held-out window only)."""
    if rc is None:
        return CheckResult("role", "n/a", "role check unavailable for this candidate")
    values: dict[str, float] = {"sleeve": rc.sleeve}
    oos = rc.oos
    if oos is not None:
        values.update(
            oos_ulcer_with=oos.ulcer_with, oos_ulcer_without=oos.ulcer_without,
            oos_cdar_with=oos.cdar_with, oos_cdar_without=oos.cdar_without,
            oos_dd_with=oos.dd_with, oos_dd_without=oos.dd_without,
            oos_vol_with=oos.vol_with, oos_vol_without=oos.vol_without,
        )
    return CheckResult("role", _ROLE_STATUS[rc.verdict], rc.reason, values)


def screen_candidates(
    candidates: list[str],
    candidate_close: dict[str, "pd.Series[float]"],
    portfolio_returns: "pd.Series[float]",
    meta: dict[str, SecurityMeta],
    held_meta: dict[str, SecurityMeta],
    held: set[str],
    *,
    asof: date,
    role: dict[str, RoleCheck] | None = None,
    held_worst: tuple[str, float] | None = None,
) -> list[CandidateScreen]:
    """Run every check per candidate. Pure; degrades per-check, never raises.

    SELECTING vs DESCRIBING. When `role` is supplied this screen is a **gate in front of**
    the held-out role check: a candidate that fails here never reaches that verdict. Judging
    the gate on full history would pick candidates using the very window `role_check` then
    holds out — held out from the final comparison, but not from the choosing, which is the
    part that makes an out-of-sample number mean anything. So the return-bearing checks
    (correlation, red-day ρ, drawdown-window return, own drawdown) are cut to the in-sample
    side. The structural facts — cost, liquidity, age, concentration, overlap,
    leveraged/inverse structure — carry no return information, so nothing can leak through
    them and they stay on full history, where they are strongest.

    The cut is **per candidate**, and takes the boundary from that candidate's OWN
    `RoleCheck` window when one exists, falling back to `in_sample_end` over the book's
    return index. Deriving it from the book alone was wrong: `role_check` splits its own
    `common` window (candidate ∩ target price history), so when a candidate's series ends
    before the book's — delisted, halted, or just a staler cache entry — its real split
    lands EARLIER and a book-derived cutoff reads months past it, back into the window this
    exists to protect. Taking the tighter of the two can only ever cut more, never less.

    Without `role` (a bare `--screen TICKER`) there is no selection step and nothing to
    protect: you named the ticker, so every check runs on everything available.
    """
    book_cutoff = (
        in_sample_end(portfolio_returns.index)
        if role is not None and not portfolio_returns.empty
        else None
    )

    def _role_cutoff(tk: str) -> "pd.Timestamp | None":
        """The end of the in-sample window `role_check` actually judged for this candidate."""
        rc = role.get(tk) if role is not None else None
        if rc is None:
            return None
        win = next((w for w in rc.windows if w.label == "in-sample"), None)
        return pd.Timestamp(win.end) if win is not None else None

    # Truncating the book's series and re-deriving its drawdown is the expensive part, and
    # most candidates share one cutoff — memoize on the date so the common case pays once.
    _ctx: dict[object, tuple["pd.Series[float]", DrawdownInfo | None]] = {}

    def _context(cut: "pd.Timestamp | None") -> tuple["pd.Series[float]", DrawdownInfo | None]:
        if cut not in _ctx:
            pr = (
                portfolio_returns
                if cut is None
                else portfolio_returns[portfolio_returns.index <= cut]
            )
            dd = (
                max_drawdown(twr_index(pr))
                if not pr.empty and float(pr.std()) > 0.0
                else None
            )
            _ctx[cut] = (pr, dd)
        return _ctx[cut]

    out: list[CandidateScreen] = []
    for tk in candidates:
        m = meta.get(tk)
        close = candidate_close.get(tk)
        # The tighter of the two boundaries — never read past what role_check held out.
        cuts = [c for c in (book_cutoff, _role_cutoff(tk)) if c is not None]
        cutoff = min(cuts) if cuts else None
        window_note = f" [in-sample through {cutoff.date()}]" if cutoff is not None else ""
        if cutoff is not None and close is not None:
            try:
                close = close[close.index <= cutoff]
            except TypeError:
                # A series without a date index cannot be windowed. The per-check guards
                # below already answer "n/a" for it; this module promises to degrade per
                # check and NEVER raise, so the slice must not be the one place it does.
                close = None
        portfolio_window, portfolio_dd = _context(cutoff)
        checks: list[CheckResult] = []
        if tk in held:
            checks.append(
                CheckResult("novelty", "fail",
                            "already in your book — the screen judges NEW tickers")
            )
        checks.append(_check_structure(m))
        checks.append(_check_cost(m))
        checks.append(_check_liquidity(m))
        checks.append(_check_age(m, asof))
        checks.append(_check_concentration(m))
        checks.append(_check_diversifier(close, portfolio_window, portfolio_dd, window_note))
        checks.append(_check_own_drawdown(close, portfolio_dd, window_note, held_worst))
        checks.append(_check_overlap(m, held_meta))
        if role is not None:  # a target was supplied → the held-out evidence row
            checks.append(_check_role(role.get(tk)))
        out.append(CandidateScreen(ticker=tk, checks=tuple(checks)))
        log.info("screened %s: %s", tk, out[-1].verdict)
    return out
