"""The narration fence: ground an LLM's prose in validated numbers (v2.0.0 P2a).

Pure functions — no I/O, no LLM. This is the *deterministic* half of the output
edge: it makes it structurally impossible for a language model to put a wrong
number in the brief. The LLM (a later phase, `app/llm.py`) only ever writes prose
with `{{token}}` placeholders; this module owns the numbers.

Two guarantees, both checked here:

- **SymGen** (substitute-generated): `render_narration` replaces each `{{token}}`
  with the *validated* rendered value from the claim set — the model never types a
  figure, it references one by name (arXiv 2311.09188).
- **PCN** (proof-carrying numbers): the model's raw prose must contain **no bare
  digit** — every numeral must arrive via a `{{token}}`. If a literal digit appears
  outside a token, the model wrote a number itself, so the narration is **rejected**
  wholesale (fail-closed) and the caller prints the plain brief (arXiv 2509.06902).

Because the only path for a digit into the final text is substitution from
`build_claim_set` (which reads the same validated core the brief does), a fabricated
or mis-scaled number cannot survive. The residual risk is purely interpretive prose.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date

from app.derive import DerivedState
from app.prices import PriceRow
from app.returns import ReturnsSummary
from app.risk import DollarDrawdown, RiskSummary


@dataclass(frozen=True)
class Claim:
    """One referenceable figure: the LLM cites it as ``{{token}}`` and the renderer
    substitutes ``rendered`` (the validated display string). ``value`` is the raw
    typed number; the fence only uses ``rendered``. ``band`` is the coarse
    qualitative label (e.g. "moderate") the FREE tier sends instead of the value —
    computed once here, *with* the claim, so the prompt builder never re-dispatches on
    token name and a new metric can't silently lose its band."""

    token: str
    value: float | int | str | date  # raw figure behind the claim; its FREE-tier band derives from it
    rendered: str
    label: str               # human description (for the LLM prompt)
    band: str | None = None  # FREE-tier qualitative band; None for dates/counts/$ amounts


# ── display formatting (mirrors report.py's conventions; numbers are the core's) ──


def _usd(x: float) -> str:
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def _pct(x: float) -> str:
    return f"{x * 100:+.2f}%"      # signed: returns, drawdown depth


def _mag(x: float) -> str:
    return f"{x * 100:.2f}%"       # unsigned magnitude: Ulcer / CDaR


def _ratio(x: float) -> str:
    return f"{x:+.2f}"             # Sharpe / Sortino / Calmar


# ── free-tier qualitative bands (the privacy dial) ────────────────────────────


@dataclass(frozen=True)
class _BandSpec:
    """A coarse band table for one metric kind. ``use_abs`` bands by magnitude
    (drawdown depth is negative)."""

    cuts: tuple[tuple[float, str], ...]
    top: str
    use_abs: bool = False

    def label_for(self, value: float) -> str:
        v = abs(value) if self.use_abs else value
        for cut, label in self.cuts:
            if v < cut:
                return label
        return self.top


_RATIO_BAND = _BandSpec(((0.0, "negative"), (1.0, "weak"), (2.0, "solid")), "strong")
_RETURN_BAND = _BandSpec(((0.0, "negative"), (0.10, "modest"), (0.20, "healthy")), "strong")
# Drawdown-magnitude cuts (5/15/30%), shared by max-drawdown depth (negative → use_abs)
# and CDaR (already a positive magnitude). All band thresholds here are ROUGH, coarse
# tone buckets — tunable; they only nudge the FREE-tier model's wording, never a figure.
_DD_CUTS = ((0.05, "mild"), (0.15, "moderate"), (0.30, "significant"))

# token → band spec. ONE table (was a per-token if-ladder in build_prompt): each
# metric's band is computed *with* its Claim, so adding a metric can't silently drop
# its FREE-tier band, and the privacy dial just reads ``c.band`` — it never
# re-dispatches on token name. (Dates/counts/$ amounts have no band → not listed.)
_BAND_SPECS: dict[str, _BandSpec] = {
    "max_drawdown": _BandSpec(_DD_CUTS, "severe", use_abs=True),  # depth is negative
    "cdar": _BandSpec(_DD_CUTS, "severe"),                        # already a positive magnitude
    "ulcer": _BandSpec(((0.02, "mild"), (0.05, "moderate"), (0.10, "significant")), "severe"),
    "sharpe": _RATIO_BAND, "sortino": _RATIO_BAND, "calmar": _RATIO_BAND,
    "twr_annual": _RETURN_BAND, "mwr_annual": _RETURN_BAND, "dietz_annual": _RETURN_BAND,
}


def _band_for(token: str, value: float | int | str | date) -> str | None:
    """The FREE-tier band for a claim, or None for a token without one. Computed once
    at claim construction and stored on ``Claim.band``. Fails closed (None) on a
    non-finite value too — a band must never assert "severe" for a figure that is
    actually n/a (matches the fence's ethos; today add() already pre-filters these)."""
    spec = _BAND_SPECS.get(token)
    if spec is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return None if not math.isfinite(value) else spec.label_for(float(value))


# ── claim set: the validated figures the LLM may reference ────────────────────


def build_claim_set(
    state: DerivedState,
    prices: dict[str, PriceRow] | None,
    returns: ReturnsSummary | None,
    risk: RiskSummary | None,
    *,
    dollar_dd: DollarDrawdown | None = None,
) -> dict[str, Claim]:
    """The frozen set of figures the narration may cite — built from the SAME typed
    core the brief renders, so a cited number is the brief's number. A metric that
    is `None` / non-finite (n/a) is **omitted** entirely: the LLM cannot reference a
    figure that doesn't honestly exist."""
    claims: dict[str, Claim] = {}

    def add(token: str, value: float | int | str | date, rendered: str, label: str) -> None:
        # Fail-safe: never expose a non-finite figure (mirrors report.py rendering n/a).
        # Most numeric claims are isfinite-gated by their caller; this central guard
        # also covers the ungated ones (drawdown depth, P&L, $ giveback) so a nan/inf
        # can never reach the fence as "nan%"/"$inf" (which PCN's \d would not catch).
        if isinstance(value, float) and not math.isfinite(value):
            return
        claims[token] = Claim(token=token, value=value, rendered=rendered, label=label,
                              band=_band_for(token, value))

    prices = prices or {}
    held = state.held()
    priced = {tk: held[tk].shares * prices[tk].close for tk in held if tk in prices}
    if priced:
        mkt = sum(priced.values())
        unreal = sum(v - held[tk].cost_basis for tk, v in priced.items())
        real = state.total_realized()
        add("total_market_value", mkt, _usd(mkt), "total market value of priced holdings")
        add("unrealized_pnl", unreal, _usd(unreal), "unrealized profit/loss")
        add("net_pnl", unreal + real, _usd(unreal + real), "net P&L (unrealized + realized)")
        add("n_holdings", len(priced), str(len(priced)), "number of priced holdings")
    cost = state.total_cost_basis()
    if cost > 0:
        add("total_cost_basis", cost, _usd(cost), "total cost basis of held positions")
    add("realized_pnl", state.total_realized(), _usd(state.total_realized()),
        "realized P&L (sells + dividends, net of fees)")

    if returns is not None:
        for token, val, label in (
            ("twr_annual", returns.true_twr_annualized, "annualized time-weighted return"),
            ("mwr_annual", returns.money_weighted_annualized, "annualized money-weighted return (IRR)"),
            ("dietz_annual", returns.modified_dietz_annualized, "annualized Modified Dietz return"),
        ):
            if val is not None and math.isfinite(val):
                add(token, val, _pct(val), label)
        if returns.period_days > 0:
            add("period_start", returns.period_start, returns.period_start.isoformat(),
                "start of the measured period")
            add("period_end", returns.asof_date, returns.asof_date.isoformat(),
                "end of the measured period")
            yrs = returns.period_days / 365.25
            add("period_years", yrs, f"{yrs:.2f}y", "length of the measured period (years)")

    if risk is not None:
        dd = risk.drawdown
        add("max_drawdown", dd.depth, _pct(dd.depth), "deepest peak-to-trough decline")
        add("drawdown_peak_date", dd.peak_date, dd.peak_date.isoformat(),
            "date of the pre-drawdown peak")
        add("drawdown_trough_date", dd.trough_date, dd.trough_date.isoformat(),
            "date of the drawdown trough")
        if dd.recovery_date is not None:
            add("drawdown_recovery_date", dd.recovery_date, dd.recovery_date.isoformat(),
                "date the drawdown recovered")
        add("drawdown_duration_days", dd.duration_days, f"{dd.duration_days} days",
            "peak-to-recovery duration")
        add("time_underwater_pct", dd.time_underwater_pct,
            f"{dd.time_underwater_pct * 100:.0f}%", "share of days below a prior peak")
        for token, ci, label in (
            ("ulcer", risk.ulcer_index, "Ulcer index (RMS drawdown)"),
            ("cdar", risk.cdar, "CDaR (mean of the worst-5% drawdowns)"),
        ):
            if math.isfinite(ci.point):
                add(token, ci.point, _mag(ci.point), label)
        for token, ci, label in (
            ("sharpe", risk.sharpe, "Sharpe ratio (risk-adjusted return)"),
            ("sortino", risk.sortino, "Sortino ratio (downside risk-adjusted)"),
            ("calmar", risk.calmar, "Calmar ratio (return over max drawdown)"),
        ):
            if math.isfinite(ci.point):
                add(token, ci.point, _ratio(ci.point), label)

    if dollar_dd is not None:
        add("gains_given_back", dollar_dd.giveback_dollars, _usd(dollar_dd.giveback_dollars),
            "largest dollar decline in cumulative market profit")

    return claims


# ── the fence: SymGen substitution + PCN verification ─────────────────────────

# A reference is `{{token}}`; the token is a lowercase identifier (optional inner
# whitespace tolerated). ONE pattern drives both the PCN strip and substitution, so
# they can never disagree about what counts as a token.
_TOKEN_RE = re.compile(r"\{\{\s*([A-Za-z][A-Za-z0-9_]*)\s*\}\}")


def _has_bare_numeral(prose: str) -> bool:
    """PCN: True if any digit appears OUTSIDE a `{{token}}` — i.e. the model typed a
    number itself. Token references (incl. ones whose name contains a digit) are
    removed first, so only a model-authored numeral trips this."""
    return bool(re.search(r"\d", _TOKEN_RE.sub("", prose)))


def render_narration(prose: str, claim_set: dict[str, Claim]) -> str | None:
    """Fail-closed SymGen + PCN. Returns the validated narration, or **None** when
    the LLM output violates the fence — a bare numeral (PCN), or a `{{token}}` that
    isn't a real claim. `None` means: withhold narration, print the plain brief.

    Numbers in the returned string come ONLY from `claim_set[...].rendered`, so they
    are the validated core's figures by construction; the model supplied only words.
    """
    if not prose.strip():
        return None
    if _has_bare_numeral(prose):
        return None  # PCN: the model wrote a literal number
    if any(m.group(1).lower() not in claim_set for m in _TOKEN_RE.finditer(prose)):
        return None  # referenced a claim that doesn't exist

    def _sub(m: re.Match[str]) -> str:
        return claim_set[m.group(1).lower()].rendered

    out = _TOKEN_RE.sub(_sub, prose)
    if "{{" in out or "}}" in out:
        return None  # a malformed/truncated placeholder (e.g. `{{tok}`) survived → fail closed
    return out


# ── prompt building (pure): claims → an LLM prompt + the privacy dial ─────────

_SYSTEM_PROMPT = (
    "You write a SHORT plain-language summary (3-4 sentences) of a personal "
    "investment portfolio, for its owner.\n"
    "HARD RULES:\n"
    "- Reference EVERY figure ONLY by its {{token}} placeholder from the list "
    "below. NEVER write a digit, percent, dollar amount, or date yourself.\n"
    "- Do NOT quantify in words either: no 'doubled', 'halved', 'a third', 'nearly "
    "all'. If a figure isn't in the list, describe direction or quality "
    "qualitatively ('rose', 'fell', 'steady') without putting a number on it.\n"
    "- Name funds and indices WITHOUT their number ('the S&P index', not 'S&P 500'): "
    "any stray digit voids the entire summary.\n"
    "- Lead with drawdown (how far it fell from its peak), then risk-adjusted "
    "performance, then return.\n"
    "- Calm, concrete, plain. No hype, no recommendations, no predictions.\n"
    "- It is a description, not financial advice.\n"
    "Output ONLY the summary prose — no preamble, no heading, no bullet points."
)


def build_prompt(claim_set: dict[str, Claim], *, tier: str) -> tuple[str, str]:
    """Pure: the (system, user) prompts for the narrator. The **privacy dial** lives
    here — PAID/LOCAL send each claim's exact rendered value (richer prose; a paid
    provider doesn't train on inputs, a local model never leaves the machine); FREE
    sends only the coarse qualitative band (``Claim.band``), so exact values stay home
    and are substituted locally by `render_narration`. Token NAMES + labels always go
    (the model must know what it may cite)."""
    send_exact = tier in ("paid", "local")
    lines: list[str] = []
    for token in sorted(claim_set):
        c = claim_set[token]
        hint = c.rendered if send_exact else (c.band or "")
        lines.append("{{" + token + "}}  — " + c.label + (f" ({hint})" if hint else ""))
    user = (
        "Available figures — cite each ONLY as its {{token}}, never the value:\n"
        + "\n".join(lines)
        + "\n\nWrite the summary now."
    )
    return _SYSTEM_PROMPT, user
