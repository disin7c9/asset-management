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

from app.backtest import BenchmarkResult, BenchmarkVerdict
from app.derive import DerivedState
from app.discover import Discovery
from app.prices import PriceRow
from app.returns import ReturnsSummary
from app.risk import DollarDrawdown, RiskSummary
from app.screen import CandidateScreen


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
_DD_BAND = _BandSpec(_DD_CUTS, "severe", use_abs=True)  # depth is negative; shared by the
#   brief's max_drawdown and the benchmark legs' {{bench_dd_*}} depths (slice 2b)
# Book exposure (0..1) to a discovery gap role — how much of this slice you already hold.
_EXPOSURE_BAND = _BandSpec(((0.005, "none"), (0.03, "very little"), (0.10, "some")), "a fair amount")

# token → band spec. ONE table (was a per-token if-ladder in build_prompt): each
# metric's band is computed *with* its Claim, so adding a metric can't silently drop
# its FREE-tier band, and the privacy dial just reads ``c.band`` — it never
# re-dispatches on token name. (Dates/counts/$ amounts have no band → not listed.)
_BAND_SPECS: dict[str, _BandSpec] = {
    "max_drawdown": _DD_BAND,                                     # depth is negative
    "cdar": _BandSpec(_DD_CUTS, "severe"),                        # already a positive magnitude
    "ulcer": _BandSpec(((0.02, "mild"), (0.05, "moderate"), (0.10, "significant")), "severe"),
    "sharpe": _RATIO_BAND, "sortino": _RATIO_BAND, "calmar": _RATIO_BAND,
    "twr_annual": _RETURN_BAND, "mwr_annual": _RETURN_BAND, "dietz_annual": _RETURN_BAND,
}

# Token-PREFIX band families: a consumer whose claims are named by a stable prefix (one
# per gap role, one per benchmark leg) shares a band by that prefix, so a new claim in the
# family is banded automatically. The brief's fixed metrics match by exact name above;
# these match by prefix only when no exact spec exists.
_PREFIX_BANDS: tuple[tuple[str, _BandSpec], ...] = (
    ("gap_", _EXPOSURE_BAND),       # discovery: the book's exposure to a gap role
    ("bench_dd_", _DD_BAND),        # benchmark: a leg's max-drawdown depth
    ("bench_ulcer_", _BAND_SPECS["ulcer"]),  # benchmark: a leg's Ulcer index (the verdict stat)
)


def _band_for(token: str, value: float | int | str | date) -> str | None:
    """The FREE-tier band for a claim, or None for a token without one. Computed once
    at claim construction and stored on ``Claim.band``. Fails closed (None) on a
    non-finite or non-numeric value — a band must never assert "severe" for a figure that
    is actually n/a (matches the fence's ethos)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    spec = _BAND_SPECS.get(token) or next(
        (s for prefix, s in _PREFIX_BANDS if token.startswith(prefix)), None
    )
    return spec.label_for(float(value)) if spec is not None else None


def _add_claim(
    claims: dict[str, Claim],
    token: str,
    value: float | int | str | date,
    rendered: str,
    label: str,
) -> None:
    """Insert ONE claim, skipping a non-finite float figure — the single fail-closed gate
    EVERY claim-builder routes through (brief / discovery / benchmark), so a nan/inf can
    never reach the fence as "nan%"/"$inf" (which PCN's ``\\d`` would not catch). The
    FREE-tier band derives once here from (token, value), so a new claim can't silently
    lose its privacy band."""
    if isinstance(value, float) and not math.isfinite(value):
        return
    claims[token] = Claim(token=token, value=value, rendered=rendered, label=label,
                          band=_band_for(token, value))


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
        _add_claim(claims, token, value, rendered, label)  # shared fail-closed guard + band

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


def _claim_listing(claim_set: dict[str, Claim], *, send_exact: bool) -> str:
    """One line per claim — ``{{token}}  — label (hint)``. The privacy dial: ``send_exact``
    (PAID/LOCAL) appends the exact rendered value; FREE appends only the coarse band (or
    nothing). Token names + labels always go — the model must know what it may cite."""
    lines: list[str] = []
    for token in sorted(claim_set):
        c = claim_set[token]
        hint = c.rendered if send_exact else (c.band or "")
        lines.append("{{" + token + "}}  — " + c.label + (f" ({hint})" if hint else ""))
    return "\n".join(lines)


def build_prompt(claim_set: dict[str, Claim], *, tier: str) -> tuple[str, str]:
    """Pure: the (system, user) prompts for the narrator. The **privacy dial** lives
    here — PAID/LOCAL send each claim's exact rendered value (richer prose; a paid
    provider doesn't train on inputs, a local model never leaves the machine); FREE
    sends only the coarse qualitative band (``Claim.band``), so exact values stay home
    and are substituted locally by `render_narration`."""
    user = (
        "Available figures — cite each ONLY as its {{token}}, never the value:\n"
        + _claim_listing(claim_set, send_exact=tier in ("paid", "local"))
        + "\n\nWrite the summary now."
    )
    return _SYSTEM_PROMPT, user


# ── discovery narration (P3b): rank/explain the screened gap-fillers ──────────
#
# The SAME fence (Claim + render_narration), pointed at --discover's output. The LLM
# ranks/explains candidates that ALREADY PASSED the deterministic screen, strictly on
# ROLE FIT — never a return forecast. Candidate facts stay qualitative (the exact
# cost/overlap/correlation figures live in the deterministic panel); only the book's
# gap exposure is a citable figure, banded for FREE like every other claim.

# Plain-English role names for the prose — the slugs ("bond-aggregate") read as jargon,
# and the model may write these directly (they carry no digit, so the fence allows them),
# keeping the {{gap_*}} token for the exposure FIGURE, not the role's name.
_ROLE_NAMES: dict[str, str] = {
    "us-large": "US large-cap stocks",
    "us-small-mid": "US small- and mid-cap stocks",
    "us-dividend": "US dividend stocks",
    "intl-developed": "developed international stocks",
    "em-equity": "emerging-market stocks",
    "sector-equity": "sector and thematic equities",
    "bond-aggregate": "total US bonds",
    "treasury": "US Treasuries",
    "tips": "inflation-protected bonds (TIPS)",
    "corporate-bond": "corporate bonds",
    "gold": "gold",
    "commodity-broad": "broad commodities",
    "reit": "real estate (REITs)",
}

# Number-free role-fit words per screen check + status — what the model may echo.
_CHECK_WORDS: dict[str, dict[str, str]] = {
    "cost": {"pass": "cheap", "warn": "a bit pricey", "fail": "expensive"},
    "liquidity": {"pass": "liquid", "warn": "thinly traded", "fail": "illiquid"},
    "overlap": {
        "pass": "little overlap with what you hold",
        "warn": "some overlap with what you hold",
        "fail": "heavy overlap with what you hold",
    },
    "diversifier": {
        "pass": "diversified your past drawdowns",
        "warn": "a weak diversifier for your book",
        "fail": "moved with your book",
    },
}


_MAX_LABEL_NAME = 80


def _one_line(text: str) -> str:
    """Collapse a provider-controlled string to a single bounded line.

    Fund names come from the metadata provider and from `ASSET_UNIVERSE`, which a user can
    repoint at any CSV — neither is ours. They are interpolated into the claim label, and
    the label is rendered into the prompt as one `{{token}} — label` LINE per claim. A name
    carrying a newline therefore writes new lines into the prompt, where anything after it
    reads as further instructions to the narrator. The fence still stops fabricated FIGURES
    (the renderer substitutes validated values), but prose direction is not numeric, so this
    is the layer that has to hold. Collapsing whitespace removes the mechanism entirely.
    """
    return " ".join(text.split())[:_MAX_LABEL_NAME]


def _fit_phrase(screen: CandidateScreen) -> str:
    """A number-free role-fit phrase from the screen's headline checks (cost / liquidity
    / overlap / diversifier) — the qualitative judgement the model may echo. The exact
    figures stay in the deterministic DISCOVERY panel, never in the prompt."""
    return ", ".join(
        _CHECK_WORDS[chk.name][chk.status]
        for chk in screen.checks
        if chk.name in _CHECK_WORDS and chk.status in _CHECK_WORDS[chk.name]
    )


def _claim_token(prefix: str, raw: str) -> str:
    """A valid {{token}} from a role/ticker: lowercased, non-alphanumerics → ``_``
    (so ``us-large`` / ``BRK.B`` can't smuggle a stray character past the fence)."""
    return prefix + re.sub(r"[^a-z0-9]+", "_", raw.lower())


def build_discovery_claims(
    discovery: Discovery, results: list[CandidateScreen]
) -> dict[str, Claim]:
    """Validated figures the discovery note may cite: the book's exposure to each gap
    role (banded for FREE) and each screened candidate's TICKER (cited by name, so even
    a digit-bearing ticker passes the PCN fence). Candidate cost/overlap/correlation stay
    qualitative — those exact figures live in the panel, not here. A candidate without a
    screen result is omitted: the note can't cite a fund the screen didn't judge."""
    by_ticker = {r.ticker: r for r in results}
    claims: dict[str, Claim] = {}
    for role in discovery.gaps:
        exposure = discovery.exposure.get(role, 0.0)
        _add_claim(  # band: the gap_ prefix → _EXPOSURE_BAND (see _PREFIX_BANDS)
            claims, _claim_token("gap_", role), exposure, f"{exposure * 100:.0f}%",
            f"the book's current exposure to {_ROLE_NAMES.get(role, role)}",
        )
    for c in discovery.candidates:
        r = by_ticker.get(c.ticker)
        if r is None:
            continue
        label = f"{_one_line(c.name)}, a {c.role} fund — the screen rates it {r.verdict.upper()}"
        fit = _fit_phrase(r)
        if fit:
            label += f" ({fit})"
        _add_claim(claims, _claim_token("cand_", c.ticker), c.ticker, c.ticker, label)
    return claims


_DISCOVERY_SYSTEM_PROMPT = (
    "You help the owner of a personal portfolio consider NEW funds for the roles "
    "their book is light in. You are given each gap role and a short list of candidate "
    "funds, each SCORED by a deterministic screen (cost, liquidity, overlap with what "
    "they hold, and whether the fund diversified their past drawdowns) — each "
    "candidate's verdict, PASS / WARN / FAIL, is in its label.\n"
    "Write a SHORT plain-language note (3-5 sentences): for each gap role, which "
    "candidate(s) look worth a closer look and WHY — strictly on ROLE FIT (how cheap, "
    "how liquid, how little it overlaps what they already hold, whether it diversified "
    "their drawdowns). Favor PASS verdicts; mention a WARN's caveat; do NOT recommend a "
    "FAIL (you may say why to steer clear).\n"
    "HARD RULES:\n"
    "- Refer to every fund and figure ONLY by its {{token}} placeholder. NEVER write a "
    "ticker symbol, digit, percent, or dollar amount yourself — one stray character "
    "voids the whole note. (Some fund names contain digits, e.g. year ranges — cite the "
    "{{token}}, never spell the fund's name.)\n"
    "- You MAY name each role in plain English ('real estate', 'bonds', 'emerging "
    "markets') — role names are words, not figures. Use a {{gap_...}} token ONLY for the "
    "exposure figure, never as the role's name.\n"
    "- Rank ONLY on role fit and the screen's verdict. NEVER predict returns or say one "
    "fund will out-perform, beat, grow more, or do better than another — you have no "
    "performance figures and past return is NOT the criterion.\n"
    "- Do NOT quantify in words ('half the cost', 'twice as liquid'); stay qualitative "
    "('cheaper', 'more liquid', 'less overlap').\n"
    "- A PASS means 'sane, cheap, liquid, genuinely different' — NOT a buy and NOT "
    "advice. Frame every candidate as 'worth a look', for the owner to judge.\n"
    "Output ONLY the note — no preamble, no heading, no bullet list."
)


def build_discovery_prompt(claim_set: dict[str, Claim], *, tier: str) -> tuple[str, str]:
    """Pure: the (system, user) prompts for the discovery note. Same privacy dial as
    `build_prompt` — FREE sends the coarse band for the gap exposures and the (already
    qualitative) candidate labels; PAID/LOCAL send the exact rendered exposures. Nothing
    sensitive about the candidates leaves on FREE — their facts are qualitative here."""
    user = (
        "Gap roles and the screened candidate funds — cite each ONLY as its {{token}}:\n"
        + _claim_listing(claim_set, send_exact=tier in ("paid", "local"))
        + "\n\nWrite the note now."
    )
    return _DISCOVERY_SYSTEM_PROMPT, user


# ── benchmark narration (slice 2b): explain the preset-vs-reference verdict ────
#
# The SAME fence, pointed at --benchmark's result. The note is drawdown-first and HONEST:
# where the posture's drawdown landed vs a canonical reference (60-40 / all-weather /
# permanent), with the held-out verdict injected as FIXED framing — never "beats" /
# "outperforms", never a forward prediction. Only the two legs' drawdown DEPTHS and the
# reference's name are citable figures; the verdict is words the prompt pins, so the model
# can't strengthen "no clear difference" into a win.

# Plain, digit-free display names for the reference (the slug "60-40" carries a digit, so
# the model MUST cite {{bench_reference}}, never type the name). Presentation — kept here
# alongside _ROLE_NAMES, not in backtest's canonical _BENCHMARKS.
_BENCHMARK_DISPLAY: dict[str, str] = {
    "60-40": "the classic 60/40 stock-and-bond mix",
    "all-weather": "the All-Weather portfolio",
    "permanent": "the Permanent Portfolio",
}

# The held-out verdict as ONE fixed, number-free sentence the model must convey AS-IS and
# never strengthen. Since v2.9.0 the verdict is judged on the ULCER index (whole-window
# drawdown pain), so the cited figures ARE the two OOS Ulcer values (see build_benchmark_claims)
# and these sentences describe what Ulcer measures WITHOUT over-claiming: a lower Ulcer does not
# on its own prove the declines were both shallower AND shorter, so we say "how deep and how
# long, combined" (what the metric is), never assert both moved.
_VERDICT_SENTENCE: dict[BenchmarkVerdict, str] = {
    "shallower": "Over the tested out-of-sample window, your posture carried LESS overall "
    "drawdown pain than the reference — a single whole-window measure of how deep and how "
    "long its declines ran, combined.",
    "deeper": "Over the tested out-of-sample window, your posture carried MORE overall "
    "drawdown pain than the reference — a single whole-window measure of how deep and how "
    "long its declines ran, combined.",
    "inconclusive": "A held-out test found NO CLEAR difference in overall drawdown pain "
    "between the two — an honest, common result on a short history.",
    "insufficient": "There was not enough overlapping history to run a held-out test, so "
    "the comparison is not yet judged.",
}


def build_benchmark_claims(result: BenchmarkResult) -> dict[str, Claim]:
    """Validated figures the benchmark note may cite, ALL from the held-out (out-of-sample)
    window the verdict is actually judged on: the two Ulcer indices — your posture's and the
    reference's — which ARE the verdict statistic (banded via the bench_ulcer_ prefix), plus
    the reference's NAME (cited by {{token}} so a digit-bearing name like "60-40" can't trip
    PCN). Ulcer, not max-DD, is cited so the numbers can never point opposite to the Ulcer-based
    verdict word (the pre-v2.9.0 failure: a shallower single drop next to a 'more pain' verdict).
    The VERDICT itself is NOT a number here — it's a word mandated as fixed framing in the
    prompt. Returns {} when there is no held-out window (an 'insufficient'/too-short run has
    nothing honest to cite)."""
    if result.oos is None:
        return {}
    oos = result.oos
    claims: dict[str, Claim] = {}
    _add_claim(claims, "bench_ulcer_preset", oos.ulcer_with, _mag(oos.ulcer_with),
               "your posture's overall drawdown pain (Ulcer index) over the held-out window")
    _add_claim(claims, "bench_ulcer_reference", oos.ulcer_without, _mag(oos.ulcer_without),
               "the reference's overall drawdown pain (Ulcer index) over the same window")
    name = _BENCHMARK_DISPLAY.get(result.reference, result.reference)
    _add_claim(claims, "bench_reference", name, name,
               "the name of the well-known reference you are compared against")
    return claims


_BENCHMARK_SYSTEM_PROMPT = (
    "You explain, in plain language for a portfolio's owner, how their chosen posture (a "
    "conservative / moderate / aggressive preset mix) compares to a well-known reference "
    "portfolio — DRAWDOWN-FIRST.\n"
    "You are given each side's overall drawdown pain (its Ulcer index) and ONE fixed verdict "
    "sentence stating what a held-out recent-window test found. Put that verdict in plain "
    "words for the owner.\n"
    "HARD RULES:\n"
    "- Refer to every figure AND the reference's name ONLY by its {{token}} placeholder. "
    "NEVER write a digit, percent, or portfolio name yourself — one stray character voids "
    "the whole note. (The reference's name may contain digits; cite {{bench_reference}}, "
    "never spell it.)\n"
    "- Lead with overall drawdown pain: your posture's {{bench_ulcer_preset}} versus the "
    "reference's {{bench_ulcer_reference}}. This is the whole-window measure the verdict is "
    "judged on (how deep AND how long declines ran, combined) — NOT the single worst drop.\n"
    "- Convey the verdict EXACTLY as given and NO stronger. NEVER say one portfolio 'beats', "
    "'outperforms', 'wins', or is 'better than' the other — the finding is about overall "
    "drawdown pain alone, and on a short history 'no clear difference' is a common, honest "
    "result.\n"
    "- NEVER predict returns or say which will do better going forward; you have no forward "
    "figures and past drawdown is not a forecast.\n"
    "- Do NOT quantify in words ('twice as deep', 'half'); stay qualitative ('a little "
    "deeper', 'about the same').\n"
    "- This is a description for the owner to judge — NOT advice, NOT a recommendation.\n"
    "Output ONLY the note (2-4 sentences) — no preamble, no heading, no bullet points."
)


def build_benchmark_prompt(
    claim_set: dict[str, Claim], result: BenchmarkResult, *, tier: str
) -> tuple[str, str]:
    """Pure: the (system, user) prompts for the benchmark note. Same privacy dial as the
    other builders. The held-out verdict is injected as a FIXED sentence the model must
    convey without strengthening — so it can never turn 'no clear difference' into a win."""
    user = (
        "Figures — cite each ONLY as its {{token}}, never the value:\n"
        + _claim_listing(claim_set, send_exact=tier in ("paid", "local"))
        + "\n\nThe held-out verdict you MUST convey (in your own words, but never stronger "
        "than this states):\n"
        + _VERDICT_SENTENCE.get(result.verdict, _VERDICT_SENTENCE["inconclusive"])
        + "\n\nWrite the note now."
    )
    return _BENCHMARK_SYSTEM_PROMPT, user
