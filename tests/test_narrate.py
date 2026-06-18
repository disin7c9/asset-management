"""Tests for the narration fence (SymGen + PCN). Pure — no LLM, no I/O.

The safety property under test: a number can enter the narration ONLY by
substitution from the validated claim set; any digit the model types itself is
rejected wholesale (fail-closed).
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from app.derive import DerivedState, Position
from app.narrate import Claim, build_claim_set, build_prompt, render_narration
from app.prices import PriceRow
from app.returns import ReturnsSummary
from app.risk import DrawdownInfo, MetricCI, RiskSummary


@pytest.fixture
def core() -> tuple[DerivedState, dict[str, PriceRow], ReturnsSummary, RiskSummary]:
    state = DerivedState()
    state.positions["AAA"] = Position(ticker="AAA", shares=10.0, cost_basis=1000.0)
    state.realized["AAA"] = 50.0
    prices = {
        "AAA": PriceRow("AAA", date(2026, 6, 10), 120.0, "cache",
                        datetime(2026, 6, 10, tzinfo=timezone.utc)),
    }
    returns = ReturnsSummary(
        period_start=date(2023, 1, 5),
        asof_date=date(2026, 6, 2),
        money_weighted_annualized=0.1828,
        modified_dietz_annualized=0.1794,
        true_twr_annualized=0.1937,
    )
    risk = RiskSummary(
        n_days=300,
        drawdown=DrawdownInfo(
            depth=-0.0984, peak_date=date(2025, 2, 19), trough_date=date(2025, 4, 8),
            recovery_date=date(2025, 6, 10), duration_days=111, time_underwater_pct=0.80,
        ),
        max_drawdown_ci=MetricCI(-0.0984, -0.1663, -0.0637),
        ulcer_index=MetricCI(0.0238, 0.0151, 0.0626),
        cdar=MetricCI(0.0673, 0.0443, 0.1401),
        sharpe=MetricCI(1.75, 0.68, 2.83),
        sortino=MetricCI(2.66, 0.96, 4.63),
        calmar=MetricCI(1.97, 0.52, 4.83),
    )
    return state, prices, returns, risk


def test_claim_set_renders_core_figures(
    core: tuple[DerivedState, dict[str, PriceRow], ReturnsSummary, RiskSummary],
) -> None:
    state, prices, returns, risk = core
    c = build_claim_set(state, prices, returns, risk)
    assert c["max_drawdown"].rendered == "-9.84%"
    assert c["sharpe"].rendered == "+1.75"
    assert c["sortino"].rendered == "+2.66"
    assert c["calmar"].rendered == "+1.97"
    assert c["ulcer"].rendered == "2.38%"
    assert c["cdar"].rendered == "6.73%"
    assert c["twr_annual"].rendered == "+19.37%"
    assert c["mwr_annual"].rendered == "+18.28%"
    assert c["total_market_value"].rendered == "$1,200"
    assert c["unrealized_pnl"].rendered == "$200"
    assert c["net_pnl"].rendered == "$250"
    assert c["realized_pnl"].rendered == "$50"
    assert c["drawdown_peak_date"].rendered == "2025-02-19"
    assert c["time_underwater_pct"].rendered == "80%"
    assert c["drawdown_duration_days"].rendered == "111 days"
    assert c["n_holdings"].rendered == "1"


def test_symgen_substitutes_validated_values(
    core: tuple[DerivedState, dict[str, PriceRow], ReturnsSummary, RiskSummary],
) -> None:
    state, prices, returns, risk = core
    c = build_claim_set(state, prices, returns, risk)
    out = render_narration(
        "You fell {{max_drawdown}} from your {{drawdown_peak_date}} peak; Sharpe {{sharpe}}.", c
    )
    assert out == "You fell -9.84% from your 2025-02-19 peak; Sharpe +1.75."


def test_pcn_rejects_a_bare_numeral(
    core: tuple[DerivedState, dict[str, PriceRow], ReturnsSummary, RiskSummary],
) -> None:
    state, prices, returns, risk = core
    c = build_claim_set(state, prices, returns, risk)
    # The model typed a fabricated figure instead of a token → rejected wholesale.
    assert render_narration("Your portfolio fell 9.8% this year.", c) is None
    # Even mixed with a valid token, any bare digit fails closed.
    assert render_narration("Drawdown {{max_drawdown}}, plus about 12% more.", c) is None


def test_rejects_unknown_token(
    core: tuple[DerivedState, dict[str, PriceRow], ReturnsSummary, RiskSummary],
) -> None:
    state, prices, returns, risk = core
    c = build_claim_set(state, prices, returns, risk)
    assert render_narration("Your {{fabricated_metric}} looks great.", c) is None


def test_prose_without_numbers_passes_through(
    core: tuple[DerivedState, dict[str, PriceRow], ReturnsSummary, RiskSummary],
) -> None:
    state, prices, returns, risk = core
    c = build_claim_set(state, prices, returns, risk)
    txt = "Your portfolio had a calm, steady run with little drama."
    assert render_narration(txt, c) == txt


def test_empty_prose_is_none(
    core: tuple[DerivedState, dict[str, PriceRow], ReturnsSummary, RiskSummary],
) -> None:
    state, prices, returns, risk = core
    c = build_claim_set(state, prices, returns, risk)
    assert render_narration("", c) is None
    assert render_narration("   \n  ", c) is None


def test_token_name_with_digit_is_not_flagged_by_pcn() -> None:
    # A digit INSIDE a token name (e.g. top10) is a reference, not a model-typed
    # number; PCN strips the whole token first, and "93%" arrives by substitution.
    cs = {"top10_overlap": Claim("top10_overlap", 0.93, "93%", "top-10 overlap")}
    assert render_narration("Overlap is {{top10_overlap}}.", cs) == "Overlap is 93%."


def test_malformed_or_truncated_token_is_rejected(
    core: tuple[DerivedState, dict[str, PriceRow], ReturnsSummary, RiskSummary],
) -> None:
    # A single-brace typo / a reply truncated mid-token leaves a stray {{ or }} that
    # _TOKEN_RE never substitutes; rather than print the raw brace artifact, fail closed.
    state, prices, returns, risk = core
    c = build_claim_set(state, prices, returns, risk)
    assert render_narration("Returns were {{twr_annual} strong.", c) is None   # one brace
    assert render_narration("You fell {{max_drawdown", c) is None              # truncated


def test_undefined_metrics_are_omitted() -> None:
    # Calmar (no drawdown) and Sortino (no downside) come back non-finite from the
    # core; they must NOT be referenceable claims, and citing one fails closed.
    state = DerivedState()
    state.positions["AAA"] = Position(ticker="AAA", shares=10.0, cost_basis=1000.0)
    risk = RiskSummary(
        n_days=300,
        drawdown=DrawdownInfo(
            depth=0.0, peak_date=date(2025, 1, 1), trough_date=date(2025, 1, 1),
            recovery_date=None, duration_days=0, time_underwater_pct=0.0,
        ),
        max_drawdown_ci=MetricCI(0.0, 0.0, 0.0),
        ulcer_index=MetricCI(0.0, 0.0, 0.0),
        cdar=MetricCI(0.0, 0.0, 0.0),
        sharpe=MetricCI(2.0, 1.0, 3.0),
        sortino=MetricCI(float("inf"), float("inf"), float("inf")),
        calmar=MetricCI(float("nan"), float("nan"), float("nan")),
    )
    c = build_claim_set(state, {}, None, risk)
    assert "sharpe" in c
    assert "calmar" not in c
    assert "sortino" not in c
    assert render_narration("Calmar is {{calmar}}.", c) is None


def test_non_finite_drawdown_depth_is_omitted() -> None:
    # The central add() guard: even an UNGATED claim (drawdown depth, P&L, $ giveback)
    # is dropped if non-finite, so the fence can never substitute "nan%"/"$inf" — which
    # PCN's bare-digit check would not catch.
    state = DerivedState()
    state.positions["AAA"] = Position(ticker="AAA", shares=10.0, cost_basis=1000.0)
    risk = RiskSummary(
        n_days=300,
        drawdown=DrawdownInfo(
            depth=float("nan"), peak_date=date(2025, 1, 1), trough_date=date(2025, 1, 1),
            recovery_date=None, duration_days=0, time_underwater_pct=0.0,
        ),
        max_drawdown_ci=MetricCI(0.0, 0.0, 0.0),
        ulcer_index=MetricCI(0.0, 0.0, 0.0),
        cdar=MetricCI(0.0, 0.0, 0.0),
        sharpe=MetricCI(2.0, 1.0, 3.0),
        sortino=MetricCI(2.0, 1.0, 3.0),
        calmar=MetricCI(2.0, 1.0, 3.0),
    )
    c = build_claim_set(state, {}, None, risk)
    assert "max_drawdown" not in c  # non-finite depth dropped, not rendered "nan%"
    assert render_narration("You fell {{max_drawdown}}.", c) is None


def test_system_prompt_forbids_words_and_index_numbers() -> None:
    # #4/#7: the prompt must not invite verbal magnitudes, and must tell the model to
    # name indices without their number (any stray digit voids the whole summary).
    cs = {"sharpe": Claim("sharpe", 1.75, "+1.75", "Sharpe ratio")}
    system, _user = build_prompt(cs, tier="paid")
    low = system.lower()
    assert "doubled" in low and "halved" in low      # verbal quantities are forbidden
    assert "s&p" in low                              # name indices without the number


def test_build_prompt_paid_sends_exact_values(
    core: tuple[DerivedState, dict[str, PriceRow], ReturnsSummary, RiskSummary],
) -> None:
    state, prices, returns, risk = core
    c = build_claim_set(state, prices, returns, risk)
    system, user = build_prompt(c, tier="paid")
    assert "never write a digit" in system.lower()  # the LLM rule is in the system prompt
    assert "{{max_drawdown}}" in user
    assert "-9.84%" in user  # paid → the exact rendered value is in the prompt


def test_build_prompt_free_sends_buckets_not_values(
    core: tuple[DerivedState, dict[str, PriceRow], ReturnsSummary, RiskSummary],
) -> None:
    state, prices, returns, risk = core
    c = build_claim_set(state, prices, returns, risk)
    _system, user = build_prompt(c, tier="free")
    assert "{{max_drawdown}}" in user
    # The privacy dial: exact values stay home; only a coarse band leaves.
    assert "-9.84%" not in user and "+1.75" not in user and "$1,200" not in user
    assert "(moderate)" in user  # max_drawdown -9.84% → moderate band
    assert "solid" in user       # sharpe +1.75 → solid band


def test_band_lives_on_the_claim(
    core: tuple[DerivedState, dict[str, PriceRow], ReturnsSummary, RiskSummary],
) -> None:
    # #9: the FREE-tier band is computed with the claim (not re-dispatched on token
    # name in build_prompt). Banded metrics carry one; dates/counts/$ amounts don't.
    state, prices, returns, risk = core
    c = build_claim_set(state, prices, returns, risk)
    assert c["max_drawdown"].band == "moderate"   # |-9.84%| ∈ [5%, 15%)
    assert c["sharpe"].band == "solid"            # +1.75 ∈ [1.0, 2.0)
    assert c["twr_annual"].band == "healthy"      # +19.37% ∈ [10%, 20%)
    assert c["ulcer"].band == "moderate"          # 2.38% ∈ [2%, 5%)
    assert c["cdar"].band == "moderate"           # 6.73% ∈ [5%, 15%)
    assert c["total_market_value"].band is None   # $ amount → no band
    assert c["drawdown_peak_date"].band is None   # date → no band


def test_band_for_fails_closed_on_non_finite() -> None:
    # #4: a band must never assert "severe"/"strong" for a non-finite (n/a) figure; an
    # unknown token has no spec → None.
    from app.narrate import _band_for

    assert _band_for("sharpe", float("nan")) is None
    assert _band_for("max_drawdown", float("inf")) is None
    assert _band_for("unknown_token", 1.0) is None


def test_build_prompt_local_sends_exact_values(
    core: tuple[DerivedState, dict[str, PriceRow], ReturnsSummary, RiskSummary],
) -> None:
    # tier=local (a model on your own machine) sends exact values like paid — nothing
    # leaves the machine, so there's no reason to withhold them.
    state, prices, returns, risk = core
    c = build_claim_set(state, prices, returns, risk)
    _system, user = build_prompt(c, tier="local")
    assert "-9.84%" in user and "+1.75" in user  # exact, like paid


def test_fence_over_a_fake_model(
    core: tuple[DerivedState, dict[str, PriceRow], ReturnsSummary, RiskSummary],
) -> None:
    # The full P2 flow with a canned model output: claims → prompt → (the "model"
    # writes prose with tokens) → fence substitutes the validated values.
    state, prices, returns, risk = core
    c = build_claim_set(state, prices, returns, risk)
    build_prompt(c, tier="paid")  # would be sent to the LLM
    model_output = "You fell {{max_drawdown}} from your {{drawdown_peak_date}} peak; Sharpe {{sharpe}}."
    assert render_narration(model_output, c) == "You fell -9.84% from your 2025-02-19 peak; Sharpe +1.75."


# A golden battery pinning the fence's behavior on realistic model prose (P2c). Every
# number in an accepted output arrives by substitution; anything else fails closed.
_GOLDEN: tuple[tuple[str, str | None], ...] = (
    # accepted — figures substituted from the validated core
    ("You fell {{max_drawdown}} from your {{drawdown_peak_date}} peak; Sharpe {{sharpe}}.",
     "You fell -9.84% from your 2025-02-19 peak; Sharpe +1.75."),
    ("Risk-adjusted, the run looks {{sharpe}} on a Sharpe basis and {{sortino}} on Sortino.",
     "Risk-adjusted, the run looks +1.75 on a Sharpe basis and +2.66 on Sortino."),
    ("A calm, steady stretch — nothing dramatic to report.",
     "A calm, steady stretch — nothing dramatic to report."),
    # rejected (→ None) — fail closed
    ("Your portfolio fell about 10% this year.", None),                  # bare digit
    ("Drawdown {{max_drawdown}}, and roughly 12% more besides.", None),  # digit beside a token
    ("Your fund tracks the S&P 500 index.", None),                       # incidental index number
    ("Your {{made_up_metric}} looks great.", None),                      # unknown token
    ("Returns were {{twr_annual} strong.", None),                        # malformed (one brace)
    ("", None),                                                          # empty
)


@pytest.mark.parametrize("prose, expected", _GOLDEN)
def test_fence_golden_set(
    core: tuple[DerivedState, dict[str, PriceRow], ReturnsSummary, RiskSummary],
    prose: str,
    expected: str | None,
) -> None:
    state, prices, returns, risk = core
    c = build_claim_set(state, prices, returns, risk)
    assert render_narration(prose, c) == expected


def test_verbal_magnitude_passes_the_fence_is_a_known_residual(
    core: tuple[DerivedState, dict[str, PriceRow], ReturnsSummary, RiskSummary],
) -> None:
    # KNOWN residual: PCN blocks model-typed DIGITS, not words. A magnitude stated in
    # words has no digit, so the fence passes it through — the system prompt is what
    # discourages it (#7), not the fence. Pinned so the boundary stays explicit.
    state, prices, returns, risk = core
    c = build_claim_set(state, prices, returns, risk)
    txt = "Your portfolio nearly doubled over the period."
    assert render_narration(txt, c) == txt


def test_compute_narration_builds_summary_section(
    core: tuple[DerivedState, dict[str, PriceRow], ReturnsSummary, RiskSummary],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import cli
    from app.llm import NarratorConfig

    state, prices, returns, risk = core
    cfg = NarratorConfig("anthropic", "claude-haiku-4-5", "k", "", "paid", None)
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(
        cli, "complete",
        lambda *_a, **_k: "You fell {{max_drawdown}} from your {{drawdown_peak_date}} peak.",
    )
    run: dict[str, object] = {}
    sec = cli._compute_narration(state, prices, returns, risk, None, run)
    assert sec is not None
    assert sec.title == "SUMMARY"
    assert sec.lines[0] == "You fell -9.84% from your 2025-02-19 peak."  # substituted
    assert any("wording by claude-haiku-4-5 (paid tier)" in line for line in sec.lines)
    assert str(run["narrate"]).startswith("claude-haiku-4-5")


def test_compute_narration_withholds_on_fabrication(
    core: tuple[DerivedState, dict[str, PriceRow], ReturnsSummary, RiskSummary],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import cli
    from app.llm import NarratorConfig

    state, prices, returns, risk = core
    cfg = NarratorConfig("anthropic", "m", "k", "", "paid", None)
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(cli, "complete", lambda *_a, **_k: "You fell 10% this year.")  # bare digit
    run: dict[str, object] = {}
    assert cli._compute_narration(state, prices, returns, risk, None, run) is None
    assert "withheld" in str(run["narrate"])
