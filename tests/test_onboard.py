"""The onboarding rubric: fixed, explainable, and every posture reachable."""

from __future__ import annotations

import pytest

from app.allocate import PRESETS
from app.onboard import MAX_SCORE, QUESTIONS, posture_from_answers


def test_questions_shape() -> None:
    # Three questions, three options each, unique keys, scores exactly {0, 1, 2}:
    # the rubric's arithmetic (thresholds, MAX_SCORE) assumes this shape.
    assert [q.key for q in QUESTIONS] == ["horizon", "loss_response", "cash_buffer"]
    for q in QUESTIONS:
        keys = [o.key for o in q.options]
        assert len(keys) == len(set(keys)) == 3
        assert sorted(o.score for o in q.options) == [0, 1, 2]
        assert all(o.label and o.note for o in q.options)
    assert MAX_SCORE == 6


def test_every_posture_reachable() -> None:
    assert posture_from_answers("3_to_10_years", "sell", "no").posture == "conservative"
    assert posture_from_answers("3_to_10_years", "hold", "partly").posture == "moderate"
    assert posture_from_answers("over_10_years", "buy_more", "comfortably").posture == "aggressive"


def test_score_thresholds() -> None:
    # 2 → conservative, 3 → moderate, 4 → moderate, 5 → aggressive (no cap in play).
    assert posture_from_answers("3_to_10_years", "hold", "no").score == 2
    assert posture_from_answers("3_to_10_years", "hold", "no").posture == "conservative"
    assert posture_from_answers("3_to_10_years", "hold", "partly").score == 3
    assert posture_from_answers("over_10_years", "hold", "partly").score == 4
    assert posture_from_answers("over_10_years", "hold", "partly").posture == "moderate"
    assert posture_from_answers("over_10_years", "buy_more", "partly").score == 5
    assert posture_from_answers("over_10_years", "buy_more", "partly").posture == "aggressive"


def test_short_horizon_caps_at_conservative() -> None:
    # Max other answers with a sub-3y horizon = score 4 (moderate by score) → capped.
    res = posture_from_answers("under_3_years", "buy_more", "comfortably")
    assert res.score == 4
    assert res.posture == "conservative"
    assert any("capped at conservative" in line for line in res.rationale)


def test_rationale_carries_each_answer_and_the_score() -> None:
    # Pin the actual score-line text (not `f"...{res.score}...{res.posture}"`, which would be
    # tautological): 4/6 → moderate by score, uncapped.
    res = posture_from_answers("over_10_years", "hold", "partly")  # 2 + 1 + 1 = 4
    assert len(res.rationale) == 4  # 3 answer notes + the score line
    assert res.score == 4 and res.posture == "moderate"
    assert res.rationale[-1] == "score 4/6 → moderate"


def test_capped_rationale_score_line_shows_the_pre_cap_posture() -> None:
    # When the short-horizon cap fires, the score line must still report the BY-SCORE
    # posture (what the answers earned) and a SEPARATE cap line — not silently rewrite the
    # score line to 'conservative'. Otherwise the trail from answers→score→cap is lost.
    res = posture_from_answers("under_3_years", "buy_more", "comfortably")  # score 4
    assert res.posture == "conservative"  # capped
    assert res.rationale[-2] == "score 4/6 → moderate"  # the pre-cap posture, verbatim
    assert "capped at conservative" in res.rationale[-1]


def test_posture_is_a_valid_preset() -> None:
    for h in ("under_3_years", "3_to_10_years", "over_10_years"):
        for lr in ("sell", "hold", "buy_more"):
            for c in ("no", "partly", "comfortably"):
                assert posture_from_answers(h, lr, c).posture in PRESETS


def test_every_preset_is_reachable_from_some_answer_set() -> None:
    # Bidirectional drift guard (stronger than posture ⊆ PRESETS): if PRESETS gains a
    # fourth posture, the 3-way threshold rubric silently can't reach it — this catches
    # that, mirroring the codebase's `set(...) == ROLES` guards.
    reachable = {
        posture_from_answers(h, lr, c).posture
        for h in ("under_3_years", "3_to_10_years", "over_10_years")
        for lr in ("sell", "hold", "buy_more")
        for c in ("no", "partly", "comfortably")
    }
    assert reachable == PRESETS  # every preset reachable, and no posture outside PRESETS


def test_unknown_answer_raises_naming_the_menu() -> None:
    with pytest.raises(ValueError, match="unknown horizon answer 'tomorrow'.*under_3_years"):
        posture_from_answers("tomorrow", "hold", "partly")
    with pytest.raises(ValueError, match="unknown loss_response"):
        posture_from_answers("over_10_years", "panic", "partly")
    with pytest.raises(ValueError, match="unknown cash_buffer"):
        posture_from_answers("over_10_years", "hold", "maybe")
