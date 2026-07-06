"""Step-0 onboarding: three plain-language risk questions → a starting posture.

Layer 2, pure: no I/O, no network. The quiz decides nothing new — it only maps
stated answers onto the existing strategic presets (``allocate.preset_target``),
so the resulting target flows through exactly the same propose-only machinery
as ``--allocate conservative|moderate|aggressive``. Deterministic and
explainable: each answer carries a fixed score and a written reason; the
mapping is a rubric anyone can read, never a model call.

Two consumers ask the questions, one rubric answers: the CLI's interactive
``--onboard`` menu and the MCP ``starter_allocation`` tool (where the chat
assistant asks, but THIS module scores).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.allocate import PresetName


@dataclass(frozen=True)
class Option:
    key: str  # stable answer token — what the CLI menu maps to and the MCP tool accepts
    label: str  # plain-English menu line
    score: int  # 0 (most cautious) .. 2 (most growth-tolerant)
    note: str  # why this answer moves the posture, in plain words


@dataclass(frozen=True)
class Question:
    key: str
    text: str
    options: tuple[Option, ...]

    def option(self, key: str) -> Option:
        """The option for an answer token; ValueError names the valid menu."""
        for o in self.options:
            if o.key == key:
                return o
        valid = ", ".join(repr(o.key) for o in self.options)
        raise ValueError(f"unknown {self.key} answer {key!r}; valid: {valid}")


# Drawdown-first questions: each one asks, in disguise, "how deep a fall could
# you actually live with?" — horizon (time to recover), gut response (behavior
# in the fall), and cash buffer (whether a fall can force a sale).
QUESTIONS: tuple[Question, ...] = (
    Question(
        key="horizon",
        text="When do you expect to spend most of this money?",
        options=(
            Option(
                key="under_3_years",
                label="within 3 years (a house deposit, tuition, a big purchase)",
                score=0,
                note="money needed within ~3 years can't wait out a deep drawdown, "
                "so preserving capital leads",
            ),
            Option(
                key="3_to_10_years",
                label="in 3-10 years",
                score=1,
                note="a medium horizon can ride out a typical drawdown, but not "
                "several in a row",
            ),
            Option(
                key="over_10_years",
                label="10+ years away (retirement-style saving)",
                score=2,
                note="a long horizon has time to recover even from deep drawdowns",
            ),
        ),
    ),
    Question(
        key="loss_response",
        text="Imagine your $10,000 falls to $8,500 in one bad month (-15%). "
        "What would you most likely do?",
        options=(
            Option(
                key="sell",
                label="sell some or all of it to stop further losses",
                score=0,
                note="selling into a fall locks the loss in — a shallower ride "
                "you can hold is worth more than a deeper one you can't",
            ),
            Option(
                key="hold",
                label="hold on and wait it out (uneasily)",
                score=1,
                note="holding through a drawdown works as long as its depth "
                "stays livable",
            ),
            Option(
                key="buy_more",
                label="buy more while prices are lower",
                score=2,
                note="buying into falls accepts deeper drawdowns in exchange "
                "for more growth exposure",
            ),
        ),
    ),
    Question(
        key="cash_buffer",
        text="If a surprise expense hit tomorrow, could you cover it WITHOUT "
        "selling these investments?",
        options=(
            Option(
                key="no",
                label="no — this money doubles as my safety net",
                score=0,
                note="a safety net must not be deep underwater on the day it's "
                "needed",
            ),
            Option(
                key="partly",
                label="partly — I have some savings elsewhere",
                score=1,
                note="some cushion elsewhere makes a forced sale during a "
                "drawdown less likely",
            ),
            Option(
                key="comfortably",
                label="comfortably — I keep a separate emergency fund",
                score=2,
                note="a separate emergency fund means a drawdown here never "
                "forces a sale",
            ),
        ),
    ),
)

MAX_SCORE = sum(max(o.score for o in q.options) for q in QUESTIONS)


@dataclass(frozen=True)
class OnboardResult:
    posture: PresetName
    score: int  # 0..MAX_SCORE (the pre-cap rubric sum)
    rationale: tuple[str, ...]  # one line per answer + the scoring line (+ the cap, if hit)


def posture_from_answers(
    horizon: str, loss_response: str, cash_buffer: str
) -> OnboardResult:
    """Map the three answer tokens to a preset posture, with the written why.

    Fixed rubric: each answer scores 0-2; the sum picks the posture
    (0-2 conservative, 3-4 moderate, 5-6 aggressive) — EXCEPT that a sub-3-year
    horizon caps the result at conservative no matter what, because no tolerance
    for losses buys back the time a deep drawdown needs to recover. Raises
    ValueError naming the valid answers on any unknown token.
    """
    chosen = [
        q.option(k)
        for q, k in zip(QUESTIONS, (horizon, loss_response, cash_buffer), strict=True)
    ]
    score = sum(o.score for o in chosen)
    by_score: PresetName
    if score <= 2:
        by_score = "conservative"
    elif score <= 4:
        by_score = "moderate"
    else:
        by_score = "aggressive"
    posture = by_score
    rationale = [o.note for o in chosen]
    rationale.append(f"score {score}/{MAX_SCORE} → {by_score}")
    if horizon == "under_3_years" and by_score != "conservative":
        posture = "conservative"
        rationale.append(
            "capped at conservative: with under 3 years there may be no time to "
            "recover from a deep drawdown before the money is needed"
        )
    return OnboardResult(posture=posture, score=score, rationale=tuple(rationale))
