"""Watch the number fence catch a fabricated figure — run it yourself.

The narration layer (SymGen + PCN) is why an LLM can narrate this tool's numbers but
never author one: the model may only cite ``{{token}}`` placeholders; a deterministic
renderer substitutes the validated figures and refuses the WHOLE narration if the model
typed even one numeral of its own — in ANY Unicode numeric category, so ``½`` and ``⑧``
fail closed alongside ``3.2``. This script drives the real production fence
(``app.narrate.render_narration``, unmodified) with five canned model outputs — one
obedient, one fabricating, two smuggling numerals a category check alone would miss, one citing a claim that
doesn't exist — so both sides of the wall are visible. Model outputs are canned so the
demo needs no API key; the fence code path is the production one.

What the fence does NOT catch is prose: a correct figure placed in a sentence that
misstates it, a magnitude in words, or advice-shaped wording. See ``app/narrate.py``.

Run:  ``uv run python scripts/demo_fence.py``   (``--fast`` skips the dramatic pauses)
"""

from __future__ import annotations

import os
import sys
import time

from app.narrate import Claim, render_narration

if os.name == "nt":  # enable ANSI (VT) processing on legacy Windows consoles
    os.system("")
# Color only a real terminal — a redirected run (file, CI log) gets clean plain text.
if sys.stdout.isatty():
    GREEN, RED, DIM, BOLD, END = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"
else:
    GREEN = RED = DIM = BOLD = END = ""


def _pause(seconds: float) -> None:
    if "--fast" not in sys.argv:
        time.sleep(seconds)


def _say(text: str = "") -> None:
    print(text, flush=True)


def _show_refusal(prose: str, claims: dict[str, Claim], reason: str) -> None:
    """Render one violating case, labelling it from what the fence ACTUALLY returned.

    The label used to be unconditional literal text, so a demo run printed "REFUSED"
    even if the fence had let the string through — and `test_scripts.py`'s assertions on
    that text passed while the fence leaked. Deriving the label from the return value is
    what makes both the demo and its test able to fail.
    """
    out = render_narration(prose, claims)
    if out is None:
        _say(f"   fence renders: {RED}None — REFUSED ({reason}){END}")
    else:
        _say(f"   fence renders: {RED}{out!r} — *** LEAKED *** (expected: {reason}){END}")


def main() -> None:
    # The validated core's figures (example values, shaped like the bundled demo book's
    # brief). In production these come from derive→returns→risk; the model never sees
    # more than the token names and labels.
    claims = {
        "max_drawdown": Claim("max_drawdown", -0.0984, "-9.84%", "worst peak-to-trough fall"),
        "ulcer": Claim("ulcer", 0.0238, "2.38%", "ulcer index (how deep+long underwater)"),
        "twr": Claim("twr", 0.18, "+18.00%", "time-weighted return, annualized"),
    }

    _say(f"{BOLD}The number fence: the model writes ONLY {{{{tokens}}}} — Python owns every digit.{END}")
    _say(f"{DIM}(driving the real app.narrate.render_narration; model outputs canned — no API key){END}")
    _pause(1.6)

    _say(f"\n{BOLD}1) An obedient model narrates:{END}")
    good = (
        "Your deepest stretch fell {{max_drawdown}} from its peak, and an ulcer index of "
        "{{ulcer}} says the ride was bumpy but shallow overall; through it all, "
        "time-weighted return landed at {{twr}} a year."
    )
    _say(f"   model wrote : {good}")
    _pause(2.2)
    _say(f"   fence renders: {GREEN}{render_narration(good, claims)}{END}")
    _say(f"   {DIM}every figure above was substituted from the validated core{END}")
    _pause(2.2)

    _say(f"\n{BOLD}2) A fabricating model tries to type its own number:{END}")
    bad = "Held steady this year: the drawdown was only about -3.2%, with returns near 18%."
    _say(f"   model wrote : {bad}")
    _pause(2.2)
    _show_refusal(bad, claims, "a bare digit outside any token")
    _pause(1.6)

    _say(f"\n{BOLD}3) A model smuggles a numeral that isn't an ASCII digit:{END}")
    unicode_numeral = "You gave back about ½ of the gain; drawdown was {{max_drawdown}}."
    _say(f"   model wrote : {unicode_numeral}")
    _pause(2.2)
    _show_refusal(unicode_numeral, claims, "any Unicode numeral, not just 0-9")
    _pause(1.6)

    _say(f"\n{BOLD}4) A model writes the number in words the reader parses as digits:{END}")
    cjk = "The fund gave back 十 percent; drawdown was {{max_drawdown}}."
    _say(f"   model wrote : {cjk}")
    _pause(2.2)
    _show_refusal(cjk, claims, "a CJK numeral ideograph — a letter by category, a number in fact")
    _pause(1.6)

    _say(f"\n{BOLD}5) A sneaky model cites a claim that doesn't exist:{END}")
    sneaky = "Risk-adjusted, this beat the market: Sharpe came in at {{sharpe_2024}}."
    _say(f"   model wrote : {sneaky}")
    _pause(2.2)
    _show_refusal(sneaky, claims, "unknown claim")
    _pause(1.6)

    _say(f"\n{BOLD}Fail-closed:{END} a violating narration is withheld entirely — the plain")
    _say("numeric brief prints instead. A NUMERAL can never be the model's — the wording")
    _say("still is, so the fence bounds arithmetic, not interpretation.")
    _say(f"{DIM}Try it: uv run python scripts/demo_fence.py{END}")
    _pause(3.5)  # hold the final frame so the looping GIF doesn't snap away instantly


if __name__ == "__main__":
    main()
