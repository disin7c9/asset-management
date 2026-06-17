"""Tests for the LLM narrator adapter. Offline — the HTTP POST is monkeypatched,
so no network and no API key are needed. Verifies env config resolution, the two
providers' wire-format parsing, and the fail-closed paths (refusal, empty, error).
"""

from __future__ import annotations

import pytest

from app import llm
from app.llm import NarratorConfig, complete, load_config

_ENV_KEYS = (
    "ASSET_NARRATE_PROVIDER", "ASSET_NARRATE_KEY", "ASSET_NARRATE_MODEL",
    "ASSET_NARRATE_BASE_URL", "ASSET_NARRATE_TIER",
)


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


def test_load_config_off_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    assert load_config() is None  # narration is opt-in / off by default


def test_load_config_anthropic_omits_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("ASSET_NARRATE_PROVIDER", "anthropic")
    monkeypatch.setenv("ASSET_NARRATE_KEY", "sk-x")
    monkeypatch.setenv("ASSET_NARRATE_MODEL", "claude-haiku-4-5")
    c = load_config()
    assert c is not None
    assert c.provider == "anthropic" and c.model == "claude-haiku-4-5"
    assert c.temperature is None  # Opus-tier rejects sampling params → omit
    assert c.tier == "free"  # privacy fail-safe: tier unset → free (values stay home)


def test_load_config_openai_needs_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("ASSET_NARRATE_PROVIDER", "openai")
    monkeypatch.setenv("ASSET_NARRATE_KEY", "k")
    monkeypatch.setenv("ASSET_NARRATE_MODEL", "m")
    assert load_config() is None  # missing ASSET_NARRATE_BASE_URL → off
    monkeypatch.setenv("ASSET_NARRATE_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("ASSET_NARRATE_TIER", "free")
    c = load_config()
    assert c is not None
    assert c.provider == "openai" and c.temperature == 0.0 and c.tier == "free"


def test_load_config_tier_fails_safe_to_free(monkeypatch: pytest.MonkeyPatch) -> None:
    # The privacy dial must fail SAFE: only an explicit, correctly-spelled "paid"
    # sends exact values. A typo ("fre") must NOT silently upgrade disclosure.
    _clear_env(monkeypatch)
    monkeypatch.setenv("ASSET_NARRATE_PROVIDER", "anthropic")
    monkeypatch.setenv("ASSET_NARRATE_KEY", "k")
    monkeypatch.setenv("ASSET_NARRATE_MODEL", "m")
    monkeypatch.setenv("ASSET_NARRATE_TIER", "fre")  # typo for "free"
    c = load_config()
    assert c is not None and c.tier == "free"  # not "paid" — the leak is closed
    monkeypatch.setenv("ASSET_NARRATE_TIER", "PAID")  # case/space tolerated
    assert (load_config() or NarratorConfig("", "", "", "", "", None)).tier == "paid"


def test_load_config_rejects_insecure_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    # An http:// (or file://) endpoint would send the API key in the clear → refuse.
    _clear_env(monkeypatch)
    monkeypatch.setenv("ASSET_NARRATE_PROVIDER", "openai")
    monkeypatch.setenv("ASSET_NARRATE_KEY", "k")
    monkeypatch.setenv("ASSET_NARRATE_MODEL", "m")
    monkeypatch.setenv("ASSET_NARRATE_BASE_URL", "http://api.groq.com/openai/v1")
    assert load_config() is None  # insecure scheme → narration off
    monkeypatch.setenv("ASSET_NARRATE_BASE_URL", "http://localhost:11434/v1")
    assert load_config() is not None  # localhost proxy (Ollama/llama.cpp) is allowed


_ANTHRO = NarratorConfig("anthropic", "claude-haiku-4-5", "k", "", "paid", None)
_OPENAI = NarratorConfig("openai", "m", "k", "https://x/v1", "free", 0.0)


def test_complete_anthropic_parses_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llm, "_post",
        lambda *_a, **_k: {"stop_reason": "end_turn", "content": [{"type": "text", "text": "hello world"}]},
    )
    assert complete(_ANTHRO, "sys", "usr") == "hello world"


def test_complete_anthropic_refusal_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "_post", lambda *_a, **_k: {"stop_reason": "refusal", "content": []})
    assert complete(_ANTHRO, "sys", "usr") is None


def test_complete_openai_parses_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llm, "_post", lambda *_a, **_k: {"choices": [{"message": {"content": "hi there"}}]}
    )
    assert complete(_OPENAI, "sys", "usr") == "hi there"


def test_complete_network_error_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_k: object) -> dict[str, object]:
        raise OSError("network down")

    monkeypatch.setattr(llm, "_post", boom)
    assert complete(_ANTHRO, "sys", "usr") is None  # fail-closed


def test_complete_empty_content_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "_post", lambda *_a, **_k: {"choices": [{"message": {"content": "   "}}]})
    assert complete(_OPENAI, "sys", "usr") is None


def test_complete_anthropic_truncation_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    # A max_tokens truncation can end mid-{{token}} → withhold rather than risk it.
    monkeypatch.setattr(
        llm, "_post",
        lambda *_a, **_k: {"stop_reason": "max_tokens",
                           "content": [{"type": "text", "text": "fell {{max_dr"}]},
    )
    assert complete(_ANTHRO, "sys", "usr") is None


def test_complete_openai_truncation_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llm, "_post",
        lambda *_a, **_k: {"choices": [{"finish_reason": "length",
                                        "message": {"content": "fell {{max_dr"}}]},
    )
    assert complete(_OPENAI, "sys", "usr") is None
