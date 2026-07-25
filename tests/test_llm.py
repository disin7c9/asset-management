"""Tests for the LLM narrator adapter. Offline — the HTTP POST is monkeypatched,
so no network and no API key are needed. Verifies env config resolution, the two
providers' wire-format parsing, and the fail-closed paths (refusal, empty, error).
"""

from __future__ import annotations

import logging

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


def test_load_config_tier_local_honored_for_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    # tier=local on a localhost model is honored — nothing leaves the machine, so it
    # may send exact figures (build_prompt treats local like paid).
    _clear_env(monkeypatch)
    monkeypatch.setenv("ASSET_NARRATE_PROVIDER", "openai")
    monkeypatch.setenv("ASSET_NARRATE_KEY", "k")
    monkeypatch.setenv("ASSET_NARRATE_MODEL", "m")
    monkeypatch.setenv("ASSET_NARRATE_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("ASSET_NARRATE_TIER", "local")
    c = load_config()
    assert c is not None and c.tier == "local"


def test_load_config_local_hosts_recognized(monkeypatch: pytest.MonkeyPatch) -> None:
    # IPv6 loopback (::1) and the Docker host are local too — they must be accepted
    # (not rejected as non-https, which would turn narration off) and honor tier=local.
    for base in ("http://[::1]:11434/v1", "http://host.docker.internal:11434/v1"):
        _clear_env(monkeypatch)
        monkeypatch.setenv("ASSET_NARRATE_PROVIDER", "openai")
        monkeypatch.setenv("ASSET_NARRATE_KEY", "k")
        monkeypatch.setenv("ASSET_NARRATE_MODEL", "m")
        monkeypatch.setenv("ASSET_NARRATE_BASE_URL", base)
        monkeypatch.setenv("ASSET_NARRATE_TIER", "local")
        c = load_config()
        assert c is not None and c.tier == "local", base


def test_load_config_tier_local_downgrades_on_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    # tier=local against a REMOTE endpoint must NOT send exact values to the cloud under
    # a "local" label → falls back to free. Anthropic (cloud-only) can't be local either.
    _clear_env(monkeypatch)
    monkeypatch.setenv("ASSET_NARRATE_PROVIDER", "openai")
    monkeypatch.setenv("ASSET_NARRATE_KEY", "k")
    monkeypatch.setenv("ASSET_NARRATE_MODEL", "m")
    monkeypatch.setenv("ASSET_NARRATE_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("ASSET_NARRATE_TIER", "local")
    c = load_config()
    assert c is not None and c.tier == "free"
    monkeypatch.setenv("ASSET_NARRATE_PROVIDER", "anthropic")
    monkeypatch.delenv("ASSET_NARRATE_BASE_URL", raising=False)
    a = load_config()
    assert a is not None and a.tier == "free"


def test_free_cloud_emits_enrollment_warning_local_does_not(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Structural trust warning: free-tier on a CLOUD provider warns (training + rougher
    # wording). A localhost endpoint does NOT — nothing leaves the machine.
    _clear_env(monkeypatch)
    monkeypatch.setenv("ASSET_NARRATE_PROVIDER", "anthropic")
    monkeypatch.setenv("ASSET_NARRATE_KEY", "k")
    monkeypatch.setenv("ASSET_NARRATE_MODEL", "m")  # tier unset → free, anthropic = cloud
    with caplog.at_level(logging.WARNING, logger="app.llm"):
        load_config()
    assert "free-tier narration" in caplog.text and "train" in caplog.text

    caplog.clear()
    _clear_env(monkeypatch)
    monkeypatch.setenv("ASSET_NARRATE_PROVIDER", "openai")
    monkeypatch.setenv("ASSET_NARRATE_KEY", "k")
    monkeypatch.setenv("ASSET_NARRATE_MODEL", "m")
    monkeypatch.setenv("ASSET_NARRATE_BASE_URL", "http://localhost:11434/v1")  # local, tier unset → free
    with caplog.at_level(logging.WARNING, logger="app.llm"):
        c = load_config()
    assert c is not None and c.tier == "free"
    assert "free-tier narration" not in caplog.text


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


def test_a_redirect_cannot_walk_the_api_key_to_another_host() -> None:
    """Regression: `_post` used a bare `urlopen`, which follows 3xx and re-sends every
    header — including the user's `Authorization` and, on the next call, their portfolio
    figures. `_is_safe_url` vets the CONFIGURED base_url and has no say over where a
    redirect points, so a user-supplied proxy or a MITM was enough. `prices.py` already
    had the fix for its own credential; the LLM path did not, because the handler lived in
    the price module rather than in a shared one.

    **The status code is load-bearing — use 302, not 307.** urllib's stock handler already
    refuses to auto-follow a 307/308 for a POST, so a 307 test passes against the *unfixed*
    code and pins nothing. 302 is where the leak actually lives: verified against a bare
    urlopen, the redirect target received `Bearer sk-SECRET`."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    received: list[str | None] = []

    class _Target(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            received.append(self.headers.get("Authorization"))
            body = json.dumps({"ok": True}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a: object) -> None: ...

    class _Redirector(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{target.server_port}/v1/x")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *a: object) -> None: ...

    target = HTTPServer(("127.0.0.1", 0), _Target)
    redirector = HTTPServer(("127.0.0.1", 0), _Redirector)
    for srv in (target, redirector):
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with pytest.raises(Exception, match="302"):
            llm._post(
                f"http://127.0.0.1:{redirector.server_port}/v1/x",
                {"Authorization": "Bearer sk-SECRET"}, {"q": 1}, 5.0,
            )
        assert received == []          # the redirect target got nothing at all
    finally:
        for srv in (target, redirector):
            srv.shutdown()
            srv.server_close()
