"""LLM narrator adapter: fetch prose from a configurable LLM backend (v2.0.0 P2b).

The ONE network concern of the narration edge — a named I/O adapter on the
`prices.py`/`metadata.py` pattern. It knows NOTHING about claims or the fence: it
takes a system + user prompt and returns the model's text, or **None on any
failure** (network, HTTP, refusal, empty) — narration is fail-closed and never
raises. The pure fence (`narrate.py`) owns the numbers; `cli` wires the two.

Two backends behind one `complete()` call:
- **OpenAI-compatible** (`provider=openai`): POST `{base_url}/chat/completions` —
  one wire format covers Gemini, Groq, OpenRouter, Mistral, OpenAI.
- **Anthropic** (`provider=anthropic`): POST the Messages API.

Config comes from the environment (.env) so a key never lives in code. stdlib
`urllib` only (no SDK) — minimal deps, matching `prices.py`'s stooq fetch.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

log = logging.getLogger(__name__)

_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"


@dataclass(frozen=True)
class NarratorConfig:
    """A resolved LLM backend. ``temperature=None`` omits the field — Anthropic's
    Opus-tier models reject sampling params (400); Haiku and the OpenAI-compatible
    providers accept it."""

    provider: str            # "anthropic" | "openai" (the wire protocol)
    model: str
    api_key: str
    base_url: str            # OpenAI-compatible endpoint; "" for Anthropic
    tier: str                # "free" | "paid" | "local" (privacy dial + provenance)
    temperature: float | None
    max_tokens: int = 600
    timeout: float = 30.0


def load_config() -> NarratorConfig | None:
    """Resolve the narrator backend from the environment, or **None** when not
    configured (the default — narration is opt-in and off). Env:
    ``ASSET_NARRATE_PROVIDER`` (anthropic|openai), ``ASSET_NARRATE_MODEL``,
    ``ASSET_NARRATE_KEY``; for openai also ``ASSET_NARRATE_BASE_URL``; optional
    ``ASSET_NARRATE_TIER`` (free|paid|local, default free — fails safe toward less
    disclosure: only an explicit ``paid``/``local`` sends exact figures)."""
    provider = os.environ.get("ASSET_NARRATE_PROVIDER", "").strip().lower()
    if not provider:
        return None
    if provider not in ("anthropic", "openai"):
        log.warning(
            "ASSET_NARRATE_PROVIDER must be 'anthropic' or 'openai' (got %r); narration off",
            provider,
        )
        return None
    key = os.environ.get("ASSET_NARRATE_KEY", "").strip()
    model = os.environ.get("ASSET_NARRATE_MODEL", "").strip()
    if not key or not model:
        log.warning("narration needs ASSET_NARRATE_KEY and ASSET_NARRATE_MODEL; narration off")
        return None
    # Resolve the endpoint first — settling the tier needs to know if it's local.
    base_url = ""
    if provider == "openai":
        base_url = os.environ.get("ASSET_NARRATE_BASE_URL", "").strip()
        if not base_url:
            log.warning("ASSET_NARRATE_PROVIDER=openai needs ASSET_NARRATE_BASE_URL; narration off")
            return None
        if not _is_safe_url(base_url):
            log.warning(
                "ASSET_NARRATE_BASE_URL must be https:// (or http://localhost); got %r; "
                "narration off (refusing to send the API key in the clear)", base_url,
            )
            return None
    is_local = provider == "openai" and _is_local_url(base_url)

    # Privacy dial fails SAFE: exact values leave only on an explicit, correctly-spelled
    # `paid` (a provider that doesn't train) or `local` (a model on your own machine, so
    # nothing leaves). Anything else — unset / blank / misspelled, or `local` against a
    # NON-local endpoint — falls back to `free` (coarse bands only). A typo must never
    # silently *upgrade* disclosure.
    raw_tier = os.environ.get("ASSET_NARRATE_TIER", "").strip().lower()
    if raw_tier == "paid":
        tier = "paid"
    elif raw_tier == "local" and is_local:
        tier = "local"
    elif raw_tier == "local":  # 'local' tier but a remote endpoint → don't send exact to cloud
        log.warning(
            "ASSET_NARRATE_TIER=local needs a localhost endpoint; %r is remote — using "
            "'free' (exact values stay home).", base_url or provider,
        )
        tier = "free"
    else:
        if raw_tier not in ("", "free"):
            log.warning(
                "ASSET_NARRATE_TIER=%r unrecognized; defaulting to 'free' (values stay home). "
                "Use 'paid' (a provider that doesn't train) or 'local' to send exact figures.",
                raw_tier,
            )
        tier = "free"

    # Structural enrollment warning: free-tier on a CLOUD provider sends (coarse) context
    # to a model the provider may train on, and small free models word things more roughly.
    if tier == "free" and not is_local:
        log.warning(
            "free-tier narration: only coarse qualitative bands leave your machine (exact "
            "figures stay home), but the provider may train on them and smaller models word "
            "things more roughly — the numbers in your brief are always the tool's, never the "
            "model's. Use ASSET_NARRATE_TIER=paid for a provider that doesn't train, or "
            "'local' for a localhost model, to send exact figures."
        )

    # openai accepts temperature; anthropic omits it (Opus-tier 400s on sampling params,
    # Haiku is fine). base_url is "" for anthropic (never set above).
    temperature = 0.0 if provider == "openai" else None
    return NarratorConfig(provider, model, key, base_url, tier, temperature=temperature)


_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "host.docker.internal"})


def _is_local_url(url: str) -> bool:
    """True if the endpoint is a local model (Ollama/llama.cpp) — nothing leaves for the
    internet, so the ``local`` tier may send exact figures. Covers loopback (localhost /
    127.0.0.1 / IPv6 ``::1``) and the Docker host (``host.docker.internal``)."""
    return urlparse(url).hostname in _LOCAL_HOSTS


def _is_safe_url(url: str) -> bool:
    """An LLM endpoint must be https (so the API key + prompt aren't sent in the
    clear); http is tolerated only for a localhost proxy. Guards a typo'd or hostile
    ASSET_NARRATE_BASE_URL (e.g. ``http://``, ``file://``)."""
    parsed = urlparse(url)
    return parsed.scheme == "https" or (parsed.scheme == "http" and _is_local_url(url))


def complete(config: NarratorConfig, system: str, user: str) -> str | None:
    """The model's text for one (system, user) prompt, or **None** on ANY failure —
    narration degrades to nothing, never raises. The caller then withholds the
    SUMMARY block and prints the plain brief."""
    try:
        if config.provider == "anthropic":
            return _anthropic(config, system, user)
        return _openai(config, system, user)
    except Exception as exc:  # noqa: BLE001 — fail-closed: any error → no narration
        log.warning("narration LLM call failed (%s): %s", config.model, exc)
        return None


def _post(
    url: str, headers: dict[str, str], body: dict[str, Any], timeout: float
) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    # Scheme is https (anthropic is a fixed constant; the openai base_url is validated
    # by _is_safe_url in load_config), so the urlopen scheme is constrained, not arbitrary.
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        parsed: dict[str, Any] = json.loads(resp.read())
        return parsed


def _anthropic(config: NarratorConfig, system: str, user: str) -> str | None:
    body: dict[str, Any] = {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if config.temperature is not None:
        body["temperature"] = config.temperature
    headers = {
        "content-type": "application/json",
        "x-api-key": config.api_key,
        "anthropic-version": _ANTHROPIC_VERSION,
    }
    resp = _post(_ANTHROPIC_URL, headers, body, config.timeout)
    if resp.get("stop_reason") in ("refusal", "max_tokens"):  # declined, or truncated
        return None  # a truncated reply can end mid-{{token}} → withhold, don't risk it
    text = "".join(
        b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text"
    ).strip()
    return text or None


def _openai(config: NarratorConfig, system: str, user: str) -> str | None:
    # NB: `max_tokens` covers Gemini/Groq/OpenRouter/Mistral + standard OpenAI chat
    # models. OpenAI's *reasoning* models (o*) reject it for `max_completion_tokens`;
    # that 400 fails closed here (→ None + a logged warning), so narration is simply
    # withheld rather than crashing. Switch the field only after verifying live (#8).
    body: dict[str, Any] = {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if config.temperature is not None:
        body["temperature"] = config.temperature
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {config.api_key}",
    }
    resp = _post(config.base_url.rstrip("/") + "/chat/completions", headers, body, config.timeout)
    choices = resp.get("choices") or []
    if not choices:
        return None
    if choices[0].get("finish_reason") == "length":  # truncated → may end mid-token
        return None
    content = (choices[0].get("message") or {}).get("content")
    if not isinstance(content, str):
        return None
    return content.strip() or None
