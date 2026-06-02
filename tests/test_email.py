"""Tests for the Resend email adapter.

Offline: every test monkey-patches ``app.email._dispatch`` (the single real SDK
call) so no network happens and no API key is needed.
"""

from __future__ import annotations

from typing import Any

import pytest

from app import email as E
from app.email import EmailResult, send_report


def test_send_success_builds_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_dispatch(payload: dict[str, Any], api_key: str) -> str:
        captured["payload"] = payload
        captured["api_key"] = api_key
        return "msg_123"

    monkeypatch.setattr(E, "_dispatch", fake_dispatch)
    result = send_report(
        subject="Brief", html="<p>hi</p>", to="me@example.com", api_key="key_abc"
    )
    assert result == EmailResult(True, "msg_123")
    assert captured["api_key"] == "key_abc"
    assert captured["payload"]["to"] == ["me@example.com"]
    assert captured["payload"]["subject"] == "Brief"
    assert captured["payload"]["html"] == "<p>hi</p>"
    # default sandbox sender when REPORT_FROM unset
    assert captured["payload"]["from"] == "onboarding@resend.dev"


def test_explicit_sender_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(E, "_dispatch", lambda p, k: captured.update(p) or "id")
    send_report(
        subject="s", html="h", to="a@b.com", sender="me@mine.com", api_key="k"
    )
    assert captured["from"] == "me@mine.com"


def test_missing_api_key_is_nonfatal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    result = send_report(subject="s", html="h", to="a@b.com", api_key=None)
    assert result.sent is False
    assert "RESEND_API_KEY" in result.detail


def test_missing_recipient_is_nonfatal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REPORT_TO", raising=False)
    result = send_report(subject="s", html="h", to=None, api_key="k")
    assert result.sent is False
    assert "REPORT_TO" in result.detail


def test_dispatch_exception_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(payload: dict[str, Any], api_key: str) -> str:
        raise RuntimeError("network down")

    monkeypatch.setattr(E, "_dispatch", boom)
    result = send_report(subject="s", html="h", to="a@b.com", api_key="k")
    assert result.sent is False
    assert "send failed" in result.detail
    assert "network down" in result.detail


def test_env_supplies_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(E, "_dispatch", lambda p, k: captured.update({"k": k, **p}) or "x")
    monkeypatch.setenv("RESEND_API_KEY", "env_key")
    monkeypatch.setenv("REPORT_TO", "env@to.com")
    result = send_report(subject="s", html="h")
    assert result.sent is True
    assert captured["k"] == "env_key"
    assert captured["to"] == ["env@to.com"]
