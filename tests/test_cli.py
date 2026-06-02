"""CLI delivery routing tests (slice 4).

Offline: ``--no-prices`` avoids the network entirely; ``--send`` is exercised by
monkey-patching ``app.email._dispatch`` so no real email leaves the box.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest

from app import email as E
from app.cli import main


def _today() -> str:
    # Must match cli.main's `today = date.today()` (local), which dates the file.
    return date.today().isoformat()


def test_save_writes_markdown(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["--no-prices", "--save", "--reports-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "=== HOLDINGS ===" in out  # stdout still printed

    saved = tmp_path / f"{_today()}.md"
    assert saved.exists()
    text = saved.read_text(encoding="utf-8")
    assert text.startswith("# Portfolio brief —")
    assert "## HOLDINGS" in text
    assert "not financial advice" in text


def test_send_invokes_dispatch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(E, "_dispatch", lambda p, k: calls.append(p) or "msg_1")
    monkeypatch.setenv("RESEND_API_KEY", "test_key")
    monkeypatch.setenv("REPORT_TO", "me@example.com")

    rc = main(["--no-prices", "--send"])
    assert rc == 0
    assert len(calls) == 1
    payload = calls[0]
    assert payload["to"] == ["me@example.com"]
    assert payload["subject"].startswith("Portfolio brief —")
    assert payload["html"].startswith("<!doctype html>")


def test_send_without_credentials_prints_but_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # No real send should happen; make the SDK call explode if it's ever reached.
    monkeypatch.setattr(E, "_dispatch", lambda p, k: (_ for _ in ()).throw(AssertionError))
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    monkeypatch.delenv("REPORT_TO", raising=False)
    # load_dotenv in main must not resurrect a key from a real .env for this test.
    monkeypatch.setattr("app.cli.load_dotenv", lambda *a, **k: False)

    rc = main(["--no-prices", "--send"])
    # Brief still printed, but the requested delivery failed → non-zero so cron alerts.
    assert rc == 1
    assert "=== HOLDINGS ===" in capsys.readouterr().out


def test_save_to_unwritable_path_prints_but_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Make reports-dir sit *under a regular file* so mkdir raises NotADirectoryError.
    blocker = tmp_path / "afile"
    blocker.write_text("not a dir", encoding="utf-8")
    rc = main(["--no-prices", "--save", "--reports-dir", str(blocker / "sub")])
    assert rc == 1  # save failed, recorded not raised
    assert "=== HOLDINGS ===" in capsys.readouterr().out  # brief still printed
