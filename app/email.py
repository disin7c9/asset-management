"""Layer 4 delivery: send the rendered report as an HTML email via Resend.

The only outbound-network module besides ``prices.py``. It isolates the Resend
SDK behind one tiny dispatch function so the rest of the app — and the tests —
never touch the network: tests monkey-patch ``app.email._dispatch``.

Config is read from the environment (Layer 4 owns config): ``RESEND_API_KEY``
(secret, from ``.env``), ``REPORT_TO`` (recipient), ``REPORT_FROM`` (sender;
defaults to Resend's sandbox address). Explicit arguments override the env so
callers/tests can inject values. Never raises on a send failure — returns an
``EmailResult`` the composition root stamps into ``run_summary``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

# Resend's sandbox sender — works without a verified domain, for testing.
_DEFAULT_SENDER = "onboarding@resend.dev"


@dataclass(frozen=True)
class EmailResult:
    sent: bool
    detail: str  # message id on success, or the reason it didn't send


def send_report(
    *,
    subject: str,
    html: str,
    to: str | None = None,
    sender: str | None = None,
    api_key: str | None = None,
) -> EmailResult:
    """Send one HTML email. Returns an EmailResult; never raises.

    Missing credentials are a normal, non-fatal outcome (e.g. ``--send`` on a
    box with no key): we log and return ``sent=False`` so the run still prints
    its report and exits cleanly.
    """
    api_key = api_key or os.environ.get("RESEND_API_KEY")
    to = to or os.environ.get("REPORT_TO")
    sender = sender or os.environ.get("REPORT_FROM") or _DEFAULT_SENDER

    if not api_key:
        log.error("cannot send email: RESEND_API_KEY is not set")
        return EmailResult(False, "missing RESEND_API_KEY")
    if not to:
        log.error("cannot send email: REPORT_TO is not set")
        return EmailResult(False, "missing REPORT_TO")

    payload: dict[str, Any] = {
        "from": sender,
        "to": [to],
        "subject": subject,
        "html": html,
    }
    try:
        message_id = _dispatch(payload, api_key)
    except Exception as exc:  # SDK / network errors must not crash the run
        log.error("email send failed: %s", exc)
        return EmailResult(False, f"send failed: {exc}")
    log.info("email sent to %s (id=%s)", to, message_id)
    return EmailResult(True, message_id)


def _dispatch(payload: dict[str, Any], api_key: str) -> str:
    """The single real Resend call. Patched out in tests.

    Imported lazily so importing this module never requires the SDK and so the
    network dependency is contained to exactly this function.
    """
    import resend

    resend.api_key = api_key
    resp = resend.Emails.send(payload)  # type: ignore[arg-type]
    # Resend (v2) returns {"id": "..."}. Read the id; never fall back to
    # stringifying the whole response (that would log garbage as a message id
    # and hide an SDK shape change behind an apparent success).
    if isinstance(resp, dict):
        return str(resp.get("id", ""))
    return str(getattr(resp, "id", ""))
