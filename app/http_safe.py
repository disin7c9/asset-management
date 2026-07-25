"""Layer 1: the one hardened `urllib` opener every outbound adapter uses.

This exists as its own module for a reason found by review. The no-redirect handler was
written correctly in `prices.py`, with a docstring explaining exactly which credential it
protects — and the LLM path, the *other* place in this codebase that puts an API key in an
outbound header, used a bare `urlopen` and did not have it. One adapter was hardened and its
twin was not, because there was no shared seam to be hardened in.

No internal imports (a leaf), so any layer may use it.
"""

from __future__ import annotations

import urllib.request
from http.client import HTTPMessage
from typing import IO

__all__ = ["NoRedirect", "no_redirect_opener"]


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse 3xx rather than follow it.

    `urllib` re-sends every header on a redirect and — unlike `requests` — does NOT strip
    `Authorization` when the host changes, so a redirect (a compromised host, a MITM'd
    proxy, a user-supplied base URL) would hand the user's API key to the target. None of
    the endpoints called here redirect, so refusing costs nothing: the 3xx surfaces as an
    HTTPError, which each caller already treats as a failed request.
    """

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        return None


def no_redirect_opener() -> urllib.request.OpenerDirector:
    """An opener that refuses redirects. Build one per adapter at import time."""
    return urllib.request.build_opener(NoRedirect)
