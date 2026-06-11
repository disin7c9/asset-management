"""Smoke test for the `python -m app` module entry point.

Runs the real entry as a subprocess (so `app/__main__.py` is actually executed,
not just `cli.main`). Offline by construction: --no-prices touches no network.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_python_m_app_runs_and_prints_holdings() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "app", "--csv", "data/sample_data/transactions.csv", "--no-prices"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "=== HOLDINGS ===" in proc.stdout
