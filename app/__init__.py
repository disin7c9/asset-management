"""Asset-management v0 — personal tool for transaction-log holdings + drawdown-first reporting."""

from __future__ import annotations

import logging

from app.log_config import setup_logging

__version__ = "0.0.1"
__all__ = ["__version__", "setup_logging"]

# Library best practice: attach a NullHandler so importers that don't configure
# logging don't see "No handlers could be found" warnings. Production paths
# call setup_logging() (from app/cli.py) to attach a real handler.
logging.getLogger(__name__).addHandler(logging.NullHandler())
