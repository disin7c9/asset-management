"""Build the Claude-Desktop MCPB bundle: one-click install for the MCP addon.

An ``.mcpb`` is a zip the host unpacks and launches: ``manifest.json`` + the files
``uv`` needs to run the server on the user's machine. ``server.type: "uv"`` means
Claude Desktop provisions uv itself, and uv resolves Python 3.12 + the locked
dependencies from ``pyproject.toml``/``uv.lock`` at install — nothing is vendored,
so the bundle stays ~small and platform-neutral (pandas/pyarrow wheels come from
PyPI per-platform instead of being baked in).

Contents are an explicit ALLOWLIST, never a glob over the repo, so a secret
(``.env``) or private data (``data/my_data/``) can't ride along by accident.

Run:  ``uv run python scripts/build_mcpb.py``  →  ``dist/asset-management-<version>.mcpb``
Install: Claude Desktop → Settings → Extensions → Install Extension… → pick the file.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Path SEGMENTS that must never appear in a bundled path (belt-and-braces on top of the
# allowlist): secrets, private books, repo internals, the local price cache. Matched per
# path part (not substring — `app/prices.py` is fine; `data/prices/…` is not).
_FORBIDDEN_PARTS = frozenset({"my_data", ".git", ".git-private", "reports", "prices"})


def _version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as fh:
        v = tomllib.load(fh)["project"]["version"]
    return str(v)


# The bundle's entry script, generated at the bundle ROOT. Two hard-won constraints
# (07-04/05 click-tests) shape it:
# 1. It must NOT live inside app/ and must NOT be launched as `python -m`: running a
#    script inside app/ puts app/ at sys.path[0], where app/email.py shadows the stdlib
#    `email` package (fatal import crash) — while args containing a literal "python"
#    token made Claude Desktop probe `python`/`python3` on the system PATH and fail
#    with exit 9009 on machines that (by design) have no Python installed.
# 2. A root-level script launched by `uv run <script>` satisfies both: sys.path[0] is
#    the bundle root (app/ is a normal package) and the args match the official
#    hello-world-uv example shape exactly.
LAUNCHER_PY = '''\
"""Bundle entry point — launched by Claude Desktop as `uv run server_entry.py`.

Lives at the bundle ROOT, never inside app/: a script's directory becomes
sys.path[0], and inside app/ the local email.py would shadow the stdlib `email`
package. Do not rename to `-m app.mcp_server` in the manifest either — a literal
"python" arg makes the host probe the system PATH for Python (absent by design).
"""

from app.mcp_server import run

run()
'''


def bundle_pyproject() -> str:
    """The bundle's pyproject: the repo's, minus the ``[dependency-groups]`` section.

    Claude Desktop's UV runtime syncs the DEFAULT groups — dev included — so shipping
    the repo pyproject installs ruff/pytest/mypy on every end-user machine. Worse than
    waste: ruff's script-bearing ``.data`` wheel layout is exactly what Desktop's two
    concurrent setup passes race over (the observed os-error-32 install loop, 07-03/04).
    End users get the runtime dependencies only; the repo's dev workflow is untouched.
    """
    raw = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    lines = raw.splitlines(keepends=True)
    out: list[str] = []
    skipping = False
    for line in lines:
        if line.strip() == "[dependency-groups]":
            skipping = True
            continue
        if skipping and line.startswith("["):  # next section header ends the cut
            skipping = False
        if not skipping:
            out.append(line)
    trimmed = "".join(out)
    parsed = tomllib.loads(trimmed)  # verify the surgery
    if "dependency-groups" in parsed or parsed["project"]["version"] != _version():
        raise SystemExit("bundle pyproject transformation went wrong — refusing to build")
    return trimmed


def _bundle_lock(stage: Path) -> str:
    """Regenerate ``uv.lock`` against the trimmed pyproject (in ``stage``), so the
    bundle's ``uv run --frozen`` sees a lockfile that MATCHES its pyproject — uv
    refuses a frozen run against a stale lock. Then verify no dev tool survived."""
    subprocess.run(
        ["uv", "lock", "--directory", str(stage)], check=True, capture_output=True
    )
    lock = (stage / "uv.lock").read_text(encoding="utf-8")
    for dev_pkg in ('name = "ruff"', 'name = "pytest"', 'name = "mypy"'):
        if dev_pkg in lock:
            raise SystemExit(f"dev package leaked into the bundle lock ({dev_pkg})")
    return lock


def _manifest(version: str) -> dict[str, object]:
    """The MCPB manifest (schema v0.4, additionalProperties=false — documented keys only)."""
    return {
        "$schema": (
            "https://raw.githubusercontent.com/anthropics/mcpb/main/schemas/"
            "mcpb-manifest-v0.4.schema.json"
        ),
        "manifest_version": "0.4",
        "name": "asset-management",
        "display_name": "Asset Management — chat with your portfolio",
        "version": version,
        "description": (
            "Read-only portfolio brief for your own stock/ETF transaction log: "
            "drawdown-first numbers from a validated deterministic core — the AI "
            "narrates, it never does the math."
        ),
        "long_description": (
            "Point it at a Ghostfolio-compatible CSV (or Ghostfolio JSON export) and ask "
            "about your drawdown, returns, rebalance drift, portfolio gaps, or a preset "
            "allocation.\n\n"
            "**The guarantees** (enforced in code, pinned by tests — ask the model to read "
            "`portfolio://guarantees`):\n"
            "1. **Read-only** — no tool can trade, move money, or edit your log.\n"
            "2. **Every number is computed, never generated** — the assistant narrates; a "
            "deterministic Python core (reconciled to the cent vs a real brokerage export) "
            "does all arithmetic.\n"
            "3. **Your data stays home** — local stdio server, no telemetry; the only "
            "network use is downloading public market data (price history + published "
            "fund facts).\n"
            "4. **Descriptions, not advice** — suggestions are paired to named rules, "
            "metrics carry confidence intervals, verdicts never say \"beats\".\n\n"
            "The pre-filled transaction file is a bundled DEMO portfolio — explore with "
            "fake data first, then swap in your own export under Settings.\n\n"
            "*First launch provisions Python 3.12 and the locked dependencies (one-time, "
            "a few minutes, needs network — even with \"Strictly offline\" ticked, which "
            "only governs the server's own price fetches). If the first start times out, "
            "give it a minute and retry — the provisioning keeps running.*"
        ),
        "author": {"name": "disin7c9", "url": "https://github.com/disin7c9"},
        "repository": {"type": "git", "url": "https://github.com/disin7c9/asset-management"},
        "keywords": ["portfolio", "etf", "drawdown", "finance", "read-only", "backtest"],
        # Declare the tool surface (display metadata per MANIFEST.md — the runtime list
        # still comes from tools/list). Some Desktop builds appear to expose only
        # DECLARED capabilities to the chat (the prompts_generated lesson below); the
        # official example manifests all declare tools, so ship the anomaly-free shape.
        # tests/test_scripts.py locks these names to the server's real registrations.
        "tools": [
            {
                "name": "portfolio_summary",
                "description": "Current holdings, P&L and annualized returns from your "
                "transaction log (read-only).",
            },
            {
                "name": "risk_report",
                "description": "Drawdown-first risk panel: max drawdown depth/dates/"
                "recovery, Ulcer, CDaR, Sharpe/Sortino/Calmar — with bootstrap "
                "confidence intervals.",
            },
            {
                "name": "rebalance_check",
                "description": "Drift vs your target allocation and named, rule-paired "
                "rebalance suggestions (propose-only).",
            },
            {
                "name": "securities_facts",
                "description": "Fund facts for what you hold: expense ratio, AUM, "
                "liquidity, age.",
            },
            {
                "name": "discover_gaps",
                "description": "Role gaps in your book and screened candidate ETFs that "
                "could fill them (propose-only).",
            },
            {
                "name": "screen_candidate",
                "description": "Judge one NEW ticker against your book: cost, liquidity, "
                "age, overlap, diversification.",
            },
            {
                "name": "propose_allocation",
                "description": "Build a conservative/moderate/aggressive preset target "
                "and validate it against a benchmark with a walk-forward held-out "
                "verdict (propose-only).",
            },
            {
                "name": "starter_allocation",
                "description": "New user onboarding: map 3 risk answers to a posture and "
                "return that starting allocation, validated against a benchmark "
                "(propose-only).",
            },
        ],
        # Our 6 conversation starters are runtime-registered (FastMCP decorators), not
        # static manifest templates. Without this flag Claude Desktop LISTS them in the
        # + menu but refuses every prompts/get as "attempted undeclared prompt" (click-test
        # lesson, 07-05: "Failed to attach prompt" on every machine). Declaring template
        # text instead would be worse: the host then requires the server's rendered prompt
        # to byte-match the declared text, which any wording drift breaks.
        "prompts_generated": True,
        "server": {
            "type": "uv",
            "entry_point": "server_entry.py",
            "mcp_config": {
                "command": "uv",
                # Args mirror the official hello-world-uv example: `uv run <script at
                # bundle root>`. See LAUNCHER_PY for why it must be a ROOT-level script
                # (app/email.py stdlib shadowing) and must not say "python" (the host
                # probes any python token against the system PATH → exit 9009 on
                # Python-less machines → "Unable to connect" everywhere).
                #
                # NO --no-dev and NO UV_PROJECT_ENVIRONMENT (click-test lesson, 07-03):
                # Claude Desktop provisions <extension>/.venv ITSELF at install. A
                # pruning launch flag makes our `uv run` fight the host's sync — an
                # endless os-error-32 lock loop. Launch ACCEPTS the host env as-is.
                "args": [
                    "run",
                    "--frozen",
                    "--directory",
                    "${__dirname}",
                    "server_entry.py",
                ],
                "env": {
                    "ASSET_BOOK": "${user_config.book}",
                    "ASSET_CACHE_DIR": "${user_config.cache_dir}",
                    "ASSET_TARGET": "${user_config.target}",
                    "ASSET_MCP_OFFLINE": "${user_config.offline}",
                },
            },
        },
        "user_config": {
            "book": {
                "type": "file",
                "title": "Transaction file (your book)",
                "description": (
                    "Your Ghostfolio-compatible CSV or Ghostfolio JSON export. Optional: if "
                    "left unset the server runs on a bundled DEMO portfolio (fake data) so you "
                    "can explore immediately — swap in your own file when ready. (Kept optional "
                    "so the extension launches without a forced Configure-and-Save first.)"
                ),
                "default": "data/sample_data/transactions.csv",
                "required": False,
            },
            "cache_dir": {
                "type": "directory",
                "title": "Price-cache folder (optional)",
                # No "${HOME}/…" default: Claude Desktop shows such defaults LITERALLY,
                # unsubstituted (observed in the click-test) — confusing at best, a junk
                # path at worst. Left empty, the server itself defaults to
                # ~/.asset-management/prices (a real, stable per-user dir).
                "description": (
                    "Where downloaded price history is stored (created if missing). "
                    "Leave empty for the default: .asset-management/prices in your home "
                    "folder. Only set it if you already warmed a cache elsewhere."
                ),
                "required": False,
            },
            "target": {
                "type": "file",
                "title": "Target allocation CSV (optional)",
                "description": (
                    "Only needed by rebalance_check: a CSV of your target weights. Leave "
                    "empty otherwise."
                ),
                "required": False,
            },
            "offline": {
                "type": "boolean",
                "title": "Strictly offline (no price downloads)",
                "description": (
                    "Forbid ALL network use — no first-call auto-warm, no on-demand "
                    "ticker fetch. Only useful for a cache you've already warmed via the "
                    "CLI; a cold cache then answers n/a instead of fetching."
                ),
                "default": False,
                "required": False,
            },
        },
        "compatibility": {
            "platforms": ["darwin", "win32", "linux"],
            "runtimes": {"python": ">=3.12"},
        },
    }


# Fixed member metadata → two builds of the same tree are byte-identical, so a released
# bundle can be verified against a self-build by hash ("correctness you can check").
_ZIP_DATE = (2026, 1, 1, 0, 0, 0)


def _add(zf: zipfile.ZipFile, arcname: str, data: bytes) -> None:
    zi = zipfile.ZipInfo(arcname, date_time=_ZIP_DATE)
    zi.compress_type = zipfile.ZIP_DEFLATED
    zi.external_attr = 0o644 << 16
    zf.writestr(zi, data)


def _git_tracked(paths: list[Path]) -> None:
    """Refuse to bundle any file git doesn't track — in this dual-repo layout a private
    stray (e.g. a local-only app/scratch.py hidden via .git/info/exclude) is invisible to
    normal git hygiene yet would ride a glob straight into the distributed zip."""
    tracked = set(
        subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "-z"],
            check=True, capture_output=True, text=True,
        ).stdout.split("\0")
    )
    loose = [str(f.relative_to(ROOT)) for f in paths if f.relative_to(ROOT).as_posix() not in tracked]
    if loose:
        raise SystemExit(f"refusing to bundle files git does not track: {', '.join(loose)}")


def main() -> None:
    version = _version()
    # Repo-sourced members (git-tracked, shipped verbatim). pyproject.toml and uv.lock
    # are NOT here — the bundle carries GENERATED dev-free versions (see bundle_pyproject).
    files: list[Path] = [
        ROOT / "README.md",
        ROOT / "data" / "universe.csv",  # discover_gaps / propose_allocation read it
        ROOT / "data" / "sample_data" / "transactions.csv",  # the demo default book
        *sorted((ROOT / "app").glob("*.py")),
    ]
    for f in files:
        if not f.exists():
            raise SystemExit(f"missing bundle file: {f}")
        rel = f.relative_to(ROOT)
        if set(rel.parts) & _FORBIDDEN_PARTS or rel.name.startswith(".env"):
            raise SystemExit(f"refusing to bundle a forbidden path: {f}")
    _git_tracked(files)

    trimmed = bundle_pyproject()
    with tempfile.TemporaryDirectory() as td:
        stage = Path(td)
        (stage / "pyproject.toml").write_text(trimmed, encoding="utf-8")
        lock = _bundle_lock(stage)

    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    out = dist / f"asset-management-{version}.mcpb"
    with zipfile.ZipFile(out, "w") as zf:
        _add(zf, "manifest.json", (json.dumps(_manifest(version), indent=2) + "\n").encode())
        _add(zf, "server_entry.py", LAUNCHER_PY.encode())
        _add(zf, "pyproject.toml", trimmed.encode())
        _add(zf, "uv.lock", lock.encode())
        for f in files:
            _add(zf, f.relative_to(ROOT).as_posix(), f.read_bytes())

    kb = out.stat().st_size / 1024
    print(f"built {out.relative_to(ROOT)}  ({kb:.0f} KB, {len(files) + 4} files)")
    print("install: Claude Desktop → Settings → Extensions → Install Extension… → this file")


if __name__ == "__main__":
    main()
