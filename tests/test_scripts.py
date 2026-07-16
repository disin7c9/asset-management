"""The distribution scripts: the MCPB manifest contract + the fence demo.

These lock the launch/packaging decisions a unit test of `app/` can't see — the
README tells every visitor to run `demo_fence.py`, and one wrong manifest field
bricks the Claude-Desktop bundle for every installer.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _load_build_mcpb() -> Any:
    spec = importlib.util.spec_from_file_location("build_mcpb", ROOT / "scripts" / "build_mcpb.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_manifest_launch_line_dodges_both_launch_traps() -> None:
    # REGRESSION LOCK for two click-test fatalities:
    # (1) never launch `app/mcp_server.py` as a script — app/ lands at sys.path[0]
    #     where app/email.py shadows the stdlib `email` package (import crash);
    # (2) never put a literal "python" token in args — Claude Desktop probes it
    #     against the system PATH and dies with exit 9009 on Python-less machines.
    # The sanctioned shape is the official uv example's: a ROOT-level launcher script.
    # Also locked OUT: "--no-dev" — Desktop provisions <ext>/.venv itself WITH dev
    # groups it decides on; a pruning launch sync fights it (os-error-32 lock loop).
    # Locked IN: "--no-sync" — `--frozen` alone still SYNCS the env at launch, and that
    # launch-time sync is the writer that races Desktop's provisioning over pywin32's
    # locked .data dir; --no-sync makes launch touch nothing (the zero-tools regression fix).
    m = _load_build_mcpb()._manifest("9.9.9")
    args = m["server"]["mcp_config"]["args"]
    assert args == ["run", "--no-sync", "--frozen", "--directory", "${__dirname}", "server_entry.py"]
    assert "--no-dev" not in args  # the pruning flag that fought the host sync stays out
    assert m["server"]["entry_point"] == "server_entry.py"


def test_bundle_launcher_starts_from_the_bundle_root() -> None:
    # The launcher must import app.* with the BUNDLE ROOT as sys.path[0] — this is
    # the email-shadow dodge working end-to-end. Materialize it at the repo root
    # (same layout as the bundle: launcher next to app/) and run it for real.
    launcher = ROOT / "tmp_test_server_entry.py"
    launcher.write_text(_load_build_mcpb().LAUNCHER_PY, encoding="utf-8")
    try:
        env = dict(
            os.environ,
            ASSET_BOOK=str(ROOT / "data" / "sample_data" / "transactions.csv"),
            ASSET_MCP_OFFLINE="1",
            ASSET_CACHE_DIR="",
        )
        proc = subprocess.run(
            [sys.executable, str(launcher)],
            env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120,
        )
        assert "ModuleNotFoundError" not in proc.stderr
        assert "Traceback" not in proc.stderr
    finally:
        launcher.unlink()


def test_manifest_contract_fields() -> None:
    # The fields a schema-validating host rejects the install over, plus the env wiring
    # the server's config resolution depends on.
    m = _load_build_mcpb()._manifest("9.9.9")
    assert m["version"] == "9.9.9"
    assert isinstance(m["repository"], dict) and m["repository"]["url"]  # object, not string
    env = m["server"]["mcp_config"]["env"]
    assert env["ASSET_BOOK"] == "${user_config.book}"
    assert "UV_PROJECT_ENVIRONMENT" not in env  # host manages the env; don't steer uv
    # Runtime-registered starters: without this flag Desktop lists the prompts but
    # rejects every prompts/get as "attempted undeclared prompt" (07-05 click-test).
    assert m["prompts_generated"] is True
    cfg = m["user_config"]
    # book is OPTIONAL (required:false) so the extension launches without a forced
    # Configure-and-Save (Desktop's Save-disabled-until-changed trap); the server falls
    # back to the bundled demo book when it's unset.
    assert cfg["book"]["required"] is False and cfg["book"]["default"]  # demo pre-fill
    # cache_dir: optional and DEFAULT-FREE — Desktop displays "${HOME}/…" defaults
    # literally (unsubstituted); empty routes to the server's own safe per-user default.
    assert cfg["cache_dir"]["required"] is False and "default" not in cfg["cache_dir"]
    assert cfg["offline"]["type"] == "boolean" and cfg["offline"]["default"] is False


def test_manifest_declared_tools_match_the_server() -> None:
    # Some Desktop builds only expose DECLARED capabilities to the chat (the
    # prompts_generated lesson) — a tool added to the server but missing from the
    # manifest would silently vanish for those installers, and a stale declaration
    # would advertise a tool that doesn't exist.
    import anyio
    from mcp.shared.memory import create_connected_server_and_client_session as client_session

    from app.mcp_server import mcp

    async def _server_tools() -> set[str]:
        async with client_session(mcp) as client:
            return {t.name for t in (await client.list_tools()).tools}

    declared = [t["name"] for t in _load_build_mcpb()._manifest("9.9.9")["tools"]]
    assert len(declared) == len(set(declared))  # no duplicate declarations
    assert set(declared) == anyio.run(_server_tools)


def test_bundle_pyproject_is_dev_free() -> None:
    # The bundle ships a TRIMMED pyproject: Desktop's UV runtime syncs default groups
    # (dev included), and dev tools on end-user machines are worse than waste — ruff's
    # script-bearing wheel is what Desktop's concurrent setup passes race over (the
    # os-error-32 install loop from the 07-03 click-test). Repo pyproject is untouched.
    import tomllib

    trimmed = _load_build_mcpb().bundle_pyproject()
    parsed = tomllib.loads(trimmed)
    assert "dependency-groups" in tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "dependency-groups" not in parsed
    assert parsed["project"]["dependencies"]  # runtime deps intact
    assert parsed["build-system"]["build-backend"]  # build metadata intact


def test_demo_fence_runs_keyless_and_fails_closed() -> None:
    # The README's first "run it yourself" command: offline, no API key, and both
    # refusal paths visible. Redirected output must be plain (no ANSI escapes).
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "demo_fence.py"), "--fast"],
        capture_output=True, text=True, timeout=120, check=True,
    ).stdout
    assert out.count("REFUSED") == 2
    assert "-9.84%" in out  # the obedient act rendered the substituted figure
    assert "\033[" not in out  # piped run → color gated off


def test_mcp_server_starts_in_module_mode_offline() -> None:
    # End-to-end twin of the regression lock: the exact interpreter invocation the
    # bundle uses must import cleanly and serve stdio (EOF on stdin → clean shutdown).
    env = dict(
        os.environ,
        ASSET_BOOK=str(ROOT / "data" / "sample_data" / "transactions.csv"),
        ASSET_MCP_OFFLINE="1",  # hermetic: no auto-warm, no network
        ASSET_CACHE_DIR="",  # empty → per-request default; nothing is fetched offline
    )
    proc = subprocess.run(
        [sys.executable, "-m", "app.mcp_server"],
        env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120,
    )
    assert "ModuleNotFoundError" not in proc.stderr
    assert "Traceback" not in proc.stderr


def test_privacy_policy_url_resolves_to_a_real_readme_anchor() -> None:
    # The connectors directory rejects a local connector whose privacy policy is missing —
    # and a dead link is indistinguishable from a live one until a human clicks it. The
    # manifest points at a README anchor, so decorating that heading (an emoji shifts the
    # slug GitHub generates) would silently break the submission. Pin it: the fragment must
    # resolve, either to an explicit <a id> or to a heading whose GitHub slug matches.
    import re

    manifest = _load_build_mcpb()._manifest("9.9.9")
    urls = manifest["privacy_policies"]
    assert urls, "the manifest declares no privacy policy"

    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    def github_slug(title: str) -> str:  # verified against GitHub's markdown API
        title = title.strip().lower()
        return re.sub(r"\s", "-", re.sub(r"[^\w\s-]", "", title))

    targets = {github_slug(t) for t in re.findall(r"^#{1,6} (.+)$", readme, re.M)}
    targets |= set(re.findall(r'<a id="([^"]+)"></a>', readme))

    for url in urls:
        assert url.startswith("https://"), f"{url} must be HTTPS"
        _, _, fragment = url.partition("#")
        assert fragment, f"{url} names no anchor"
        assert fragment in targets, (
            f"privacy policy URL points at #{fragment}, which no README heading or anchor "
            f"provides — the directory submission would fail on a dead link"
        )
    assert re.search(r"^#{1,6} .*Privacy Policy", readme, re.M | re.I), (
        "README has no Privacy Policy section"
    )


def test_readme_desktop_config_pins_the_current_release_wheel() -> None:
    # The Desktop install config pins the release-wheel URL — a git+ source re-resolves on
    # every launch, and on Windows that install work races whatever holds freshly written
    # cache files (os error 32, hit live 2026-07-16; the wheel form is the fix). The cost of
    # a pin is that it goes stale silently: bump pyproject, forget the README, and every new
    # Desktop user installs the previous release. Fail the gate until the README catches up.
    import re
    import tomllib

    with (ROOT / "pyproject.toml").open("rb") as fh:
        version = tomllib.load(fh)["project"]["version"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    pins = re.findall(
        r"releases/download/v([\d.]+)/asset_management-([\d.]+)-py3-none-any\.whl", readme
    )
    assert pins, "README no longer shows the release-wheel Desktop config"
    for tag, wheel in pins:
        assert tag == version and wheel == version, (
            f"README pins wheel {wheel} (tag v{tag}) but pyproject says {version} — "
            f"new Desktop installs would get the old release"
        )
