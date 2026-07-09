"""CLI delivery routing tests (slice 4).

Offline: ``--no-prices`` avoids the network entirely; ``--send`` is exercised by
monkey-patching ``app.email._dispatch`` so no real email leaves the box.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from app import email as E
from app.cli import main
from app.prices import PriceRow, PricesResult, SeriesResult

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample_data" / "transactions.csv"
TARGET = Path(__file__).resolve().parents[1] / "data" / "sample_data" / "target.csv"


def _run_summary(caplog: pytest.LogCaptureFixture) -> dict[str, Any]:
    """Parse the last structured ``run_summary`` JSON line cli.main emitted."""
    for rec in reversed(caplog.records):
        msg = rec.getMessage()
        if rec.name == "app.cli" and msg.startswith("run_summary "):
            return json.loads(msg[len("run_summary ") :])  # type: ignore[no-any-return]
    raise AssertionError("no run_summary line was logged")


@pytest.fixture(autouse=True)
def _no_network_splits(monkeypatch: pytest.MonkeyPatch) -> None:
    # The brief loads + prices via app.pipeline (load_book + compute_prices_returns_
    # risk); --screen/--backtest still fetch in app.cli. Tests drive a run by patching
    # the app.cli fetch seam, so mirror it onto app.pipeline — one patch then controls
    # the whole run (the brief AND any cli-path fetch), and no run silently hits the
    # network. Splits are stubbed to "none known" (no adjustment; the price-basis guard
    # stays the net); a split-specific test overrides pipeline's fetch_series.
    import app.cli as _cli

    monkeypatch.setattr(
        "app.pipeline.fetch_series", lambda *a, **k: _cli.fetch_series(*a, **k)
    )
    monkeypatch.setattr(
        "app.pipeline.fetch_latest", lambda *a, **k: _cli.fetch_latest(*a, **k)
    )
    monkeypatch.setattr("app.pipeline.fetch_splits", lambda *a, **k: {})


@pytest.fixture(autouse=True)
def _hermetic_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    # cli reads ASSET_BOOK / ASSET_CSV / ASSET_TARGET as personal .env fallbacks. Pin
    # them to "" (treated as unset; load_dotenv never overrides an existing var) so the
    # contract tests keep working once the developer's real .env sets them. The
    # env-fallback tests below setenv real values over this.
    monkeypatch.setenv("ASSET_BOOK", "")
    monkeypatch.setenv("ASSET_CSV", "")
    monkeypatch.setenv("ASSET_TARGET", "")
    monkeypatch.setenv("ASSET_UNIVERSE", "")  # --warm full reads it; pin so a real .env can't leak
    monkeypatch.setenv("ASSET_CACHE_DIR", "")  # --cache-dir falls back to it; pin so .env can't leak


def _today() -> str:
    # Must match cli.main's `today = date.today()` (local), which dates the file.
    return date.today().isoformat()


def test_save_writes_markdown(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["--csv", str(SAMPLE), "--no-prices", "--save", "--reports-dir", str(tmp_path)])
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

    rc = main(["--csv", str(SAMPLE), "--no-prices", "--send"])
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

    rc = main(["--csv", str(SAMPLE), "--no-prices", "--send"])
    # Brief still printed, but the requested delivery failed → non-zero so cron alerts.
    assert rc == 1
    assert "=== HOLDINGS ===" in capsys.readouterr().out


def test_save_to_unwritable_path_prints_but_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Make reports-dir sit *under a regular file* so mkdir raises NotADirectoryError.
    blocker = tmp_path / "afile"
    blocker.write_text("not a dir", encoding="utf-8")
    rc = main(["--csv", str(SAMPLE), "--no-prices", "--save", "--reports-dir", str(blocker / "sub")])
    assert rc == 1  # save failed, recorded not raised
    assert "=== HOLDINGS ===" in capsys.readouterr().out  # brief still printed


def test_rebalance_emits_suggestions(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Canned prices for the sample portfolio (held == target tickers) so no network.
    now = datetime.now(timezone.utc)
    canned = {
        tk: PriceRow(tk, date.today(), px, "test", now)
        for tk, px in {"BND": 73.0, "IAU": 84.0, "VEA": 72.0, "VOO": 697.0}.items()
    }

    def fake_fetch_latest(tickers, *a, **k):  # type: ignore[no-untyped-def]
        rows = {tk: canned[tk] for tk in tickers if tk in canned}
        return PricesResult(rows=rows, missing=[tk for tk in tickers if tk not in canned])

    monkeypatch.setattr("app.cli.fetch_latest", fake_fetch_latest)
    rc = main(["--csv", str(SAMPLE), "--no-risk", "--rebalance", "to_total", "--target", str(TARGET)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "SUGGESTED ACTIONS (rebalance to target)" in out
    # The actions panel leads the brief, before HOLDINGS.
    assert out.index("SUGGESTED ACTIONS") < out.index("=== HOLDINGS ===")


def test_rebalance_requires_target() -> None:
    with pytest.raises(SystemExit):  # --rebalance without --target → parser.error
        main(["--rebalance", "to_total"])


def test_rebalance_negative_new_cash_rejected() -> None:
    with pytest.raises(SystemExit):  # argparse parser.error → exit 2
        main(["--rebalance", "fixed_dca", "--new-cash", "-100", "--target", str(TARGET)])


def test_band_rel_nonpositive_rejected() -> None:
    with pytest.raises(SystemExit):  # --band-rel must be a positive fraction of target
        main(["--csv", str(SAMPLE), "--rebalance", "bands", "--target", str(TARGET), "--band-rel", "0"])


def test_dump_target_writes_current_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    now = datetime.now(timezone.utc)
    canned = {
        tk: PriceRow(tk, date.today(), px, "test", now)
        for tk, px in {"BND": 73.0, "IAU": 84.0, "VEA": 72.0, "VOO": 697.0}.items()
    }

    def fake_fetch_latest(tickers, *a, **k):  # type: ignore[no-untyped-def]
        rows = {tk: canned[tk] for tk in tickers if tk in canned}
        return PricesResult(rows=rows, missing=[tk for tk in tickers if tk not in canned])

    monkeypatch.setattr("app.cli.fetch_latest", fake_fetch_latest)
    out_csv = tmp_path / "mytarget.csv"
    rc = main(["--csv", str(SAMPLE), "--no-risk", "--dump-target", str(out_csv)])
    assert rc == 0
    text = out_csv.read_text(encoding="utf-8")
    assert text.startswith("Ticker,Weight")
    assert "VOO," in text and "BND," in text
    # Weights are percentages summing to ~100.
    weights = [float(line.split(",")[1]) for line in text.strip().splitlines()[1:]]
    assert sum(weights) == pytest.approx(100.0, abs=0.1)


def test_rebalance_skips_when_holdings_unpriced(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # All fetches miss → held tickers have no usable price → suggestions skipped,
    # but the rest of the brief must still print (don't emit partial-book trades).
    def empty_fetch(tickers, *a, **k):  # type: ignore[no-untyped-def]
        return PricesResult(rows={}, missing=list(tickers))

    monkeypatch.setattr("app.cli.fetch_latest", empty_fetch)
    rc = main(["--csv", str(SAMPLE), "--no-risk", "--rebalance", "to_total", "--target", str(TARGET)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "SUGGESTED ACTIONS" not in out  # skipped, not partial/wrong
    assert "=== HOLDINGS ===" in out


def _canned_sample_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime.now(timezone.utc)
    canned = {
        tk: PriceRow(tk, date.today(), px, "test", now)
        for tk, px in {"BND": 73.0, "IAU": 84.0, "VEA": 72.0, "VOO": 697.0}.items()
    }
    monkeypatch.setattr(
        "app.cli.fetch_latest",
        lambda tickers, *a, **k: PricesResult(
            rows={tk: canned[tk] for tk in tickers if tk in canned},
            missing=[tk for tk in tickers if tk not in canned],
        ),
    )


def test_dump_target_never_writes_zero_for_a_holding(tmp_path: Path) -> None:
    # A dust position must not serialize as 0 (which would reload as a deliberate
    # exit), so a no-edit dump → rebalance round-trip never sells it.
    from app.cli import _dump_target
    from app.derive import DerivedState, Position
    from app.events import load_target
    from app.strategy import suggest

    now = datetime.now(timezone.utc)
    state = DerivedState()
    state.positions["VOO"] = Position("VOO", shares=1000.0, cost_basis=1.0)
    state.positions["DUST"] = Position("DUST", shares=1.0, cost_basis=1.0)
    prices = {
        "VOO": PriceRow("VOO", date.today(), 100.0, "t", now),   # $100,000
        "DUST": PriceRow("DUST", date.today(), 3.0, "t", now),   # $3 → 0.003%
    }
    out = tmp_path / "dump.csv"
    run: dict[str, Any] = {}
    _dump_target(state, prices, out, run)
    assert run["dump_target"] == str(out)  # the write is recorded in the run summary (#4)

    t = load_target(out)
    assert t["DUST"] > 0.0  # not written as 0.00 → not an accidental exit
    hv = {tk: state.held()[tk].shares * prices[tk].close for tk in state.held()}
    pps = {tk: prices[tk].close for tk in prices}
    s = {x.ticker: x for x in suggest("to_total", hv, pps, t)}
    assert s["DUST"].action != "sell"  # round-trip is HOLD, not a liquidation


def test_rebalance_directory_target_is_nonfatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _canned_sample_prices(monkeypatch)
    d = tmp_path / "adir"
    d.mkdir()
    rc = main(["--csv", str(SAMPLE), "--no-risk", "--rebalance", "to_total", "--target", str(d)])
    assert rc == 0  # IsADirectoryError caught; the rest of the brief prints
    out = capsys.readouterr().out
    assert "SUGGESTED ACTIONS" not in out
    assert "=== HOLDINGS ===" in out


def test_cli_backtest_renders_section(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import pandas as pd

    dates = pd.bdate_range("2024-01-01", periods=120)
    rows = {
        "VOO": pd.Series([100.0 + i * 0.5 for i in range(120)], index=dates, dtype=float),
        "VEA": pd.Series([50.0 + i * 0.2 for i in range(120)], index=dates, dtype=float),
        "BND": pd.Series([70.0] * 120, index=dates, dtype=float),
        "IAU": pd.Series([40.0 + i * 0.1 for i in range(120)], index=dates, dtype=float),
    }

    def fake_fetch_series(tickers, *a, **k):  # type: ignore[no-untyped-def]
        present = {tk: rows[tk] for tk in tickers if tk in rows}
        return SeriesResult(rows=present, missing=[tk for tk in tickers if tk not in rows])

    monkeypatch.setattr("app.cli.fetch_series", fake_fetch_series)
    # --no-prices skips the holdings price panel; --backtest still fetches series.
    rc = main(["--no-prices", "--backtest", "--rebalance-every", "monthly",
               "--target", str(TARGET)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "BACKTEST" in out
    assert "rebalanced (monthly)" in out and "buy & hold" in out


def test_backtest_requires_target() -> None:
    with pytest.raises(SystemExit):
        main(["--backtest"])


def test_backtest_start_in_future_rejected() -> None:
    with pytest.raises(SystemExit):
        main(["--backtest", "--target", str(TARGET), "--backtest-start", "2999-01-01"])


def test_book_action_without_csv_errors() -> None:
    # No silent sample fallback: a book-dependent action with no --csv must error
    # loudly (exit 2), not quietly run on the bundled example.
    for argv in (
        ["--rebalance", "to_total", "--target", str(TARGET)],
        ["--allocate", "equal_weight"],
        ["--dump-target", "/tmp/whatever.csv"],
        ["--save"],
        ["--send"],
        ["--metadata"],
        ["--screen", "QQQM"],
    ):
        with pytest.raises(SystemExit):
            main(argv)


def test_bare_run_prints_hint_not_a_brief(capsys: pytest.CaptureFixture[str]) -> None:
    # No --csv and no action → a usage hint, not a fabricated brief on the sample.
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "No portfolio given" in out and "--book" in out
    assert "=== HOLDINGS ===" not in out  # nothing fabricated


def test_demo_runs_the_brief_with_zero_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The try-before-you-trust path: no --book, no repo checkout — the bundled example
    # book is materialized from a package constant into the cache dir and run as-is.
    # Fetchers stubbed as a belt: --no-prices skips them today, but this test must stay
    # network-proof even if someone later exercises the priced demo path here.
    monkeypatch.setattr("app.cli.fetch_latest", lambda *a, **k: PricesResult())
    monkeypatch.setattr("app.cli.fetch_series", lambda *a, **k: SeriesResult())
    with caplog.at_level(logging.INFO):
        rc = main(["--demo", "--no-prices", "--cache-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "=== HOLDINGS ===" in out and "VOO" in out
    assert (tmp_path / "demo_book.csv").exists()
    assert _run_summary(caplog)["source"].endswith("demo_book.csv")


def test_demo_conflicts_with_an_explicit_book() -> None:
    with pytest.raises(SystemExit):  # parser.error → exit 2: --demo brings its own book
        main(["--demo", "--book", str(SAMPLE)])


def test_demo_ignores_a_personal_env_book(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A personal ASSET_BOOK must never leak into a demo run: --demo fills the book slot
    # before the .env fallback, so the bogus path below would explode if it were read.
    monkeypatch.setenv("ASSET_BOOK", "/nonexistent/personal.csv")
    rc = main(["--demo", "--no-prices", "--cache-dir", str(tmp_path)])
    assert rc == 0
    assert "=== HOLDINGS ===" in capsys.readouterr().out


def test_demo_ignores_a_personal_env_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The other half of demo insulation: a personal ASSET_TARGET must not silently steer
    # a demo rebalance ("sell the demo book, buy MY tickers" is not a demo). With the env
    # fallback skipped, --demo --rebalance without an explicit --target hits the normal
    # "requires --target" contract error instead of the personal file.
    monkeypatch.setenv("ASSET_TARGET", str(SAMPLE))  # a real, loadable file — must be ignored
    with pytest.raises(SystemExit):
        main(["--demo", "--no-prices", "--cache-dir", str(tmp_path),
              "--rebalance", "to_total"])
    # And the refused run left no demo residue: the hook runs AFTER the contract errors.
    assert not (tmp_path / "demo_book.csv").exists()


def _mock_backtest_series(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch fetch_series with 120 business days of gently rising sample-ticker prices."""
    import pandas as pd

    dates = pd.bdate_range("2024-01-01", periods=120)
    rows = {
        tk: pd.Series([100.0 + i * 0.3 for i in range(120)], index=dates, dtype=float)
        for tk in ("VOO", "VEA", "BND", "IAU")
    }
    monkeypatch.setattr(
        "app.cli.fetch_series",
        lambda tickers, *a, **k: SeriesResult(
            rows={tk: rows[tk] for tk in tickers if tk in rows},
            missing=[tk for tk in tickers if tk not in rows],
        ),
    )


def test_backtest_without_csv_is_simulation_only(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # --backtest is notional (target-only): with no --csv it prints ONLY the
    # simulation — no fabricated holdings/status from the bundled sample.
    _mock_backtest_series(monkeypatch)
    rc = main(["--backtest", "--target", str(TARGET)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "=== BACKTEST" in out          # the simulation prints
    assert "=== HOLDINGS ===" not in out  # but NO holdings/status (no book given)
    assert "=== DRAWDOWN" not in out


def test_screen_refuses_act_or_simulate_combos() -> None:
    # Propose-only: judge candidates first, act in a separate command.
    for extra in (
        ["--rebalance", "to_total", "--target", str(TARGET)],
        ["--backtest", "--target", str(TARGET)],
        ["--allocate", "equal_weight"],
    ):
        with pytest.raises(SystemExit):
            main(["--csv", str(SAMPLE), "--screen", "QQQM", *extra])


def test_screen_panel_renders_with_verdicts(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    # Plumbing test (the screen MATH is golden-tested in test_screen.py): the
    # panel renders, every check ran, and the run summary records the verdict.
    # With this fixture the deterministic outcome is WARN — the synthetic flat
    # series gives a low ρ (the sample book's dividend/sell days spike the
    # flow-neutralized returns), and the canned metadata has no look-through
    # holdings, so overlap falls back to "same category as held" → warn.
    # (The split guard would exclude tickers whose synthetic series disagrees
    # with the sample book's execution prices — irrelevant here; neutralize it.)
    from app.metadata import MetadataResult, SecurityMeta

    monkeypatch.setattr("app.cli.fetch_series", _flat_series)
    monkeypatch.setattr("app.pipeline.price_basis_mismatches", lambda *a, **k: [])

    def fake_meta(tickers, **k):  # type: ignore[no-untyped-def]
        rows = {
            tk: SecurityMeta(
                ticker=tk, expense_ratio=0.0003, aum=5e9, avg_volume=2e6,
                category="Large Blend", family="Vanguard", legal_type="Exchange Traded Fund",
                quote_type="ETF", inception=date(2015, 1, 1),
            )
            for tk in tickers
        }
        return MetadataResult(rows=rows)

    monkeypatch.setattr("app.pipeline.fetch_metadata", fake_meta)
    with caplog.at_level(logging.INFO):
        rc = main(["--csv", str(SAMPLE), "--screen", "QQQM"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "CANDIDATES (deterministic screen)" in out
    assert "QQQM — WARN" in out
    assert "ρ=" in out                       # the diversifier test ran against the book
    assert "same category" in out            # overlap fell back to category (no holdings)
    assert "[pass] cost: 0.03%" in out       # metadata checks ran
    assert _run_summary(caplog)["screen"] == "QQQM:warn"


def test_screen_with_target_adds_role_row(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    # --screen + --target runs the walk-forward role check per candidate. With the
    # flat synthetic series both portfolios have ~zero drawdown → deterministic
    # "inconclusive" → a [warn] role row with the OOS window in its reason.
    from app.metadata import MetadataResult

    monkeypatch.setattr("app.cli.fetch_series", _flat_series)
    monkeypatch.setattr("app.pipeline.price_basis_mismatches", lambda *a, **k: [])
    monkeypatch.setattr("app.pipeline.fetch_metadata", lambda tickers, **k: MetadataResult())
    with caplog.at_level(logging.INFO):
        rc = main(["--csv", str(SAMPLE), "--screen", "QQQM", "--target", str(TARGET)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "] role:" in out and "OOS" in out
    assert "held-out simulation, not a prediction" in out  # footer switched
    assert _run_summary(caplog)["screen"] == "QQQM:warn"


def test_screen_drops_cash_pseudo_ticker(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    # Regression (review 2026-06-12): CASH in the candidate list must be dropped
    # before any fetch (it's not a screenable security) — a CASH-only screen
    # degrades to "no tickers" instead of a doomed network fetch + partial status.
    with caplog.at_level(logging.INFO):
        rc = main(["--csv", str(SAMPLE), "--no-prices", "--screen", "CASH"])
    capsys.readouterr()
    assert rc == 0
    assert _run_summary(caplog)["screen"] == "skipped: no tickers"
    assert "cash pseudo-ticker" in caplog.text


def test_screen_skipped_without_price_pipeline(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    # The diversifier needs the portfolio return series; --no-prices can't supply it.
    with caplog.at_level(logging.INFO):
        rc = main(["--csv", str(SAMPLE), "--no-prices", "--screen", "QQQM"])
    capsys.readouterr()
    assert rc == 0
    assert _run_summary(caplog)["screen"] == "skipped: needs the price pipeline"


def test_discover_refuses_act_or_simulate_combos() -> None:
    # Propose-only, same discipline as --screen: judge first, act in a separate command.
    for extra in (
        ["--rebalance", "to_total", "--target", str(TARGET)],
        ["--backtest", "--target", str(TARGET)],
        ["--allocate", "equal_weight"],
    ):
        with pytest.raises(SystemExit):
            main(["--csv", str(SAMPLE), "--discover", *extra])


def test_discover_skipped_without_price_pipeline(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    # Exposure (and the diversifier) need the price pipeline; --no-prices can't supply it.
    with caplog.at_level(logging.INFO):
        rc = main(["--csv", str(SAMPLE), "--no-prices", "--discover"])
    capsys.readouterr()
    assert rc == 0
    assert _run_summary(caplog)["discover"] == "skipped: needs the price pipeline"


def test_discover_panel_renders(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    # --discover finds the book's gaps from the curated universe and screens the fillers.
    # Target one role (reit) so the screened set is small + deterministic.
    from app.metadata import MetadataResult, SecurityMeta

    now = datetime.now(timezone.utc)
    canned = {
        tk: PriceRow(tk, date.today(), px, "test", now)
        for tk, px in {"BND": 73.0, "IAU": 84.0, "VEA": 72.0, "VOO": 697.0}.items()
    }
    monkeypatch.setattr(
        "app.cli.fetch_latest",
        lambda tickers, *a, **k: PricesResult(
            rows={tk: canned[tk] for tk in tickers if tk in canned},
            missing=[tk for tk in tickers if tk not in canned],
        ),
    )
    monkeypatch.setattr("app.cli.fetch_series", _flat_series)
    monkeypatch.setattr("app.pipeline.price_basis_mismatches", lambda *a, **k: [])
    monkeypatch.setattr(
        "app.pipeline.fetch_metadata",
        lambda tickers, **k: MetadataResult(
            rows={
                tk: SecurityMeta(
                    ticker=tk, expense_ratio=0.0003, aum=5e9, avg_volume=2e6,
                    category="Real Estate", family="Vanguard",
                    legal_type="Exchange Traded Fund", quote_type="ETF",
                    inception=date(2015, 1, 1),
                )
                for tk in tickers
            }
        ),
    )
    with caplog.at_level(logging.INFO):
        rc = main(["--csv", str(SAMPLE), "--discover", "reit"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "DISCOVERY (roles you're light in" in out
    assert "reit  — you currently hold 0%" in out
    assert "VNQ" in out  # a reit candidate from the universe was screened
    assert "VNQ:" in _run_summary(caplog)["discover"]  # verdict recorded


def test_discover_with_target_runs_the_role_check(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # --discover --target: discovered gap-fillers now also get the walk-forward role check
    # (#1) — the same held-out evidence a hand-typed --screen candidate gets.
    from app.metadata import MetadataResult, SecurityMeta

    now = datetime.now(timezone.utc)
    canned = {
        tk: PriceRow(tk, date.today(), px, "test", now)
        for tk, px in {"BND": 73.0, "IAU": 84.0, "VEA": 72.0, "VOO": 697.0}.items()
    }
    monkeypatch.setattr(
        "app.cli.fetch_latest",
        lambda tickers, *a, **k: PricesResult(
            rows={tk: canned[tk] for tk in tickers if tk in canned},
            missing=[tk for tk in tickers if tk not in canned],
        ),
    )
    monkeypatch.setattr("app.cli.fetch_series", _flat_series)
    monkeypatch.setattr("app.pipeline.price_basis_mismatches", lambda *a, **k: [])
    monkeypatch.setattr(
        "app.pipeline.fetch_metadata",
        lambda tickers, **k: MetadataResult(
            rows={
                tk: SecurityMeta(
                    ticker=tk, expense_ratio=0.0003, aum=5e9, avg_volume=2e6,
                    category="Real Estate", family="Vanguard",
                    legal_type="Exchange Traded Fund", quote_type="ETF",
                    inception=date(2015, 1, 1),
                )
                for tk in tickers
            }
        ),
    )
    rc = main(["--csv", str(SAMPLE), "--discover", "reit", "--target", str(TARGET)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "DISCOVERY (roles you're light in" in out
    assert "role:" in out  # the walk-forward role check ran for the discovered candidates


def test_metadata_not_double_fetched_with_screen(monkeypatch: pytest.MonkeyPatch) -> None:
    # --metadata + --screen: the held tickers' facts are fetched ONCE — the run's --metadata
    # fetch is reused by the screen, not re-fetched (#3).
    from app.metadata import MetadataResult, SecurityMeta

    now = datetime.now(timezone.utc)
    canned = {
        tk: PriceRow(tk, date.today(), px, "test", now)
        for tk, px in {"BND": 73.0, "IAU": 84.0, "VEA": 72.0, "VOO": 697.0}.items()
    }
    monkeypatch.setattr(
        "app.cli.fetch_latest",
        lambda tickers, *a, **k: PricesResult(
            rows={tk: canned[tk] for tk in tickers if tk in canned},
            missing=[tk for tk in tickers if tk not in canned],
        ),
    )
    monkeypatch.setattr("app.cli.fetch_series", _flat_series)
    monkeypatch.setattr("app.pipeline.price_basis_mismatches", lambda *a, **k: [])

    calls: list[list[str]] = []

    def counting_meta(tickers, **k):  # type: ignore[no-untyped-def]
        calls.append(list(tickers))
        return MetadataResult(
            rows={
                tk: SecurityMeta(
                    ticker=tk, expense_ratio=0.0003, aum=5e9, avg_volume=2e6,
                    category="X", family="Y", legal_type="ETF", quote_type="ETF",
                    inception=date(2015, 1, 1),
                )
                for tk in tickers
            }
        )

    # The --metadata handler fetches held via app.cli; the screen's candidate fetch goes through
    # pipeline — patch both with the one counter so reuse (no held re-fetch) is provable + hermetic.
    monkeypatch.setattr("app.cli.fetch_metadata", counting_meta)
    monkeypatch.setattr("app.pipeline.fetch_metadata", counting_meta)
    rc = main(["--csv", str(SAMPLE), "--metadata", "--screen", "QQQM"])
    assert rc == 0
    assert any("VOO" in c for c in calls)              # held facts were fetched
    assert sum("VOO" in c for c in calls) == 1         # exactly once, not twice (#3)


def test_discover_narrate_leads_the_panel(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    # --discover --narrate: a fence-validated LLM note leads the deterministic panel.
    from app.llm import NarratorConfig
    from app.metadata import MetadataResult, SecurityMeta

    now = datetime.now(timezone.utc)
    canned = {
        tk: PriceRow(tk, date.today(), px, "test", now)
        for tk, px in {"BND": 73.0, "IAU": 84.0, "VEA": 72.0, "VOO": 697.0}.items()
    }
    monkeypatch.setattr(
        "app.cli.fetch_latest",
        lambda tickers, *a, **k: PricesResult(
            rows={tk: canned[tk] for tk in tickers if tk in canned},
            missing=[tk for tk in tickers if tk not in canned],
        ),
    )
    monkeypatch.setattr("app.cli.fetch_series", _flat_series)
    monkeypatch.setattr("app.pipeline.price_basis_mismatches", lambda *a, **k: [])
    monkeypatch.setattr(
        "app.pipeline.fetch_metadata",
        lambda tickers, **k: MetadataResult(
            rows={
                tk: SecurityMeta(
                    ticker=tk, expense_ratio=0.0003, aum=5e9, avg_volume=2e6,
                    category="Real Estate", family="Vanguard",
                    legal_type="Exchange Traded Fund", quote_type="ETF",
                    inception=date(2015, 1, 1),
                )
                for tk in tickers
            }
        ),
    )

    def fake_complete(_cfg: object, system: str, _user: str) -> str:
        if "NEW funds" in system:  # the discovery note
            return "You hold {{gap_reit}} of real estate; the screened picks are worth a look."
        return "A calm summary of the book."  # the brief SUMMARY

    monkeypatch.setattr(
        "app.cli.load_config",
        lambda: NarratorConfig("openai", "test-model", "k", "http://x", "paid", 0.0),
    )
    monkeypatch.setattr("app.cli.complete", fake_complete)

    with caplog.at_level(logging.INFO):
        rc = main(["--csv", str(SAMPLE), "--discover", "reit", "--narrate"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "DISCOVERY — worth a closer look" in out          # the narrated note section leads
    assert "the screened picks are worth a look" in out      # the model's words, rendered
    assert "wording by test-model (paid tier)" in out        # source-labeled provenance
    assert "DISCOVERY (roles you're light in" in out         # deterministic panel still follows
    assert _run_summary(caplog)["discover_narrate"] == "test-model (paid)"


def test_benchmark_narrate_leads_the_panel(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    # --backtest --benchmark --narrate: a fence-validated note leads the BENCHMARK panel.
    from app.llm import NarratorConfig

    monkeypatch.setattr("app.cli.fetch_series", _flat_series)

    def fake_complete(_cfg: object, system: str, _user: str) -> str:
        if "reference portfolio" in system:  # the benchmark note
            return (
                "Your posture's overall drawdown pain was {{bench_ulcer_preset}}, versus "
                "{{bench_ulcer_reference}} for {{bench_reference}}; a held-out test "
                "couldn't tell them apart."
            )
        return "A calm summary of the book."  # the (empty-book) SUMMARY block

    monkeypatch.setattr(
        "app.cli.load_config",
        lambda: NarratorConfig("openai", "test-model", "k", "http://x", "paid", 0.0),
    )
    monkeypatch.setattr("app.cli.complete", fake_complete)

    with caplog.at_level(logging.INFO):
        rc = main([
            "--csv", str(SAMPLE), "--no-prices", "--backtest", "--target", str(TARGET),
            "--benchmark", "60-40", "--narrate",
        ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "BENCHMARK — how your posture compares" in out         # the narrated note section leads
    assert "the classic 60/40 stock-and-bond mix" in out          # the reference name, rendered
    assert "couldn't tell them apart" in out                      # the model's words
    assert "wording by test-model (paid tier)" in out             # source-labeled provenance
    assert "BENCHMARK (preset vs 60-40" in out                    # deterministic panel still follows
    assert _run_summary(caplog)["benchmark_narrate"] == "test-model (paid)"


def _canned_meta() -> object:
    from app.metadata import MetadataResult, SecurityMeta

    row = SecurityMeta(
        ticker="VOO", expense_ratio=0.0003, aum=1.7e12, avg_volume=8.8e6,
        category="Large Blend", family="Vanguard", legal_type="Exchange Traded Fund",
        quote_type="ETF", inception=date(2010, 9, 16),
    )
    return MetadataResult(rows={"VOO": row}, missing=["IAU"])


def test_metadata_requires_csv() -> None:
    # --metadata reads YOUR holdings → it's a book action (exit 2 without --csv).
    with pytest.raises(SystemExit):
        main(["--metadata"])


def test_metadata_panel_renders_and_flips_partial(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("app.cli.fetch_metadata", lambda tickers, **k: _canned_meta())
    with caplog.at_level(logging.INFO):
        rc = main(["--csv", str(SAMPLE), "--no-prices", "--metadata"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "SECURITIES (know your holdings)" in out
    assert "0.03%" in out and "$1.70T" in out and "Large Blend — Vanguard" in out
    assert "no metadata for: IAU" in out
    summary = _run_summary(caplog)
    assert summary["metadata"] == "1 fetched, 1 missing"
    assert summary["status"] == "partial"  # a missing ticker degrades honestly


def test_metadata_asks_only_held_tickers(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    asked: list[list[str]] = []

    def spy(tickers: list[str], **k: object) -> object:
        asked.append(list(tickers))
        from app.metadata import MetadataResult
        return MetadataResult()

    monkeypatch.setattr("app.cli.fetch_metadata", spy)
    rc = main(["--csv", str(SAMPLE), "--no-prices", "--metadata"])
    capsys.readouterr()
    assert rc == 0
    assert asked == [["BND", "IAU", "VEA", "VOO"]]  # held tickers, sorted, no CASH


def test_env_asset_csv_makes_bare_run_a_brief(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # ASSET_CSV in .env is the personal default: a run with no --csv uses it
    # instead of printing the usage hint.
    monkeypatch.setenv("ASSET_CSV", str(SAMPLE))
    rc = main(["--no-prices"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "=== HOLDINGS ===" in out
    assert "No portfolio given" not in out


def test_book_flag_loads_the_book(capsys: pytest.CaptureFixture[str]) -> None:
    # --book is the canonical flag (--csv/--json are aliases of the same dest).
    rc = main(["--book", str(SAMPLE), "--no-prices"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "=== HOLDINGS ===" in out


def test_json_flag_reads_a_ghostfolio_export(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    # The loader auto-detects format by content, so --json is a label; pointing it at
    # a real Ghostfolio JSON export must derive a holding through the CLI brief path.
    export = tmp_path / "ghostfolio-export.json"
    export.write_text(
        '{"activities": [{"type": "BUY", "symbol": "VOO", "quantity": 10, '
        '"unitPrice": 365, "fee": 1, "currency": "USD", "dataSource": "YAHOO", '
        '"date": "2023-01-04T15:00:00.000Z", "comment": null}]}',
        encoding="utf-8",
    )
    rc = main(["--json", str(export), "--no-prices"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "=== HOLDINGS ===" in out and "VOO" in out


def test_env_asset_book_makes_bare_run_a_brief(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # ASSET_BOOK is the canonical .env default (ASSET_CSV stays honored for back-compat):
    # a run with no flag uses it instead of printing the usage hint.
    monkeypatch.setenv("ASSET_BOOK", str(SAMPLE))
    rc = main(["--no-prices"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "=== HOLDINGS ===" in out
    assert "No portfolio given" not in out


def test_env_asset_csv_does_not_reach_pure_backtest(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # The pure `--backtest --target` run is notional by contract: it stays
    # book-free even when ASSET_CSV is set (pass --csv explicitly to include it).
    _mock_backtest_series(monkeypatch)
    monkeypatch.setenv("ASSET_CSV", str(SAMPLE))
    rc = main(["--backtest", "--target", str(TARGET)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "=== BACKTEST" in out
    assert "=== HOLDINGS ===" not in out


def test_narrate_does_not_pull_book_into_pure_backtest(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    # C3 (privacy): a pure `--backtest --target` is notional by contract — adding --narrate must
    # NOT drag in ASSET_BOOK (the user's real portfolio). Regression for the leak where --narrate
    # sat in needs_book, so it loaded ASSET_BOOK and shipped a SUMMARY of it to the LLM.
    _mock_backtest_series(monkeypatch)
    monkeypatch.setenv("ASSET_BOOK", str(SAMPLE))
    monkeypatch.setenv("ASSET_CSV", str(SAMPLE))
    with caplog.at_level(logging.INFO):
        rc = main(["--backtest", "--target", str(TARGET), "--narrate"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "=== BACKTEST" in out
    assert "=== HOLDINGS ===" not in out                       # book-free: no portfolio loaded
    assert _run_summary(caplog)["source"] == "(no book; backtest-only)"
    assert "nothing to narrate" in caplog.text                 # no book SUMMARY, no benchmark note


def test_narrate_backtest_benchmark_is_book_free(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    # C3 (the useful case the fix enables): `--backtest --benchmark --narrate` with NO book
    # narrates the benchmark verdict book-free — and even with ASSET_BOOK set, the personal book
    # is never loaded and the book-derived SUMMARY never runs.
    from app.llm import NarratorConfig

    monkeypatch.setattr("app.cli.fetch_series", _flat_series)
    monkeypatch.setenv("ASSET_BOOK", str(SAMPLE))  # set, but must NOT be loaded

    def fake_complete(_cfg: object, system: str, _user: str) -> str:
        assert "reference portfolio" in system  # ONLY the benchmark note — never a book SUMMARY
        return (
            "Your posture's overall drawdown pain was {{bench_ulcer_preset}}, versus "
            "{{bench_ulcer_reference}} for {{bench_reference}}; a held-out test "
            "couldn't tell them apart."
        )

    monkeypatch.setattr(
        "app.cli.load_config",
        lambda: NarratorConfig("openai", "test-model", "k", "http://x", "paid", 0.0),
    )
    monkeypatch.setattr("app.cli.complete", fake_complete)

    with caplog.at_level(logging.INFO):
        rc = main(["--backtest", "--target", str(TARGET), "--benchmark", "60-40", "--narrate"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "=== HOLDINGS ===" not in out                       # book-free: no personal portfolio
    assert _run_summary(caplog)["source"] == "(no book; backtest-only)"
    assert "couldn't tell them apart" in out                   # the benchmark note still rendered
    assert _run_summary(caplog)["benchmark_narrate"] == "test-model (paid)"


def test_explicit_csv_wins_over_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # An explicit --csv always beats the .env default (which here points nowhere).
    monkeypatch.setenv("ASSET_CSV", str(tmp_path / "does_not_exist.csv"))
    rc = main(["--csv", str(SAMPLE), "--no-prices"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "=== HOLDINGS ===" in out


def test_env_asset_csv_expands_tilde(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # ~ in an .env path means the home directory, not a repo-relative "~" folder.
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "book.csv").write_text(SAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv("ASSET_CSV", "~/book.csv")
    rc = main(["--no-prices"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "=== HOLDINGS ===" in out


def test_env_asset_target_fills_rebalance(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # ASSET_TARGET in .env fills --target when --rebalance/--backtest need one.
    _canned_sample_prices(monkeypatch)
    monkeypatch.setenv("ASSET_TARGET", str(TARGET))
    rc = main(["--csv", str(SAMPLE), "--no-risk", "--rebalance", "to_total"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "SUGGESTED ACTIONS" in out


def test_omitting_a_holding_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    _canned_sample_prices(monkeypatch)
    t = tmp_path / "omit.csv"
    t.write_text("Ticker,Weight\nVOO,45\nVEA,20\nBND,35\n", encoding="utf-8")  # IAU omitted
    with caplog.at_level(logging.WARNING):
        main(["--csv", str(SAMPLE), "--no-risk", "--rebalance", "to_total", "--target", str(t)])
    capsys.readouterr()
    assert "omits held tickers" in caplog.text and "IAU" in caplog.text


def test_explicit_zero_weight_does_not_warn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    _canned_sample_prices(monkeypatch)
    t = tmp_path / "explicit.csv"
    t.write_text("Ticker,Weight\nVOO,45\nVEA,20\nBND,35\nIAU,0\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        main(["--csv", str(SAMPLE), "--no-risk", "--rebalance", "to_total", "--target", str(t)])
    out = capsys.readouterr().out
    assert "omits held tickers" not in caplog.text  # explicit 0 → no omission warning
    assert "IAU" in out  # still shown as a full-exit sell


# ── --offline end-to-end (no network) ───────────────────────────────────────


def test_offline_never_touches_the_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # --offline must serve from cache only. Make BOTH network wrappers fail the
    # test if reached; an empty cache → all prices missing, but the brief prints.
    import app.prices as P

    monkeypatch.setattr(P, "_fetch_yf", lambda *a, **k: pytest.fail("network call in --offline"))
    monkeypatch.setattr(
        P, "_fetch_stooq_csv", lambda *a, **k: pytest.fail("network call in --offline")
    )
    rc = main(["--csv", str(SAMPLE), "--offline", "--cache-dir", str(tmp_path)])
    assert rc == 0
    assert "=== HOLDINGS ===" in capsys.readouterr().out


# ── run_summary structured-log paths ────────────────────────────────────────


def test_missing_csv_returns_2_and_logs_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        rc = main(["--csv", str(tmp_path / "nope.csv")])
    assert rc == 2
    run = _run_summary(caplog)
    assert run["status"] == "error" and run["error"] == "book_not_found"


def test_run_summary_counts_prices_on_series_path(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Regression for P1#1: the common risk-on path must record n_prices_fetched
    # (it was stuck at 0 because only the --no-risk branch set it), and
    # fallbacks_used must reflect the series provenance.
    import pandas as pd

    dates = pd.bdate_range("2024-01-01", periods=400)

    def fake_series(tickers, *a, **k):  # type: ignore[no-untyped-def]
        rows = {
            tk: pd.Series([100.0 + i * 0.1 for i in range(len(dates))], index=dates, dtype=float)
            for tk in tickers
        }
        prov = {tk: ("stooq", datetime.now(timezone.utc)) for tk in tickers}
        return SeriesResult(rows=rows, missing=[], provenance=prov)

    monkeypatch.setattr("app.cli.fetch_series", fake_series)
    with caplog.at_level(logging.INFO):
        rc = main(["--csv", str(SAMPLE)])  # default sample CSV, risk panel ON
    capsys.readouterr()
    assert rc == 0
    run = _run_summary(caplog)
    assert run["n_prices_fetched"] > 0      # was 0 before the fix
    assert run["n_prices_missing"] == 0
    assert run["n_series_fetched"] > 0
    assert run["fallbacks_used"] == run["n_series_fetched"]  # all provenance is stooq


def test_rebalance_missing_target_file_skip_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    _canned_sample_prices(monkeypatch)
    with caplog.at_level(logging.INFO):
        rc = main(["--csv", str(SAMPLE), "--no-risk", "--rebalance", "to_total", "--target", str(tmp_path / "nope.csv")])
    capsys.readouterr()
    assert rc == 0
    assert _run_summary(caplog)["rebalance"] == "skipped: no target file"


def test_rebalance_no_prices_skip_reason(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    # --no-prices leaves nothing to size trades against → skip (no network either).
    with caplog.at_level(logging.INFO):
        rc = main(["--csv", str(SAMPLE), "--no-prices", "--rebalance", "to_total", "--target", str(TARGET)])
    capsys.readouterr()
    assert rc == 0
    assert _run_summary(caplog)["rebalance"] == "skipped: --no-prices"


def test_backtest_bad_target_skip_reason(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    d = tmp_path / "adir"
    d.mkdir()  # a directory → load_target raises OSError → non-fatal skip
    with caplog.at_level(logging.INFO):
        rc = main(["--no-prices", "--backtest", "--target", str(d)])
    capsys.readouterr()
    assert rc == 0
    assert _run_summary(caplog)["backtest"] == "skipped: bad target"


def test_email_failure_marks_status_partial(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(E, "_dispatch", lambda p, k: (_ for _ in ()).throw(AssertionError))
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    monkeypatch.delenv("REPORT_TO", raising=False)
    monkeypatch.setattr("app.cli.load_dotenv", lambda *a, **k: False)
    with caplog.at_level(logging.INFO):
        rc = main(["--csv", str(SAMPLE), "--no-prices", "--send"])
    capsys.readouterr()
    assert rc == 1
    run = _run_summary(caplog)
    assert run["status"] == "partial" and run["email_sent"] is False


# ── P1 regressions: cash-mode warning + --no-risk scope ──────────────────────


def test_cash_mode_without_cash_warns(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    # P1#3: fixed_dca / cash_flow_only with --new-cash 0 deploy nothing → warn.
    _canned_sample_prices(monkeypatch)
    with caplog.at_level(logging.WARNING):
        main(["--csv", str(SAMPLE), "--no-risk", "--rebalance", "fixed_dca", "--target", str(TARGET)])
    capsys.readouterr()
    assert "nothing to invest" in caplog.text


def test_no_risk_still_runs_explicit_backtest(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # P1#2 (documented scope): --no-risk drops the HOLDINGS risk panel but the
    # explicitly-requested --backtest still reports its own risk metrics.
    import pandas as pd

    _canned_sample_prices(monkeypatch)  # held latest prices, no network
    dates = pd.bdate_range("2024-01-01", periods=120)
    rows = {
        tk: pd.Series([100.0 + i * 0.5 for i in range(120)], index=dates, dtype=float)
        for tk in ("VOO", "VEA", "BND", "IAU")
    }
    monkeypatch.setattr(
        "app.cli.fetch_series",
        lambda tickers, *a, **k: SeriesResult(
            rows={tk: rows[tk] for tk in tickers if tk in rows},
            missing=[tk for tk in tickers if tk not in rows],
        ),
    )
    rc = main(["--csv", str(SAMPLE), "--no-risk", "--backtest", "--target", str(TARGET)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "=== BACKTEST" in out          # backtest panel present despite --no-risk
    assert "=== RISK-ADJUSTED" not in out  # holdings risk panel suppressed
    assert "=== DRAWDOWN" not in out


def test_main_resolves_today_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Regression (v1.7.0 bug-class): cli.main must read the clock ONCE and thread
    # it, so the title, the returns-window end, and the --backtest end can never
    # straddle local midnight. _compute_prices_returns_risk and _compute_backtest
    # used to re-read date.today(); now they receive it. A full run exercising both
    # must make exactly one cli-side clock read. (Line 250's validation read only
    # fires with --backtest-start, which this run omits.)
    class _ClockSpy(date):
        calls = 0

        @classmethod
        def today(cls) -> date:
            _ClockSpy.calls += 1
            return date(2026, 6, 13)

    monkeypatch.setattr("app.cli.date", _ClockSpy)
    monkeypatch.setattr("app.cli.fetch_series", _flat_series)
    monkeypatch.setattr("app.pipeline.price_basis_mismatches", lambda *a, **k: [])
    with caplog.at_level(logging.INFO):
        rc = main(["--csv", str(SAMPLE), "--backtest", "--target", str(TARGET)])
    capsys.readouterr()
    assert rc == 0
    assert _ClockSpy.calls == 1
    assert _run_summary(caplog)["date"] == "2026-06-13"


def test_nonpositive_priced_holding_suppresses_money_weighted(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Review #1: a held ticker priced with a non-usable close (NaN/<=0) is dropped
    # from market value, so the book is only PARTIALLY priced. Money-weighted
    # figures (MWR / Modified Dietz) must then be n/a — not a confident wrong
    # number computed over a partial book — while path-based TWR is preserved.
    import pandas as pd

    dates = pd.bdate_range("2024-01-01", periods=400)

    def fake_series(tickers, *a, **k):  # type: ignore[no-untyped-def]
        rows = {}
        for tk in tickers:
            s = pd.Series([100.0 + i * 0.1 for i in range(400)], index=dates, dtype=float)
            if tk == "VOO":          # one held ticker gets an unusable tail price
                s = s.copy()
                s.iloc[-1] = float("nan")
            rows[tk] = s
        prov = {tk: ("cache", datetime.now(timezone.utc)) for tk in tickers}
        return SeriesResult(rows=rows, missing=[], provenance=prov)

    monkeypatch.setattr("app.cli.fetch_series", fake_series)
    with caplog.at_level(logging.INFO):
        rc = main(["--csv", str(SAMPLE)])  # risk-on (so true TWR is computed and the RETURNS panel renders)
    out = capsys.readouterr().out
    assert rc == 0
    twr_line = next(line for line in out.splitlines() if "Time-weighted" in line)
    mwr_line = next(line for line in out.splitlines() if "Money-weighted" in line)
    dietz_line = next(line for line in out.splitlines() if "Modified Dietz" in line)
    assert not twr_line.strip().endswith("n/a")  # path-based TWR survives
    assert mwr_line.strip().endswith("n/a")       # money-weighted suppressed
    assert dietz_line.strip().endswith("n/a")
    run = _run_summary(caplog)
    assert run["status"] == "partial"
    assert run["n_prices_missing"] >= 1  # VOO counted unpriced (dropped from value)


def test_unhandled_split_ticker_excluded_from_twr_and_noted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    # Real-data guard (Shinhan/NVDA): a ticker bought at ~10× its split-adjusted
    # close (an unhandled split) must be EXCLUDED from the time-weighted series so
    # it can't fabricate a return — with a warning and a report note. Mirrors the
    # NVDA case: the split ticker is sold out, so current holdings stay clean.
    import pandas as pd

    csv = tmp_path / "book.csv"
    csv.write_text(
        "Date,Code,Action,Quantity,Price,Fee\n"
        "2024-01-02,VOO,buy,10,100,0\n"
        "2024-01-02,SPLT,buy,1,1000,0\n"   # exec 1000 vs adjusted close ~100 → 10×
        "2024-01-03,SPLT,sell,1,1010,0\n"  # sold out → not a current holding
        "2024-02-01,VOO,buy,5,100,0\n",
        encoding="utf-8",
    )
    dates = pd.bdate_range("2024-01-02", periods=400)

    def fake_series(tickers, *a, **k):  # type: ignore[no-untyped-def]
        rows = {
            tk: pd.Series([100.0 + i * 0.05 for i in range(400)], index=dates, dtype=float)
            for tk in tickers
        }
        prov = {tk: ("cache", datetime.now(timezone.utc)) for tk in tickers}
        return SeriesResult(rows=rows, missing=[], provenance=prov)

    monkeypatch.setattr("app.cli.fetch_series", fake_series)
    with caplog.at_level(logging.WARNING):
        rc = main(["--csv", str(csv)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "SPLT" in caplog.text and "split" in caplog.text.lower()  # warned
    assert "excluded from TWR" in out and "SPLT" in out              # noted in the brief
    twr_line = next(line for line in out.splitlines() if "Time-weighted" in line)
    assert not twr_line.strip().endswith("n/a")  # TWR computed over the clean sub-book
    # The dollar P&L curve drops the excluded ticker too, so "Gains given back"
    # would silently omit a holding → it must be suppressed (not shown), like MWR.
    assert "Gains given back" not in out


def test_cash_pseudo_ticker_not_fetched_no_false_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    # A book with deposit/withdraw (CASH legs) must NOT ask the price fetcher for
    # "CASH" — that would land in series.missing and falsely flip status to partial.
    import pandas as pd

    csv = tmp_path / "book.csv"
    csv.write_text(
        "Date,Code,Action,Quantity,Price,Fee\n"
        "2024-01-02,CASH,deposit,0,5000,0\n"
        "2024-01-02,VOO,buy,10,100,0\n"
        "2024-06-03,CASH,withdraw,0,200,0\n",
        encoding="utf-8",
    )
    dates = pd.bdate_range("2024-01-02", periods=400)
    asked: list[str] = []

    def fake_series(tickers, *a, **k):  # type: ignore[no-untyped-def]
        asked.extend(tickers)
        rows = {
            tk: pd.Series([100.0 + i * 0.05 for i in range(400)], index=dates, dtype=float)
            for tk in tickers
        }
        prov = {tk: ("cache", datetime.now(timezone.utc)) for tk in tickers}
        return SeriesResult(rows=rows, missing=[], provenance=prov)

    monkeypatch.setattr("app.cli.fetch_series", fake_series)
    with caplog.at_level(logging.INFO):
        rc = main(["--csv", str(csv)])
    capsys.readouterr()
    assert rc == 0
    assert "CASH" not in asked              # pseudo-ticker excluded from the fetch set
    run = _run_summary(caplog)
    assert run["status"] == "ok"            # not falsely "partial"
    assert run["n_series_missing"] == 0


def test_split_handled_by_adjustment_keeps_ticker_in_twr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    # Slice 7 wiring: when split data IS available, the adjustment fixes the basis
    # so the ticker is NOT excluded from TWR (guard stays silent) — the real fix,
    # not the guard fallback. Mirrors NVDA once corporate actions are handled.
    import pandas as pd

    csv = tmp_path / "book.csv"
    csv.write_text(
        "Date,Code,Action,Quantity,Price,Fee\n"
        "2024-01-02,VOO,buy,10,100,0\n"
        "2024-01-02,SPLT,buy,1,1000,0\n"   # raw pre-split: 1 share @ $1000
        "2024-02-01,VOO,buy,5,100,0\n",
        encoding="utf-8",
    )
    dates = pd.bdate_range("2024-01-02", periods=400)

    def fake_series(tickers, *a, **k):  # type: ignore[no-untyped-def]
        rows = {
            tk: pd.Series([100.0 + i * 0.05 for i in range(400)], index=dates, dtype=float)
            for tk in tickers
        }
        prov = {tk: ("cache", datetime.now(timezone.utc)) for tk in tickers}
        return SeriesResult(rows=rows, missing=[], provenance=prov)

    monkeypatch.setattr("app.cli.fetch_series", fake_series)
    # SPLT did a 10:1 AFTER the buy → adjustment makes it 10 @ $100, matching the series.
    monkeypatch.setattr(
        "app.pipeline.fetch_splits", lambda *a, **k: {"SPLT": [(date(2024, 1, 15), 10.0)]}
    )
    with caplog.at_level(logging.INFO):
        rc = main(["--csv", str(csv)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "excluded from TWR" not in out          # adjustment handled it, not the guard
    assert "excluding SPLT" not in caplog.text
    assert "split-adjusted share counts for: SPLT" in caplog.text  # the adjustment actually ran
    splt_line = next(line for line in out.splitlines() if line.startswith("SPLT"))
    assert "10.000" in splt_line                   # 1 raw share → 10 split-adjusted (not still 1)


# ── --allocate (slice 9: the non-AI strategy engine) ────────────────────────


def _flat_series(tickers, *_a, **_k):  # type: ignore[no-untyped-def]
    import pandas as pd

    dates = pd.bdate_range("2024-01-01", periods=300)
    rows = {
        tk: pd.Series([100.0 + i * 0.1 for i in range(300)], index=dates, dtype=float)
        for tk in tickers
    }
    prov = {tk: ("cache", datetime.now(timezone.utc)) for tk in tickers}
    return SeriesResult(rows=rows, missing=[], provenance=prov)


def _split_vol_series(tickers, *_a, **_k):  # type: ignore[no-untyped-def]
    import pandas as pd

    dates = pd.bdate_range("2024-01-01", periods=300)
    rows = {}
    for tk in tickers:  # BND/IAU calm, the rest bumpy → different volatilities
        step = 0.05 if tk in ("BND", "IAU") else 8.0
        rows[tk] = pd.Series(
            [100.0 + (i % 2) * step for i in range(300)], index=dates, dtype=float
        )
    return SeriesResult(rows=rows, missing=[], provenance={})


def _weights_from(path: Path) -> dict[str, float]:
    rows = [ln.split(",") for ln in path.read_text().splitlines()[1:]]  # skip header
    return {tk: float(w) for tk, w in rows}


def test_allocate_equal_weight_previews_and_dumps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("app.cli.fetch_series", _flat_series)
    out_csv = tmp_path / "t.csv"
    with caplog.at_level(logging.INFO):
        rc = main(["--csv", str(SAMPLE), "--allocate", "equal_weight", "--allocate-out", str(out_csv)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "PROPOSED ALLOCATION: equal_weight" in out
    w = _weights_from(out_csv)
    assert len(w) == 4 and all(abs(v - 25.0) < 0.01 for v in w.values())  # sample = 4 holdings
    assert _run_summary(caplog)["allocate"] == "equal_weight"


def test_allocate_inverse_vol_overweights_calm_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("app.cli.fetch_series", _split_vol_series)
    out_csv = tmp_path / "t.csv"
    rc = main(["--csv", str(SAMPLE), "--allocate", "inverse_vol", "--allocate-out", str(out_csv)])
    capsys.readouterr()
    assert rc == 0
    w = _weights_from(out_csv)
    assert min(w["BND"], w["IAU"]) > max(w["VOO"], w["VEA"])  # lower vol → larger weight


def test_allocate_cap_limits_concentration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("app.cli.fetch_series", _split_vol_series)
    out_csv = tmp_path / "t.csv"
    rc = main(
        ["--csv", str(SAMPLE), "--allocate", "inverse_vol", "--allocate-cap", "0.4", "--allocate-out", str(out_csv)]
    )
    capsys.readouterr()
    assert rc == 0
    assert max(_weights_from(out_csv).values()) <= 40.0 + 0.01  # capped at 40%


def test_allocate_skips_without_prices(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    with caplog.at_level(logging.INFO):
        rc = main(["--csv", str(SAMPLE), "--allocate", "equal_weight", "--no-prices"])
    capsys.readouterr()
    assert rc == 0
    assert _run_summary(caplog)["allocate"] == "skipped: --no-prices"


def test_allocate_cap_out_of_range_rejected() -> None:
    # A cap is a fraction in (0, 1]; "30" (meaning 30%) or 0 must be rejected loudly,
    # not silently no-op.
    with pytest.raises(SystemExit):
        main(["--allocate", "inverse_vol", "--allocate-cap", "30"])
    with pytest.raises(SystemExit):
        main(["--allocate", "inverse_vol", "--allocate-cap", "0"])


def test_allocate_surfaces_omitted_holding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # A held ticker with no price history is absent from the target → on a to_total
    # rebalance it would be sold. The preview must surface that before the user acts.
    import pandas as pd

    dates = pd.bdate_range("2024-01-01", periods=300)

    def drop_voo(tickers, *_a, **_k):  # type: ignore[no-untyped-def]
        rows = {
            tk: pd.Series([100.0 + i * 0.1 for i in range(300)], index=dates, dtype=float)
            for tk in tickers if tk != "VOO"
        }
        missing = ["VOO"] if "VOO" in tickers else []
        return SeriesResult(rows=rows, missing=missing, provenance={})

    monkeypatch.setattr("app.cli.fetch_series", drop_voo)
    rc = main(["--csv", str(SAMPLE), "--allocate", "equal_weight", "--allocate-out", str(tmp_path / "t.csv")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "would SELL these: VOO" in out
    assert "VOO" not in _weights_from(tmp_path / "t.csv")  # correctly absent from the file


def test_allocate_refuses_combination_with_act_or_simulate() -> None:
    # propose != act/simulate: --allocate may not be combined with --rebalance/--backtest.
    with pytest.raises(SystemExit):
        main(["--allocate", "equal_weight", "--rebalance", "to_total", "--target", str(TARGET)])
    with pytest.raises(SystemExit):
        main(["--allocate", "equal_weight", "--backtest", "--target", str(TARGET)])


def test_allocate_out_requires_allocate() -> None:
    with pytest.raises(SystemExit):
        main(["--allocate-out", "/tmp/x.csv"])  # no --allocate → rejected


def test_allocate_inverse_vol_reuses_fetched_series(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # inverse_vol must REUSE the history already fetched for the brief — one fetch, not two.
    calls: list[list[str]] = []

    def counting(tickers, *_a, **_k):  # type: ignore[no-untyped-def]
        calls.append(sorted(tickers))
        return _flat_series(tickers)

    monkeypatch.setattr("app.cli.fetch_series", counting)
    rc = main(["--csv", str(SAMPLE), "--allocate", "inverse_vol"])
    capsys.readouterr()
    assert rc == 0
    assert len(calls) == 1, calls  # reused the first fetch, did not refetch


def test_allocate_edge_rule_refused_before_prices_check(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    # Ordering contract: an (hypothetical) edge rule is REFUSED before the prices
    # check or any series fetch — the run log must say "refused", never
    # "skipped: --no-prices", and no network work may happen first. The CLI
    # fast-fail mirrors the dispatcher's authoritative gate.
    import app.allocate as A

    monkeypatch.setitem(A._RULE_KIND, "equal_weight", "edge")  # make a real choice edge
    with caplog.at_level(logging.INFO):
        rc = main(["--csv", str(SAMPLE), "--no-prices", "--allocate", "equal_weight"])
    capsys.readouterr()
    assert rc == 0
    assert _run_summary(caplog)["allocate"] == "refused: equal_weight unvalidated edge"


def test_allocate_survives_unwritable_out_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    # A bad --allocate-out path must be REPORTED, not crash the run: the preview still
    # prints, the run is logged, exit 0 (a failed sink is non-fatal).
    monkeypatch.setattr("app.cli.fetch_series", _flat_series)
    a_file = tmp_path / "afile"
    a_file.write_text("x")  # a FILE where a parent directory is needed → mkdir fails
    with caplog.at_level(logging.INFO):
        rc = main(["--csv", str(SAMPLE), "--allocate", "equal_weight", "--allocate-out", str(a_file / "t.csv")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "PROPOSED ALLOCATION: equal_weight" in out   # preview still printed
    assert "could not write target" in caplog.text       # error reported, not raised
    assert _run_summary(caplog)["allocate"] == "equal_weight"  # run completed + logged


def test_allocate_excludes_cash_pseudo_ticker_from_the_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # A book with CASH legs (deposit + interest on idle cash) must NOT re-weight the
    # cash pseudo-ticker: the target is over real holdings only. Exclusion is enforced
    # at several layers (CASH is dropped from the series fetch, carries no price, and
    # holds 0 shares) plus an explicit guard in _compute_allocation — this pins the
    # user-visible contract so none of those can silently regress (e.g. if cash were
    # ever priced at $1, it must still stay out of the weights).
    book = tmp_path / "book.csv"
    book.write_text(
        "Date,Code,Action,Quantity,Price,Fee\n"
        "2024-01-02,CASH,deposit,0,10000,0\n"
        "2024-01-02,VOO,buy,10,100,0\n"
        "2024-01-02,BND,buy,10,80,0\n"
        "2024-01-02,IAU,buy,10,40,0\n"
        "2024-01-02,VEA,buy,10,50,0\n"
        "2024-03-01,CASH,interest,0,12,0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("app.cli.fetch_series", _flat_series)
    out_csv = tmp_path / "t.csv"
    rc = main(["--csv", str(book), "--allocate", "equal_weight", "--allocate-out", str(out_csv)])
    capsys.readouterr()
    assert rc == 0
    w = _weights_from(out_csv)
    assert set(w) == {"VOO", "BND", "IAU", "VEA"}      # the 4 real holdings only
    assert "CASH" not in w                              # the pseudo-ticker is not re-weighted
    assert all(abs(v - 25.0) < 0.01 for v in w.values())


def test_target_writers_refuse_to_overwrite_the_transactions_csv(tmp_path: Path) -> None:
    # Read-only invariant: --dump-target / --allocate-out must never clobber the input
    # transaction log. The guard fires during arg validation, before any fetch.
    book = tmp_path / "book.csv"
    book.write_text(
        "Date,Code,Action,Quantity,Price,Fee\n2024-01-02,VOO,buy,10,100,0\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        main(["--csv", str(book), "--dump-target", str(book)])
    with pytest.raises(SystemExit):
        main(["--csv", str(book), "--allocate", "equal_weight", "--allocate-out", str(book)])


def test_allocate_moderate_preset_keeps_held_funds_and_fills_gaps(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    # --allocate moderate: a strategic template that keeps the user's fund per role
    # (VOO/BND/VEA/IAU) and fills the gap roles with universe defaults (new tickers).
    import re

    now = datetime.now(timezone.utc)
    held = {"BND": 73.0, "IAU": 84.0, "VEA": 72.0, "VOO": 697.0}
    monkeypatch.setattr(
        "app.cli.fetch_latest",
        lambda tickers, *a, **k: PricesResult(
            rows={
                tk: PriceRow(tk, date.today(), held[tk], "test", now)
                for tk in tickers if tk in held
            },
            missing=[tk for tk in tickers if tk not in held],
        ),
    )
    with caplog.at_level(logging.INFO):
        rc = main(["--csv", str(SAMPLE), "--allocate", "moderate", "--no-risk"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "VOO" in out  # the held us-large fund anchors the equity bucket
    m = re.search(r"PROPOSED ALLOCATION: moderate \((\d+) holdings\)", out)
    assert m is not None and int(m.group(1)) > 4  # held funds + gap-fills, not just the 4 held
    assert _run_summary(caplog)["allocate"] == "moderate"


def _price_held_funds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Latest prices for the sample's held tickers — a preset anchors on the held fund
    per role, so it needs them priced."""
    now = datetime.now(timezone.utc)
    held = {"BND": 73.0, "IAU": 84.0, "VEA": 72.0, "VOO": 697.0}
    monkeypatch.setattr(
        "app.cli.fetch_latest",
        lambda tickers, *a, **k: PricesResult(
            rows={
                tk: PriceRow(tk, date.today(), held[tk], "test", now)
                for tk in tickers if tk in held
            },
            missing=[tk for tk in tickers if tk not in held],
        ),
    )


def test_allocate_preset_prints_benchmark_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # A preset is a posture → after writing it, print the ready-to-run validation command
    # (real path) so the generate → validate loop the design intends isn't left implicit.
    _price_held_funds(monkeypatch)
    out_csv = tmp_path / "moderate.csv"
    rc = main(["--csv", str(SAMPLE), "--allocate", "moderate", "--allocate-out", str(out_csv), "--no-risk"])
    out = capsys.readouterr().out
    assert rc == 0
    assert f"--backtest --benchmark 60-40 --target {out_csv}" in out


def test_allocate_benchmark_handoff_is_preset_and_output_gated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # The handoff is for strategic presets that wrote a file. A mechanical rule isn't a
    # posture (no --benchmark line), and a preset with no --allocate-out has no path to run.
    monkeypatch.setattr("app.cli.fetch_series", _flat_series)
    rc = main(["--csv", str(SAMPLE), "--allocate", "equal_weight", "--allocate-out", str(tmp_path / "t.csv")])
    assert rc == 0 and "--benchmark" not in capsys.readouterr().out  # mechanical → no handoff

    _price_held_funds(monkeypatch)
    rc = main(["--csv", str(SAMPLE), "--allocate", "moderate", "--no-risk"])  # no --allocate-out
    assert rc == 0 and "--benchmark 60-40 --target" not in capsys.readouterr().out


def test_resolve_role_tickers_prefers_a_held_fund_then_the_universe_default() -> None:
    from app.allocate import _resolve_role_tickers  # moved here from cli (shared by mcp too)
    from app.universe import Candidate

    universe = [
        Candidate("VOO", "Vanguard S&P 500", "us-large", ""),  # AUM-ordered: VOO is the default
        Candidate("IVV", "iShares Core S&P 500", "us-large", ""),
        Candidate("BND", "Vanguard Total Bond", "bond-aggregate", ""),
    ]
    rt = _resolve_role_tickers(universe, {"IVV": 1000.0})  # holds IVV (us-large), no bonds
    assert rt["us-large"] == "IVV"        # the held fund wins its role over the VOO default
    assert rt["bond-aggregate"] == "BND"  # a gap role → the universe default (top-AUM)


def _answer_with(monkeypatch: pytest.MonkeyPatch, answers: list[str]) -> None:
    """Feed the --onboard interactive prompts a scripted sequence of answers."""
    it = iter(answers)
    monkeypatch.setattr("builtins.input", lambda *_a: next(it))


def test_onboard_maps_cautious_answers_to_conservative_and_proposes(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    # The full step-0 path: 3 cautious answers → conservative posture → the SAME preset
    # allocation --allocate conservative would build (holdings-aware, propose-only).
    _price_held_funds(monkeypatch)
    _answer_with(monkeypatch, ["1", "1", "1"])  # under_3_years / sell / no
    with caplog.at_level(logging.INFO):
        rc = main(["--csv", str(SAMPLE), "--onboard", "--no-risk"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Matched posture: CONSERVATIVE" in out
    assert "PROPOSED ALLOCATION: conservative" in out
    assert _run_summary(caplog)["onboard"] == "conservative"


def test_onboard_accepts_option_keys_and_maps_to_aggressive(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    # Answers may be the option key, not just the number; growth-tolerant → aggressive.
    _price_held_funds(monkeypatch)
    _answer_with(monkeypatch, ["over_10_years", "buy_more", "comfortably"])
    with caplog.at_level(logging.INFO):
        rc = main(["--csv", str(SAMPLE), "--onboard", "--no-risk"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Matched posture: AGGRESSIVE" in out
    assert _run_summary(caplog)["onboard"] == "aggressive"


def test_onboard_reprompts_on_a_bad_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    _price_held_funds(monkeypatch)
    # "9" and "maybe" are invalid for the first question → re-prompt, then accept "2".
    _answer_with(monkeypatch, ["9", "maybe", "2", "2", "2"])
    rc = main(["--csv", str(SAMPLE), "--onboard", "--no-risk"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "enter 1-3" in out  # the re-prompt hint fired
    assert "Matched posture: MODERATE" in out


def test_onboard_writes_target_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    _price_held_funds(monkeypatch)
    _answer_with(monkeypatch, ["2", "2", "2"])
    out_csv = tmp_path / "starter.csv"
    rc = main(["--csv", str(SAMPLE), "--onboard", "--allocate-out", str(out_csv), "--no-risk"])
    capsys.readouterr()
    assert rc == 0
    w = _weights_from(out_csv)
    assert abs(sum(w.values()) - 100.0) < 0.1  # a normalized target was written


def test_onboard_aborts_cleanly_on_eof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Piped/closed stdin (no answers) must not loop — it errors out via parser.error.
    def _raise_eof(*_a: object) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", _raise_eof)
    with pytest.raises(SystemExit):
        main(["--csv", str(SAMPLE), "--onboard", "--no-risk"])


def test_onboard_rejects_explicit_allocate() -> None:
    # --onboard chooses the rule; passing --allocate too is contradictory.
    with pytest.raises(SystemExit):
        main(["--csv", str(SAMPLE), "--onboard", "--allocate", "moderate"])


def test_onboard_is_propose_only() -> None:
    with pytest.raises(SystemExit):
        main(["--csv", str(SAMPLE), "--onboard", "--rebalance", "to_total", "--target", str(SAMPLE)])


def test_onboard_rejected_with_screen_or_discover() -> None:
    # --onboard resolves to an --allocate proposal at runtime, so the screen/discover
    # propose-only guard must reject it too (parse-time), exactly like --screen --allocate.
    with pytest.raises(SystemExit):
        main(["--csv", str(SAMPLE), "--screen", "QQQM", "--onboard"])
    with pytest.raises(SystemExit):
        main(["--csv", str(SAMPLE), "--discover", "--onboard"])


def test_onboard_matches_the_equivalent_allocate_preset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    # The 'no second code path' guarantee on the CLI: onboarding with cautious answers must
    # write the SAME target file as `--allocate conservative` (not just the same header).
    _price_held_funds(monkeypatch)
    _answer_with(monkeypatch, ["1", "1", "1"])  # → conservative
    onb = tmp_path / "onb.csv"
    main(["--csv", str(SAMPLE), "--onboard", "--allocate-out", str(onb), "--no-risk"])
    capsys.readouterr()
    alloc = tmp_path / "alloc.csv"
    main(["--csv", str(SAMPLE), "--allocate", "conservative", "--allocate-out", str(alloc), "--no-risk"])
    capsys.readouterr()
    assert _weights_from(onb) == _weights_from(alloc)


def test_onboard_on_a_holdingless_book_builds_a_universe_starter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    # The core onboarding audience: a brand-new book with nothing held yet (deposits only).
    # A preset is a prior, so it must still produce a universe-default starter — not skip
    # with a misleading '--no-prices' message.
    book = tmp_path / "cash.csv"
    book.write_text("Date,Code,Action,Quantity,Price,Fee\n2024-01-02,CASH,deposit,0,1000,0\n", encoding="utf-8")
    _answer_with(monkeypatch, ["1", "1", "1"])
    with caplog.at_level(logging.INFO):
        rc = main(["--csv", str(book), "--onboard", "--offline"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "PROPOSED ALLOCATION: conservative" in out  # universe defaults, no holdings needed
    assert _run_summary(caplog)["allocate"] == "conservative"


def test_dry_run_previews_the_book_without_fetching(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    # --dry-run reads the file and reports; it must fetch nothing. Make any fetch explode
    # to PROVE no network path runs, then assert the preview content.
    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("--dry-run must not fetch prices")

    monkeypatch.setattr("app.cli.fetch_series", _boom)
    monkeypatch.setattr("app.cli.fetch_latest", _boom)
    with caplog.at_level(logging.INFO):
        rc = main(["--csv", str(SAMPLE), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "format: csv" in out
    assert "derives to 4 holding(s)" in out  # the sample's 4 held tickers
    assert "before split-adjustment" in out  # raw-shares honesty label
    assert "import would succeed" in out
    assert _run_summary(caplog)["dry_run"].startswith("ok: csv")


def test_dry_run_reports_a_row_level_error_and_exits_2(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    # Not just a missing-column (parse-level) error: a well-formed header with a bad row
    # (unknown action) raises inside _rows_to_events and must still funnel to rc 2 cleanly.
    bad = tmp_path / "badrow.csv"
    bad.write_text("Date,Code,Action,Quantity,Price,Fee\n2024-01-02,VOO,teleport,1,100,0\n", encoding="utf-8")
    with caplog.at_level(logging.INFO):
        rc = main(["--csv", str(bad), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "cannot import" in out
    assert _run_summary(caplog)["dry_run"] == "invalid"


def test_dry_run_reports_a_malformed_file_and_exits_2(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("Date,Code,Nope\n2024-01-01,VOO,x\n", encoding="utf-8")
    with caplog.at_level(logging.INFO):
        rc = main(["--csv", str(bad), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "cannot import" in out
    assert _run_summary(caplog)["dry_run"] == "invalid"


def test_dry_run_rejects_a_truncated_row_with_rc2(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    # D1 headline: a truncated row (missing trailing cells) used to crash --dry-run with a raw
    # AttributeError traceback + rc 1 (and crashed the real brief too). It must now be a clean
    # "cannot import" + rc 2.
    bad = tmp_path / "truncated.csv"
    bad.write_text(
        "Date,Code,DataSource,Currency,Price,Quantity,Action,Fee,Note\n"
        "2024-01-02,VOO,YAHOO,USD,400\n",  # only 5 of 9 columns
        encoding="utf-8",
    )
    with caplog.at_level(logging.INFO):
        rc = main(["--csv", str(bad), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "cannot import" in out
    assert _run_summary(caplog)["dry_run"] == "invalid"


def test_dry_run_surfaces_ghostfolio_skips(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    import json

    p = tmp_path / "export.json"
    p.write_text(json.dumps({"activities": [
        {"symbol": "VOO", "currency": "USD", "unitPrice": 400, "quantity": 10,
         "type": "BUY", "fee": 0, "dataSource": "YAHOO", "date": "2024-01-02T00:00:00Z"},
        {"symbol": "SAP.DE", "currency": "EUR", "unitPrice": 120, "quantity": 3,
         "type": "BUY", "fee": 0, "dataSource": "YAHOO", "date": "2024-01-03T00:00:00Z"},
    ]}), encoding="utf-8")
    rc = main(["--book", str(p), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "format: ghostfolio-json" in out
    assert "skipped 1 row(s)" in out and "EUR" in out


def test_benchmark_needs_backtest() -> None:
    # --benchmark compares --target against a reference; the contract requires --backtest.
    with pytest.raises(SystemExit):
        main(["--csv", str(SAMPLE), "--benchmark", "60-40"])


def test_backtest_benchmark_compares_target_vs_reference(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    # --backtest --target X --benchmark 60-40: a drawdown-first two-leg comparison + the
    # walk-forward held-out verdict, instead of rebalanced-vs-buy-and-hold.
    import math

    import pandas as pd

    target = tmp_path / "preset.csv"
    target.write_text("Ticker,Weight\nVOO,0.5\nGLD,0.5\n", encoding="utf-8")
    n = 320
    dates = pd.bdate_range("2022-01-03", periods=n)
    rows = {
        "VOO": pd.Series([300.0 + i for i in range(n)], index=dates, dtype=float),
        "GLD": pd.Series([170.0 + 10.0 * math.sin(i / 5.0) for i in range(n)], index=dates, dtype=float),
        "BND": pd.Series([72.0 + 2.0 * math.sin(i / 9.0) for i in range(n)], index=dates, dtype=float),
    }
    monkeypatch.setattr(
        "app.cli.fetch_series",
        lambda tickers, *a, **k: SeriesResult(
            rows={tk: rows[tk] for tk in tickers if tk in rows},
            missing=[tk for tk in tickers if tk not in rows],
        ),
    )
    with caplog.at_level(logging.INFO):
        rc = main(["--backtest", "--target", str(target), "--benchmark", "60-40"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "BENCHMARK (preset vs 60-40" in out      # the comparison panel, not BACKTEST
    assert "Walk-forward (held-out):" in out         # the held-out verdict row
    assert "vs 60-40:" in _run_summary(caplog)["backtest"]


# ── --warm (offline-cache onboarding) ────────────────────────────────────────

_REFS = {"VOO", "BND", "VTI", "TLT", "IEI", "GLD", "DBC", "BIL"}


def _record_warm_fetchers(monkeypatch: pytest.MonkeyPatch) -> dict[str, set[str]]:
    """Mock the pipeline fetch adapters `warm_cache` calls; record which tickers each is
    asked for (never the network). Overrides the autouse no-network fixture for this run."""
    from app.metadata import MetadataResult

    seen: dict[str, set[str]] = {}

    def series(tickers: object, start: object, end: object, **k: object) -> SeriesResult:
        seen["series"] = set(tickers)  # type: ignore[arg-type]
        return SeriesResult()

    def latest(tickers: object, **k: object) -> PricesResult:
        seen["latest"] = set(tickers)  # type: ignore[arg-type]
        return PricesResult()

    def splits(tickers: object, **k: object) -> dict[str, list[object]]:
        seen["splits"] = set(tickers)  # type: ignore[arg-type]
        return {}

    def meta(tickers: object, **k: object) -> "MetadataResult":
        seen["meta"] = set(tickers)  # type: ignore[arg-type]
        return MetadataResult()

    monkeypatch.setattr("app.pipeline.fetch_series", series)
    monkeypatch.setattr("app.pipeline.fetch_latest", latest)
    monkeypatch.setattr("app.pipeline.fetch_splits", splits)
    monkeypatch.setattr("app.pipeline.fetch_metadata", meta)
    return seen


def test_warm_core_fetches_book_and_refs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen = _record_warm_fetchers(monkeypatch)
    with caplog.at_level(logging.INFO):
        rc = main(["--book", str(SAMPLE), "--warm", "--cache-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert seen["series"] == {"VOO", "BND", "VEA", "IAU"} | _REFS  # book ∪ refs
    assert seen["latest"] == {"VOO", "BND", "VEA", "IAU"}          # book only, refs excluded
    assert "Warmed the offline cache (core)" in out
    assert _run_summary(caplog)["warm"].startswith("core:")


def test_warm_full_adds_the_universe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _record_warm_fetchers(monkeypatch)
    rc = main(["--book", str(SAMPLE), "--warm", "full", "--cache-dir", str(tmp_path)])
    assert rc == 0
    assert {"VOO", "BND", "VEA", "IAU"} | _REFS <= seen["series"]
    assert "QQQM" in seen["series"] and "SCHD" in seen["series"]  # real universe tickers landed
    assert len(seen["series"]) > 100


def test_warm_without_a_book_warms_refs_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    seen = _record_warm_fetchers(monkeypatch)
    rc = main(["--warm", "--cache-dir", str(tmp_path)])  # ASSET_BOOK pinned "" by the fixture
    capsys.readouterr()
    assert rc == 0
    assert seen["series"] == _REFS         # just the references
    assert set(seen) == {"series"}         # no book → no latest/splits/metadata fetch at all


def test_warm_rejects_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    _record_warm_fetchers(monkeypatch)
    with pytest.raises(SystemExit):  # parser.error → SystemExit(2)
        main(["--book", str(SAMPLE), "--warm", "--offline"])


def test_warm_rejects_a_malformed_book_with_rc2(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    # E1: --warm loads the book to collect tickers, so a malformed book must fail cleanly HERE
    # too (matching the brief / --dry-run) — not escape as a raw traceback + rc 1. A negative-qty
    # sell warmed fine (rc 0) before D1; now it's a clean "cannot import" + rc 2, before any fetch.
    bad = tmp_path / "bad.csv"
    bad.write_text(
        "Date,Code,DataSource,Currency,Price,Quantity,Action,Fee,Note\n"
        "2024-01-02,VOO,YAHOO,USD,400,-3,sell,0,\n",  # negative sell quantity
        encoding="utf-8",
    )
    with caplog.at_level(logging.INFO):
        rc = main(["--book", str(bad), "--warm", "--cache-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 2
    assert "cannot import" in out


def _record_warm_cache_dir(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Record the cache_dir the warm fetch was handed (never the network)."""
    from app.metadata import MetadataResult

    seen: dict[str, object] = {}

    def series(tickers: object, start: object, end: object, **k: object) -> SeriesResult:
        seen["cache_dir"] = k.get("cache_dir")
        return SeriesResult()

    monkeypatch.setattr("app.pipeline.fetch_series", series)
    monkeypatch.setattr("app.pipeline.fetch_latest", lambda *a, **k: PricesResult())
    monkeypatch.setattr("app.pipeline.fetch_splits", lambda *a, **k: {})
    monkeypatch.setattr("app.pipeline.fetch_metadata", lambda *a, **k: MetadataResult())
    return seen


def test_warm_nonzero_when_the_whole_book_fails_even_if_refs_succeed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # The refs are liquid and usually cache; the book is the point of warming. If every book
    # ticker fails to price (delisted / partial outage) while refs succeed, --warm must surface a
    # non-zero exit and an honest message — not a false "Warmed the offline cache".
    from app.metadata import MetadataResult

    book = {"VOO", "BND", "VEA", "IAU"}  # SAMPLE's holdings

    def series(tickers: object, start: object, end: object, **k: object) -> SeriesResult:
        miss = [t for t in tickers if t in book]  # the whole book comes back unpriced; refs land
        return SeriesResult(missing=miss)

    monkeypatch.setattr("app.pipeline.fetch_series", series)
    monkeypatch.setattr("app.pipeline.fetch_latest", lambda *a, **k: PricesResult())
    monkeypatch.setattr("app.pipeline.fetch_splits", lambda *a, **k: {})
    monkeypatch.setattr("app.pipeline.fetch_metadata", lambda *a, **k: MetadataResult())
    rc = main(["--book", str(SAMPLE), "--warm", "--cache-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 2                                   # not a false success
    assert "warm failed" in out.lower() and "could be priced" in out.lower()
    assert "Warmed the offline cache" not in out     # the success headline is suppressed


def test_cache_dir_falls_back_to_asset_cache_dir_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # No --cache-dir → honor ASSET_CACHE_DIR, so a CLI `--warm` and the MCP server (which reads
    # the same var) share ONE cache instead of silently warming different directories.
    monkeypatch.setenv("ASSET_CACHE_DIR", str(tmp_path))
    seen = _record_warm_cache_dir(monkeypatch)
    rc = main(["--book", str(SAMPLE), "--warm"])
    capsys.readouterr()
    assert rc == 0
    assert seen["cache_dir"] == tmp_path  # not the data/prices default


def test_explicit_cache_dir_wins_over_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    explicit = tmp_path / "explicit"
    monkeypatch.setenv("ASSET_CACHE_DIR", str(tmp_path / "env"))
    seen = _record_warm_cache_dir(monkeypatch)
    rc = main(["--book", str(SAMPLE), "--warm", "--cache-dir", str(explicit)])
    capsys.readouterr()
    assert rc == 0
    assert seen["cache_dir"] == explicit  # an explicit flag still beats the env var
