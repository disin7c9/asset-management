"""CLI delivery routing tests (slice 4).

Offline: ``--no-prices`` avoids the network entirely; ``--send`` is exercised by
monkey-patching ``app.email._dispatch`` so no real email leaves the box.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from app import email as E
from app.cli import main
from app.prices import PriceRow, PricesResult, SeriesResult

TARGET = Path(__file__).resolve().parents[1] / "data" / "sample_data" / "target.csv"


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
    rc = main(["--no-risk", "--rebalance", "to_total", "--target", str(TARGET)])
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
    rc = main(["--no-risk", "--dump-target", str(out_csv)])
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
    rc = main(["--no-risk", "--rebalance", "to_total", "--target", str(TARGET)])
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
    from app.strategy import load_target, suggest

    now = datetime.now(timezone.utc)
    state = DerivedState()
    state.positions["VOO"] = Position("VOO", shares=1000.0, cost_basis=1.0)
    state.positions["DUST"] = Position("DUST", shares=1.0, cost_basis=1.0)
    prices = {
        "VOO": PriceRow("VOO", date.today(), 100.0, "t", now),   # $100,000
        "DUST": PriceRow("DUST", date.today(), 3.0, "t", now),   # $3 → 0.003%
    }
    out = tmp_path / "dump.csv"
    _dump_target(state, prices, out)

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
    rc = main(["--no-risk", "--rebalance", "to_total", "--target", str(d)])
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


def test_default_csv_with_real_intent_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # No --csv + a real-intent flag → warn that holdings are the bundled example.
    monkeypatch.setattr(
        "app.cli.fetch_series", lambda *a, **k: SeriesResult(rows={}, missing=[])
    )
    with caplog.at_level(logging.WARNING):
        main(["--no-prices", "--backtest", "--target", str(TARGET)])
    capsys.readouterr()
    assert "EXAMPLE portfolio" in caplog.text


def test_explicit_csv_suppresses_footgun_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    csv = tmp_path / "mine.csv"
    csv.write_text("Date,Code,Action,Quantity,Price,Fee\n2024-01-02,VOO,buy,1,100,0\n",
                   encoding="utf-8")
    monkeypatch.setattr(
        "app.cli.fetch_series", lambda *a, **k: SeriesResult(rows={}, missing=[])
    )
    with caplog.at_level(logging.WARNING):
        main(["--csv", str(csv), "--no-prices", "--backtest", "--target", str(TARGET)])
    capsys.readouterr()
    assert "EXAMPLE portfolio" not in caplog.text


def test_omitting_a_holding_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    _canned_sample_prices(monkeypatch)
    t = tmp_path / "omit.csv"
    t.write_text("Ticker,Weight\nVOO,45\nVEA,20\nBND,35\n", encoding="utf-8")  # IAU omitted
    with caplog.at_level(logging.WARNING):
        main(["--no-risk", "--rebalance", "to_total", "--target", str(t)])
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
        main(["--no-risk", "--rebalance", "to_total", "--target", str(t)])
    out = capsys.readouterr().out
    assert "omits held tickers" not in caplog.text  # explicit 0 → no omission warning
    assert "IAU" in out  # still shown as a full-exit sell
