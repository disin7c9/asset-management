"""Multi-source price fetch with provenance and on-disk cache.

Provider chain (per ticker):
    1. on-disk Parquet cache (fresh-within-TTL hit returns source="cache")
    2. yfinance
    3. stooq (free CSV; no API key)

Every returned PriceRow records `source` and `fetched_at` so any displayed
number can be traced back to its origin. Cache rows clamp their age to ≥ 0
(future-stamped entries are refused, not silently treated as fresh).

`fetch_latest` never raises on a per-ticker miss: it returns a `PricesResult`
with `.rows` (successful) and `.missing` (failed). The caller can render the
report and continue.

Tests are offline by default: each test monkey-patches the two underlying
network wrappers `_fetch_yf` and `_fetch_stooq_csv` so no real HTTP happens.
"""

from __future__ import annotations

import csv
import io
import logging
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

_CACHE_DIR_DEFAULT = Path("data/prices")
_CACHE_TTL = timedelta(hours=20)  # roughly one business day


@dataclass(frozen=True)
class PriceRow:
    """A single price observation with provenance."""

    ticker: str
    asof_date: date
    close: float
    source: str            # "cache" | "yfinance" | "stooq"
    fetched_at: datetime   # tz-aware (UTC)

    @property
    def cache_age(self) -> timedelta:
        """Age of this observation right now (clamped to ≥ 0 against clock skew)."""
        delta = _now_utc() - self.fetched_at
        return delta if delta >= timedelta(0) else timedelta(0)


@dataclass(frozen=True)
class PricesResult:
    """Outcome of a `fetch_latest` call. Never raises on per-ticker failure."""

    rows: dict[str, PriceRow] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    @property
    def n_cache(self) -> int:
        return sum(1 for r in self.rows.values() if r.source == "cache")

    @property
    def n_yfinance(self) -> int:
        return sum(1 for r in self.rows.values() if r.source == "yfinance")

    @property
    def n_stooq(self) -> int:
        return sum(1 for r in self.rows.values() if r.source == "stooq")

    @property
    def fallbacks_used(self) -> int:
        """Tickers that needed the secondary (stooq) provider after yfinance failed."""
        return self.n_stooq


def fetch_latest(
    tickers: Iterable[str],
    asof_date: date | None = None,
    *,
    cache_dir: Path | None = None,
    online: bool = True,
) -> PricesResult:
    """Fetch the close at or before `asof_date` for each ticker (de-duplicated).

    Provider order: cache → yfinance → stooq. On live success the row is
    written back to the cache for next time. Never raises on a per-ticker
    failure — those tickers appear in the returned `missing` list.
    """
    asof = asof_date or date.today()
    cache = cache_dir if cache_dir is not None else _CACHE_DIR_DEFAULT
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.warning("cache dir unwritable (%s) — proceeding without cache", exc)
        cache = None  # type: ignore[assignment]

    rows: dict[str, PriceRow] = {}
    missing: list[str] = []

    for ticker in dict.fromkeys(tickers):  # dedup, preserve order
        cached = _from_cache(ticker, asof, cache) if cache else None
        if cached is not None:
            rows[ticker] = cached
            continue
        if not online:
            missing.append(ticker)
            continue
        live = _from_yfinance(ticker, asof) or _from_stooq(ticker, asof)
        if live is None:
            missing.append(ticker)
            continue
        if cache:
            _write_cache(live, cache)
        rows[ticker] = live

    result = PricesResult(rows=rows, missing=missing)
    log.info(
        "prices fetched: returned=%d cache=%d yfinance=%d stooq=%d missing=%d",
        len(result.rows), result.n_cache, result.n_yfinance,
        result.n_stooq, len(result.missing),
    )
    return result


# ── provider wrappers (monkey-patchable in tests) ─────────────────────────


def _fetch_yf(ticker: str, start: date, end: date) -> pd.DataFrame | None:
    """Thin wrapper around yfinance.download. Returns None on any failure."""
    try:
        df = yf.download(
            ticker,
            start=start.isoformat(),
            end=end.isoformat(),
            progress=False,
            auto_adjust=False,
        )
    except Exception as exc:  # noqa: BLE001 — yfinance raises many specific things
        log.warning("yfinance fetch failed for %s: %s", ticker, exc)
        return None
    if df is None or df.empty:
        return None
    return df


def _fetch_stooq_csv(ticker: str) -> str | None:
    """Fetch the stooq daily-history CSV for a US ticker. None on failure."""
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310 — fixed scheme
            data: bytes = resp.read()
    except Exception as exc:  # noqa: BLE001
        log.warning("stooq fetch failed for %s: %s", ticker, exc)
        return None
    return data.decode("utf-8")


# ── individual sources ────────────────────────────────────────────────────


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _from_cache(ticker: str, asof: date, cache_dir: Path) -> PriceRow | None:
    path = cache_dir / f"{ticker}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        log.warning("cache read failed for %s: %s", ticker, exc)
        return None
    match = df[df["asof_date"] == pd.Timestamp(asof)]
    if match.empty:
        return None
    row = match.sort_values("fetched_at").iloc[-1]
    fetched_at_ts = row["fetched_at"]
    fetched_at: datetime
    if isinstance(fetched_at_ts, pd.Timestamp):
        ts = fetched_at_ts.to_pydatetime()
        fetched_at = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    else:
        fetched_at = fetched_at_ts

    now = _now_utc()
    # Refuse future-stamped entries (clock skew or tampered file).
    if fetched_at > now:
        log.warning(
            "cache for %s has fetched_at in the future (%s > %s); refusing",
            ticker, fetched_at, now,
        )
        return None
    if now - fetched_at > _CACHE_TTL:
        return None
    return PriceRow(
        ticker=ticker,
        asof_date=asof,
        close=float(row["close"]),
        source="cache",
        fetched_at=fetched_at,
    )


def _from_yfinance(ticker: str, asof: date) -> PriceRow | None:
    end = asof + timedelta(days=1)
    start = asof - timedelta(days=10)
    df = _fetch_yf(ticker, start, end)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)
    close = df["Close"].dropna()
    if close.empty:
        return None
    last_idx = close.index[-1]
    last_date: date = last_idx.date() if hasattr(last_idx, "date") else asof
    return PriceRow(
        ticker=ticker,
        asof_date=last_date,
        close=float(close.iloc[-1]),
        source="yfinance",
        fetched_at=_now_utc(),
    )


def _from_stooq(ticker: str, asof: date) -> PriceRow | None:
    raw = _fetch_stooq_csv(ticker)
    if not raw:
        return None
    reader = csv.DictReader(io.StringIO(raw))
    candidates: list[tuple[date, float]] = []
    for row in reader:
        # Stooq header is documented as title-case ("Date,Open,...") but we
        # accept any case to be forgiving of upstream changes.
        d_str = row.get("Date") or row.get("date")
        close_str = row.get("Close") or row.get("close")
        if d_str is None or close_str is None:
            continue
        try:
            d = date.fromisoformat(d_str)
            if d <= asof:
                candidates.append((d, float(close_str)))
        except ValueError:
            continue
    if not candidates:
        return None
    candidates.sort()
    d, close = candidates[-1]
    return PriceRow(
        ticker=ticker,
        asof_date=d,
        close=close,
        source="stooq",
        fetched_at=_now_utc(),
    )


def _write_cache(row: PriceRow, cache_dir: Path) -> None:
    path = cache_dir / f"{row.ticker}.parquet"
    # The cache stores only what the reader needs (no redundant `source`).
    new_df = pd.DataFrame(
        [
            {
                "asof_date": pd.Timestamp(row.asof_date),
                "close": row.close,
                "fetched_at": pd.Timestamp(row.fetched_at),
            }
        ]
    )
    if path.exists():
        try:
            existing = pd.read_parquet(path)
            # Tolerate older files that may still carry the dropped column.
            if "source" in existing.columns:
                existing = existing.drop(columns=["source"])
            combined = pd.concat([existing, new_df], ignore_index=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("cache append failed for %s (rewriting): %s", row.ticker, exc)
            combined = new_df
    else:
        combined = new_df
    combined.to_parquet(path, index=False)
