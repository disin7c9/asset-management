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
class SeriesResult:
    """Outcome of a `fetch_series` call. Never raises on per-ticker failure.

    `rows` maps ticker → a pandas Series of daily close prices indexed by a
    normalized (midnight) DatetimeIndex. `provenance` maps ticker → the real
    `(source, fetched_at)` of that series — `source ∈ {cache, yfinance, stooq}`
    and `fetched_at` is when the data was actually obtained (the cache's stored
    timestamp on a hit) — so the caller can stamp honest provenance instead of
    a placeholder.
    """

    rows: dict[str, "pd.Series[float]"] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    provenance: dict[str, tuple[str, datetime]] = field(default_factory=dict)

    @property
    def fallbacks_used(self) -> int:
        """Tickers served by the secondary (stooq) provider after yfinance failed.

        Mirrors `PricesResult.fallbacks_used`: counts the stooq wins on *this*
        run. A cache hit reports source "cache" (its original provider isn't
        re-derived), so historical fallbacks aren't double-counted."""
        return sum(1 for src, _ in self.provenance.values() if src == "stooq")


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
            # Offline only: the dedicated latest cache missed, so fall back to the
            # *series* cache tail — a risk-on run writes only <T>_series.parquet,
            # so the series can hold a price the latest cache lacks. Online we
            # prefer a fresh network fetch over a possibly stale-dated tail.
            from_series = _latest_from_series_cache(ticker, asof, cache) if cache else None
            if from_series is not None:
                rows[ticker] = from_series
            else:
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


def _coerce_fetched_at(value: object) -> datetime | None:
    """Normalize a cache cell's ``fetched_at`` to a tz-aware UTC datetime, or None
    if missing / NaT / unparseable (legacy or corrupt rows)."""
    ts = pd.to_datetime(value, errors="coerce")  # scalar in → Timestamp or NaT
    if pd.isna(ts):
        return None
    dt: datetime = ts.to_pydatetime()
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _fresh(fetched_at: datetime, ttl: timedelta, *, what: str) -> bool:
    """The one freshness gate for every on-disk cache: reject a future-stamped row
    (clock skew / tampered file) and anything older than ``ttl``. Centralized so the
    rule can't drift between the latest-price, series, and splits caches."""
    now = _now_utc()
    if fetched_at > now:
        log.warning(
            "cache for %s has fetched_at in the future (%s > %s); refusing",
            what, fetched_at, now,
        )
        return False
    return now - fetched_at <= ttl


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
    fetched_at = _coerce_fetched_at(row["fetched_at"])
    if fetched_at is None:
        return None  # NaT / corrupt — unusable, refetch
    if not _fresh(fetched_at, _CACHE_TTL, what=ticker):
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


def fetch_series(
    tickers: Iterable[str],
    start: date,
    end: date,
    *,
    cache_dir: Path | None = None,
    online: bool = True,
) -> SeriesResult:
    """Fetch the daily close *history* in [start, end] for each ticker.

    Provider order per ticker: cache → yfinance → stooq. The whole series is
    cached as `data/prices/<TICKER>_series.parquet` (columns: date, close,
    fetched_at) and reused if fresh within TTL and covering `end`. Never
    raises on a per-ticker miss — those appear in `missing`.
    """
    cache = cache_dir if cache_dir is not None else _CACHE_DIR_DEFAULT
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.warning("cache dir unwritable (%s) — proceeding without cache", exc)
        cache = None  # type: ignore[assignment]

    rows: dict[str, pd.Series[float]] = {}
    missing: list[str] = []
    provenance: dict[str, tuple[str, datetime]] = {}

    for ticker in dict.fromkeys(tickers):
        cached = _series_from_cache(ticker, start, end, cache) if cache else None
        if cached is not None:
            rows[ticker], cached_at = cached
            provenance[ticker] = ("cache", cached_at)
            continue
        if not online:
            missing.append(ticker)
            continue
        live = _series_from_yfinance(ticker, start, end)
        source = "yfinance"
        if live is None or live.empty:
            live = _series_from_stooq(ticker, start, end)
            source = "stooq"
        if live is None or live.empty:
            missing.append(ticker)
            continue
        if cache:
            _write_series_cache(ticker, live, cache)
        rows[ticker] = live
        provenance[ticker] = (source, _now_utc())

    log.info(
        "series fetched: returned=%d missing=%d range=%s..%s",
        len(rows), len(missing), start, end,
    )
    return SeriesResult(rows=rows, missing=missing, provenance=provenance)


def _normalize_close(df: pd.DataFrame) -> "pd.Series[float] | None":
    """Extract a clean daily-close Series (normalized DatetimeIndex) from a yf frame."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)
    if "Close" not in df.columns:
        return None
    close = df["Close"].dropna()
    if close.empty:
        return None
    close.index = pd.to_datetime(close.index).normalize()
    return close.astype(float)


def _series_from_yfinance(ticker: str, start: date, end: date) -> "pd.Series[float] | None":
    df = _fetch_yf(ticker, start, end + timedelta(days=1))
    if df is None or df.empty:
        return None
    return _normalize_close(df)


def _series_from_stooq(ticker: str, start: date, end: date) -> "pd.Series[float] | None":
    raw = _fetch_stooq_csv(ticker)
    if not raw:
        return None
    dates: list[pd.Timestamp] = []
    closes: list[float] = []
    for row in csv.DictReader(io.StringIO(raw)):
        d_str = row.get("Date") or row.get("date")
        close_str = row.get("Close") or row.get("close")
        if d_str is None or close_str is None:
            continue
        try:
            d = date.fromisoformat(d_str)
            if start <= d <= end:
                dates.append(pd.Timestamp(d))
                closes.append(float(close_str))
        except ValueError:
            continue
    if not dates:
        return None
    return pd.Series(closes, index=pd.DatetimeIndex(dates), dtype=float).sort_index()


def _read_fresh_series_cache(
    path: Path, ticker: str
) -> "tuple[pd.Series[float], datetime] | None":
    """Read a series-cache parquet → (sorted series, fetched_at), or None.

    Shared by `_series_from_cache` (price history) and `_latest_from_series_cache`
    (the latest-from-series fallback). Returns None if the file is absent,
    unreadable, malformed (missing date/close/fetched_at), future-stamped, or
    stale (older than the TTL). Does NOT apply any date-range filter — callers
    slice for what they need."""
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        log.warning("series cache read failed for %s: %s", ticker, exc)
        return None
    if df.empty or not {"date", "close", "fetched_at"} <= set(df.columns):
        return None
    fetched = _coerce_fetched_at(pd.to_datetime(df["fetched_at"]).max())
    if fetched is None:
        return None  # NaT (corrupt/empty) — unusable, refetch
    if not _fresh(fetched, _CACHE_TTL, what=ticker):
        return None  # stale or future-stamped → refetch
    idx = pd.to_datetime(df["date"]).dt.normalize()
    series = pd.Series(
        df["close"].astype(float).to_numpy(), index=pd.DatetimeIndex(idx)
    ).sort_index()
    return series, fetched


def _series_from_cache(
    ticker: str, start: date, end: date, cache_dir: Path
) -> "tuple[pd.Series[float], datetime] | None":
    """Return (series, fetched_at) from the cache, or None if absent/stale."""
    read = _read_fresh_series_cache(cache_dir / f"{ticker}_series.parquet", ticker)
    if read is None:
        return None
    series, fetched = read
    # The cached series must cover the requested end date to be usable as-is.
    if series.index.max() < pd.Timestamp(end - timedelta(days=4)):
        return None
    mask = (series.index >= pd.Timestamp(start)) & (series.index <= pd.Timestamp(end))
    return series[mask], fetched


def _latest_from_series_cache(
    ticker: str, asof: date, cache_dir: Path
) -> PriceRow | None:
    """Derive a latest-price row from the *series* cache tail (source="cache").

    Unifies the two on-disk caches so a risk-on run's series cache can serve a
    latest price the dedicated latest cache lacks — notably under `--no-risk
    --offline`. Returns the last close at/before `asof` from a fresh cached
    series (same TTL + future-stamp rules as the latest cache), else None."""
    read = _read_fresh_series_cache(cache_dir / f"{ticker}_series.parquet", ticker)
    if read is None:
        return None
    series, fetched = read
    series = series[series.index <= pd.Timestamp(asof)]
    if series.empty:
        return None
    return PriceRow(
        ticker=ticker,
        asof_date=series.index[-1].date(),
        close=float(series.iloc[-1]),
        source="cache",
        fetched_at=fetched,
    )


def _write_series_cache(ticker: str, series: "pd.Series[float]", cache_dir: Path) -> None:
    path = cache_dir / f"{ticker}_series.parquet"
    fetched = pd.Timestamp(_now_utc())
    out = pd.DataFrame(
        {
            "date": series.index,
            "close": series.to_numpy(),
            "fetched_at": fetched,
        }
    )
    try:
        out.to_parquet(path, index=False)
    except Exception as exc:  # noqa: BLE001
        log.warning("series cache write failed for %s: %s", ticker, exc)


# ── corporate actions: stock splits (slice 7) ──────────────────────────────

_SPLITS_TTL = timedelta(days=7)  # splits are stable facts; a weekly refetch catches new ones


def _fetch_yf_splits(ticker: str) -> "pd.Series[float] | None":
    """Split history for a ticker (index=date, value=ratio). None on failure.

    Thin wrapper around `yf.Ticker(...).splits`, monkey-patched in tests so no
    real HTTP happens.
    """
    try:
        s = yf.Ticker(ticker).splits
    except Exception as exc:  # noqa: BLE001 — yfinance raises many specific things
        log.warning("yfinance splits fetch failed for %s: %s", ticker, exc)
        return None
    return s


def _parse_splits(s: "pd.Series[float] | None") -> list[tuple[date, float]]:
    """yfinance splits Series → sorted [(effective_date, ratio)].

    Keeps real splits (ratio > 0 and ≠ 1.0); a 1:1 ratio is a no-op (and our
    no-split cache placeholder), so it's dropped → a clean empty list.
    """
    if s is None or len(s) == 0:
        return []
    rows: list[tuple[date, float]] = []
    for ts, ratio in s.items():
        if pd.isna(ts):
            continue  # NaT index: .date() returns NaT (no raise) → would crash sort/compare
        try:
            d = pd.Timestamp(ts).date()
            r = float(ratio)
        except (ValueError, TypeError):
            continue
        if r > 0 and r != 1.0:
            rows.append((d, r))
    return sorted(rows)


def _splits_from_cache(ticker: str, cache_dir: Path) -> list[tuple[date, float]] | None:
    """Cached split history, or None if absent/stale. A no-split ticker is cached
    as a harmless identity placeholder so we don't refetch it every run."""
    path = cache_dir / f"{ticker}_splits.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        log.warning("splits cache read failed for %s: %s", ticker, exc)
        return None
    if df.empty or not {"date", "ratio", "fetched_at"} <= set(df.columns):
        return None  # malformed (missing a column) → refetch, don't KeyError
    fetched = _coerce_fetched_at(pd.to_datetime(df["fetched_at"]).max())
    if fetched is None:
        return None
    if not _fresh(fetched, _SPLITS_TTL, what=ticker):
        return None  # stale or future-stamped → refetch
    return _parse_splits(pd.Series(df["ratio"].to_numpy(), index=pd.to_datetime(df["date"])))


def _write_splits_cache(
    ticker: str, rows: list[tuple[date, float]], cache_dir: Path
) -> None:
    fetched = pd.Timestamp(_now_utc())
    if rows:
        df = pd.DataFrame(
            {
                "date": [pd.Timestamp(d) for d, _ in rows],
                "ratio": [r for _, r in rows],
                "fetched_at": fetched,
            }
        )
    else:
        # No splits — store an identity placeholder (ratio 1.0, epoch date) so the
        # cache records "checked, none"; it's a no-op in cumulative_split_factor.
        df = pd.DataFrame(
            {"date": [pd.Timestamp("1970-01-01")], "ratio": [1.0], "fetched_at": fetched}
        )
    try:
        df.to_parquet(cache_dir / f"{ticker}_splits.parquet", index=False)
    except Exception as exc:  # noqa: BLE001
        log.warning("splits cache write failed for %s: %s", ticker, exc)


def fetch_splits(
    tickers: Iterable[str],
    *,
    cache_dir: Path | None = None,
    online: bool = True,
) -> dict[str, list[tuple[date, float]]]:
    """Fetch stock-split history per ticker → {ticker: [(effective_date, ratio)]}.

    Cache → yfinance, same as prices. Never raises: a ticker that fails (or is
    unknown offline) maps to `[]` (no adjustment), and the price-basis-mismatch
    guard remains the safety net for any split we couldn't fetch.
    """
    cache = cache_dir if cache_dir is not None else _CACHE_DIR_DEFAULT
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.warning("cache dir unwritable (%s) — proceeding without cache", exc)
        cache = None  # type: ignore[assignment]

    out: dict[str, list[tuple[date, float]]] = {}
    for ticker in dict.fromkeys(tickers):
        cached = _splits_from_cache(ticker, cache) if cache else None
        if cached is not None:
            out[ticker] = cached
            continue
        if not online:
            out[ticker] = []  # unknown offline → no adjustment (guard catches splits)
            continue
        rows = _parse_splits(_fetch_yf_splits(ticker))
        if cache:
            _write_splits_cache(ticker, rows, cache)
        out[ticker] = rows
    return out


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
