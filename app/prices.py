"""Multi-source price fetch with provenance and on-disk cache.

Provider chain (per ticker):
    1. on-disk Parquet cache (fresh-within-TTL hit returns source="cache")
    2. yfinance (one spaced retry — Yahoo throttles bursts with empty frames)
    3. Tiingo (free API key via TIINGO_API_KEY; skipped, logged once, without one)

**Both live providers must return the same price basis** — split-adjusted, dividend-
UNadjusted closes — because the cache stores whichever won and the rest of the codebase
(`corporate_actions.adjust_for_splits`) re-expresses share counts to match it. yfinance
serves that basis directly; Tiingo does not, so `_parse_tiingo_rows` rebuilds it from the
`splitFactor` Tiingo ships per row. See that function — the basis is the subtle part.

Every returned PriceRow records `source` and `fetched_at` so any displayed
number can be traced back to its origin. Cache rows clamp their age to ≥ 0
(future-stamped entries are refused, not silently treated as fresh).

`fetch_latest` never raises on a per-ticker miss: it returns a `PricesResult`
with `.rows` (successful) and `.missing` (failed). The caller can render the
report and continue.

Tests are offline by default: each test monkey-patches the two underlying
network wrappers `_fetch_yf` and `_fetch_tiingo_json` so no real HTTP happens.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import pandas as pd
import yfinance as yf

from app.http_safe import no_redirect_opener

log = logging.getLogger(__name__)

_CACHE_DIR_DEFAULT = Path("data/prices")
_CACHE_TTL = timedelta(hours=20)  # roughly one business day
# Don't pass off a close older than this as a "current" price — beyond it, report the holding
# unpriced rather than quote a stale close. The brief is weekly, so ~10 days covers a normal gap
# (plus a little slack); older than that, calling the price "current" would mislead — a
# delisted/halted ticker's last-ever close would otherwise be served as today's value, forever.
# Public: enforced on the offline latest path here AND on the risk-on series-tail path in
# `pipeline.compute_prices_returns_risk` (which imports it), so the two can't disagree.
STALE_PRICE_FLOOR = timedelta(days=10)
# The longest NORMAL no-trading gap (weekend + a holiday cluster): a close within this many
# days of the target date is current-enough to raise no eyebrows. Public and single-sourced
# because two rules share it and must stay in lockstep: the series-cache end-coverage check
# below (a cached series "covers" `end` if its last row is within the grace) and the report's
# stale-close display (`report._stale_close_lag` starts naming close dates beyond it).
STALE_CLOSE_GRACE = timedelta(days=4)

# WHICH close a series carries. The two are not interchangeable and the choice is forced by
# whether the caller has a transaction log:
#   "raw"          — split-adjusted only (yfinance auto_adjust=False / Tiingo `close` rebuilt
#                    from splitFactor). The PORTFOLIO path: dividends already arrive as rows in
#                    the user's log, so a dividend-adjusted close would count every one twice.
#   "total_return" — split- AND dividend-adjusted (yfinance auto_adjust=True / Tiingo
#                    `adjClose`). The NOTIONAL-SIMULATION path (`backtest.simulate` and every
#                    verdict built on it): it holds funds without a log, so there is nowhere
#                    else for income to come from. On a raw basis a bond sleeve shows the
#                    coupon-stripped price drift as pure loss — BND ≈ -1.5%/yr over the cached
#                    decade against ~+4% real, and BIL (whose return is ENTIRELY coupon) ≈ 0.
PriceBasis = Literal["raw", "total_return"]


@dataclass(frozen=True)
class PriceRow:
    """A single price observation with provenance."""

    ticker: str
    asof_date: date
    close: float
    source: str            # "cache" | "yfinance" | "tiingo"
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
    `(source, fetched_at)` of that series — `source ∈ {cache, yfinance, tiingo}`
    and `fetched_at` is when the data was actually obtained (the cache's stored
    timestamp on a hit) — so the caller can stamp honest provenance instead of
    a placeholder.
    """

    rows: dict[str, "pd.Series[float]"] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    provenance: dict[str, tuple[str, datetime]] = field(default_factory=dict)

    @property
    def fallbacks_used(self) -> int:
        """Tickers served by the secondary (Tiingo) provider after yfinance failed.

        Mirrors `PricesResult.fallbacks_used`: counts the Tiingo wins on *this*
        run. A cache hit reports source "cache" (its original provider isn't
        re-derived), so historical fallbacks aren't double-counted."""
        return sum(1 for src, _ in self.provenance.values() if src == "tiingo")


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
    def n_tiingo(self) -> int:
        return sum(1 for r in self.rows.values() if r.source == "tiingo")

    @property
    def fallbacks_used(self) -> int:
        """Tickers that needed the secondary (Tiingo) provider after yfinance failed."""
        return self.n_tiingo


def fetch_latest(
    tickers: Iterable[str],
    asof_date: date | None = None,
    *,
    cache_dir: Path | None = None,
    online: bool = True,
) -> PricesResult:
    """Fetch the close at or before `asof_date` for each ticker (de-duplicated).

    Provider order: cache → yfinance → Tiingo. On live success the row is
    written back to the cache for next time. Never raises on a per-ticker
    failure — those tickers appear in the returned `missing` list.
    """
    asof = asof_date or date.today()
    cache = ensure_cache_dir(cache_dir)

    rows: dict[str, PriceRow] = {}
    missing: list[str] = []
    circuit = _RetryCircuit()  # batch-scoped yfinance retry breaker

    for ticker in dict.fromkeys(tickers):  # dedup, preserve order
        cached = _from_cache(ticker, asof, cache, allow_stale=not online) if cache else None
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
        live = _from_yfinance(ticker, asof, circuit=circuit) or _from_tiingo(ticker, asof)
        if live is None:
            missing.append(ticker)
            continue
        if cache:
            _write_cache(live, cache)
        rows[ticker] = live

    result = PricesResult(rows=rows, missing=missing)
    log.info(
        "prices fetched: returned=%d cache=%d yfinance=%d tiingo=%d missing=%d",
        len(result.rows), result.n_cache, result.n_yfinance,
        result.n_tiingo, len(result.missing),
    )
    return result


# ── provider wrappers (monkey-patchable in tests) ─────────────────────────

_YF_RETRY_WAIT = 1.5   # seconds between the two yfinance attempts
_YF_BLOCKED_AFTER = 3  # consecutive failures in one batch → assume a block, stop retrying


class _RetryCircuit:
    """Per-batch yfinance retry state — created fresh by each `fetch_latest` / `fetch_series`.

    The spaced retry clears a transient single-ticker throttle, but callers loop over tickers
    one at a time, so under a *wholesale* Yahoo block retrying every ticker just doubles the
    request volume against a volume-throttling host and burns `_YF_RETRY_WAIT × N` of dead sleep
    (~9 min on a 375-ticker warm). After `_YF_BLOCKED_AFTER` consecutive failures the circuit
    opens and the rest of the batch skips the retry. It is deliberately **batch-scoped**, not a
    module global: a fresh batch (a later, unrelated MCP tool call) re-assesses from zero, and two
    concurrent batches can't corrupt each other's count."""

    def __init__(self) -> None:
        self.consecutive_failures = 0

    @property
    def armed(self) -> bool:
        return self.consecutive_failures < _YF_BLOCKED_AFTER

    def record(self, *, ok: bool) -> None:
        self.consecutive_failures = 0 if ok else self.consecutive_failures + 1
        if not ok and self.consecutive_failures == _YF_BLOCKED_AFTER:
            log.info(
                "yfinance failed %d times in a row this batch — treating Yahoo as blocked "
                "and skipping the retry for the rest of this batch",
                _YF_BLOCKED_AFTER,
            )


def _fetch_yf(
    ticker: str, start: date, end: date, *, adjusted: bool = False
) -> pd.DataFrame | None:
    """One raw yfinance.download attempt — the mockable network seam. None on failure/empty.

    ``adjusted`` picks the BASIS. Default False = split-adjusted only, which is what the
    portfolio path needs (dividends already arrive as rows in the transaction log, so an
    adjusted close would count them twice). True = split- AND dividend-adjusted, i.e. total
    return — what a *notional* simulation needs, since it has no transaction log to draw
    income from. See `PriceBasis`.
    """
    try:
        df = yf.download(
            ticker,
            start=start.isoformat(),
            end=end.isoformat(),
            progress=False,
            auto_adjust=adjusted,
        )
    except Exception as exc:  # noqa: BLE001 — yfinance raises many specific things
        log.warning("yfinance fetch failed for %s: %s", ticker, exc)
        return None
    return df if df is not None and not df.empty else None


def _fetch_yf_retrying(
    ticker: str, start: date, end: date, *,
    circuit: _RetryCircuit | None = None, adjusted: bool = False,
) -> pd.DataFrame | None:
    """`_fetch_yf` with one spaced retry, gated by a batch `circuit`.

    Yahoo throttles bursts (a cache warm or a multi-tool chat turn fires dozens of requests),
    and a throttled call comes back empty rather than raising, so a brief pause clears the
    transient case. The `circuit` (owned by the batch loop) opens after a run of failures so a
    wholesale block doesn't retry every ticker; with no circuit (a lone call) the retry is
    always armed. The raw one-shot `_fetch_yf` stays the seam tests mock, so this policy layer
    is transparent to them.
    """
    armed = circuit is None or circuit.armed
    attempts = 2 if armed else 1
    for attempt in range(1, attempts + 1):
        df = _fetch_yf(ticker, start, end, adjusted=adjusted)
        if df is not None and not df.empty:
            if circuit is not None:
                circuit.record(ok=True)
            return df
        if attempt < attempts:
            time.sleep(_YF_RETRY_WAIT)
    if circuit is not None:
        circuit.record(ok=False)
    return None


_warned_no_tiingo_key = False


def _warn_no_tiingo_key() -> None:
    """One INFO per process, the first time the fallback is skipped for lack of a key."""
    global _warned_no_tiingo_key
    if _warned_no_tiingo_key:
        return
    _warned_no_tiingo_key = True
    log.info(
        "yfinance failed and TIINGO_API_KEY is not set — skipping the Tiingo fallback "
        "(a free key from tiingo.com enables a second price source)"
    )


def _env_secret(var: str) -> str:
    """A secret from the environment, or "" when it isn't really set.

    An MCPB host substitutes an optional `user_config` field the user left blank with
    the LITERAL `"${user_config.x}"` template text. That must count as unset — never be
    posted as a credential — else a blank key field sends a doomed authenticated request
    per ticker instead of skipping cleanly. `mcp_server._env_raw` applies the same rule
    to the path vars; this is its counterpart for secrets (Layer 2 imports no app module).
    """
    raw = os.environ.get(var, "").strip()
    return "" if raw.startswith("${") else raw


# Tiingo's API does not redirect, so refusing one is free — and `urllib` would otherwise
# re-send the Authorization header to wherever the 3xx pointed. See `http_safe.NoRedirect`.
_TIINGO_OPENER = no_redirect_opener()


def _fetch_tiingo_json(
    ticker: str, start: date, end: date
) -> list[dict[str, object]] | None:
    """Fetch Tiingo daily EOD rows for [start, end]. None on failure or no key.

    The secondary provider behind yfinance. Auth is a free API key read from
    TIINGO_API_KEY (loaded from .env by cli/mcp_server; the Desktop addon passes
    it from its settings) — sent as a header so it can never leak into a logged
    URL. Without a key the fallback is skipped and that is logged once.
    """
    key = _env_secret("TIINGO_API_KEY")
    if not key:
        _warn_no_tiingo_key()
        return None
    # Quote the ticker: it comes from the user's book / --screen, where it is only
    # upper-cased, so a stray '?', '#' or '/' would otherwise rewrite the request.
    url = (
        f"https://api.tiingo.com/tiingo/daily/{quote(ticker.lower(), safe='')}/prices"
        f"?startDate={start.isoformat()}&endDate={end.isoformat()}"
    )
    req = urllib.request.Request(url, headers={"Authorization": f"Token {key}"})  # noqa: S310 — fixed https scheme
    try:
        with _TIINGO_OPENER.open(req, timeout=10) as resp:
            rows = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.warning("tiingo fetch failed for %s: %s", ticker, exc)
        return None
    return rows if isinstance(rows, list) else None


# ── individual sources ────────────────────────────────────────────────────


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def ensure_cache_dir(cache_dir: Path | None) -> Path | None:
    """Resolve and create the cache dir; None means "run without cache".

    The one mkdir-or-degrade gate for every on-disk cache (latest / series /
    splits / metadata — public: `metadata.py` imports it), so the unwritable-dir
    fallback can't drift between them.
    """
    cache = cache_dir if cache_dir is not None else _CACHE_DIR_DEFAULT
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.warning("cache dir unwritable (%s) — proceeding without cache", exc)
        return None
    return cache


def _coerce_fetched_at(value: object) -> datetime | None:
    """Normalize a cache cell's ``fetched_at`` to a tz-aware UTC datetime, or None
    if missing / NaT / unparseable (legacy or corrupt rows)."""
    ts = pd.to_datetime(value, errors="coerce")  # scalar in → Timestamp or NaT
    if pd.isna(ts):
        return None
    dt: datetime = ts.to_pydatetime()
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def fresh(
    fetched_at: datetime, ttl: timedelta, *, what: str, allow_stale: bool = False
) -> bool:
    """The one freshness gate for every on-disk cache: reject a future-stamped row
    (clock skew / tampered file) and anything older than ``ttl``. Centralized so the
    rule can't drift between the latest-price, series, splits, and metadata caches
    (public: `metadata.py` imports it). ``allow_stale`` (set by offline reads) keeps the
    future-stamp rejection but waives the age limit: the TTL exists to force a re-fetch,
    and offline there is nothing to fetch — serving the newest cached value (honestly
    dated) beats returning nothing."""
    now = _now_utc()
    if fetched_at > now:
        log.warning(
            "cache for %s has fetched_at in the future (%s > %s); refusing",
            what, fetched_at, now,
        )
        return False
    return allow_stale or now - fetched_at <= ttl


def usable_price(value: float) -> bool:
    """Whether a close is usable as a price: finite AND strictly positive.

    The ONE price-validity rule, shared by every ingest point (Tiingo, yfinance, and both
    cache reads) and by the market-value / rebalance sinks — so no consumer has to re-derive
    it and the checks can't drift apart (they did: a sink on `> 0` and one on `isfinite` once
    disagreed about the same ticker). `> 0` alone is wrong (`inf > 0` is True); `isfinite`
    alone is wrong (0.0 and negatives aren't prices); NaN fails both. A 0 / negative / ±inf
    close is a bad feed row or a corrupt cache cell, never a tradeable price.
    """
    return math.isfinite(value) and value > 0.0


def _usable_closes(series: "pd.Series[float]") -> "pd.Series[float]":
    """`series` keeping only usable prices — the Series form of `usable_price`, so the
    history paths (yfinance + cache) drop bad closes by the exact same rule as the scalar
    paths. A dropped day makes its neighbours span a 2-day return, far better than the
    -100%/±inf a 0 or non-finite close fabricates into the drawdown-first figures."""
    return series[series.map(usable_price)]


def _from_cache(
    ticker: str, asof: date, cache_dir: Path, *, allow_stale: bool = False
) -> PriceRow | None:
    """The newest cached close at or before `asof`, or None.

    Reads match on `asof_date <= asof`, NOT on equality. The writer stores the provider's
    ACTUAL last close date, which is only equal to the requested date when `asof` is itself a
    completed trading day. The README's flagship workflow is an 08:00 Monday cron — before the
    US open — so yfinance returns Friday's bar and an equality match missed on every single
    run, forever: two identical requests made two network calls and appended two duplicate
    rows, while the 20h TTL never got a chance to apply. Weekends and holidays were the same.
    """
    path = cache_dir / f"{ticker}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        log.warning("cache read failed for %s: %s", ticker, exc)
        return None
    match = df[df["asof_date"] <= pd.Timestamp(asof)]
    if match.empty:
        return None
    # Newest close date first, then newest fetch of that date.
    row = match.sort_values(["asof_date", "fetched_at"]).iloc[-1]
    close_date = pd.Timestamp(row["asof_date"]).date()
    if asof - close_date > STALE_PRICE_FLOOR:
        return None  # too old to pass off as a current price (a delisted/halted tail)
    fetched_at = _coerce_fetched_at(row["fetched_at"])
    if fetched_at is None:
        return None  # NaT / corrupt — unusable, refetch
    if not fresh(fetched_at, _CACHE_TTL, what=ticker, allow_stale=allow_stale):
        return None
    close = float(row["close"])
    if not usable_price(close):
        return None  # 0 / negative / non-finite cell (corrupt or legacy cache) — refetch
    return PriceRow(
        ticker=ticker,
        # The close's OWN date, not the requested one — the report's stale-close display
        # reads this, so claiming `asof` would hide a Friday price behind a Monday label.
        asof_date=close_date,
        close=close,
        source="cache",
        fetched_at=fetched_at,
    )


def _from_yfinance(
    ticker: str, asof: date, *, circuit: _RetryCircuit | None = None
) -> PriceRow | None:
    end = asof + timedelta(days=1)
    start = asof - timedelta(days=10)
    df = _fetch_yf_retrying(ticker, start, end, circuit=circuit)
    if df is None or df.empty:
        return None
    close = _normalize_close(df)  # shared extractor: one MultiIndex/"Close"/empty guard
    if close is None:
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


def _tiingo_float(value: object) -> float | None:
    """A finite float from a Tiingo JSON scalar, else None.

    Rejects `bool` (a subclass of `int` — `float(True)` would be a silent $1.00 price)
    and non-finite values: `json.loads` accepts the bare `NaN`/`Infinity` tokens, and
    `float("inf")`/`float("nan")` return rather than raise, so neither is caught by the
    except clause. `OverflowError` (a huge JSON int) is an ArithmeticError, not a
    ValueError, so it too must be named or it escapes the per-ticker loop and aborts the
    batch. The yfinance path drops these via `_normalize_close`'s `dropna`; this mirrors it.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        out = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _tiingo_date(value: object) -> date | None:
    """The day out of a Tiingo timestamp ("2024-01-05T00:00:00.000Z"), else None."""
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _parse_tiingo_rows(
    rows: list[dict[str, object]], *, total_return: bool = False
) -> list[tuple[date, float]]:
    """Tiingo JSON rows → ascending [(date, split-adjusted close)]; bad rows skipped.

    With ``total_return=True`` the split reconstruction below is skipped entirely and
    `adjClose` is served as-is: Tiingo already ships it split- AND dividend-adjusted, which
    IS the total-return basis (see `PriceBasis`). That makes the third basis mentioned below
    the *right* one for the notional-simulation path, and a mismatch only for the portfolio
    path, which never asks for it.

    **The basis.** Tiingo's `close` is the RAW as-traded price, while yfinance's `Close`
    (`auto_adjust=False`) is **split-adjusted** — the basis this codebase assumes
    everywhere (`corporate_actions.py`: "price history from yfinance is split-adjusted",
    and `adjust_for_splits` re-expresses share counts to match). Serving raw closes would
    fabricate a vast one-day return across any split — NVDA's 10:1 turns $1208 → $121
    overnight, a phantom −90% day straight into the drawdown and Ulcer figures.

    So we rebuild yfinance's basis from the `splitFactor` Tiingo ships on every row:
    divide each close by the product of the split factors of all *later* rows — the same
    "splits effective strictly after this date" rule as
    `corporate_actions.cumulative_split_factor`. (`adjClose` is split- AND dividend-
    adjusted — a third basis, ≈0.17% off yfinance here — so it is not a drop-in.)

    Forgiving by contract: a row that isn't a dict, or whose date/close is missing,
    malformed, boolean, or non-finite, is skipped rather than raising — `fetch_latest`
    and `fetch_series` promise never to raise on a per-ticker miss.
    """
    key = "adjClose" if total_return else "close"
    parsed: list[tuple[date, float, float]] = []  # (date, close, split factor)
    for row in rows:
        if not isinstance(row, dict):
            continue  # a JSON array of non-dicts (an error page, a format shift)
        day, close = _tiingo_date(row.get("date")), _tiingo_float(row.get(key))
        if day is None or close is None or not usable_price(close):
            continue  # a 0 / negative close is a halted / bad-feed row — skip it
        factor = _tiingo_float(row.get("splitFactor"))
        parsed.append((day, close, factor if factor and factor > 0.0 else 1.0))

    if total_return:
        parsed.sort()
        return [(d, c) for d, c, _ in parsed]  # adjClose already carries every adjustment

    parsed.sort()
    out: list[tuple[date, float]] = []
    later_splits = 1.0  # product of the split factors effective AFTER the row in hand
    for day, close, factor in reversed(parsed):
        out.append((day, close / later_splits))
        later_splits *= factor
    out.reverse()
    return out


def _tiingo_closes(
    ticker: str, start: date, end: date, *, total_return: bool = False
) -> list[tuple[date, float]] | None:
    """Split-adjusted (date, close) pairs within [start, end], ascending. None if empty.

    Fetches through **today** whenever `end` is older, then slices: yfinance's adjusted
    Close reflects every split up to now, so a split that happened *after* `end` still
    rescales the whole requested window. Asking only for [start, end] would miss it and
    leave the two providers on different bases again — the exact bug this guards.

    Every current caller passes `end == today` (fetch_latest defaults asof to today;
    fetch_series is always called with `today`), so the reach-forward is a no-op in
    practice and fetches nothing extra. If a dated-history feature ever passes an `end`
    well in the past, prefer deriving the post-`end` split factor from the splits cache
    (`fetch_splits`) over downloading the intervening prices only to discard them.
    """
    raw = _fetch_tiingo_json(ticker, start, max(end, date.today()))
    if not raw:
        return None
    parsed = _parse_tiingo_rows(raw, total_return=total_return)
    pairs = [(d, c) for d, c in parsed if start <= d <= end]
    return pairs or None


def _from_tiingo(ticker: str, asof: date) -> PriceRow | None:
    # Same 10-day lookback as `_from_yfinance`: a close older than that is not a
    # "current" price anywhere else here either (cf. `STALE_PRICE_FLOOR`).
    pairs = _tiingo_closes(ticker, asof - timedelta(days=10), asof)
    if not pairs:
        return None
    day, close = pairs[-1]  # ascending, already bounded at asof
    return PriceRow(
        ticker=ticker,
        asof_date=day,
        close=close,
        source="tiingo",
        fetched_at=_now_utc(),
    )


def fetch_series(
    tickers: Iterable[str],
    start: date,
    end: date,
    *,
    cache_dir: Path | None = None,
    online: bool = True,
    basis: PriceBasis = "raw",
) -> SeriesResult:
    """Fetch the daily close *history* in [start, end] for each ticker.

    Provider order per ticker: cache → yfinance → Tiingo. The whole series is
    cached as `data/prices/<TICKER>_series.parquet` (columns: date, close,
    fetched_at) and reused if fresh within TTL and covering `end`. Never
    raises on a per-ticker miss — those appear in `missing`.

    ``basis`` picks WHICH close (see `PriceBasis`), and each basis gets its own cache
    file, so warming one never serves the other. `"total_return"` lands in
    `<TICKER>_series_tr.parquet`.
    """
    cache = ensure_cache_dir(cache_dir)
    total_return = basis == "total_return"

    rows: dict[str, pd.Series[float]] = {}
    missing: list[str] = []
    provenance: dict[str, tuple[str, datetime]] = {}
    circuit = _RetryCircuit()  # batch-scoped yfinance retry breaker

    for ticker in dict.fromkeys(tickers):
        cached = (
            _series_from_cache(
                ticker, start, end, cache, allow_stale=not online, total_return=total_return
            )
            if cache
            else None
        )
        if cached is not None:
            rows[ticker], cached_at = cached
            provenance[ticker] = ("cache", cached_at)
            continue
        if not online:
            missing.append(ticker)
            continue
        live = _series_from_yfinance(ticker, start, end, circuit=circuit, total_return=total_return)
        source = "yfinance"
        if live is None or live.empty:
            live = _series_from_tiingo(ticker, start, end, total_return=total_return)
            source = "tiingo"
        if live is None or live.empty:
            missing.append(ticker)
            continue
        if cache:
            _write_series_cache(ticker, live, cache, total_return=total_return)
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
    close = _usable_closes(close.astype(float))  # dropna leaves ±inf and 0/negative; drop them too
    return close if not close.empty else None


def _series_from_yfinance(
    ticker: str, start: date, end: date, *,
    circuit: _RetryCircuit | None = None, total_return: bool = False,
) -> "pd.Series[float] | None":
    df = _fetch_yf_retrying(
        ticker, start, end + timedelta(days=1), circuit=circuit, adjusted=total_return
    )
    if df is None or df.empty:
        return None
    return _normalize_close(df)


def _series_from_tiingo(
    ticker: str, start: date, end: date, *, total_return: bool = False
) -> "pd.Series[float] | None":
    pairs = _tiingo_closes(ticker, start, end, total_return=total_return)
    if not pairs:
        return None
    return pd.Series(
        [c for _, c in pairs],
        index=pd.DatetimeIndex([pd.Timestamp(d) for d, _ in pairs]),
        dtype=float,
    )


def _read_fresh_series_cache(
    path: Path, ticker: str, *, allow_stale: bool = False
) -> "tuple[pd.Series[float], datetime] | None":
    """Read a series-cache parquet → (sorted series, fetched_at), or None.

    Shared by `_series_from_cache` (price history) and `_latest_from_series_cache`
    (the latest-from-series fallback). Returns None if the file is absent,
    unreadable, malformed (missing date/close/fetched_at), future-stamped, or
    (unless ``allow_stale``) stale beyond the TTL. Does NOT apply any date-range
    filter — callers slice for what they need."""
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
    if not fresh(fetched, _CACHE_TTL, what=ticker, allow_stale=allow_stale):
        return None  # stale or future-stamped → refetch
    idx = pd.to_datetime(df["date"]).dt.normalize()
    series = pd.Series(
        df["close"].astype(float).to_numpy(), index=pd.DatetimeIndex(idx)
    ).sort_index()
    series = _usable_closes(series)  # a corrupt/legacy cell (0, negative, NaN, inf) must not re-serve
    if series.empty:
        return None
    return series, fetched


def series_cache_path(ticker: str, cache_dir: Path, *, total_return: bool = False) -> Path:
    """On-disk series cache for one ticker on one basis.

    The two bases MUST NOT share a file: a raw close and a dividend-adjusted close for the
    same day are different numbers, and mixing them into one parquet would silently corrupt
    both the portfolio path and the simulation path."""
    suffix = "_series_tr.parquet" if total_return else "_series.parquet"
    return cache_dir / f"{ticker}{suffix}"


def _series_from_cache(
    ticker: str, start: date, end: date, cache_dir: Path, *,
    allow_stale: bool = False, total_return: bool = False,
) -> "tuple[pd.Series[float], datetime] | None":
    """Return (series, fetched_at) from the cache, or None if absent/stale."""
    read = _read_fresh_series_cache(
        series_cache_path(ticker, cache_dir, total_return=total_return),
        ticker, allow_stale=allow_stale,
    )
    if read is None:
        return None
    series, fetched = read
    # The cached series must cover the requested end date to be usable as-is — UNLESS the
    # cache itself is fresh within TTL: then the provider had no newer rows when we asked
    # (every caller passes end=today), so falling short of `end` means the data genuinely
    # stops there (a delisted/halted ticker). Refetching would re-ask the network on every
    # run, forever, for rows that will never come; serve the cache and let the pipeline's
    # staleness floor decide whether the tail may count as a current price. A STALE short
    # cache (reachable only via allow_stale) still returns None — offline, "no usable
    # series" stays the honest answer there.
    if series.index.max() < pd.Timestamp(end - STALE_CLOSE_GRACE) and _now_utc() - fetched > _CACHE_TTL:
        return None
    mask = (series.index >= pd.Timestamp(start)) & (series.index <= pd.Timestamp(end))
    sliced = series[mask]
    if sliced.empty:
        return None  # nothing in-window (e.g. a short cache ending before `start`)
    return sliced, fetched


def _latest_from_series_cache(
    ticker: str, asof: date, cache_dir: Path
) -> PriceRow | None:
    """Derive a latest-price row from the *series* cache tail (source="cache").

    Unifies the two on-disk caches so a risk-on run's series cache can serve a
    latest price the dedicated latest cache lacks — notably under `--no-risk
    --offline`. Returns the last close at/before `asof` from the cached series,
    age-tolerant (offline fallback); future-stamped rows are still rejected, else None."""
    read = _read_fresh_series_cache(
        series_cache_path(ticker, cache_dir), ticker, allow_stale=True
    )
    if read is None:
        return None
    series, fetched = read
    series = series[series.index <= pd.Timestamp(asof)]
    if series.empty:
        return None
    tail = series.index[-1].date()
    if asof - tail > STALE_PRICE_FLOOR:
        return None  # too old to pass off as a current price, even offline
    return PriceRow(
        ticker=ticker,
        asof_date=series.index[-1].date(),
        close=float(series.iloc[-1]),
        source="cache",
        fetched_at=fetched,
    )


def _write_series_cache(
    ticker: str, series: "pd.Series[float]", cache_dir: Path, *, total_return: bool = False
) -> None:
    path = series_cache_path(ticker, cache_dir, total_return=total_return)
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


def _splits_from_cache(
    ticker: str, cache_dir: Path, *, allow_stale: bool = False
) -> list[tuple[date, float]] | None:
    """Cached split history, or None if absent (or stale, unless ``allow_stale``). A no-split
    ticker is cached as a harmless identity placeholder so we don't refetch it every run."""
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
    if not fresh(fetched, _SPLITS_TTL, what=ticker, allow_stale=allow_stale):
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
    cache = ensure_cache_dir(cache_dir)

    out: dict[str, list[tuple[date, float]]] = {}
    for ticker in dict.fromkeys(tickers):
        cached = _splits_from_cache(ticker, cache, allow_stale=not online) if cache else None
        if cached is not None:
            out[ticker] = cached
            continue
        if not online:
            out[ticker] = []  # no cached split offline → no adjustment (guard catches splits)
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
