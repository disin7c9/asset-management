"""Securities metadata ("know your securities") with provenance and on-disk cache.

The v2 foundation layer: per-ticker facts that prices don't carry — expense
ratio, AUM, average volume, category, fund family, legal type, inception date,
and the top-10 holdings (the ingredient for v1.8's near-equivalent overlap
test). The screening layer judges candidates on these; the brief can display
them for the user's own holdings (``--metadata``).

Pattern is `prices.py`'s, deliberately: provider wrapper that tests
monkey-patch (`_fetch_yf_meta` — no real HTTP in tests), per-ticker JSON cache
under the same cache dir, the ONE shared freshness gate (`prices.fresh`), and
a `fetch_metadata` that never raises on a per-ticker failure — failures land in
`.missing` and the run degrades honestly.

Values are pulled LIVE (or from a fresh cache), never from anyone's memory —
a recorded v2 rule: expense ratios and AUM drift, and stale cost data is how
a "cheap" fund quietly isn't. Metadata moves slowly, so the TTL is a week.

yfinance is the primary (spike-verified 2026-06-12 against issuer pages:
VOO 0.03% / GLDM 0.10% / BNDX 0.07%). A second provider can slot in beside
`_meta_from_yfinance` exactly like stooq did for prices, when needed.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yfinance as yf

from app.events import CASH_TICKER
from app.prices import ensure_cache_dir, fresh  # the shared cache gates (see prices.py)

log = logging.getLogger(__name__)

_CACHE_DIR_DEFAULT = Path("data/prices")
_META_TTL = timedelta(days=7)  # expense ratio / AUM / holdings drift slowly


@dataclass(frozen=True)
class SecurityMeta:
    """Per-security facts with provenance. Any field can be None — a niche or
    non-fund ticker (an equity, a physical-commodity trust) legitimately lacks
    some; consumers must degrade per-field, not per-ticker."""

    ticker: str
    expense_ratio: float | None    # annual, as a fraction (0.0003 = 0.03%)
    aum: float | None              # total net assets, USD
    avg_volume: float | None       # average daily share volume
    category: str | None           # e.g. "Large Blend"
    family: str | None             # e.g. "Vanguard"
    legal_type: str | None         # e.g. "Exchange Traded Fund"
    quote_type: str | None         # e.g. "ETF" | "EQUITY"
    inception: date | None
    top_holdings: dict[str, float] = field(default_factory=dict)  # symbol → weight fraction
    source: str = "yfinance"       # "cache" | "yfinance"
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def age_years(self, asof: date) -> float | None:
        """Fund age in years at `asof` (None if inception unknown)."""
        if self.inception is None:
            return None
        return max((asof - self.inception).days, 0) / 365.25


@dataclass(frozen=True)
class MetadataResult:
    """Outcome of a `fetch_metadata` call. Never raises on per-ticker failure."""

    rows: dict[str, SecurityMeta] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    @property
    def n_cache(self) -> int:
        return sum(1 for r in self.rows.values() if r.source == "cache")


def fetch_metadata(
    tickers: Iterable[str],
    *,
    cache_dir: Path | None = None,
    online: bool = True,
) -> MetadataResult:
    """Fetch metadata per ticker (de-duplicated): cache → yfinance.

    Never raises on a per-ticker failure — those tickers appear in `.missing`.
    The CASH pseudo-ticker is skipped (it is not a security). Offline, only a
    fresh cache can serve; everything else is missing.
    """
    cache = ensure_cache_dir(cache_dir if cache_dir is not None else _CACHE_DIR_DEFAULT)

    rows: dict[str, SecurityMeta] = {}
    missing: list[str] = []

    for ticker in dict.fromkeys(tickers):
        if ticker == CASH_TICKER:
            continue  # cash is not a security; nothing to know
        cached = _meta_from_cache(ticker, cache) if cache else None
        if cached is not None:
            rows[ticker] = cached
            continue
        if not online:
            missing.append(ticker)
            continue
        live = _meta_from_yfinance(ticker)
        if live is None:
            missing.append(ticker)
            continue
        if cache:
            _write_meta_cache(live, cache)
        rows[ticker] = live

    result = MetadataResult(rows=rows, missing=missing)
    log.info(
        "metadata fetched: returned=%d cache=%d live=%d missing=%d",
        len(result.rows), result.n_cache, len(result.rows) - result.n_cache,
        len(result.missing),
    )
    return result


# ── provider wrapper (monkey-patchable in tests) ───────────────────────────


def _fetch_yf_meta(ticker: str) -> dict[str, Any] | None:
    """All network access for one ticker, materialized into plain objects.

    Returns {"info": dict, "ops": DataFrame|None, "overview": dict|None,
    "holdings": DataFrame|None} or None when even `info` fails. The funds_data
    attributes each lazy-load over HTTP and raise for non-fund tickers
    (equities), so each is guarded separately — a missing piece is a None
    field, not a missing ticker.
    """
    try:
        t = yf.Ticker(ticker)
        info: dict[str, Any] = dict(t.info or {})
    except Exception as exc:  # noqa: BLE001 — yfinance raises many specific things
        log.warning("yfinance metadata fetch failed for %s: %s", ticker, exc)
        return None
    out: dict[str, Any] = {"info": info, "ops": None, "overview": None, "holdings": None}
    try:
        fd = t.funds_data
        out["ops"] = fd.fund_operations
        out["overview"] = fd.fund_overview
        out["holdings"] = fd.top_holdings
    except Exception as exc:  # noqa: BLE001 — non-fund tickers (equities) land here
        log.debug("no funds data for %s (%s)", ticker, exc)
    return out


# ── normalization ──────────────────────────────────────────────────────────


def _opt_float(value: object) -> float | None:
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None  # NaN/±inf → None (never render "inf")


def _opt_str(value: object) -> str | None:
    return str(value).strip() if isinstance(value, str) and value.strip() else None


def _epoch_to_date(value: object) -> date | None:
    """yfinance's fundInceptionDate is epoch seconds (UTC)."""
    f = _opt_float(value)
    if f is None or f <= 0:
        return None
    try:
        return datetime.fromtimestamp(f, tz=timezone.utc).date()
    except (OverflowError, OSError, ValueError):
        return None


def _expense_ratio_from_ops(ops: Any) -> float | None:
    """fund_operations frame → the 'Annual Report Expense Ratio' cell."""
    try:
        if ops is None or ops.empty:
            return None
        row = ops.loc["Annual Report Expense Ratio"]
        return _opt_float(row.iloc[0])
    except (KeyError, IndexError, AttributeError):
        return None


def _holdings_from_frame(holdings: Any) -> dict[str, float]:
    """top_holdings frame (index=symbol) → {symbol: weight fraction}.

    Empty for funds without look-through holdings (physical-commodity trusts
    like GLDM) — that absence is itself signal for the overlap test.
    """
    out: dict[str, float] = {}
    try:
        if holdings is None or holdings.empty:
            return out
        col = "Holding Percent" if "Holding Percent" in holdings.columns else None
        for sym in holdings.index:
            raw = holdings.loc[sym, col] if col else holdings.loc[sym].iloc[-1]
            w = _opt_float(raw)
            if w is not None and isinstance(sym, str):
                out[sym] = w
    except (KeyError, IndexError, AttributeError, TypeError):
        return out
    return out


def _meta_from_yfinance(ticker: str) -> SecurityMeta | None:
    raw = _fetch_yf_meta(ticker)
    if raw is None:
        return None
    info: dict[str, Any] = raw.get("info") or {}
    overview: dict[str, Any] = raw.get("overview") or {}
    if not info and not overview:
        return None
    return SecurityMeta(
        ticker=ticker,
        expense_ratio=_expense_ratio_from_ops(raw.get("ops")),
        aum=_opt_float(info.get("totalAssets")),
        avg_volume=_opt_float(info.get("averageVolume")),
        category=_opt_str(overview.get("categoryName")) or _opt_str(info.get("category")),
        family=_opt_str(overview.get("family")) or _opt_str(info.get("fundFamily")),
        legal_type=_opt_str(overview.get("legalType")),
        quote_type=_opt_str(info.get("quoteType")),
        inception=_epoch_to_date(info.get("fundInceptionDate")),
        top_holdings=_holdings_from_frame(raw.get("holdings")),
        source="yfinance",
        fetched_at=datetime.now(timezone.utc),
    )


# ── JSON cache (one file per ticker; the shared freshness gate applies) ────


def _meta_cache_path(ticker: str, cache_dir: Path) -> Path:
    return cache_dir / f"{ticker}_meta.json"


def _parse_iso_utc(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _meta_from_cache(ticker: str, cache_dir: Path) -> SecurityMeta | None:
    path = _meta_cache_path(ticker, cache_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("metadata cache read failed for %s: %s", ticker, exc)
        return None
    if not isinstance(payload, dict):
        return None
    fetched = _parse_iso_utc(payload.get("fetched_at"))
    if fetched is None:
        return None
    if not fresh(fetched, _META_TTL, what=ticker):
        return None  # stale or future-stamped → refetch
    inception = payload.get("inception")
    try:
        # Inside the try: never raise on a corrupt cache. A bad holding weight
        # (non-numeric, NaN, ±inf) degrades per-field via _opt_float — same as the
        # live path — and is dropped; a bad inception date below refetches the row.
        holdings_raw = payload.get("top_holdings")
        holdings = (
            {str(k): w for k, v in holdings_raw.items() if (w := _opt_float(v)) is not None}
            if isinstance(holdings_raw, dict)
            else {}
        )
        return SecurityMeta(
            ticker=ticker,
            expense_ratio=_opt_float(payload.get("expense_ratio")),
            aum=_opt_float(payload.get("aum")),
            avg_volume=_opt_float(payload.get("avg_volume")),
            category=_opt_str(payload.get("category")),
            family=_opt_str(payload.get("family")),
            legal_type=_opt_str(payload.get("legal_type")),
            quote_type=_opt_str(payload.get("quote_type")),
            inception=date.fromisoformat(inception) if isinstance(inception, str) else None,
            top_holdings=holdings,
            source="cache",
            fetched_at=fetched,
        )
    except (TypeError, ValueError) as exc:
        log.warning("metadata cache malformed for %s: %s", ticker, exc)
        return None


def _write_meta_cache(meta: SecurityMeta, cache_dir: Path) -> None:
    payload = asdict(meta)
    payload["fetched_at"] = meta.fetched_at.isoformat()
    payload["inception"] = meta.inception.isoformat() if meta.inception else None
    payload.pop("source", None)  # a cache hit reports source="cache"; don't store it
    try:
        _meta_cache_path(meta.ticker, cache_dir).write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    except OSError as exc:
        log.warning("metadata cache write failed for %s: %s", meta.ticker, exc)
