#!/usr/bin/env python3
"""Dev tool (NOT runtime) — seed/refresh ``app/data/universe.csv`` from yfinance.

For each ticker: fetch the fund's name + Morningstar-style category + business summary,
best-effort map the category to our role vocabulary, and emit a universe CSV row for
**human review**. Run occasionally by the maintainer to grow/refresh the universe; the
app never rebuilds it at runtime. Anything the mapping can't place is tagged ``REVIEW``
(which ``universe.load_universe`` rejects), so a guessed/missing role can't slip in
uncommitted-unchecked — yfinance categories are messy and sometimes absent.

Usage:
    python scripts/build_universe.py VOO BND IAU VEA                # seed these tickers (-> stdout)
    python scripts/build_universe.py --from app/data/universe.csv   # refresh the existing tickers
    python scripts/build_universe.py VWO LQD --out /tmp/seed.csv
"""

from __future__ import annotations

import argparse
import csv
import sys

import yfinance as yf

# Morningstar categories whose funds are a TILT within their role rather than a plain
# representative of it: Growth/Value styles, single regions, foreign small/mid, high-yield.
# Everything else seeds core=1 — blends, diversified EM, investment-grade bonds, and the
# sector/thematic categories (there the ROLE is the tilt; within it every fund represents it).
# Seeds the universe's `core` column for human review; discovery surfaces core funds first.
_NON_CORE_CATEGORIES: frozenset[str] = frozenset(
    {
        "Large Growth", "Mid-Cap Growth", "Mid-Cap Value", "Small Growth", "Small Value",
        "Foreign Large Growth", "Foreign Large Value", "Foreign Small/Mid Blend",
        "Europe Stock", "Japan Stock", "Pacific/Asia ex-Japan Stk",
        "High Yield Bond",
    }
)


def _seed_core(category: str) -> str:
    return "0" if category in _NON_CORE_CATEGORIES else "1"


# Category -> flavor (SHELF) seed. The shelf groups near-substitutes within a role for
# discovery's menus; sector categories seed the block-level token and the human review
# refines finer shelves (semis/software/ai/... — the committed file is the record).
# Blank = the role's one unnamed shelf.
_CATEGORY_FLAVOR: dict[str, str] = {
    "Large Blend": "blend", "Large Growth": "growth",
    "Mid-Cap Blend": "mid-blend", "Mid-Cap Growth": "mid-growth", "Mid-Cap Value": "mid-value",
    "Small Blend": "small-blend", "Small Growth": "small-growth", "Small Value": "small-value",
    "Foreign Large Blend": "blend", "Foreign Large Growth": "growth",
    "Foreign Large Value": "value", "Foreign Small/Mid Blend": "small-mid",
    "Europe Stock": "europe", "Japan Stock": "japan",
    "Diversified Emerging Mkts": "diversified", "Pacific/Asia ex-Japan Stk": "asia-pacific",
    "Long Government": "long", "Intermediate Government": "intermediate",
    "Short Government": "short",
    "Corporate Bond": "investment-grade", "High Yield Bond": "high-yield",
    "Real Estate": "us", "Global Real Estate": "global",
    "Technology": "tech", "Health": "health", "Financial": "financial",
    "Utilities": "utilities", "Equity Energy": "energy",
    "Natural Resources": "nat-resources", "Infrastructure": "infrastructure",
}


def _seed_flavor(category: str) -> str:
    return _CATEGORY_FLAVOR.get(category, "")


# Mis-shelved funds a category query keeps re-pulling: SPHB is a large-cap high-beta
# factor fund tagged "Mid-Cap Blend"; MTBA/LMBS are mortgage funds tagged as treasuries
# (the committed file re-roles them to bond-aggregate tilts by hand). Dropped from pulls
# so a refresh can't silently restore the wrong shelf.
_DROP_TICKERS = frozenset({"SPHB", "MTBA", "LMBS"})


# Category substring -> role. First match wins; order matters (specific before general).
# Unmapped -> "REVIEW". Gold is handled by name/category before this table.
_CATEGORY_RULES: tuple[tuple[str, str], ...] = (
    ("inflation-protected", "tips"),
    ("emerging", "em-equity"),
    ("foreign", "intl-developed"),
    ("real estate", "reit"),
    ("corporate", "corporate-bond"),
    ("government", "treasury"),
    ("treasury", "treasury"),
    ("commodities", "commodity-broad"),
    ("bond", "bond-aggregate"),
    ("small", "us-small-mid"),
    ("mid-cap", "us-small-mid"),
    ("large", "us-large"),
    ("technology", "sector-equity"),
    ("health", "sector-equity"),
    ("financial", "sector-equity"),
    ("energy", "sector-equity"),
    ("utilities", "sector-equity"),
)


def _seed_role(category: str, name: str) -> str:
    cat, nm = category.lower(), name.lower()
    if "gold" in nm or "gold" in cat:
        return "gold"
    for needle, role in _CATEGORY_RULES:
        if needle in cat:
            return role
    return "REVIEW"  # load_universe rejects this -> forces a human decision


def _short_summary(text: str, limit: int = 160) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def _fetch(ticker: str) -> dict[str, str]:
    info = yf.Ticker(ticker).info
    name = info.get("longName") or info.get("shortName") or ""
    category = info.get("category") or ""
    role = _seed_role(category, name)
    summary = _short_summary(info.get("longBusinessSummary") or "")
    if role == "REVIEW":  # leave the raw category visible so the human can place it
        summary = (summary + f"  [category: {category or 'n/a'}]").strip()
    return {
        "ticker": ticker.upper(), "name": name, "role": role, "summary": summary,
        "core": _seed_core(category), "flavor": _seed_flavor(category),
    }


# Names that mark a leveraged/inverse product (out of scope — long-only) → dropped from the pull.
_EXCLUDE_NAME = (
    "leveraged", "inverse", "ultrapro", "ultrashort", "-1x", "2x", "3x", " bear", "bull 3",
)

# Our role → Morningstar categorynames (ETFQuery filters on these, so the role is known from the
# query — no per-ticker fetch). Covers 10 of 13 roles. The other 3 stay curated below: us-dividend
# (a strategy inside "Large Value"), gold bullion ("Commodities Focused" — not an allowed value),
# and bond-aggregate (its funds are tagged "Intermediate Core Bond", which the query enum rejects —
# the only allowed near-match, "Intermediate-Term Bond", matches zero funds). Theme funds
# ("Miscellaneous Sector" junk-drawer) are also curated, as shelves of sector-equity.
# All names below verified to return ETFs.
_ROLE_CATEGORIES: dict[str, tuple[str, ...]] = {
    "us-large": ("Large Blend", "Large Growth"),  # "Large Value" omitted (where dividend funds sit)
    "us-small-mid": ("Mid-Cap Blend", "Mid-Cap Growth", "Mid-Cap Value",
                     "Small Blend", "Small Growth", "Small Value"),
    "intl-developed": ("Foreign Large Blend", "Foreign Large Growth", "Foreign Large Value",
                       "Foreign Small/Mid Blend", "Europe Stock", "Japan Stock"),
    "em-equity": ("Diversified Emerging Mkts", "Pacific/Asia ex-Japan Stk"),
    "sector-equity": ("Technology", "Health", "Financial", "Utilities", "Equity Energy",
                      "Natural Resources", "Infrastructure"),
    # Intermediate first: the role's lead shelf (and preset default) is the standard
    # market-duration sleeve, so the refresh keeps intermediates at the top of the block.
    "treasury": ("Intermediate Government", "Long Government", "Short Government"),
    "tips": ("Inflation-Protected Bond",),
    "corporate-bond": ("Corporate Bond", "High Yield Bond"),
    "reit": ("Real Estate", "Global Real Estate"),
    "commodity-broad": ("Commodities Broad Basket",),
}

# The 3 roles with no clean/queryable category + the theme shelves — a small, stable curated set (wins on ticker conflict).
_CURATED: tuple[dict[str, str], ...] = (
    {"ticker": "SCHD", "name": "Schwab U.S. Dividend Equity ETF", "role": "us-dividend",
     "summary": "High-dividend, quality-screened US stocks (Dow Jones US Dividend 100)."},
    {"ticker": "VYM", "name": "Vanguard High Dividend Yield ETF", "role": "us-dividend",
     "summary": "Higher-yielding US large-caps (FTSE High Dividend Yield)."},
    {"ticker": "VIG", "name": "Vanguard Dividend Appreciation ETF", "role": "us-dividend",
     "summary": "US stocks with a long record of growing dividends."},
    {"ticker": "FDVV", "name": "Fidelity High Dividend ETF", "role": "us-dividend",
     "summary": "Higher-yielding US stocks with a quality tilt."},
    {"ticker": "IAU", "name": "iShares Gold Trust", "role": "gold", "summary": "Physical gold bullion."},
    {"ticker": "GLD", "name": "SPDR Gold Shares", "role": "gold",
     "summary": "Physical gold bullion — the largest gold ETF."},
    {"ticker": "GLDM", "name": "SPDR Gold MiniShares Trust", "role": "gold",
     "summary": "Physical gold bullion (low-cost MiniShares)."},
    {"ticker": "IAUM", "name": "iShares Gold Trust Micro", "role": "gold",
     "summary": "Physical gold bullion (low-cost Micro)."},
    {"ticker": "ICLN", "name": "iShares Global Clean Energy ETF", "role": "sector-equity",
     "flavor": "clean-energy",
     "summary": "~100 global clean-energy companies (S&P Global Clean Energy)."},
    {"ticker": "TAN", "name": "Invesco Solar ETF", "role": "sector-equity",
     "flavor": "clean-energy",
     "summary": "Global solar-energy companies (MAC Global Solar Energy Index)."},
    # The ARK family: cross-sector theme funds the category queries mis-file (ARKK as
    # "Technology", ARKQ/ARKW as mid-caps) — curated so a refresh keeps them on the
    # innovation shelf of the sector/thematic aisle.
    {"ticker": "ARKK", "name": "ARK Innovation ETF", "role": "sector-equity",
     "flavor": "innovation",
     "summary": "Disruptive innovation across sectors, actively managed (ARK)."},
    {"ticker": "ARKQ", "name": "ARK Autonomous Technology & Robotics ETF",
     "role": "sector-equity", "flavor": "innovation",
     "summary": "Autonomous tech & robotics, actively managed (ARK)."},
    {"ticker": "ARKW", "name": "ARK Next Generation Internet ETF",
     "role": "sector-equity", "flavor": "innovation",
     "summary": "Next-generation internet, actively managed (ARK)."},
    {"ticker": "BND", "name": "Vanguard Total Bond Market ETF", "role": "bond-aggregate",
     "summary": "Broad US investment-grade taxable bond market."},
    {"ticker": "AGG", "name": "iShares Core U.S. Aggregate Bond ETF", "role": "bond-aggregate",
     "summary": "Broad US investment-grade bond market (Bloomberg US Aggregate)."},
    {"ticker": "SCHZ", "name": "Schwab U.S. Aggregate Bond ETF", "role": "bond-aggregate",
     "summary": "Broad US investment-grade bond market (Bloomberg US Aggregate)."},
    {"ticker": "FBND", "name": "Fidelity Total Bond ETF", "role": "bond-aggregate",
     "summary": "Broad US bonds, actively managed (core-plus)."},
)


def _auto_pull(min_aum: float, per_category: int) -> dict[str, dict[str, str]]:
    """Auto-build the universe via ETFQuery: for each role's Morningstar categories, the biggest
    (AUM-sorted) US-listed ETFs. Role is known from the query → no per-ticker fetch. Returns a
    ticker→row map (biggest-first, first wins). Drops foreign cross-listings (``.`` in the symbol)
    and leveraged/inverse by name."""
    seen: dict[str, dict[str, str]] = {}
    for role, cats in _ROLE_CATEGORIES.items():
        for cat in cats:
            try:
                q = yf.ETFQuery("and", [
                    yf.ETFQuery("eq", ["categoryname", cat]),
                    yf.ETFQuery("gt", ["fundnetassets", min_aum]),
                ])
                res = yf.screen(q, count=per_category, sortField="fundnetassets", sortAsc=False)
            except Exception as exc:  # noqa: BLE001 — a dev tool: report the bad category and keep going
                print(f"WARN ETFQuery {cat!r}: {exc}", file=sys.stderr)
                continue
            quotes = (res.get("quotes") if isinstance(res, dict) else res) or []
            kept = 0
            for r in quotes:
                sym = (r.get("symbol") or "").upper()
                nm = r.get("shortName") or r.get("longName") or sym
                if (sym and "." not in sym and r.get("quoteType") == "ETF" and sym not in seen
                        and sym not in _DROP_TICKERS
                        and not any(x in nm.lower() for x in _EXCLUDE_NAME)):
                    seen[sym] = {"ticker": sym, "name": nm, "role": role,
                                 "summary": f"{cat} ETF.", "core": _seed_core(cat),
                                 "flavor": _seed_flavor(cat),
                                 # sort key only (DictWriter drops it); a str keeps the row homogeneous
                                 "_aum": str(float(r.get("fundnetassets") or r.get("marketCap") or 0.0))}
                    kept += 1
            print(f"  {role:16} {cat:28} +{kept}", file=sys.stderr)
    return seen


def _tickers_from_screens(screens: list[str], count: int) -> list[str]:
    """Auto-pull ETF tickers from yfinance's predefined screens (e.g. ``top_etfs_us``,
    ``bond_etfs``). Returns ``quoteType==ETF`` symbols, dropping leveraged/inverse by name;
    each one's category/role is derived later by the per-ticker enrichment. Network — a
    failing screen is skipped (a dev tool)."""
    seen: dict[str, None] = {}
    for name in screens:
        try:
            res = yf.screen(name, count=count)
        except Exception as exc:  # noqa: BLE001 — a dev tool: report the bad screen and keep going
            print(f"WARN screen {name}: {exc}", file=sys.stderr)
            continue
        quotes = (res.get("quotes") if isinstance(res, dict) else res) or []
        kept = 0
        for r in quotes:
            sym = (r.get("symbol") or "").upper()
            nm = (r.get("shortName") or r.get("longName") or "").lower()
            if (sym and r.get("quoteType") == "ETF"
                    and not any(x in nm for x in _EXCLUDE_NAME) and sym not in seen):
                seen[sym] = None
                kept += 1
        print(f"screen {name}: {len(quotes)} rows -> +{kept} new ETFs", file=sys.stderr)
    return list(seen)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Seed/refresh the curated universe from yfinance (output is for human review)."
    )
    ap.add_argument("tickers", nargs="*", help="tickers to seed")
    ap.add_argument("--from", dest="src", help="also read tickers from an existing universe CSV")
    ap.add_argument("--out", help="write CSV here (default: stdout)")
    ap.add_argument(
        "--from-screen", dest="screens",
        help="auto-pull ETF tickers from comma-separated predefined yfinance screens "
        "(e.g. top_etfs_us,bond_etfs); each is then enriched + role-mapped like any other",
    )
    ap.add_argument("--count", type=int, default=100,
                    help="max ETFs per --from-screen, or per category for --auto (default 100)")
    ap.add_argument(
        "--auto", action="store_true",
        help="auto-build the WHOLE universe via ETFQuery-by-category-by-AUM (10 roles) + the "
        "curated odd roles (dividend/gold/bond-aggregate + theme shelves). Writes complete rows; no per-ticker fetch.",
    )
    ap.add_argument("--min-aum", type=float, default=1e9, help="AUM floor for --auto (default $1B)")
    args = ap.parse_args(argv)

    if args.auto:
        rows_by_ticker = _auto_pull(args.min_aum, args.count)
        for c in _CURATED:  # curated wins on a conflict (SCHD is us-dividend, not auto's us-large)
            # pop first so a curated row takes the CURATED list's position, not the auto
            # pull's (a dict update keeps the old slot — that artifact once let VIG jump
            # ahead of SCHD). Anchors lead their role; curated rows are core by construction.
            rows_by_ticker.pop(c["ticker"], None)
            rows_by_ticker[c["ticker"]] = {
                **c, "core": "1", "flavor": c.get("flavor", ""), "_aum": "inf",
            }
        # Group by role, biggest-AUM first within each role → discovery's "top 3" = the giants.
        out_rows = sorted(rows_by_ticker.values(), key=lambda r: (r["role"], -float(r.get("_aum", "0"))))
        out = open(args.out, "w", newline="", encoding="utf-8") if args.out else sys.stdout  # noqa: SIM115 — closed in the finally below; a `with` can't also yield stdout  # noqa: SIM115 — closed in the finally below; a `with` can't also yield stdout
        try:
            writer = csv.DictWriter(out, fieldnames=("ticker", "name", "role", "summary", "core", "flavor"),
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(out_rows)
        finally:
            if args.out:
                out.close()
        roles = sorted({r["role"] for r in out_rows})
        print(f"auto-built {len(out_rows)} ETFs across {len(roles)} roles", file=sys.stderr)
        return 0

    tickers = list(args.tickers)
    if args.src:
        with open(args.src, newline="", encoding="utf-8-sig") as fh:
            tickers += [(r.get("ticker") or "").strip() for r in csv.DictReader(fh)]
    if args.screens:
        tickers += _tickers_from_screens(
            [s.strip() for s in args.screens.split(",") if s.strip()], args.count
        )
    tickers = list(dict.fromkeys(t.upper() for t in tickers if t.strip()))  # dedup, keep order
    if not tickers:
        ap.error("give tickers, --from a CSV, or --from-screen NAMES")

    rows: list[dict[str, str]] = []
    for t in tickers:
        try:
            rows.append(_fetch(t))
        except Exception as exc:  # noqa: BLE001 — a dev tool: report and keep going
            print(f"WARN {t}: {exc}", file=sys.stderr)
            rows.append({"ticker": t, "name": "", "role": "REVIEW", "summary": "[fetch failed]"})

    out = open(args.out, "w", newline="", encoding="utf-8") if args.out else sys.stdout  # noqa: SIM115 — closed in the finally below; a `with` can't also yield stdout
    try:
        writer = csv.DictWriter(out, fieldnames=("ticker", "name", "role", "summary", "core", "flavor"))
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.out:
            out.close()

    n_review = sum(1 for r in rows if r["role"] == "REVIEW")
    print(
        f"seeded {len(rows)} ticker(s); {n_review} need a role assigned (REVIEW) before committing",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
