# asset-management

A small Python tool for tracking a personal portfolio of stocks and ETFs.

Reads an append-only CSV transaction log (date, ticker, action, quantity, price, fee per row), replays it, and prints current holdings + market value, cost basis, realized and unrealized P&L, and per-period returns. Holdings are *derived* from the log — never stored — so the same input always produces the same output. Prices are fetched from multiple sources with fallback and an on-disk cache; every displayed number is traceable to its source.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/) for dependency management

## Install

```bash
uv sync
```

## Run

```bash
uv run python -m app                         # bundled example log; fetches live prices
uv run python -m app --csv path/to/your.csv  # your own CSV
uv run python -m app --no-prices             # skip pricing entirely (holdings + realized only)
uv run python -m app --offline               # serve from on-disk cache; no network
uv run python -m app --cache-dir /some/dir   # override the default cache location
```

The CSV is expected to have columns: `Date, Code, DataSource, Currency, Price, Quantity, Action, Fee, Note`. `Action` is one of `buy`, `sell`, `dividend`, `fee`, `interest`. Empty cells in numeric columns are treated as zero. Non-ISO dates are rejected with a clear error. UTF-8 BOM is tolerated.

Example output:

```
ticker     shares  avg cost    price    mkt value      unreal    realized
-------------------------------------------------------------------------
BND        50.000     72.26    73.46      3673.00      +60.00      +45.10
IAU        25.000     37.23    85.49      2137.25    +1206.62     +220.62
VEA        15.000     40.27    71.77      1076.55     +472.55       +0.00
VOO        12.000    375.13   695.49      8345.88    +3844.28     +295.80
-------------------------------------------------------------------------
Total cost basis (held): $9,649.23
Market value (priced):   $15,232.68
Unrealized P&L:          $+5,583.45
Realized P&L (sells+div): $+561.52
Fees paid (informational): $8.00
Net P&L (unrealized + realized): $+6,144.98

Period: 2023-01-05 → 2026-06-01 (1243 days, ~3.40y)
Money-weighted return (IRR, annualized):    +18.35%
Modified Dietz (annualized, approx TWR):    +18.01%

Prices: 4 yfinance  (age: 0s .. 1s old as of 2026-06-01 04:34 UTC)
```

Each run emits one structured JSON log line (`run_summary`) on stderr summarizing what happened: `{date, source, n_events_replayed, n_prices_fetched, n_prices_missing, fallbacks_used, status}`.

## Develop

```bash
uv run pytest                  # unit + property-based + regression tests
uv run mypy app/               # strict type-checking
uv run ruff check app/ tests/  # lint
```

## Layout

```
asset-management/
├── app/
│   ├── events.py     CSV → typed event list
│   ├── derive.py     events → holdings + cost basis + realized P&L
│   ├── prices.py     multi-source price fetch with provenance + on-disk cache
│   ├── returns.py    events + value → MWR (XIRR) + Modified Dietz returns
│   ├── report.py     state + prices + returns → text summary
│   ├── cli.py        argparse + entry composition + structured run log
│   ├── log_config.py logging setup
│   ├── __main__.py   python -m app
│   └── __init__.py
├── tests/            unit, property, and regression tests
├── pyproject.toml    dependencies, mypy/ruff config
└── README.md
```

## License

Personal project, no license granted. All rights reserved.
