# asset-management

A small Python tool for tracking a personal portfolio of stocks and ETFs.

Reads an append-only CSV transaction log (date, ticker, action, quantity, price, fee per row), replays it, and prints current holdings with cost basis and realized profit/loss. Holdings are *derived* from the log — never stored — so the same input always produces the same output.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/) for dependency management

## Install

```bash
uv sync
```

## Run

```bash
uv run python -m app                         # use the bundled example log
uv run python -m app --csv path/to/your.csv  # use your own
```

The CSV is expected to have columns: `Date, Code, DataSource, Currency, Price, Quantity, Action, Fee, Note`. `Action` is one of `buy`, `sell`, `dividend`, `fee`, `interest`. Empty cells in numeric columns are treated as zero. Non-ISO dates are rejected with a clear error.

Example output:

```
ticker     shares  avg cost   cost basis     realized
-----------------------------------------------------
BND        50.000     72.26      3613.00       +45.10
IAU        25.000     37.23       930.62      +220.62
VEA        15.000     40.27       604.00        +0.00
VOO        12.000    375.13      4501.60      +295.80
-----------------------------------------------------
Total cost basis (held): $9,649.23
Total realized P&L:      $+561.52
Total fees paid:         $8.00
```

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
│   ├── report.py     state → text summary
│   ├── cli.py        argparse + entry composition
│   ├── log_config.py logging setup
│   ├── __main__.py   python -m app
│   └── __init__.py
├── tests/            unit, property, and regression tests
├── pyproject.toml    dependencies, mypy/ruff config
└── README.md
```

## License

Personal project, no license granted. All rights reserved.
