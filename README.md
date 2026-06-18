# asset-management

A small Python tool for tracking a personal portfolio of stocks and ETFs.

Reads an append-only CSV transaction log (date, ticker, action, quantity, price, fee per row), replays it, and prints a **drawdown-first** brief: how far the portfolio fell from its peak (with a bootstrap confidence band), risk-adjusted ratios, time- and money-weighted returns, and current holdings. Holdings are *derived* from the log — never stored — so the same input always produces the same output. Prices are fetched from multiple sources with fallback and an on-disk cache; every displayed number is traceable to its source, and figures that can't be computed honestly (too-short a window, no real solution) print `n/a` rather than a fabricated number. **Stock splits are adjusted automatically** (share counts are reconciled with the split-adjusted price history), so a split during your holding period doesn't distort the returns.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/) for dependency management

## Install

```bash
uv sync
```

## Run

```bash
uv run python -m app --csv data/sample_data/transactions.csv  # the bundled example book (opt-in)
uv run python -m app --csv path/to/your.csv  # your own book: holdings + returns + risk
uv run python -m app --csv your.csv --no-risk    # skip the holdings drawdown/risk panel (a --backtest still shows its own)
uv run python -m app --csv your.csv --no-prices  # holdings + realized P&L only (no network)
uv run python -m app --csv your.csv --offline    # serve from on-disk cache; no network (latest falls back to the series cache)
uv run python -m app --csv your.csv --cache-dir /some/dir   # override the default cache location
uv run python -m app --csv your.csv --save       # also write reports/<asof>.md (markdown)
uv run python -m app --csv your.csv --send       # also email the brief as HTML via Resend
uv run python -m app --csv your.csv --metadata   # + SECURITIES panel: expense ratio, AUM, liquidity, age per holding
uv run python -m app --csv your.csv --screen QQQM,SCHD  # judge NEW candidate tickers against your book (propose-only)
uv run python -m app --csv your.csv --screen SCHD --target target.csv  # + walk-forward ROLE check (held-out evidence)
uv run python -m app --csv your.csv --narrate    # + a plain-language SUMMARY (opt-in; bring your own LLM key in .env)
uv run python -m app --csv your.csv --dump-target target.csv                    # write current allocation to edit
uv run python -m app --csv your.csv --allocate inverse_vol --allocate-out t.csv  # PROPOSE a target (re-weight holdings)
uv run python -m app --csv your.csv --rebalance to_total --target target.csv    # suggestions toward a target
uv run python -m app --csv your.csv --rebalance cash_flow_only --target target.csv --new-cash 1000
uv run python -m app --backtest --target target.csv             # notional backtest — needs only --target, no --csv
uv run python -m app --csv your.csv --rebalance bands --backtest --target target.csv  # panels stack
```

The report is **composable panels**, not exclusive modes — combine flags and the panels stack (the last example prints SUGGESTED ACTIONS *and* BACKTEST). What each action needs:

| action | needs | what it does |
|---|---|---|
| status brief (default) | `--csv` | your holdings + returns + drawdown/risk |
| `--rebalance MODE` | `--csv` + `--target` | buy/sell suggestions toward the target (`--new-cash` sizes a deposit) |
| `--allocate RULE` | `--csv` | propose a target by re-weighting your holdings (write it with `--allocate-out`) |
| `--metadata` | `--csv` | published fund facts per holding (expense ratio, AUM, volume, age, category), cached 7 days |
| `--screen TICKERS` | `--csv` + prices | judge NEW candidates vs your book: diversifier (incl. your red days + worst drawdown), cost, liquidity, age, concentration, leveraged/inverse auto-reject, holdings-overlap dedup — each verdict with its reason. Add `--target` for the **walk-forward role check**: did a 5% sleeve improve drawdown/vol on a held-out window? (paired-bootstrap honesty gate; "inconclusive" when inside the noise band). Propose-only; a PASS is "sane, cheap, liquid, genuinely different", never a prediction |
| `--backtest` | `--target` | notional rebalance-vs-buy-and-hold — **no `--csv`**; prints the simulation alone |
| `--narrate` | `--csv` + an LLM key | a plain-language **SUMMARY** at the top of the brief; the model writes only the words, every number is substituted and verified from the core (opt-in, off by default — see [Narration](#narration-optional-plain-language-summary)) |

There is **no silent built-in default**: a book-dependent action without `--csv` errors out, and a bare `python -m app` prints a hint — the bundled example is opt-in (`--csv data/sample_data/transactions.csv`), never assumed. `--allocate` is **propose-only** and cannot be combined with `--rebalance`/`--backtest` (review the written file, then act on it in a separate command). A `target.csv` is one you create with `--dump-target` / `--allocate-out` (or use `data/sample_data/target.csv`).

**Your own default** lives in a gitignored `.env` at the repo root: set `ASSET_CSV=path/to/your.csv` (and optionally `ASSET_TARGET=path/to/target.csv`; `~` is expanded, relative paths resolve against the repo root) and a bare `python -m app` becomes your brief, `--rebalance MODE` alone works, etc. Explicit flags always win, and the pure `--backtest --target` run stays book-free by contract — pass `--csv` explicitly to stack your status panel onto a backtest.

## Choose a target — the strategy engine

`--allocate <rule>` builds a target allocation **over your current holdings** and prints it next to your present weights; add `--allocate-out path` to save it (a dedicated flag — `--dump-target` always means your current holdings). It re-weights what you already own (no new tickers — finding new ones is a later, AI-assisted step), so it's *discipline*, not prediction. Rules:

- **`equal_weight`** — 1/N. The robust baseline (it beat optimization out-of-sample in our own studies).
- **`inverse_vol`** — weight ∝ 1 ÷ volatility, so each holding contributes roughly the *same risk* rather than the same dollars (a calm bond and a wild theme ETF get balanced by how much they move). Cap any single weight with `--allocate-cap 0.30`.

Deliberately **not** included: return-forecasting optimizers (mean-variance / max-Sharpe) — they overfit and lost to equal-weight out-of-sample in our tests, so they stay behind an *edge* gate for a later version.

**Propose, simulate, and act are separate steps**, enforced: `--allocate` only *proposes* (and optionally writes the file), and it **may not be combined** with `--backtest`/`--rebalance` in one command. You review the file, then run `--backtest --target file` to simulate it, or `--rebalance <mode> --target file` to get orders — in a separate command. The strategy never silently becomes trades.

## Suggestions

With `--rebalance <mode>` the brief leads with a **SUGGESTED ACTIONS** panel: per-ticker buy/sell/hold to move your holdings toward a target allocation, each line paired to the **named rule** that produced it (you learn the rule, not trust a bot). These are *discipline, not predictions* — a rebalance makes no claim to beat the market, so it needs no backtest to be honest. The tool never trades; it suggests.

A **target is a complete spec** (`--target path`, columns `Ticker,Weight`; weights are relative and normalized): any held ticker **not** listed is treated as an exit and sold to $0. So `--target` is *required* with `--rebalance` (no silent default), and the run **warns** listing any held tickers the target omits. To **close a position on purpose**, give it weight `0` — that's an explicit, warning-free exit; *omitting* it does the same but triggers the safety warning (the tool can't tell "forgot" from "meant it"). To create a target that matches your real holdings, run `--dump-target target.csv` — it writes your **current** allocation, which you then edit toward your desired mix. Modes:

- `to_total` — sell + buy to hit the target exactly (cash-neutral; deploys `--new-cash` too)
- `cash_flow_only` — invest `--new-cash` into underweights; never sell (tax-friendly)
- `fixed_dca` — buy the target mix with `--new-cash`, ignoring drift
- `bands` — like `to_total` but only act when a ticker's drift exceeds its band; rebalances existing holdings only (ignores `--new-cash`). The band is the **smaller** of an absolute `--band` (default 5pp) or `--band-rel` × the ticker's target weight (the **"5/25 rule"**, default 25%) — so a small sleeve isn't handed a band many times its own size (a flat 5pp would let a 1% holding vanish or 6× untouched); a 0% target → 0 band → always exits

## Backtest

`--backtest --target T.csv` runs a **notional $10,000** historical simulation of that target and prints a **BACKTEST** panel comparing **rebalanced** (schedule via `--rebalance-every {monthly,quarterly,annually}`, default quarterly) vs **buy-and-hold** — drawdown-first, with bootstrap CIs. It's *notional*: it starts a clean $10k at the target weights on the earliest date all tickers have prices (`--backtest-start` to override), so it tests the *strategy*, independent of your actual buy timing. Labeled **a historical simulation, not a prediction**.

A fixed rebalance policy fits no parameters, so the whole history is out-of-sample-clean (nothing to overfit). The **walk-forward train/test *selection*** machinery — needed only once a strategy *searches* (tunes parameters or picks among candidates: an optimizer, or an *edge* timing strategy) — is deliberately deferred; a **discipline-vs-edge gate** enforces that any future edge strategy must pass a walk-forward backtest before it may surface a suggestion. Today's rebalance modes are all *discipline*, so they suggest freely.

The same report can leave three ways from one build: plain text on stdout (always),
markdown to `reports/<asof>.md` (`--save`), and an HTML email (`--send`). `--send`
reads `RESEND_API_KEY` and `REPORT_TO` (and optional `REPORT_FROM`) from `.env`.
A failed sink never crashes the run — the brief still prints to stdout and the
failure is logged — but if a sink you *requested* (`--save`/`--send`) fails, the
process exits non-zero so a scheduler notices the missed delivery. A weekly
Monday brief is just this on cron:

```cron
# 08:00 every Monday — email the brief and archive the markdown
0 8 * * 1  cd /path/to/asset-management && /path/to/uv run python -m app --send --save
```

The CSV is expected to have columns: `Date, Code, DataSource, Currency, Price, Quantity, Action, Fee, Note`. `Action` is one of `buy`, `sell`, `dividend`, `fee`, `interest`, `deposit`, `withdraw`. Cash flows (`deposit`/`withdraw`) use a `CASH` code and put the amount in the `Price` column. Empty cells in numeric columns are treated as zero. Non-ISO dates are rejected with a clear error. UTF-8 BOM is tolerated.

The drawdown panel also reports **Gains given back** — the largest dollar decline in your cumulative market profit (the felt "how much did I watch evaporate"). It's flow-neutral: deposits, withdrawals, and trades cancel, so funding and broker transfers don't distort it (the raw account balance would dip on every transfer).

Example output (drawdown leads, then risk-adjusted ratios, then returns, then holdings):

```
=== DRAWDOWN (investment, time-weighted) ===
Max drawdown:      -9.84%  (95% CI -16.63% .. -6.37%)
  peak 2025-02-19 → trough 2025-04-08 → 2025-06-10  (111 days)
Ulcer index:       2.38%  (95% CI 1.51% .. 6.26%)
CDaR (worst 5%):   6.73%  (95% CI 4.43% .. 14.01%)
You've spent 80% of this period below a previous high.

=== RISK-ADJUSTED (annualized, 252-day basis, risk-free 0%, ± bootstrap CI) ===
Sharpe:   +1.75  (95% CI +0.68 .. +2.83)
Sortino:  +2.66  (95% CI +0.96 .. +4.63)
Calmar:   +1.97  (95% CI +0.52 .. +4.83)

=== RETURNS (annualized, 252-day basis) ===
Period: 2023-01-05 → 2026-06-02 (1244 days, ~3.41y)
Time-weighted (true TWR):                +19.37%
Money-weighted (IRR):                    +18.28%
Modified Dietz (approx TWR):             +17.94%
  (point figures: accounting identities over your cash flows, not sampled
  statistics → no band; see RISK-ADJUSTED for bootstrapped CIs)

=== HOLDINGS ===
ticker     shares  avg cost    price    mkt value      unreal    realized
-------------------------------------------------------------------------
BND        50.000     72.26    73.18      3659.00      +46.00      +45.10
...
Total cost basis (held): $9,649.23
Market value (priced):   $15,211.40
Unrealized P&L:          $+5,562.17
Realized P&L (sells+div): $+561.52
Fees paid (informational): $8.00
Net P&L (unrealized + realized): $+6,123.70

Prices: 4 cache  (age: 6.2h .. 6.2h old as of 2026-06-02 02:42 UTC)
Generated by asset-management. Figures are deterministic and reconciled against ghostfolio + quantstats; this is not financial advice.
```

Confidence bands come from a moving-block bootstrap. Drawdown is *investment* (time-weighted) drawdown, not account-balance drawdown. Each run also emits one structured JSON log line (`run_summary`) on stderr: `{date, source, n_events_replayed, n_prices_fetched, n_prices_missing, n_series_fetched, n_series_missing, fallbacks_used, status, report_saved, email_sent, rebalance, backtest, allocate, metadata, screen, narrate}` (with `email_detail`/`error` present when relevant).

## MCP server (read-only, for AI assistants)

Expose your portfolio to an AI assistant (Claude Desktop, Claude Code, …) as **read-only tools** it can call — so you can "chat with your portfolio" while every number still comes from the validated core, not the model. The server is **local, offline, and read-only**: it serves from the on-disk cache (no network), exposes **no write tools**, and is bound to your `ASSET_CSV` book (it takes no file paths). Three tools:

- **`portfolio_summary`** — holdings, P&L, and annualized returns.
- **`risk_report`** — drawdown-first risk: max drawdown (depth/dates/recovery), Ulcer, CDaR, Sharpe/Sortino/Calmar, all with bootstrap confidence intervals.
- **`rebalance_check`** — buy/sell/hold suggestions toward your `ASSET_TARGET` (it suggests, never trades; refuses to size over a partially-cached book).

Run it directly, or register it with Claude Code:

```bash
uv run python -m app.mcp_server                                    # serve over stdio (set ASSET_CSV in .env)
claude mcp add asset-management -- uv run python -m app.mcp_server  # register, then /mcp to use it
```

Warm the cache first (run the brief online once) so the offline tools have prices. The server runs no LLM itself — an assistant calls it; this is not financial advice.

## Narration (optional plain-language summary)

`--narrate` adds a short **SUMMARY** in plain English at the top of the brief — what happened to your drawdown, risk, and return, in sentences. It's **opt-in and off by default**, and it's built so a language model can never put a wrong number in your brief: the model writes only prose with `{{placeholder}}` tokens, and the tool substitutes the *validated* figures from its own core — rejecting the whole summary if the model tries to write any number itself, or names a figure that doesn't exist. The wording is the model's; **every figure is the tool's**, and the block is labeled with the model that produced it. It is a description, not financial advice.

You bring your own LLM key in `.env` — nothing is sent anywhere unless you turn this on:

```bash
ASSET_NARRATE_PROVIDER=anthropic        # or: openai (any OpenAI-compatible endpoint)
ASSET_NARRATE_MODEL=claude-haiku-4-5
ASSET_NARRATE_KEY=sk-...
# for provider=openai, also set the endpoint (https required; http only for localhost):
ASSET_NARRATE_BASE_URL=https://api.groq.com/openai/v1
ASSET_NARRATE_TIER=paid                  # free | paid | local — privacy dial (see below); default: free
```

**Privacy dial.** The `tier` controls what leaves your machine. On **`free`** (the default — for keys from providers whose free tiers may train on your inputs) only *coarse qualitative bands* ("moderate", "solid") are sent; your exact dollar amounts, returns, and dates stay home and are filled in locally. On **`paid`** (providers that contractually don't train on your data) the exact figures are sent for richer wording. A third tier, **`local`**, is for a model running on your own machine (Ollama / llama.cpp at `http://localhost` — also `::1` or `host.docker.internal`): it sends exact figures too, since nothing leaves the machine, and it's honored only against a genuine local endpoint (otherwise it falls back to `free`). The dial **fails safe**: only an explicit `paid` or `local` ever sends exact values — a blank or misspelled tier stays on `free`, and on `free` the tool logs a one-line reminder that the provider may train on what it is sent. If narration isn't configured, or the model call fails, the brief simply prints without the SUMMARY.

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
│   ├── corporate_actions.py  split-adjust raw share counts (stock splits)
│   ├── prices.py     multi-source price fetch (latest + history + splits) with provenance + cache
│   ├── metadata.py   published fund facts (expense ratio, AUM, volume, age, holdings) + cache
│   ├── returns.py    events + prices → equity curve, true TWR, MWR (XIRR), Modified Dietz
│   ├── risk.py       drawdown family + Sharpe/Sortino/Calmar with bootstrap CIs
│   ├── strategy.py   holdings + target → named buy/sell suggestions (rebalance modes) + edge gate
│   ├── allocate.py   choose a target: equal_weight / inverse_vol + per-asset caps
│   ├── screen.py     judge NEW candidate tickers (diversifier / cost / liquidity / age / overlap)
│   ├── backtest.py   notional rebalanced-vs-buy-and-hold simulation
│   ├── report.py     suggestions + backtest + state + prices + returns + risk → ReportData → text/markdown/HTML
│   ├── narrate.py    fenced narration: validated figures → {{token}} prose → SUMMARY (pure; no number can be the model's)
│   ├── llm.py        optional narrator backend (OpenAI-compatible + Anthropic); fail-closed, opt-in
│   ├── email.py      send the HTML brief via Resend (--send)
│   ├── cli.py        argparse + entry composition + delivery routing + structured run log
│   ├── mcp_server.py read-only stdio MCP server: 3 tools over the core (offline, no network)
│   ├── log_config.py logging setup
│   ├── __main__.py   python -m app
│   └── __init__.py
├── tests/            automated suite (unit, property, regression) — offline, run on every change
├── reconcile/        manual cross-validation against external tools (ghostfolio, quantstats)
├── pyproject.toml    dependencies, mypy/ruff config
└── README.md
```

## Validation

The numbers are cross-checked against two independent tools on the bundled example data (harness in [`reconcile/`](reconcile/)):

- **ghostfolio** reconstructs holdings, market value, and P&L from the same transaction log and matches **to the cent**.
- **quantstats** independently computes Sharpe, Sortino, and max drawdown from the return series, matching **to 4 decimals**.

Every formula the tool computes — returns, the drawdown family (Ulcer / CDaR), risk-adjusted ratios, bootstrap confidence bands, and the allocation/screening math — is defined in one place: [MATH.md](MATH.md).

Together they validate the whole pipeline: ghostfolio confirms the holdings/value reconstruction; quantstats confirms the risk/return formulas. Full comparison in [`reconcile/RESULTS.md`](reconcile/RESULTS.md).

```bash
uv run --with quantstats python reconcile/reconcile_quantstats.py   # metric cross-check
# end-to-end (ghostfolio): see reconcile/RESULTS.md
```

## License

Personal project, no license granted. All rights reserved.
