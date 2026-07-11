# asset-management

[![gate](https://github.com/disin7c9/asset-management/actions/workflows/ci.yml/badge.svg)](https://github.com/disin7c9/asset-management/actions/workflows/ci.yml)

**TL;DR:** `uvx --from git+https://github.com/disin7c9/asset-management asset-management --demo` — a drawdown-first portfolio brief on a bundled example book, one command, no setup. USD-only, long-only stock/ETF; you keep your own transaction log.

**Track a personal stock/ETF portfolio and get suggestions you can audit. Python computes every number; the optional AI narrates — and is structurally unable to write a figure of its own.**

Many AI finance tools let the model produce the numbers. Here the model may only place `{{token}}` placeholders: a deterministic renderer substitutes figures from the validated core and **refuses the entire narration if the model typed even one digit itself**:

```text
 the model wrote                             │  the reader gets
─────────────────────────────────────────────┼─────────────────────────────────────────────
 "Your deepest stretch fell                  │  "Your deepest stretch fell -9.84% from
  {{max_drawdown}} from its peak, and an     │   its peak, and an ulcer index of 2.38%
  ulcer index of {{ulcer}} says the ride     │   says the ride stayed shallow."
  stayed shallow."                           │  ✔ every figure substituted from the
                                             │    validated core
─────────────────────────────────────────────┼─────────────────────────────────────────────
 "Your portfolio fell 12% but recovered      │  (no summary at all)
  nicely — don't worry."                     │  ✘ REFUSED — one model-typed digit voids
                                             │    the entire note; the brief prints
                                             │    without it
```

Run both sides yourself — it drives the real production fence, no API key needed: `uv run python scripts/demo_fence.py`

## What you get

- A **drawdown-first brief** of your real holdings — how far you fell from your peak, how long underwater, what came back — with a bootstrap confidence band on every sampled risk statistic (returns are accounting identities, so they honestly carry none).
- **Deterministic buy/sell suggestions** toward a target you choose, each line paired to the **named rule** that produced it — you learn the rule rather than trust a bot.
- Optional, fenced **AI narration**, and a read-only **Claude Desktop addon** ("chat with your portfolio") over the same validated core.

Holdings are *derived* from an append-only transaction log (date, ticker, action, quantity, price, fee per row) — never stored — so the same input always produces the same output. Prices are fetched with a provider fallback (Yahoo Finance primary; Tiingo secondary, via a free API key) and an on-disk cache; every displayed number is traceable to its source, and figures that can't be computed honestly (too short a window, no real solution) print `n/a` rather than a fabricated number. **Stock splits are adjusted automatically** (share counts are reconciled with the split-adjusted price history), so a split during your holding period doesn't distort the returns.

## Try it in 60 seconds (bundled fake portfolio, no setup)

```bash
uvx --from git+https://github.com/disin7c9/asset-management asset-management --demo
```

or from a clone: `uv sync && uv run python -m app --demo` (needs Python 3.12 and [uv](https://docs.astral.sh/uv/)).

**The full tour** (still the bundled book, ~a minute online) — the two commands below show the tool's characteristic features end to end: a preset target is *proposed*, then *validated* against a known 60-40 reference with a walk-forward held-out verdict, and threshold-band rebalance suggestions are laid out with the named rule behind every line, plus per-fund facts:

```bash
uvx --from git+https://github.com/disin7c9/asset-management asset-management \
    --demo --allocate moderate --allocate-out demo_target.csv
uvx --from git+https://github.com/disin7c9/asset-management asset-management \
    --demo --backtest --target demo_target.csv --benchmark 60-40 --rebalance bands --metadata
```

Note what it *doesn't* say: the verdict reads like "no clear drawdown difference from 60-40; the paired bootstrap does not confirm the gap" when the evidence is thin — never "beats the benchmark". When the output earns your trust, the four steps below point it at your own money.

## Use it with your own money — four steps

### Step 1 — your book (the input)

Everything is derived from one transaction log. The CSV format is **Ghostfolio's own CSV-import schema** (so a book you keep here imports straight into Ghostfolio too) — columns `Date, Code, DataSource, Currency, Price, Quantity, Action, Fee, Note`. `Action` is one of `buy`, `sell`, `dividend`, `fee`, `interest`, `deposit`, `withdraw`. Cash flows (`deposit`/`withdraw`) use a `CASH` code and put the amount in the `Price` column. **This tool is USD-only** (long-only stock/ETF): `Currency` must be `USD` — a non-USD row is refused with an error naming the row, never silently booked as dollars 1:1. Empty cells in numeric columns are treated as zero. Non-ISO dates are rejected with a clear error. UTF-8 BOM is tolerated. The bundled example ([data/sample_data/transactions.csv](data/sample_data/transactions.csv)) shows every row type.

**Already use [Ghostfolio](https://ghostfol.io)?** Point the input straight at a Ghostfolio **JSON** export (Portfolio → Activities → ⋯ → Export) — the loader detects it and reads it directly, no conversion step. It reads the activities (a dividend's cash = `quantity × unitPrice`; Ghostfolio's UTC timestamps are rounded back to your local date) and skips non-USD, crypto, and non-security (`ITEM`/`LIABILITY`) rows with a warning (USD-only, long-only equity/ETF for now). Brokers without a native Ghostfolio account can run a community converter such as [Export-To-Ghostfolio](https://github.com/dickwolff/Export-To-Ghostfolio) (26 brokers → a Ghostfolio JSON) first.

```bash
uv run python -m app --book your.csv                     # or ghostfolio-export.json — auto-detected
uv run python -m app --book your.csv --dry-run           # preview an import BEFORE trusting it:
                                                         # format, events, skipped rows with reasons,
                                                         # derived holdings — fetches nothing
```

**Set your default once** in a gitignored `.env` at the repo root — `ASSET_BOOK=path/to/your.csv` (and optionally `ASSET_TARGET=path/to/target.csv`) — and a bare `python -m app` becomes your brief. Explicit flags always win. There is **no silent built-in default**: without `--book` or `ASSET_BOOK`, a book-dependent action errors out and a bare run prints a hint — the bundled example is opt-in (`--demo`), never assumed.

### Step 2 — warm the cache (once)

The core is offline-first: prices come from an on-disk cache, refreshed when you run online. After a fresh clone, fill it once:

```bash
uv run python -m app --book your.csv --warm        # your tickers + the benchmark references
uv run python -m app --book your.csv --warm full   # + the ~375-ETF discovery universe (slow) —
                                                   # only needed for offline --discover
```

After that, `--offline` runs and the Claude Desktop addon serve entirely from this cache (`--cache-dir` / `ASSET_CACHE_DIR` override the location).

**Optional but recommended:** add a free [Tiingo](https://www.tiingo.com) API key to `.env` — `TIINGO_API_KEY=...` — to enable the second price source. Yahoo Finance throttles bursts of requests now and then; with a key the fetch falls back to Tiingo instead of reporting tickers missing. Without one, the tool fetches from Yahoo only.

### Step 3 — choose a target

Most of the decision features (`--rebalance`, `--backtest`, the walk-forward checks) work toward a **target allocation** — a small CSV (`Ticker,Weight`) that *you* own and edit. Three ways to get one, by where you're starting from:

- **You already hold a portfolio** → `--dump-target target.csv` writes your **current** allocation; edit it toward the mix you want. (A target is a *complete spec* — see [Rebalance modes](#rebalance-modes) for the exit semantics.)
- **Start from a risk posture** → `--onboard` asks three plain questions in the terminal (horizon / loss response / cash buffer) and builds the matched preset; or pick it yourself with `--allocate conservative|moderate|aggressive`. Save with `--allocate-out target.csv`. These are strategic **role-bucket templates** (stocks / bonds / diversifiers split by posture, core-satellite within each bucket): a role you already hold resolves to *your* largest fund in it, a role you're missing is filled with a sensible default ETF from the curated universe.
- **Explore and build it yourself** → `--discover` suggests screened ETFs for the roles you're light in; `--screen QQQM,SCHD` judges any candidate against your book (cost, liquidity, age, overlap, and whether it actually diversified *your* worst drawdowns). Then write the CSV by hand.

Two simpler re-weighting rules also exist — `--allocate equal_weight` (1/N; the robust baseline) and `--allocate inverse_vol` (each holding contributes roughly the same *risk* rather than the same dollars; cap any single weight with `--allocate-cap 0.30`). Deliberately **not** included: return-forecasting optimizers (mean-variance / max-Sharpe) — they overfit, so they stay behind an *edge* gate for a later version.

**Validate the target before following it:**

```bash
uv run python -m app --backtest --target target.csv --benchmark 60-40   # or all-weather | permanent
```

simulates your target and the reference over their common history (notional $10k, drawdown-first legs with CIs) and adds a **walk-forward held-out verdict**: `shallower` / `deeper` / `inconclusive` / `insufficient` — never "beats". The verdict is judged on the **Ulcer index** (whole-window drawdown pain — how deep *and* how long), with **CDaR** (the worst-tail average) required not to contradict it and a paired-bootstrap confidence interval to confirm; **max drawdown is reported as context, not the decider** — one worst event is too noisy to decide anything on a short history. When it can't call it, it says `inconclusive` and names exactly which gate blocked it.

**Propose, simulate, and act are separate steps**, enforced: `--allocate`/`--onboard` only *propose* (and optionally write the file) and **may not be combined** with `--backtest`/`--rebalance` in one command. You review the file, then simulate it, then ask for orders — each in its own command. A strategy never silently becomes trades.

### Step 4 — the weekly brief

One command covers the routine check-in — the status brief, suggested actions when a band is breached, the backtest + benchmark verdict, and fund facts:

```bash
uv run python -m app --book your.csv --backtest --target target.csv \
  --benchmark permanent --rebalance bands --metadata        # + --narrate --save --send to taste
```

- Pick the `--benchmark` reference nearest **your** posture (`60-40` / `all-weather` / `permanent`).
- `bands` acts only when a holding drifts past its band (the "5/25 rule") and **ignores `--new-cash`** — on a week you're depositing, swap in `--rebalance cash_flow_only --new-cash 500` (invest into underweights, never sell; tax-friendly).
- Panels are **composable** — every flag adds a panel and they stack (see the [Reference](#reference)).

The same report leaves three ways from one build: plain text on stdout (always), markdown to `reports/<asof>.md` (`--save`), and an HTML email (`--send`; `RESEND_API_KEY` + `REPORT_TO` in `.env`). A failed sink never crashes the run — the brief still prints and the failure is logged — but if a sink you *requested* fails, the process exits non-zero so a scheduler notices. A weekly Monday brief is just this on cron:

```cron
# 08:00 every Monday — email the brief and archive the markdown.
# cron's PATH is minimal and won't find uv — point PATH at it (see `which uv`):
PATH=/home/you/.local/bin:/usr/bin:/bin
0 8 * * 1  cd /path/to/asset-management && uv run python -m app --book your.csv --backtest --target target.csv --benchmark permanent --rebalance bands --metadata --save --send >> reports/cron.log 2>&1
```

(cron only fires while the machine is on at that moment — on WSL, Windows Task Scheduler running `wsl.exe` is the always-fires alternative.)

## Not financial advice — structurally, not as fine print

The *shape* of the output is what makes this a description rather than advice:

- every suggestion is paired to the **named rule** that fired ("5/25 band breached — trim X"), never a bare "buy X";
- every sampled risk metric carries a **confidence interval**, and too-little-data says so (`inconclusive`, `insufficient`, `n/a`) instead of pretending;
- benchmark verdicts only say **`shallower` / `deeper` / `inconclusive` / `insufficient`** — the vocabulary has no "beats";
- anything claiming an *edge* must pass a **walk-forward (out-of-sample) gate** before it may surface a suggestion — in-sample-only numbers are refused by design;
- the tool is **read-only**: it never trades, and the AI can never write to your ledger.

## Correctness is a claim you can check

- the [gate](.github/workflows/ci.yml) runs the full test suite + `mypy --strict` + ruff on every push to `main` and every PR — the badge at the top is that gate, live;
- holdings, market value, and P&L are reconciled **to the cent** against [Ghostfolio](https://ghostfol.io), and Sharpe/Sortino/drawdown **to 4 decimals** against quantstats → [reconcile/RESULTS.md](reconcile/RESULTS.md);
- every formula is written down in [MATH.md](MATH.md);
- the number fence is a script you can poke yourself: [`scripts/demo_fence.py`](scripts/demo_fence.py).

## Claude Desktop addon — chat with your portfolio (read-only MCP)

Expose your portfolio to an AI assistant (Claude Desktop, Claude Code, …) as **read-only tools** it can call — so you can "chat with your portfolio" while every number still comes from the validated core, not the model. The server is **read-only** — no write tools, bound to your `ASSET_BOOK` book (no file-path args) — and offline **except two bounded, opt-out fetches**: a cold cache **auto-warms the core set once** (your tickers + benchmark refs, ~30–60s), and screening/proposing a ticker that isn't cached fetches it on demand. Set `ASSET_MCP_OFFLINE=1` to disable both (for an *already-warmed* cache; pointed at a cold one it just degrades to honest `n/a`). Eight tools:

- **`portfolio_summary`** — holdings, P&L, and annualized returns.
- **`risk_report`** — drawdown-first risk: max drawdown (depth/dates/recovery), Ulcer, CDaR, Sharpe/Sortino/Calmar, all with bootstrap confidence intervals.
- **`rebalance_check`** — buy/sell/hold suggestions toward your `ASSET_TARGET` (it suggests, never trades; refuses to size over a partially-cached book).
- **`securities_facts`** — published fund facts per holding (expense ratio, AUM, volume, age, category).
- **`discover_gaps`** — suggest NEW ETFs for the roles you hold ≤3% of (propose-only; needs `--warm full`).
- **`screen_candidate`** — judge a NEW candidate ticker against your book (diversifier/cost/liquidity/age/overlap, each with a reason).
- **`propose_allocation`** — a strategic target for a posture (`conservative`/`moderate`/`aggressive`) over your book + the universe, validated against a reference with the same walk-forward, Ulcer-first held-out verdict — propose-only, numbers from the core, never a recommendation.
- **`starter_allocation`** — new to this? answer three plain risk questions → a starting posture and its validated proposal (the onboarding path into `propose_allocation`).

**One-click install (Claude Desktop):** grab `asset-management-<version>.mcpb` from the
[Releases page](https://github.com/disin7c9/asset-management/releases) (or build it yourself:
`uv run python scripts/build_mcpb.py`), then Claude Desktop → Settings → Extensions →
**Install Extension…** → pick the file. The install dialog asks for your transaction file
(pre-filled with the **bundled demo portfolio** so you can explore on fake data first), a
price-cache folder, an optional target CSV, a strict-offline toggle, and an optional
[Tiingo](https://www.tiingo.com) API key (the second price source for when Yahoo throttles).
Nothing else to install — Claude Desktop's `uv` runtime resolves Python and the locked
dependencies itself.

**First launch is slow — this is expected.** On install, that `uv` runtime builds a ~360 MB
Python environment (pandas / NumPy / PyArrow / SciPy) before the server can answer, so give it
**2–5 minutes the first time** and don't worry if the extension briefly shows as *disconnected*
while it works. Every launch after that starts in under a second; the first tool call then warms
the price cache once (~30–60s), and that cache lives in your home folder so it survives reinstalls.

**In chat:** open the **+** menu for ready-made starters — *Portfolio checkup*, *What's my
drawdown?*, *Should I rebalance?*, *Fill my gaps*, *Find my starting allocation*, *Propose a
posture* — each one pre-loads the figures-only framing. The server also publishes `portfolio://guarantees` (its four
enforced guarantees, versioned, shipped with the code): attach it from the same **+** menu —
or, in clients that let the model read resources itself, just ask "can I trust these
numbers?" and it answers from the manifest instead of improvising.

**If the extension says "Unable to connect":** check `%APPDATA%\Claude\logs\main.log` —
- `Failed to read version of python binary "python3"/"python" … 9009`: your Claude
  Desktop build refuses to start uv extensions without *a* system Python answering on
  PATH (even though the extension ships its own). Install any Python 3 (`winget install
  Python.Python.3.12`) **and** turn OFF the two `python` App Execution Aliases
  (Settings → Apps → Advanced app settings → App execution aliases) — Microsoft's Store
  stubs otherwise keep answering the probe with garbage. Restart Claude Desktop.
- `No MCP config found for extension … skipping`: Desktop only launches the server once
  the Configure form has been **saved**, and it ignores the pre-filled defaults until
  then — but the Save button stays disabled if you change nothing. Toggle any field
  (e.g. *Strictly offline* on and off) so Save activates, save, restart Claude Desktop.
- An *install* failing with `os error 32` (file in use): don't rapid-retry — fully quit
  Claude Desktop, re-open, install once; or use the registration route below.
- Extension connects and the **+** starters attach, but the model says it has no tools
  and the tools menu is empty: the extension itself is fine — check your Claude Desktop
  configuration for the signed-in *account* (Settings → Extensions and the chat's tools
  menu). In our testing the same machine behaved differently under two accounts on the
  same plan, so if another account is available, trying it isolates the problem fast.

Or run the server directly / register it with Claude Code:

```bash
uv run python -m app.mcp_server                                    # serve over stdio (set ASSET_BOOK in .env)
claude mcp add asset-management -- uv run python -m app.mcp_server  # register, then /mcp to use it
```

The server runs no LLM itself — an assistant calls it; this is not financial advice.

## Narration (optional plain-language summary)

`--narrate` adds a short **SUMMARY** in plain English at the top of the brief — what happened to your drawdown, risk, and return, in sentences. It's **opt-in and off by default**, and it's built so a language model can never put a wrong number in your brief: the model writes only prose with `{{placeholder}}` tokens, and the tool substitutes the *validated* figures from its own core — rejecting the whole summary if the model tries to write any number itself, or names a figure that doesn't exist. The wording is the model's; **every figure is the tool's**, and the block is labeled with the model that produced it. The same fence narrates the discovery panel (`--discover --narrate`: the model ranks and explains the screened picks by role-fit, never forecasts) and the benchmark verdict (`--backtest --benchmark … --narrate`). It is a description, not financial advice.

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

## Reference

The report is **composable panels**, not exclusive modes — combine flags and the panels stack. What each action needs:

| action | needs | what it does |
|---|---|---|
| status brief (default) | `--book` | your holdings + returns + drawdown/risk |
| `--rebalance MODE` | `--book` + `--target` | buy/sell suggestions toward the target (`--new-cash` sizes a deposit) |
| `--allocate RULE` | `--book` | propose a target — re-weight your holdings (`equal_weight`/`inverse_vol`) or build a strategic role template (`conservative`/`moderate`/`aggressive`); write it with `--allocate-out` |
| `--onboard` | `--book` (or `--demo`) | step 0 for a new user: answer 3 plain risk questions in the terminal → the matched posture builds its `--allocate` preset automatically (propose-only; save with `--allocate-out`) |
| `--dry-run` | `--book` (or `--demo`) | preview an import before trusting it: detected format, events parsed, rows skipped/flagged with reasons, and the holdings they derive to — fetches nothing, computes no brief |
| `--metadata` | `--book` | published fund facts per holding (expense ratio, AUM, volume, age, category), cached 7 days |
| `--screen TICKERS` | `--book` + prices | judge NEW candidates vs your book: diversifier (incl. your red days + worst drawdown), cost, liquidity, age, concentration, leveraged/inverse auto-reject, holdings-overlap dedup — each verdict with its reason. Add `--target` for the **walk-forward role check**: did a 5% sleeve reduce drawdown pain (Ulcer, with a CDaR check) on a held-out window? "Inconclusive" names the gate that blocked it. Propose-only; a PASS is "sane, cheap, liquid, genuinely different", never a prediction |
| `--discover [roles]` | `--book` + prices | suggest **new** ETFs for the roles you hold ≤3% of, run through the same screen — propose-only (see [Discovery](#discovery--the-curated-universe)) |
| `--backtest` | `--target` | notional rebalance-vs-buy-and-hold — **no `--book`**; prints the simulation alone |
| `--backtest --benchmark REF` | `--target` | validate a target vs a canonical reference (`60-40` / `all-weather` / `permanent`) — drawdown-first legs + the walk-forward, Ulcer-first held-out verdict |
| `--narrate` | `--book` + an LLM key | a plain-language **SUMMARY** at the top of the brief; the model writes only the words, every number is substituted and verified from the core (opt-in, off by default — see [Narration](#narration-optional-plain-language-summary)) |

All flags, grouped:

```bash
# input & cache
uv run python -m app --demo                        # zero-setup test drive on the bundled example book
uv run python -m app --book your.csv               # your book (CSV or Ghostfolio JSON, auto-detected)
uv run python -m app --book your.csv --dry-run     # preview an import — fetches nothing
uv run python -m app --book your.csv --warm        # ONE-TIME: fill the offline cache (add `full` for the universe)
uv run python -m app --book your.csv --offline     # serve from the on-disk cache; no network
uv run python -m app --book your.csv --no-prices   # holdings + realized P&L only (no network)
uv run python -m app --book your.csv --cache-dir /some/dir   # override the cache location
# panels (stack freely)
uv run python -m app --book your.csv --no-risk     # skip the holdings drawdown/risk panel
uv run python -m app --book your.csv --metadata    # + SECURITIES panel (fund facts)
uv run python -m app --book your.csv --screen QQQM,SCHD            # judge NEW candidates
uv run python -m app --book your.csv --screen SCHD --target t.csv  # + the walk-forward role check
uv run python -m app --book your.csv --discover    # + DISCOVERY panel (gap-filling ETFs)
uv run python -m app --book your.csv --narrate     # + the fenced plain-language SUMMARY
# targets (propose-only)
uv run python -m app --book your.csv --dump-target target.csv                     # write your CURRENT allocation
uv run python -m app --book your.csv --allocate moderate --allocate-out t.csv     # strategic preset
uv run python -m app --book your.csv --allocate inverse_vol --allocate-out t.csv  # re-weight holdings
uv run python -m app --demo --onboard                                             # 3-question quiz → a preset
# act & validate (separate commands, by design)
uv run python -m app --book your.csv --rebalance to_total --target target.csv     # suggestions toward the target
uv run python -m app --book your.csv --rebalance cash_flow_only --target target.csv --new-cash 1000
uv run python -m app --backtest --target target.csv                # notional backtest — target-only, no --book
uv run python -m app --backtest --target target.csv --benchmark 60-40   # + the reference comparison & verdict
uv run python -m app --book your.csv --rebalance bands --backtest --target target.csv  # panels stack
# delivery
uv run python -m app --book your.csv --save        # also write reports/<asof>.md (markdown)
uv run python -m app --book your.csv --send        # also email the brief as HTML via Resend
```

A `target.csv` is one you create with `--dump-target` / `--allocate-out` (or use `data/sample_data/target.csv`). `--allocate` is **propose-only** and cannot be combined with `--rebalance`/`--backtest`.

### Rebalance modes

A **target is a complete spec** (`--target path`, columns `Ticker,Weight`; weights are relative and normalized): any held ticker **not** listed is treated as an exit and sold to $0. So `--target` is *required* with `--rebalance` (no silent default), and the run **warns** listing any held tickers the target omits. To **close a position on purpose**, give it weight `0` — that's an explicit, warning-free exit; *omitting* it does the same but triggers the safety warning (the tool can't tell "forgot" from "meant it"). Modes:

- `to_total` — sell + buy to hit the target exactly (cash-neutral; deploys `--new-cash` too)
- `cash_flow_only` — invest `--new-cash` into underweights; never sell (tax-friendly)
- `fixed_dca` — buy the target mix with `--new-cash`, ignoring drift
- `bands` — like `to_total` but only act when a ticker's drift exceeds its band; rebalances existing holdings only (ignores `--new-cash`). The band is the **smaller** of an absolute `--band` (default 5pp) or `--band-rel` × the ticker's target weight (the **"5/25 rule"**, default 25%) — so a small sleeve isn't handed a band many times its own size; a 0% target → 0 band → always exits

### Backtest details

`--backtest --target T.csv` runs a **notional $10,000** historical simulation of that target and prints a **BACKTEST** panel comparing **rebalanced** (schedule via `--rebalance-every {monthly,quarterly,annually}`, default quarterly) vs **buy-and-hold** — drawdown-first (max drawdown, Ulcer, CDaR — each with a bootstrap CI), plus Sharpe/Sortino and returns. It's *notional*: it starts a clean $10k at the target weights on the earliest date all tickers have prices (`--backtest-start` to override), so it tests the *strategy*, independent of your actual buy timing. Labeled **a historical simulation, not a prediction**.

A fixed rebalance policy fits no parameters, so the whole history is out-of-sample-clean (nothing to overfit). The **walk-forward train/test *selection*** machinery — needed only once a strategy *searches* (tunes parameters or picks among candidates: an optimizer, or an *edge* timing strategy) — is deliberately deferred; a **discipline-vs-edge gate** enforces that any future edge strategy must pass a walk-forward backtest before it may surface a suggestion. Today's rebalance modes are all *discipline*, so they suggest freely.

With `--benchmark`, the held-out verdict resolves through three named gates: the **Ulcer gain** must clear a noise margin, **CDaR** (the worst-tail average) may tie but must not contradict the direction, and a **paired moving-block bootstrap CI** must confirm it — otherwise `inconclusive`, with the blocking gate named in the reason. Max drawdown and volatility are reported as context, voting nowhere: a single worst event is an extreme-value statistic, far too noisy on a short history to decide anything.

### Example output

The shape `--demo` prints (drawdown leads, then risk-adjusted ratios, then returns, then holdings):

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

Confidence bands come from a moving-block bootstrap. Drawdown is *investment* (time-weighted) drawdown, not account-balance drawdown. The panel also reports **Gains given back** — the largest dollar decline in your cumulative market profit (the felt "how much did I watch evaporate"); it's flow-neutral, so deposits, withdrawals, and transfers don't distort it. Each run also emits one structured JSON log line (`run_summary`) on stderr: `{date, source, n_events_replayed, n_prices_fetched, n_prices_missing, n_series_fetched, n_series_missing, fallbacks_used, status, report_saved, email_sent, rebalance, backtest, allocate, dump_target, metadata, screen, narrate, discover, discover_narrate, benchmark_narrate, warm, onboard, dry_run}` (with `email_detail`/`error` present when relevant).

### Discovery & the curated universe

`--discover` maps your holdings to roles (US large, emerging markets, TIPS, REITs, …), finds the roles you hold ≤3% of, takes the biggest funds in each from a **curated universe** (`data/universe.csv`, ~375 low-cost ETFs), and runs them through the **same screen** as `--screen` — printing a DISCOVERY panel, each candidate with its verdict (PASS/WARN/FAIL) and reasons. `--discover reit,tips` limits the roles; add `--narrate` for a fenced note ranking the picks by role-fit. A PASS is "sane, cheap, liquid, and genuinely different from what you hold" — never a prediction.

The universe is **auto-built** (and refreshable) — no hand-maintenance:

```bash
uv run python scripts/build_universe.py --auto --out data/universe.csv
```

It pulls the largest US ETFs per asset-class category from a screener (by fund *size*, not past returns — chasing performance is exactly the trap this avoids), drops leveraged/inverse, and keeps a small curated set for the few categories a screener can't isolate. Point `--discover` at your own list with `ASSET_UNIVERSE=path/to/universe.csv` in `.env`.

## Develop

```bash
uv run pytest                  # unit + property-based + regression tests
uv run mypy app/               # strict type-checking
uv run ruff check app/ tests/  # lint
uv run python scripts/build_mcpb.py  # package the Claude-Desktop addon → dist/asset-management-<v>.mcpb
```

## Layout

```
asset-management/
├── app/
│   ├── events.py     CSV / Ghostfolio-JSON → typed event list
│   ├── derive.py     events → holdings + cost basis + realized P&L
│   ├── corporate_actions.py  split-adjust raw share counts (stock splits)
│   ├── prices.py     multi-source price fetch (latest + history + splits) with provenance + cache
│   ├── metadata.py   published fund facts (expense ratio, AUM, volume, age, holdings) + cache
│   ├── returns.py    events + prices → equity curve, true TWR, MWR (XIRR), Modified Dietz
│   ├── risk.py       drawdown family (max-DD / Ulcer / CDaR) + Sharpe/Sortino/Calmar with bootstrap CIs
│   ├── strategy.py   holdings + target → named buy/sell suggestions (rebalance modes) + edge gate
│   ├── allocate.py   choose a target: equal_weight / inverse_vol + risk-posture presets + per-asset caps
│   ├── onboard.py    step-0 risk quiz (pure): 3 questions → a conservative / moderate / aggressive posture
│   ├── screen.py     judge NEW candidate tickers (diversifier / cost / liquidity / age / overlap)
│   ├── backtest.py   notional simulation + the walk-forward role/benchmark verdicts (Ulcer-first)
│   ├── pipeline.py   the shared book→prices→returns→risk bundle (cli + mcp_server) + cache warm + the --demo book
│   ├── report.py     suggestions + backtest + state + prices + returns + risk → ReportData → text/markdown/HTML
│   ├── narrate.py    fenced narration: validated figures → {{token}} prose → SUMMARY (pure; no number can be the model's)
│   ├── llm.py        optional narrator backend (OpenAI-compatible + Anthropic); fail-closed, opt-in
│   ├── email.py      send the HTML brief via Resend (--send)
│   ├── cli.py        argparse + entry composition + delivery routing + structured run log
│   ├── mcp_server.py read-only stdio MCP server: 8 tools over the core (offline; one-time cold-call auto-warm, ASSET_MCP_OFFLINE=1 opts out)
│   ├── universe.py   curated ETF universe loader (Candidate + roles); data/universe.csv, auto-built
│   ├── discover.py   book → role gaps → top-by-AUM candidates for the screen (--discover, propose-only)
│   ├── log_config.py logging setup
│   ├── __main__.py   python -m app
│   └── __init__.py
├── tests/            automated suite (unit, property, regression) — offline, run on every change
├── reconcile/        manual cross-validation against external tools (ghostfolio, quantstats)
├── scripts/          build_universe.py (refresh the ETF universe) · build_mcpb.py (Claude-Desktop bundle) · demo_fence.py (poke the fence)
├── .github/          the gate: ruff + mypy --strict + pytest on every push to main + every PR
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
