# asset-management

[![gate](https://github.com/disin7c9/asset-management/actions/workflows/ci.yml/badge.svg)](https://github.com/disin7c9/asset-management/actions/workflows/ci.yml) [![asset-management MCP server](https://glama.ai/mcp/servers/disin7c9/asset-management/badges/score.svg)](https://glama.ai/mcp/servers/disin7c9/asset-management)

**TL;DR:** `uvx --from git+https://github.com/disin7c9/asset-management asset-management --demo` — a drawdown-first portfolio brief on a bundled example book, one command, no setup. USD-only, long-only stock/ETF; you keep your own transaction log.

**Track a personal stock/ETF portfolio and get suggestions you can audit. Python computes every number; the optional AI narrates — and is structurally unable to write a figure of its own.**

## 🔢 The number fence

Many AI finance tools let the model produce the numbers. Here the model may only place `{{token}}` placeholders: a deterministic renderer substitutes figures from the validated core and **refuses the entire narration if the model typed any decimal digit itself**:

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

## 🎁 What you get

- A **drawdown-first brief** of your real holdings — how far you fell from your peak, how long underwater, what came back — with a bootstrap confidence band on every sampled risk statistic (returns are accounting identities, so they honestly carry none).
- **Deterministic buy/sell suggestions** toward a target you choose, each line paired to the **named rule** that produced it — you learn the rule rather than trust a bot.
- Optional, fenced **AI narration**, and a read-only **Claude Desktop addon** ("chat with your portfolio") over the same validated core.

Holdings are *derived* from an append-only transaction log (date, ticker, action, quantity, price, fee per row) — never stored — so the same input always produces the same output. Prices are fetched with a provider fallback (Yahoo Finance primary; Tiingo secondary, via a free API key) and an on-disk cache; every displayed number is traceable to its source, and figures that can't be computed honestly (too short a window, no real solution) print `n/a` rather than a fabricated number. **Stock splits are adjusted automatically** (share counts are reconciled with the split-adjusted price history), so a split during your holding period doesn't distort the returns. When the split feed is unavailable, the mismatch detector catches ratios of 2:1 or larger.

## 📑 Contents

- [🚀 Start here](#-start-here)
  - [⚡ Try it in 60 seconds (bundled fake portfolio, no setup)](#-try-it-in-60-seconds-bundled-fake-portfolio-no-setup)
  - [💵 Use it with your own money — four steps](#-use-it-with-your-own-money--four-steps)
- [🔍 Why you can trust the numbers](#-why-you-can-trust-the-numbers)
  - [🚫 Not financial advice — structurally, not as fine print](#-not-financial-advice--structurally-not-as-fine-print)
  - [✅ Correctness is a claim you can check](#-correctness-is-a-claim-you-can-check)
  - [🔬 Validation — reconciled against two independent tools](#-validation--reconciled-against-two-independent-tools)
- [💬 Chat with your portfolio — the Claude Desktop addon (read-only MCP)](#-chat-with-your-portfolio--the-claude-desktop-addon-read-only-mcp)
  - [🛟 If something goes wrong](#-if-something-goes-wrong)
- [📖 Reference](#-reference)
  - [💻 Core brief](#-core-brief)
  - [🔁 Rebalance modes](#-rebalance-modes)
  - [📈 Backtest details](#-backtest-details)
  - [🔭 Discovery & the curated universe](#-discovery--the-curated-universe)
  - [🔍 Screen a candidate](#-screen-a-candidate)
  - [📝 Narration (optional plain-language summary)](#-narration-optional-plain-language-summary)
  - [🔑 Configuration — every key in one place](#-configuration--every-key-in-one-place)
- [🔧 Project](#-project)
  - [🧪 Develop](#-develop)
  - [📁 Layout](#-layout)
- [🔒 Privacy Policy](#-privacy-policy)
- [📜 License](#-license)

## 🚀 Start here

### ⚡ Try it in 60 seconds (bundled fake portfolio, no setup)

```bash
uvx --from git+https://github.com/disin7c9/asset-management asset-management --demo
```

or from a clone: `uv sync && uv run python -m app --demo` (needs Python 3.12 and [uv](https://docs.astral.sh/uv/)).

**The full tour** (still the bundled book, ~a minute online) — the two commands below show the tool's characteristic features end to end: a preset target is *proposed*, then *validated* against a known 60-40 reference with a held-out recent-window verdict, and threshold-band rebalance suggestions are laid out with the named rule behind every line, plus per-fund facts:

```bash
uvx --from git+https://github.com/disin7c9/asset-management asset-management --demo --allocate moderate --allocate-out demo_target.csv
uvx --from git+https://github.com/disin7c9/asset-management asset-management --demo --backtest --target demo_target.csv --benchmark 60-40 --rebalance bands --metadata
```

Note what it *doesn't* say: the verdict reads like "no clear drawdown difference from 60-40; the paired bootstrap does not confirm the gap" when the evidence is thin — never "beats the benchmark". When the output earns your trust, the four steps below point it at your own money.

### 💵 Use it with your own money — four steps

#### Step 1 — your book (the input)

Everything is derived from one transaction log. The CSV format is **Ghostfolio's own CSV-import schema** (so a book you keep here imports straight into Ghostfolio too) — columns `Date, Code, DataSource, Currency, Price, Quantity, Action, Fee, Note`. `Action` is one of `buy`, `sell`, `dividend`, `fee`, `interest`, `deposit`, `withdraw`. Cash flows (`deposit`/`withdraw`) use a `CASH` code and put the amount in the `Price` column. **This tool is USD-only** (long-only stock/ETF): `Currency` must be `USD` — a non-USD row is refused with an error naming the row, never silently booked as dollars 1:1. Empty cells in numeric columns are treated as zero. Non-ISO dates are rejected with a clear error. UTF-8 BOM is tolerated. The bundled example ([data/sample_data/transactions.csv](data/sample_data/transactions.csv)) shows every row type.

**Already use [Ghostfolio](https://ghostfol.io)?** Point the input straight at a Ghostfolio **JSON** export (Portfolio → Activities → ⋯ → Export) — the loader detects it and reads it directly, no conversion step. It reads the activities (a dividend's cash = `quantity × unitPrice`; Ghostfolio's UTC timestamps are rounded back to your local date) and skips non-USD, crypto, and non-security (`ITEM`/`LIABILITY`) rows with a warning (USD-only, long-only equity/ETF for now). Brokers without a native Ghostfolio account can run a community converter such as [Export-To-Ghostfolio](https://github.com/dickwolff/Export-To-Ghostfolio) (26 brokers → a Ghostfolio JSON) first.

```bash
uv run python -m app --book your.csv                     # or ghostfolio-export.json — auto-detected
uv run python -m app --book your.csv --dry-run           # preview an import BEFORE trusting it:
                                                         # format, events, skipped rows with reasons,
                                                         # derived holdings — fetches nothing
```

**Set your default once** in a gitignored `.env` at the repo root — `ASSET_BOOK=path/to/your.csv` (and optionally `ASSET_TARGET=path/to/target.csv`) — and a bare `python -m app` becomes your brief. Explicit flags always win. There is **no silent built-in default**: without `--book` or `ASSET_BOOK`, a book-dependent action errors out and a bare run prints a hint — the bundled example is opt-in (`--demo`), never assumed.

#### Step 2 — warm the cache (once)

The core is offline-first: prices come from an on-disk cache, refreshed when you run online. After a fresh clone, fill it once:

```bash
uv run python -m app --book your.csv --warm        # your tickers + the benchmark references
uv run python -m app --book your.csv --warm full   # + the ~375-ETF discovery universe (slow) —
                                                   # only needed for offline --discover
```

After that, `--offline` runs and the Claude Desktop addon serve entirely from this cache (`--cache-dir` / `ASSET_CACHE_DIR` override the location).

**Optional but recommended:** add a free [Tiingo](https://www.tiingo.com) API key to `.env` — `TIINGO_API_KEY=...` — to enable the second price source. Yahoo Finance throttles bursts of requests now and then; with a key the fetch falls back to Tiingo instead of reporting tickers missing. Without one, the tool fetches from Yahoo only.

#### Step 3 — choose a target

Most of the decision features (`--rebalance`, `--backtest`, the held-out checks) work toward a **target allocation** — a small CSV (`Ticker,Weight`) that *you* own and edit. Three ways to get one, by where you're starting from:

- **You already hold a portfolio** → `--dump-target target.csv` writes your **current** allocation; edit it toward the mix you want. (A target is a *complete spec* — see [Rebalance modes](#-rebalance-modes) for the exit semantics.)
- **Start from a risk posture** → `--onboard` asks three plain questions in the terminal (horizon / loss response / cash buffer) and builds the matched preset; or pick it yourself with `--allocate conservative|moderate|aggressive`. Save with `--allocate-out target.csv`. Both run **with no book at all** — before your first trade, every role fills from the curated universe. These are strategic **role-bucket templates** (stocks / bonds / diversifiers split by posture, core-satellite within each bucket): a role you already hold resolves to *your* largest fund in it, a role you're missing is filled with a sensible default ETF from the curated universe.
- **Explore and build it yourself** → `--discover` suggests screened ETFs for the roles you're light in; `--screen QQQM,SCHD` judges any candidate against your book (cost, liquidity, age, overlap, and whether it actually diversified *your* worst drawdowns). Then write the CSV by hand.

Two simpler re-weighting rules also exist — `--allocate equal_weight` (1/N; the robust baseline) and `--allocate inverse_vol` (each holding contributes roughly the same *risk* rather than the same dollars; cap any single weight with `--allocate-cap 0.30`). Deliberately **not** included: return-forecasting optimizers (mean-variance / max-Sharpe) — they overfit, so they stay behind an *edge* gate for a later version.

**Validate the target before following it:**

```bash
uv run python -m app --backtest --target target.csv --benchmark 60-40   # or all-weather | permanent
```

simulates your target and the reference over their common history (notional $10k, drawdown-first legs with CIs) and adds a **held-out recent-window verdict**: `shallower` / `deeper` / `inconclusive` / `insufficient` — never "beats". The verdict is judged on the **Ulcer index** (whole-window drawdown pain — how deep *and* how long), with **CDaR** (the worst-tail average) required not to contradict it and a paired-bootstrap confidence interval to confirm; **max drawdown is reported as context, not the decider** — one worst event is too noisy to decide anything on a short history. When it can't call it, it says `inconclusive` and names exactly which gate blocked it.

**Propose, simulate, and act are separate steps**, enforced: `--allocate`/`--onboard` only *propose* (and optionally write the file) and **may not be combined** with `--backtest`/`--rebalance` in one command. You review the file, then simulate it, then ask for orders — each in its own command. A strategy never silently becomes trades.

#### Step 4 — the weekly brief

One command covers the routine check-in — the status brief, suggested actions when a band is breached, the backtest + benchmark verdict, and fund facts:

```bash
uv run python -m app --book your.csv --backtest --target target.csv --benchmark permanent --rebalance bands --metadata        # + --narrate --save --send to taste
```

- Pick the `--benchmark` reference nearest **your** posture (`60-40` / `all-weather` / `permanent`).
- `bands` fires only when some holding has drifted past its band (the "5/25 rule"), then rebalances the whole book — on a week you're depositing and don't want to sell, swap in `--rebalance cash_flow_only --new-cash 500` (invest into underweights only; tax-friendly).
- Panels are **composable** — every flag adds a panel and they stack (see the [Reference](#-reference)).

The same report leaves three ways from one build: plain text on stdout (always), markdown to `reports/<asof>.md` (`--save`), and an HTML email (`--send`; `RESEND_API_KEY` + `REPORT_TO` in `.env`). A failed sink never crashes the run — the brief still prints and the failure is logged — but if a sink you *requested* fails, the process exits non-zero so a scheduler notices. A weekly Monday brief is just this on cron:

```cron
# 08:00 every Monday — email the brief and archive the markdown.
# cron's PATH is minimal and won't find uv — point PATH at it (see `which uv`):
PATH=/home/you/.local/bin:/usr/bin:/bin
0 8 * * 1  cd /path/to/asset-management && uv run python -m app --book your.csv --backtest --target target.csv --benchmark permanent --rebalance bands --metadata --save --send >> reports/cron.log 2>&1
```

(cron only fires while the machine is on at that moment — on WSL, Windows Task Scheduler running `wsl.exe` is the always-fires alternative.) Every key the tool reads is listed in [Configuration](#-configuration--every-key-in-one-place).

On Windows without WSL, everything above works unchanged (`uv` brings its own Python — install it with `irm https://astral.sh/uv/install.ps1 | iex`, clone, write `.env`, run). Only the scheduler differs:

```powershell
# the Windows counterpart of the cron line above, from your clone.
# -StartWhenAvailable: a Monday the machine was asleep runs on wake instead of skipping.
Register-ScheduledTask -TaskName "asset-brief" `
  -Trigger (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At 8:00am) `
  -Settings (New-ScheduledTaskSettingsSet -StartWhenAvailable) `
  -Action (New-ScheduledTaskAction -Execute "$env:USERPROFILE\.local\bin\uv.exe" `
    -Argument "run python -m app --book your.csv --backtest --target target.csv --benchmark permanent --rebalance bands --metadata --save --send" `
    -WorkingDirectory "C:\path\to\asset-management")
```

## 🔍 Why you can trust the numbers

### 🚫 Not financial advice — structurally, not as fine print

The *shape* of the output is what makes this a description rather than advice:

- every suggestion is paired to the **named rule** that fired ("5/25 band breached — trim X"), never a bare "buy X";
- every sampled risk metric carries a **confidence interval**, and too-little-data says so (`inconclusive`, `insufficient`, `n/a`) instead of pretending;
- benchmark verdicts only say **`shallower` / `deeper` / `inconclusive` / `insufficient`** — the vocabulary has no "beats";
- anything claiming an *edge* must pass an **out-of-sample gate** before it may surface a suggestion — in-sample-only numbers are refused by design;
- the tool is **read-only**: it never trades, and the AI can never write to your ledger.

### ✅ Correctness is a claim you can check

- the [gate](.github/workflows/ci.yml) runs the full test suite + `mypy --strict` + ruff on every push to `main` and every PR — the badge at the top is that gate, live;
- holdings, market value, and P&L are reconciled **to the cent** against [Ghostfolio](https://ghostfol.io), and Sharpe/Sortino/drawdown **to 4 decimals** against quantstats → [reconcile/RESULTS.md](reconcile/RESULTS.md);
- every formula is written down in [MATH.md](MATH.md);
- the number fence is a script you can poke yourself: [`scripts/demo_fence.py`](scripts/demo_fence.py).

### 🔬 Validation — reconciled against two independent tools

The numbers are cross-checked against two independent tools on the bundled example data (harness in [`reconcile/`](reconcile/)):

- **ghostfolio** reconstructs holdings, market value, and P&L from the same transaction log and matches **to the cent**.
- **quantstats** independently computes Sharpe, Sortino, and max drawdown from the return series, matching **to 4 decimals**.

Every formula the tool computes — returns, the drawdown family (Ulcer / CDaR), risk-adjusted ratios, bootstrap confidence bands, and the allocation/screening math — is defined in one place: [MATH.md](MATH.md).

Together they validate the whole pipeline: ghostfolio confirms the holdings/value reconstruction; quantstats confirms the risk/return formulas. Full comparison in [`reconcile/RESULTS.md`](reconcile/RESULTS.md).

```bash
uv run --with quantstats python reconcile/reconcile_quantstats.py   # metric cross-check
# end-to-end (ghostfolio): see reconcile/RESULTS.md
```

## 💬 Chat with your portfolio — the Claude Desktop addon (read-only MCP)

[![asset-management MCP server](https://glama.ai/mcp/servers/disin7c9/asset-management/badges/card.svg)](https://glama.ai/mcp/servers/disin7c9/asset-management)

Expose your portfolio to an AI assistant (Claude Desktop, Claude Code, …) as **read-only tools** it can call — so you can "chat with your portfolio" while every number still comes from the validated core, not the model. The server is **read-only** — no write tools, bound to your `ASSET_BOOK` book (no file-path args) — and offline **except two bounded, opt-out fetches**: a cold cache **auto-warms the core set once** (your tickers + benchmark refs, ~30–60s), and screening/proposing a ticker that isn't cached fetches it on demand. Set `ASSET_MCP_OFFLINE=1` to disable both (for an *already-warmed* cache; pointed at a cold one it just degrades to honest `n/a`). Eight tools:

- **`portfolio_summary`** — holdings, P&L, and annualized returns.
- **`risk_report`** — drawdown-first risk: max drawdown (depth/dates/recovery), Ulcer, CDaR, Sharpe/Sortino/Calmar, all with bootstrap confidence intervals.
- **`rebalance_check`** — buy/sell/hold suggestions toward your `ASSET_TARGET` (it suggests, never trades; refuses to size over a partially-cached book).
- **`securities_facts`** — published fund facts per holding (expense ratio, AUM, volume, age, category).
- **`discover_gaps`** — suggest NEW ETFs for the roles you hold ≤3% of; optional `role`/`flavor` args drill one shelf (propose-only; judge a pick with `screen_candidate`).
- **`screen_candidate`** — judge a NEW candidate ticker against your book (diversifier/cost/liquidity/age/overlap, each with a reason).
- **`propose_allocation`** — a strategic target for a posture (`conservative`/`moderate`/`aggressive`) over your book + the universe, validated against a reference with the same held-out recent-window, Ulcer-first verdict — propose-only, numbers from the core, never a recommendation.
- **`starter_allocation`** — new to this? answer three plain risk questions → a starting posture and its validated proposal (the onboarding path into `propose_allocation`).

**Install (Claude Desktop):** Settings → Developer → **Edit Config**, and add:

```json
{
  "mcpServers": {
    "asset-management": {
      "command": "uvx",
      "args": ["--from",
               "https://github.com/disin7c9/asset-management/releases/download/v2.12.0/asset_management-2.12.0-py3-none-any.whl",
               "asset-management-mcp"],
      "env": {
        "ASSET_BOOK": "C:\\path\\to\\your\\transactions.csv",
        "ASSET_TARGET": "C:\\path\\to\\your\\target.csv",
        "TIINGO_API_KEY": "your-free-tiingo-key(optional_secondary_source)"
      }
    }
  }
}
```

Restart Claude Desktop. You need [uv](https://docs.astral.sh/uv/) on your PATH and nothing else —
`uvx` resolves Python and the locked dependencies itself. **The first launch is slow** (it builds
a ~500 MB environment: pandas / NumPy / PyArrow), so give it a minute; every launch after starts
in seconds. The first tool call then warms the price cache once (~30–60s), and that cache lives
in your home folder, so it survives reinstalls. Works on the free plan. The URL pins a release —
nothing changes under you between launches; to upgrade, swap both version numbers for the
[newest release](https://github.com/disin7c9/asset-management/releases/latest).

**Every `env` entry is optional — delete what you don't use.** Drop `ASSET_BOOK` to explore the
**bundled demo portfolio** on fake data first. `ASSET_TARGET` is what `rebalance_check` compares
your holdings against (without it, that one tool errors with a hint). `TIINGO_API_KEY` (free
account) adds the second price source for when Yahoo throttles. Rarer: `ASSET_CACHE_DIR` (move
the price cache), `ASSET_MCP_OFFLINE=1` (never fetch — for an already-warmed cache). **No LLM key
goes here**: the assistant reading these tools *is* the narrator; the server itself never calls a
model.

**The `.mcpb` bundle — Claude Code only, for now.** Build it with `uv run python
scripts/build_mcpb.py` → `dist/asset-management-<version>.mcpb`, and install it in one click via
Settings → Extensions. Its tools work in **Claude Code**. They do **not** work in Claude
Desktop's chat window: Desktop does not offer tools from *sideloaded* extensions to the model —
they appear in the tool menu, keep their permission toggles, and are simply never called.
Directory-installed extensions and config-registered servers both work, which is why the config
route above is the one to use for chat.

**In chat:** open the **+** menu for ready-made starters — *Portfolio checkup*, *What's my
drawdown?*, *Should I rebalance?*, *Fill my gaps*, *Find my starting allocation*, *Propose a
posture* — each one pre-loads the figures-only framing. The server also publishes `portfolio://guarantees` (its four
enforced guarantees, versioned, shipped with the code): attach it from the same **+** menu —
or, in clients that let the model read resources itself, just ask "can I trust these
numbers?" and it answers from the manifest instead of improvising.

### 🛟 If something goes wrong

- **Claude says it can only see one tool, or none.** Go to Settings → **Connectors** →
  *asset-management* and switch the tools on. A newly registered server arrives with its tools
  **blocked**, and approving one at a permission prompt enables only that one. Claude then
  honestly reports the short list it was handed — it has no way to know the rest exist, so it
  will tell you the server "only has one tool." It has eight.
- **The tools appear in the menu, but Claude never calls them.** You installed the `.mcpb`
  bundle. Claude Desktop's chat window does not offer tools from *sideloaded* extensions to the
  model — the menu and the permission toggles are drawn from the extension itself, so everything
  *looks* connected while the model is never told the tools exist. Remove the extension and use
  the config route above.
- **The server never appears at all.** Claude Desktop resolves `"command"` on your PATH and
  can't find `uvx`. Install [uv](https://docs.astral.sh/uv/) and restart Desktop, or put the
  absolute path in `"command"` (Windows: `C:\Users\<you>\.local\bin\uvx.exe`).
- **It sits *disconnected* on the very first launch.** That launch is building the environment
  (~500 MB). Give it a minute, then restart Claude Desktop. Later launches take seconds.
- **The first launch dies with `os error 32` ("another process is using the file").** Windows:
  while uv builds the environment, another process — usually the antivirus scanner — briefly
  holds a freshly written file in uv's cache, and the install loses the race. Run the `uvx`
  command from the config once in a terminal yourself (if it trips again, just rerun it — a
  retry costs nothing there). When it sits silent, the environment is built: Ctrl-C, restart
  Desktop. Every launch after reuses the built environment and never races.
- **Every figure comes back `n/a`.** The price cache is cold and `ASSET_MCP_OFFLINE=1` is
  forbidding the fetch that would warm it. Drop that variable, or warm the cache once from the
  CLI: `uvx --from git+https://github.com/disin7c9/asset-management asset-management --book your.csv --warm`.
- **(`.mcpb` only) `No MCP config found for extension … skipping`.** Desktop won't launch the
  server until the Configure form is *saved* — and Save stays disabled until you change
  something. Toggle any field, save, restart.

Or run the server directly / register it with Claude Code:

```bash
uv run python -m app.mcp_server                                    # serve over stdio (set ASSET_BOOK in .env)
asset-management-mcp                                              # the same server as an installed entry point
claude mcp add asset-management -- uv run python -m app.mcp_server  # register, then /mcp to use it
```

The server runs no LLM itself — an assistant calls it; this is not financial advice.

## 📖 Reference

The report is **composable panels**, not exclusive modes — combine flags and the panels stack. What each action needs:

| action | needs | what it does |
|---|---|---|
| status brief (default) | `--book` | your holdings + returns + drawdown/risk |
| `--rebalance MODE` | `--book` + `--target` | buy/sell suggestions toward the target (`--new-cash` sizes a deposit) |
| `--allocate RULE` | none for the presets; `--book` for the re-weight rules | propose a target — re-weight your holdings (`equal_weight`/`inverse_vol`, which need a book) or build a strategic role template (`conservative`/`moderate`/`aggressive`, which works with no book at all); write it with `--allocate-out` |
| `--onboard` | none (a book anchors the roles on what you already hold) | step 0 for a new user: answer 3 plain risk questions in the terminal → the matched posture builds its `--allocate` preset automatically (propose-only; save with `--allocate-out`) |
| `--dry-run` | `--book` (or `--demo`) | preview an import before trusting it: detected format, events parsed, rows skipped/flagged with reasons, and the holdings they derive to — fetches nothing, computes no brief |
| `--metadata` | `--book` | published fund facts per holding (expense ratio, AUM, volume, age, category), cached 7 days |
| `--screen TICKERS` | `--book` + prices | judge NEW candidates vs your book: diversifier (incl. your red days + worst drawdown), cost, liquidity, age, concentration, leveraged/inverse auto-reject, holdings-overlap dedup — each verdict with its reason. Add `--target` for the **held-out role check**: did a 5% sleeve reduce drawdown pain (Ulcer, with a CDaR check) on a held-out window? "Inconclusive" names the gate that blocked it. Propose-only; a PASS is "sane, cheap, liquid, genuinely different", never a prediction |
| `--discover [roles]` | `--book` + prices | suggest **new** ETFs for the roles you hold ≤3% of, run through the same screen — propose-only (see [Discovery](#-discovery--the-curated-universe)) |
| `--backtest` | `--target` | notional rebalance-vs-buy-and-hold — **no `--book`**; prints the simulation alone |
| `--backtest --benchmark REF` | `--target` | validate a target vs a canonical reference (`60-40` / `all-weather` / `permanent`) — drawdown-first legs + the held-out recent-window, Ulcer-first verdict |
| `--narrate` | `--book` + an LLM key | a plain-language **SUMMARY** at the top of the brief; the model writes only the words, every number is substituted and verified from the core (opt-in, off by default — see [Narration](#-narration-optional-plain-language-summary)) |

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
uv run python -m app --book your.csv --screen SCHD --target t.csv  # + the held-out role check
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
uv run python -m app --book your.csv --save --reports-dir /elsewhere   # ...somewhere other than reports/
uv run python -m app --book your.csv --send        # also email the brief as HTML via Resend
```

A `target.csv` is one you create with `--dump-target` / `--allocate-out` (or use `data/sample_data/target.csv`). `--allocate` is **propose-only** and cannot be combined with `--rebalance`/`--backtest`.

### 💻 Core brief

<details>
<summary>Sample output — the full <code>--demo</code> brief (drawdown, ratios, returns, holdings)</summary>

```text
=== DRAWDOWN (investment, time-weighted) ===
Max drawdown:      -9.84%  (95% CI -17.49% .. -6.86%)
  peak 2025-02-19 → trough 2025-04-08 → 2025-06-10  (111 days)
Ulcer index:       2.39%  (95% CI 1.62% .. 6.17%)
CDaR (worst 5%):   6.67%  (95% CI 4.56% .. 14.04%)
You've spent 81% of this period below a previous high.
Gains given back: -$1,240  — peak profit $3,336 (2025-02-19) → $2,096 (2025-04-08) → 2025-06-10
  (dollars of profit given back from a peak; flow-neutral — funding & transfers don't distort it)

=== RISK-ADJUSTED (annualized, 252-day basis, risk-free 0%, ± bootstrap CI) ===
Sharpe:   +1.57  (95% CI +0.59 .. +2.49)
Sortino:  +2.35  (95% CI +0.85 .. +3.97)
Calmar:   +1.75  (95% CI +0.39 .. +4.07)

=== RETURNS (annualized, 252-day basis) ===
Period: 2023-01-05 → 2026-07-24 (1296 days, ~3.55y)
Time-weighted (true TWR):                +17.26%
Money-weighted (IRR):                    +16.15%
Modified Dietz (approx TWR):             +15.87%
  (point figures: accounting identities over your cash flows, not sampled statistics → no band; see RISK-ADJUSTED for bootstrapped CIs)

=== HOLDINGS ===
ticker     shares  avg cost    price    mkt value      unreal    realized
-------------------------------------------------------------------------
BND        50.000     72.26    72.25      3612.50       -0.50      +45.10
IAU        25.000     37.23    76.15      1903.75     +973.13     +220.62
VEA        15.000     40.27    69.78      1046.70     +442.70       +0.00
VOO        12.000    375.13   678.61      8143.32    +3641.72     +295.80
-------------------------------------------------------------------------
Total cost basis (held): $9,649.23
Market value (priced):   $14,706.27
Unrealized P&L:          $+5,057.04
Realized P&L (sells+div): $+549.37
Fees paid (informational): $33.00
Net P&L (unrealized + realized): $+5,606.42

Prices: 4 cache  (age: 5.9h .. 5.9h old as of 2026-07-24 13:51 UTC)
Generated by asset-management. Figures are deterministic and reconciled against ghostfolio + quantstats; this is not financial advice.
```

</details>

Confidence bands come from a moving-block bootstrap. Drawdown is *investment* (time-weighted) drawdown, not account-balance drawdown. The panel also reports **Gains given back** — the largest dollar decline in your cumulative market profit (the felt "how much did I watch evaporate"); it's flow-neutral, so deposits, withdrawals, and transfers don't distort it. Each run also emits one structured JSON log line (`run_summary`) on stderr: `{date, source, n_events_replayed, n_prices_fetched, n_prices_missing, n_series_fetched, n_series_missing, fallbacks_used, status, report_saved, email_sent, rebalance, backtest, allocate, dump_target, metadata, screen, narrate, discover, discover_narrate, benchmark_narrate, warm, onboard, dry_run}` (with `email_detail`/`error` present when relevant).

### 🔁 Rebalance modes

A **target is a complete spec** (`--target path`, columns `Ticker,Weight`; weights are relative and normalized): any held ticker **not** listed is treated as an exit and sold to $0. So `--target` is *required* with `--rebalance` (no silent default), and the run **warns** listing any held tickers the target omits. To **close a position on purpose**, give it weight `0` — that's an explicit, warning-free exit; *omitting* it does the same but triggers the safety warning (the tool can't tell "forgot" from "meant it"). Modes:

- `to_total` — sell + buy to hit the target exactly (cash-neutral; deploys `--new-cash` too)
- `cash_flow_only` — invest `--new-cash` into underweights; never sell (tax-friendly)
- `fixed_dca` — buy the target mix with `--new-cash`, ignoring drift
- `bands` — `to_total`, but gated on a drift **trigger**: if no ticker is outside its band nothing trades, and if any ticker is, every leg goes back to target (including `--new-cash`). Trading only the breached leg would sell with nothing to buy, since the offsetting drift sits in the legs still inside their bands. The band is the **smaller** of an absolute `--band` (default 5pp) or `--band-rel` × the ticker's target weight (the **"5/25 rule"**, default 25%) — so a small sleeve isn't handed a band many times its own size; a 0% target → 0 band → always exits

<details>
<summary>Sample output — rebalance suggestions (demo book, target VOO 40 / BND 30 / IAU 15 / VEA 15)</summary>

**`to_total`** — sell + buy to hit the target exactly:

```text
=== SUGGESTED ACTIONS (rebalance to target) ===
ticker    cur%   tgt% action            $      shares   why
------------------------------------------------------------------------------
VOO       55.4   40.0   SELL     -2260.81      -3.332   55.4% vs 40.0% target
VEA        7.1   15.0    BUY     +1159.24     +16.613   7.1% vs 15.0% target
BND       24.6   30.0    BUY      +799.38     +11.064   24.6% vs 30.0% target
IAU       12.9   15.0    BUY      +302.19      +3.968   12.9% vs 15.0% target
Buy $2,260.81 · Sell $2,260.81 · net +$0.00 (cash-neutral)
```

**`bands`** (the 5/25 rule) — same target. The band is a **trigger**: VOO sits 15.4pp outside its band, so the whole book goes back to target. IAU is inside its own band and still trades, because that's where the offsetting drift lives — its row says so:

```text
=== SUGGESTED ACTIONS (threshold-band rebalance) ===
ticker    cur%   tgt% action            $      shares   why
------------------------------------------------------------------------------
VOO       55.4   40.0   SELL     -2260.81      -3.332   drift +15.4pp exceeds 5.00pp band
VEA        7.1   15.0    BUY     +1159.24     +16.613   drift -7.9pp exceeds 3.75pp band
BND       24.6   30.0    BUY      +799.38     +11.064   drift -5.4pp exceeds 5.00pp band
IAU       12.9   15.0    BUY      +302.19      +3.968   12.9% vs 15.0% target (rebalance triggered)
Buy $2,260.81 · Sell $2,260.81 · net +$0.00 (cash-neutral)
```

If nothing is outside its band, nothing trades at all. Propose-only — no trade is executed; every row cites the rule that fired.
</details>

### 📈 Backtest details

`--backtest --target T.csv` runs a **notional $10,000** historical simulation of that target and prints a **BACKTEST** panel comparing **rebalanced** (schedule via `--rebalance-every {monthly,quarterly,annually}`, default quarterly) vs **buy-and-hold** — drawdown-first (max drawdown, Ulcer, CDaR — each with a bootstrap CI), plus Sharpe/Sortino and returns. It's *notional*: it starts a clean $10k at the target weights on the earliest date all tickers have prices (`--backtest-start` to override), so it tests the *strategy*, independent of your actual buy timing. Labeled **a historical simulation, not a prediction**.

A fixed rebalance policy fits no parameters, so the whole history is out-of-sample-clean (nothing to overfit). The **walk-forward train/test *selection*** machinery — needed only once a strategy *searches* (tunes parameters or picks among candidates: an optimizer, or an *edge* timing strategy) — is deliberately deferred; a **discipline-vs-edge gate** enforces that any future edge strategy must pass a walk-forward backtest before it may surface a suggestion. Today's rebalance modes are all *discipline*, so they suggest freely.

With `--benchmark`, the held-out verdict resolves through three named gates: the **Ulcer gain** must clear a noise margin, **CDaR** (the worst-tail average) may tie but must not contradict the direction, and a **paired moving-block bootstrap CI** must confirm it — otherwise `inconclusive`, with the blocking gate named in the reason. Max drawdown and volatility are reported as context, voting nowhere: a single worst event is an extreme-value statistic, far too noisy on a short history to decide anything.

<details>
<summary>Sample output — held-out verdict vs 60-40</summary>

```text
=== BENCHMARK (preset vs 60-40 · 2016-07-25 → 2026-07-23 — propose-only) ===
                                            preset                   60-40
--------------------------------------------------------------------------
Max drawdown                               -22.01%                 -21.14%
  95% CI                          -35.2% .. -10.3%        -34.5% .. -10.9%
Ulcer index                                  5.57%                   5.74%
  95% CI                             2.6% .. 13.8%           2.6% .. 13.7%
CDaR (worst 5%)                             16.25%                  16.59%
  95% CI                             7.1% .. 30.5%           7.3% .. 29.8%
Sharpe (252-basis)                           +0.88                   +0.89
  95% CI                            +0.25 .. +1.63          +0.28 .. +1.67
Sortino                                      +1.23                   +1.26
  95% CI                            +0.33 .. +2.46          +0.38 .. +2.49
Annualized return                           +9.42%                  +9.81%
Final value                                $24,543                 $25,428
  prices: cache · oldest fetch 2026-07-24

Walk-forward (held-out): OOS (2023-07-24→2026-07-23, 752d): Ulcer 2.17% vs 2.29%
60-40; CDaR 6.64% vs 6.86%; max DD -9.4% vs -11.2% (context); return +14.1% vs
+13.0%/yr (+1.1pp, context) — no clear drawdown difference from 60-40 (the Ulcer gap
is within the noise margin)
```

Read what it doesn't say. The preset carries slightly *less* drawdown pain than 60-40
(Ulcer 5.57% vs 5.74%) and slightly *less* return (+9.42% vs +9.81%/yr) — a trade, not a
win. And even the drawdown edge doesn't survive the held-out window: the verdict is
**"no clear drawdown difference"**, because the Ulcer gain is inside the noise margin.
The held-out line also prints the *return* cost alongside the drawdown gain, so a
less-painful path can't be sold to you without its price tag.
</details>

### 🔭 Discovery & the curated universe

`--discover` maps your holdings to roles (US large, emerging markets, TIPS, REITs, …), finds the roles you hold ≤3% of, takes the biggest **core** funds in each from a **curated universe** (bundled `app/data/universe.csv`, ~375 low-cost ETFs), and runs them through the **same screen** as `--screen` — printing a DISCOVERY panel, each candidate with its verdict (PASS/WARN/FAIL) and reasons. `--discover reit,tips` limits the roles; add `--narrate` for a fenced note ranking the picks by role-fit. A PASS is "sane, cheap, liquid, and genuinely different from what you hold" — never a prediction.

Four honesty rules keep the panel from overreaching. A **gap means no dedicated fund** — broad funds you hold may already include the role at market weight (a total-market fund already holds mid/smalls; an aggregate bond fund already holds treasuries and IG corporates), so the panel says so instead of implying a hole. **Candidates come in shelves of near-substitutes** — each universe row carries a machine-readable `flavor` (its *shelf*: a treasury duration, a sector, REIT geography), and every menu shows one shelf's comparable funds (≥3, a genuine choice) while *naming* the role's other shelves with counts instead of hiding or ranking them; the default shelf is the one the presets already buy from, so it carries no new opinion, and drilling any other is one flag away (`--discover treasury:long`). **Core funds surface first** — a `core` flag (plain blend / diversified / investment-grade vs a Growth/Value style, single region, or high-yield tilt) keeps a junk-bond fund from ever filling a "corporate-bond gap" by AUM accident; core-less shelves stay index lines until you name them (`--discover corporate-bond:high-yield` is consent to see junk, labeled as junk). And **the sector/thematic aisle is never flagged as a gap** — not holding a tech bet is a stance, not a hole; `--discover sector-equity` hands you the shelf *map* (tech · semis · clean-energy · …) and refuses to pick a sector, because that choice is yours; drill a shelf to screen its funds. A custom universe without the new columns simply degrades to the plain behavior.

<details>
<summary>Sample output — a lead shelf, the satellite index, a drill</summary>

**Default** — the lead shelf's comparable funds, with the role's other shelves *named* (not ranked):

```text
corporate-bond  — you currently hold 0%
  · investment-grade — the standard sleeve
  VCIT  PASS  Vanguard Intermediate-Term Corp    (0 fail / 0 warn / 4 pass / 4 n-a)
  LQD   PASS  iShares iBoxx $ Investment Grad     (0 fail / 0 warn / 5 pass / 3 n-a)
  IGIB  PASS  iShares 5-10 Year Investment Gr     (0 fail / 0 warn / 4 pass / 4 n-a)
  also here: high-yield — junk — equity-like drawdowns (19) — drill with --discover corporate-bond:<shelf>
```

**Satellite (`--discover sector-equity`)** — the tool maps the shelves and refuses to pick one:

```text
sector-equity  — you currently hold 0%
  a tactical bet, not a hole — picking the sector/theme is the decision here, and it is yours.
    tech (6) · semis (3) · ai (2) · innovation (3) · health (5) · biotech (4) · financial (6) ·
    banks (3) · utilities (5) · energy (6) · miners (4) · uranium (3) · water (2) · clean-energy (2) · …
  name one to see its screened funds, e.g. --discover sector-equity:tech
```

**Drill one shelf (`--discover treasury:long`)** — the shelf label discloses the risk up front:

```text
treasury  — you currently hold 0%
  · long — 20y+ — rate-sensitive, equity-scale drawdowns
  TLT   PASS  iShares 20+ Year Treasury Bond      (0 fail / 0 warn / 4 pass / 4 n-a)
  VGLT  PASS  Vanguard Long-Term Treasury ETF     (0 fail / 0 warn / 4 pass / 4 n-a)
  also here: intermediate — the standard sleeve (~3-10y) (6) · short — 1-3y — cash-like (4)
```

</details>

The universe is **auto-built** (and refreshable) — no hand-maintenance:

```bash
uv run python scripts/build_universe.py --auto --out app/data/universe.csv
```

It pulls the largest US ETFs per asset-class category from a screener (by fund *size*, not past returns — chasing performance is exactly the trap this avoids), drops leveraged/inverse, and keeps a small curated set for the few categories a screener can't isolate. Point `--discover` at your own list with `ASSET_UNIVERSE=path/to/universe.csv` in `.env`.

### 🔍 Screen a candidate

`--screen TICKER` judges a *new* ticker against your book — cost, liquidity, age, concentration, overlap with what you hold, whether it diversified your past drawdowns, and its **own** worst drawdown — each with a reason. Propose-only; a PASS is "sane, cheap, liquid, genuinely different", never a prediction.

Add `--target` and it also runs the **held-out role check**: give the candidate a 5% sleeve, replay it, and judge only on a recent window the check didn't look at while deciding. Two honest limits on that verdict. **It measures drawdown pain, not return** — an uncorrelated fund makes the ride smoother even when it earns nothing, so the line prints the return cost beside the drawdown gain and you weigh both. And **when the candidate came from `--discover` rather than from you, the choosing happened over the full history**, including the window the check then holds out — so the verdict is not fully independent of it. Naming the ticker yourself (`--screen SCHD`) has no selection step and no such caveat. Details and the measured size of both effects: [MATH.md §12.3](MATH.md).

<details>
<summary>Sample output — same role, different risk (IEF vs TLT)</summary>

Both fill the treasury role and both pass on correlation — but `own-drawdown` tells the rest of the story:

```text
IEF — WARN  (0 fail / 2 warn / 5 pass / 2 n-a)
  [pass] diversifier: ρ=+0.22 vs your book; during your worst drawdown (2025-02-19→2025-04-08, -9.8%) it returned +2.3%
  [warn] own-drawdown: worst fall -11.5% (2023-04-06→2023-10-19) in 3.5y — deeper than your book's worst (-9.8%)

TLT — WARN  (0 fail / 2 warn / 5 pass / 2 n-a)
  [pass] diversifier: ρ=+0.26 vs your book; during your worst drawdown (2025-02-19→2025-04-08, -9.8%) it returned +0.3%
  [warn] own-drawdown: worst fall -23.8% (2023-01-18→2023-10-19) in 3.5y — deeper than your book's worst (-9.8%)
```

Correlation alone would wave both through as clean diversifiers; own-drawdown flags that each carries a fall deeper than your whole book — TLT far more so.
</details>

### 📝 Narration (optional plain-language summary)

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

### 🔑 Configuration — every key in one place

All configuration is environment variables. Running from a checkout, set them once in a
gitignored `.env` at the repo root; running via `uvx` (no checkout to read a `.env` from), set
the same names as real environment variables — or in the Claude Desktop config's `env` block.
Explicit flags always win.

```bash
# your book + target (Steps 1 & 3)
ASSET_BOOK=data/my_data/transactions.csv
# ASSET_CSV=...                       # legacy alias for ASSET_BOOK, still honored
ASSET_TARGET=data/my_data/target.csv

# prices — optional free second source for when Yahoo throttles (Step 2)
TIINGO_API_KEY=...

# email delivery for --send (Step 4)
RESEND_API_KEY=...
REPORT_TO=you@example.com
# REPORT_FROM=briefs@yourdomain.com   # optional; defaults to Resend's onboarding sender

# narration for --narrate (optional — your own LLM key; see Narration above)
ASSET_NARRATE_PROVIDER=anthropic      # or: openai (any OpenAI-compatible endpoint)
ASSET_NARRATE_MODEL=claude-haiku-4-5
ASSET_NARRATE_KEY=sk-...
# ASSET_NARRATE_BASE_URL=https://...  # provider=openai only
# ASSET_NARRATE_TIER=free             # free (default: only coarse bands leave) | paid (exact figures) | local

# MCP addon only
# ASSET_MCP_OFFLINE=1                 # never fetch — for an already-warmed cache

# rarely needed
# ASSET_CACHE_DIR=/somewhere/else     # move the price cache
# ASSET_UNIVERSE=path/to/universe.csv # your own discovery universe
```

Nothing here is required: every feature that needs a key says so when you invoke it, and
degrades cleanly without it.

## 🔧 Project

### 🧪 Develop

```bash
uv run pytest                  # unit + property-based + regression tests
uv run mypy app/               # strict type-checking
uv run ruff check app/ tests/  # lint
uv run python scripts/build_mcpb.py  # package the Claude-Desktop addon → dist/asset-management-<v>.mcpb
```

### 📁 Layout

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
│   ├── backtest.py   notional simulation + the held-out role/benchmark verdicts (Ulcer-first)
│   ├── pipeline.py   the shared book→prices→returns→risk bundle (cli + mcp_server) + cache warm + the --demo book
│   ├── report.py     suggestions + backtest + state + prices + returns + risk → ReportData → text/markdown/HTML
│   ├── narrate.py    fenced narration: validated figures → {{token}} prose → SUMMARY (pure; no number can be the model's)
│   ├── llm.py        optional narrator backend (OpenAI-compatible + Anthropic); fail-closed, opt-in
│   ├── email.py      send the HTML brief via Resend (--send)
│   ├── cli.py        argparse + entry composition + delivery routing + structured run log
│   ├── mcp_server.py read-only stdio MCP server: 8 tools over the core (offline; one-time cold-call auto-warm, ASSET_MCP_OFFLINE=1 opts out)
│   ├── universe.py   curated ETF universe loader (Candidate + roles); app/data/universe.csv, auto-built
│   ├── discover.py   book → role gaps → shelf menus: lead-shelf default / full menu / index / drill (--discover, propose-only)
│   ├── log_config.py logging setup
│   ├── http_safe.py  the one no-redirect URL opener — a 3xx must not carry an API key to another host
│   ├── __main__.py   python -m app
│   └── __init__.py
├── tests/            automated suite (unit, property, regression) — offline, run on every change
├── reconcile/        manual cross-validation against external tools (ghostfolio, quantstats)
├── scripts/          build_universe.py (refresh the ETF universe) · build_mcpb.py (Claude-Desktop bundle) · demo_fence.py (poke the fence)
├── .github/          the gate: ruff + mypy --strict + pytest on every push to main + every PR
├── pyproject.toml    dependencies, mypy/ruff config
└── README.md
```

<a id="privacy-policy"></a>

## 🔒 Privacy Policy

This tool runs entirely on your own machine. **It has no backend, no account, and no telemetry** — the author receives nothing, ever.

- **What it collects.** Nothing. Your transaction log, derived holdings, and price cache are read and written only on your computer, at the paths you pass (`--book`, `--cache-dir`, or `ASSET_BOOK` / `ASSET_CACHE_DIR`). Nothing is uploaded, and no usage data, crash report, or analytics is emitted.
- **What leaves your machine, and only these.** (1) **Price and fund data requests** — ticker symbols are sent to [Yahoo Finance](https://finance.yahoo.com) and, if you set `TIINGO_API_KEY`, to [Tiingo](https://www.tiingo.com), to fetch quotes, history, splits, and published fund facts. Ticker symbols only — never your quantities, cost basis, or balances. (2) **Narration**, `--narrate`, which is **off by default**: if you turn it on and supply your own LLM key, portfolio *figures* are sent to the provider you chose (OpenAI-compatible or Anthropic) to be written up as prose. The `ASSET_NARRATE_TIER` dial controls how much detail is sent; `local` keeps it on your machine. Turn it off and no LLM is contacted at all. (3) **Email delivery**, `--send`, which is **off by default**: if you turn it on and supply your own [Resend](https://resend.com) API key, the rendered HTML brief — which contains your holdings, share counts, cost basis, and P&L — is POSTed to Resend to be delivered to the address you set. Nothing else is sent there, and without `--send` Resend is never contacted.
- **Storage and retention.** The price cache lives in `data/prices` when you run from a clone, or `.asset-management/prices` in your home folder when the tool is installed as a package (override either with `ASSET_CACHE_DIR`), and persists until you delete it. The two are separate directories — warming from a clone does not warm the cache a wheel-installed Desktop addon reads. Reports written with `--save` go to `reports/`. Everything is a plain file you own; delete the folders and the data is gone. The author holds no copy and cannot.
- **Third-party sharing.** The author shares nothing, and receives nothing to share. The only third parties involved are the ones you opt into above — Yahoo, Tiingo, your chosen LLM provider, and Resend — each governed by its own privacy policy.
- **The MCP server.** It never writes to your ledger and never trades; it is bound to your configured book, takes no file-path arguments, and exposes no write tools. It does write its own price cache (and, with no book configured, a demo book) under `ASSET_CACHE_DIR`. It performs two bounded, opt-out network fetches (a one-time cold-cache warm and an on-demand fetch when you screen an uncached ticker); `ASSET_MCP_OFFLINE=1` disables both. It sends nothing anywhere else.
- **Contact.** Open an issue at [github.com/disin7c9/asset-management/issues](https://github.com/disin7c9/asset-management/issues).

## 📜 License

[AGPL-3.0-or-later](LICENSE). Free to use, modify, and self-host — including commercially. The one condition: if you distribute a modified version, or run one as a network service for others, you must publish your modified source under the same license. (Same license Ghostfolio uses.)
