# asset-management

Personal tool for tracking a stock/ETF portfolio honestly.

Reads an append-only transaction log (CSV), derives holdings + correct time-weighted and money-weighted returns, and emits a **drawdown-first report with confidence bands** — leading with the pain a portfolio causes its owner, not the headline return.

## Scope (v0)

- Personal use first; deterministic only — **no AI yet**.
- Long-only equity / ETF.
- Four flagship principles: walk-forward by default, confidence intervals on every metric, drawdown-first reporting, multi-source price fallback with provenance.

See [DRAFT.md](DRAFT.md) for the full design, [research/](research/) for prior-art surveys, and [examples/](examples/) for prototypes that informed v0.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # install runtime + dev deps from pyproject.toml
cp .env.example .env    # then put your real RESEND_API_KEY in .env (never commit)
```

## Run

(Coming once v0 lands.)

## Layout

```
asset-management/
├── app/             # all v0 program code
├── examples/        # prototype scripts 1–9 (study material)
└── DRAFT.md         # full design doc
```