# v0 reconciliation results

**Date:** 2026-06-02
**Input:** `examples/data/transactions.csv` (4 held tickers — VOO, BND, IAU, VEA; 11 events over ~3.4y)
**Goal:** prove our numbers are correct by cross-checking against two independent tools.

The two oracles are **complementary** and together cover the whole chain:

- **ghostfolio** reconstructs holdings → market value → P&L from the *same transactions*, independently. Validates our **inputs** (the equity-curve / holdings reconstruction: `derive`, `prices`, `build_daily_returns`).
- **quantstats** computes risk/return metrics from a *return series*, independently. Validates our **formulas** (`risk.py`, `true_twr_annualized`).

---

## 1. quantstats — metric formulas (run: `uv run --with quantstats python reconcile/reconcile_quantstats.py`)

Fed our daily TWR series into quantstats; compared its metrics to ours.

| Metric | Ours | quantstats | Δ |
|---|---|---|---|
| Sharpe | +1.7511 | +1.7511 | 0.0000 |
| Sortino | +2.6639 | +2.6639 | 0.0000 |
| Max drawdown | −9.84% | −9.84% | 0.0000 |
| Annualized (same basis) | +19.37% | +19.37% | <0.01% |

Match to 4 decimals. Confirms our √252 annualization, `risk_free=0` assumption, std convention, and index-based drawdown all agree with an independent implementation.

**Does not** validate the equity-curve reconstruction (we handed quantstats our own returns) — that's ghostfolio's job below.

---

## 2. ghostfolio — end-to-end (transactions → performance)

Imported `reconcile/ghostfolio/ghostfolio_import.csv` (our CSV converted to ghostfolio's format) into a throwaway ghostfolio instance; it fetched its own Yahoo price history and computed performance.

### Exact matches (to the cent)

| Quantity | Ours | ghostfolio |
|---|---|---|
| Total market value | $15,211.40 | $15,211.40 |
| VOO mkt value | $8,367.60 | $8,367.60 |
| BND mkt value | $3,659.00 | $3,659.00 |
| IAU mkt value | $2,106.75 | $2,106.75 |
| VEA mkt value | $1,078.05 | $1,078.05 |
| Total dividends | $87.30 | $87.30 |

### P&L identity (ghostfolio splits dividends out; we fold them into realized)

```
ghostfolio Absolute Net Performance   6,036.40
         + ghostfolio Dividend           87.30
                                     = 6,123.70
ours: unrealized 5,562.17 + realized 561.52 = 6,123.70   ✓ exact
```

Holds per-ticker too (e.g. VOO: ghostfolio +4,119.60 + its $42.20 VOO dividends = our 3,866.00 unrealized + 295.80 realized = 4,161.80).

### Return rate — different methodology, not a discrepancy

| Measure | Value |
|---|---|
| Our TWR (annualized) | +19.37% |
| Our MWR / IRR | +18.28% |
| Our Modified Dietz | +17.94% |
| ghostfolio annualized **ROAI** | +17.03% |

Ghostfolio's headline is **ROAI** (return on average investment) — a *money-weighted* measure, so it should track our MWR/Dietz, not our TWR. It does (~1% from Dietz/MWR). TWR being highest is expected for a portfolio funded over time: TWR strips out contribution timing while money-weighted measures weight by capital deployed. Ordering TWR > MWR > ROAI is what theory predicts.

---

## Verdict

**v0's numbers are trustworthy.** Holdings, market values, and absolute P&L match ghostfolio to the cent; risk/return formulas match quantstats to 4 decimals. No bug surfaced. The full pipeline is correct end-to-end on real Yahoo data.

### Caveats (honest scope)
- Validated on the **example CSV**, not yet against a real broker statement (the broker reconciliation of shares/cost-basis/realized is a per-user task; our Korean broker doesn't report TWR/Sharpe/drawdown to compare).
- Both we and ghostfolio price from **Yahoo**, so prices aren't independently sourced — but the *math* on those prices is what we set out to validate, and it's confirmed by two tools.
- ghostfolio gives ROAI, not a pure TWR, so the TWR rate itself is validated by quantstats (formula) + ghostfolio (the value series feeding it), not by a single TWR-to-TWR number.
