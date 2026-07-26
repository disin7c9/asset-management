# Mathematics Reference

> Every non-trivial calculation in `app/`, in one place — definition, formula, and where it lives.
> Trivial arithmetic (percentage scaling, float-epsilon guards, dust thresholds) is omitted on purpose.
> Math renders in VS Code's markdown preview. Source is tagged as `file.py:function`.

The four principles this serves: **walk-forward** (hold out unseen data), **confidence
intervals on every metric**, **drawdown-first**, **provenance**. Standard risk-adjusted ratios
(Sharpe/Sortino/Calmar) and the max-drawdown *scalar* delegate to `empyrical-reloaded` (correct
by construction, golden-tested vs quantstats); the drawdown *walk*, Ulcer, CDaR, and every
bootstrap are implemented in-house because the library does not provide them.

---

## 0. Notation

| Symbol | Meaning |
|---|---|
| $P_k(d)$ | split-adjusted close price of ticker $k$ on day $d$ |
| $s_k(d)$ | shares of $k$ held on day $d$ |
| $V(d)$ | portfolio market value on day $d$ |
| $r(d)$ | daily (time-weighted) portfolio return on day $d$ |
| $\Gamma(d)$ | growth-of-1 index, $\Gamma(d)=\prod_{\tau\le d}(1+r(\tau))$ |
| $w_i$ | weight (fraction) of asset $i$; $\sum_i w_i = 1$ |
| $\sigma$ | standard deviation (sample, ddof $=1$) of daily returns |
| $c_i$ | signed cash flow to the **user** ($<0$ paid out, $>0$ received) |
| $C_i$ | signed cash flow **into** the portfolio ($C_i=-c_i$) |
| $f,\,q,\,p$ | fee, quantity (shares), per-share price of a trade |
| $n$ | number of return observations; $N$ number of assets |
| $\mathbb{1}[\cdot]$ | indicator (1 if true, else 0) |

**Constants** (`returns.py`, `risk.py`, `backtest.py`): trading year $=252$ days; calendar
year $=365.25$ days; risk-free rate $=0$.

---

## 1. Cash flows & money-weighted return — `returns.py`

### 1.1 Sign convention — `cash_flows_from_events`
$$
c \;=\;
\begin{cases}
-(q\,p + f) & \text{buy}\\
\;\;\,q\,p - f & \text{sell}\\
\;\;\,\text{cash} - f & \text{dividend / interest}\\
-f & \text{standalone fee}
\end{cases}
$$
Money the user pays out is negative; money received is positive. Every money-weighted figure rides on this.

### 1.2 Money-weighted return (XIRR) — `money_weighted_return`, `_xirr_newton`
The **internal rate of return** with irregularly-dated flows: the single annual rate $r$ for which
the present value of all flows (the live portfolio value $V$ counted as a final inflow at `asof`) is zero.
$$
\mathrm{NPV}(r)=\sum_i \frac{c_i}{(1+r)^{t_i}}=0,
\qquad t_i=\frac{d_i-d_0}{365.25}\ \text{(years)} .
$$
Solved by **Newton–Raphson** (root-finding by tangent iteration):
$$
f(r)=\sum_i c_i(1+r)^{-t_i},\qquad
f'(r)=\sum_i -t_i\,c_i\,(1+r)^{-t_i-1},\qquad
r_{k+1}=r_k-\frac{f(r_k)}{f'(r_k)} .
$$
Stop when $|r_{k+1}-r_k|<10^{-10}$ (cap $100$ iterations). Returns **None** if all flows share a
sign (no real root) or it fails to converge — silence beats a fabricated number.

### 1.3 Modified Dietz return — `modified_dietz_return`
A money-weighted *approximation* of time-weighted return needing only flows and end value.
With start value $V_0=0$ (the window opens at the first investment) and each contribution
weighted by the fraction of the period $T$ it was present:
$$
R_{\text{MD}}=\frac{V-\;C_{\text{net}}}{\displaystyle\sum_i w_i\,C_i},
\qquad
C_i=-c_i,\quad
C_{\text{net}}=\sum_i C_i,\quad
w_i=\frac{T-t_i}{T},\quad t_i=d_i-d_0 .
$$
Returns **None** when the weighted denominator $\le 0$ (net withdrawals dominate — the formula
cannot honestly represent that case).

### 1.4 Annualization (CAGR) — `annualize_return`
$$
R_{\text{ann}}=(1+R)^{1/y}-1,\qquad y=\frac{\text{days}}{365.25}.
$$
Refused (→ None) for windows under $183$ days (half a year — the one floor shared by every measure in the RETURNS panel, §12.4); a total loss ($1+R\le 0$) returns $-1$.

---

## 2. Value, P&L, and time-weighted return series — `returns.py`

### 2.1 Value curve — `value_curve`
$$
V(d)=\sum_k s_k(d)\,P_k(d),\qquad
s_k(d)=\sum_{\substack{\text{trades } j\ \text{of }k\\ \text{placed on day}\le d}} \pm q_j ,
$$
a running (cumulative) share count, prices forward-filled across non-trading days. Priced-securities universe only.

### 2.2 Flow-neutral P&L curve — `pnl_curve`
$$
\mathrm{PnL}(d)=V(d)+\sum_{\tau\le d}\Delta\mathrm{cash}(\tau),
$$
where $\Delta\mathrm{cash}=\text{sells}-\text{buys}+\text{income}-\text{fees}$ and **external**
deposits/withdrawals are excluded. Funding and transfers cancel (they move balance and cost base
equally), so a drawdown of this curve is real "dollars of profit given back."

### 2.3 Daily time-weighted return — `build_daily_returns`
Neutralize external flows so the series reflects investment performance, not contribution timing.
Writing the day's net external flow as $F(d)=B(d)-S(d)$:
$$
g(d)=V(d)-V(d-1)-F(d)+I(d),
\qquad
r(d)=\frac{g(d)}{V(d-1)+F(d)}\quad\text{for } V(d-1)>0,\ V(d-1)+F(d)>0,
$$
with $B$ buy cost, $S$ sell proceeds, $I$ income (dividends/interest net of withholding, minus fees).

**The flow sits in the denominator too** — a day-level Modified Dietz (§1.3), with a full-day
weight because the day's new shares are valued at that day's close. This is forced, not stylistic:
shares bought on day $d$ enter $V(d)$ at the CLOSE while their cash leaves at the EXECUTION price,
so the fill→close move is already inside $g(d)$. Dividing by $V(d-1)$ alone credits new money's
gain to old money, with error scaling as $F(d)/V(d-1)$ — unbounded, and bounded only on a mature
book. In a flat market where no price moved, a \$99{,}500 purchase filled 0.5% under the close
against a \$100 position yields $r=+0.502\%$ here and reported $+500\%$ before this correction.
Note the sign: a fill *above* the close manufactures a negative day, i.e. a drawdown that never
happened. Income is deliberately NOT in the denominator — it is return the holdings earned, not
capital the investor added.

### 2.4 Growth-of-1 index — `twr_index`
$$
\Gamma(d)=\prod_{\tau\le d}\bigl(1+r(\tau)\bigr).
$$
**Geometric chain-linking**: returns compound multiplicatively, so $\Gamma$ is the value of \$1 invested at the start.

### 2.5 True annualized TWR — `true_twr_annualized`
Annualized on a **252-trading-day basis by observation count** (the same clock as the risk ratios,
so return and risk are comparable; a cash gap neither dilutes nor inflates it):
$$
R_{\text{TWR}}=\Bigl(\prod_{\tau}(1+r(\tau))\Bigr)^{252/n}-1 .
$$
Returns None for $n<126$ observations (half a trading year — §12.4).

### 2.6 Split-mismatch detector — `price_basis_mismatches`
Flags an unhandled split when a trade's execution price diverges from the price *history* by a factor $\phi=2$:
$$
\text{flag } k \text{ if } \frac{p_{\text{exec}}}{P_k(\le d)}\ge \phi \ \text{ or }\ \le \frac1\phi .
$$
A clean $\ge 2{:}1$ split leaves a ratio far beyond any intraday fill-vs-close gap.

---

## 3. Holdings accounting (average cost) — `derive.py`

Per position: shares $s$, cost basis $C$, average cost $\bar p = C/s$.

$$
\begin{aligned}
\textbf{buy:}\quad   & s \mathrel{+}= q, & C &\mathrel{+}= q\,p + f\\
\textbf{sell:}\quad  & \text{realized}\mathrel{+}= q\,p - q\,\bar p - f, & C&\mathrel{-}= q\,\bar p,\quad s\mathrel{-}=q\\
\textbf{dividend/interest:}\quad & \text{realized}\mathrel{+}= \text{cash}-f &&\\
\textbf{fee:}\quad   & \text{realized}\mathrel{-}= f &&
\end{aligned}
$$

A sell releases cost basis at the **average** rate $\bar p$ and books the difference as realized P&L
(proportional cost relief). Fees are capitalized into basis on a buy, expensed against realized P&L otherwise.

---

## 4. Corporate actions (splits) — `corporate_actions.py`

### 4.1 Cumulative split factor — `cumulative_split_factor`
$$
\Phi_k(d)=\prod_{\{(\delta,\rho)\,:\,\delta > d,\ \rho>0\}}\rho ,
$$
the product of split ratios effective **strictly after** trade date $d$ ($10{:}1\Rightarrow\rho=10$, reverse $1{:}10\Rightarrow\rho=0.1$).

### 4.2 Cost-invariant adjustment — `adjust_for_splits`
$$
q' = q\,\Phi_k(d),\qquad p' = \frac{p}{\Phi_k(d)},\qquad\text{so } q'p' = q\,p\ \text{(total cost unchanged)} .
$$
Puts raw share counts onto the same split-adjusted basis as the price history.

---

## 5. Drawdown family — `risk.py`

### 5.1 Drawdown curve — `_drawdown_curve`
$$
DD(d)=\frac{\Gamma(d)}{\max_{\tau\le d}\Gamma(\tau)}-1\ \le 0 .
$$
Distance below the running peak, as a fraction.

### 5.2 Maximum drawdown — `max_drawdown`, `_drawdown_walk`
$$
\mathrm{MDD}=\min_d DD(d),\qquad
d_{\text{trough}}=\arg\min_d DD(d),\qquad
d_{\text{peak}}=\arg\max_{\tau\le d_{\text{trough}}}\Gamma(\tau),
$$
recovery $=$ first $d>d_{\text{trough}}$ with $\Gamma(d)\ge\Gamma(d_{\text{peak}})$ (else None).
Time underwater $=\operatorname{mean}\bigl(\mathbb{1}[DD(d)<0]\bigr)$.

### 5.3 Dollar drawdown ("gains given back") — `dollar_drawdown`
On the flow-neutral P&L curve:
$$
\mathrm{drop}(d)=\mathrm{PnL}(d)-\max_{\tau\le d}\mathrm{PnL}(\tau),\qquad
\text{giveback}=\mathrm{PnL}(d_{\text{trough}})-\mathrm{PnL}(d_{\text{peak}}) .
$$

### 5.4 Ulcer index — `ulcer_index`
**Root-mean-square drawdown** — penalizes drawdowns that are both deep *and* long:
$$
\mathrm{UI}=\sqrt{\frac1N\sum_{d} DD(d)^2}\ \ (\ge 0).
$$

### 5.5 Conditional Drawdown-at-Risk — `cdar`
The drawdown analogue of **CVaR / Expected Shortfall**: average of the worst $\alpha$ fraction of
drawdowns. With magnitudes $X=\{-DD(d)\}\ge 0$ over $N$ days, take the $k=\lceil\alpha N\rceil$
largest (at least one):
$$
\mathrm{CDaR}_\alpha=\frac1k\!\!\sum_{x\in\operatorname{top}_k(X)}\!\! x,\qquad \alpha=0.05,\ \ k=\lceil\alpha N\rceil .
$$
The worst-$k$ days are taken **explicitly** — not the days at/above the $(1-\alpha)$ quantile $q$:
when fewer than $\alpha$ of days are underwater, $q=0$ and a "$x\ge q$" filter sweeps in every
zero-drawdown day, collapsing CDaR to the overall mean (~20× understatement on a mostly-at-peak
leg). Since v2.9.0 CDaR is a verdict gate (§12.5), so this understatement is corrected here.

---

## 6. Risk-adjusted ratios — `risk.py` (via `empyrical`)

Daily returns, risk-free $=0$, annualized on $252$ days. Definitions as computed by `empyrical-reloaded`:

$$
\mathrm{Sharpe}=\sqrt{252}\;\frac{\operatorname{mean}(r)}{\sigma(r)},
\qquad
\mathrm{Sortino}=\sqrt{252}\;\frac{\operatorname{mean}(r)}{\sigma_{\text{down}}},\quad
\sigma_{\text{down}}=\sqrt{\operatorname{mean}\bigl(\min(r,0)^2\bigr)},
$$

$$
\mathrm{Calmar}=\frac{R_{\text{ann}}}{\lvert \mathrm{MDD}\rvert}.
$$

Sharpe rewards per unit of **total** volatility; Sortino per unit of **downside** deviation only;
Calmar per unit of **worst-case loss**.

---

## 7. Volatility annualization — `risk.py`, `backtest.py`

**Square-root-of-time rule** (volatility scales with $\sqrt{\text{periods}}$):
$$
\sigma_{\text{ann}}=\sigma_{\text{daily}}\cdot\sqrt{252},\qquad \sigma\ \text{sample std, ddof}=1 .
$$

---

## 8. Bootstrap confidence intervals — `risk.py`

The honest "± band" on every risk metric, since we have only one history.

### 8.1 Moving-block resample — `moving_block_indices`
For $n$ observations and block length $b$, draw $\lceil n/b\rceil$ start indices uniformly from
$\{0,\dots,\max(1,n-b)\}$, lay down contiguous length-$b$ blocks, and truncate to $n$:
$$
\text{indices}=\bigl[(\text{start}_j + o)\bigr]_{\,j,\;o=0..b-1}\ \text{flattened, first } n .
$$
**Blocks, not single days** — returns are autocorrelated and drawdown is path-dependent; i.i.d.
resampling shatters the runs drawdown measures and yields an over-narrow (overconfident) band.

### 8.2 Percentile CI — `bootstrap_ci`
Resample $B=1000$ times, recompute the metric $\hat\theta_j$ on each, take the central band:
$$
\mathrm{CI}_c=\Bigl[\,Q_{\frac{1-c}{2}}(\{\hat\theta_j\}),\ \ Q_{\frac{1+c}{2}}(\{\hat\theta_j\})\,\Bigr],
\qquad c=0.95 .
$$
Non-finite resamples are skipped; a non-finite point estimate yields a degenerate band (rendered n/a).

### 8.3 Block-length heuristics
$$
b_{\text{ratios}}=\max\!\bigl(2,\ \operatorname{round}(n^{1/3})\bigr)
\qquad\text{(Sharpe/Sortino/Calmar)},
$$
$$
b_{\text{path}}=\max\!\bigl(2,\ \operatorname{round}(\sqrt n)\bigr)
\qquad\text{(MDD / Ulcer / CDaR — a longer block can reassemble a multi-month decline)} .
$$
**Noisy flag:** $n<504$ ($\approx$ 2 trading years) → every ratio is treated as statistically thin.

---

## 9. Allocation rules — `allocate.py`

### 9.1 Equal weight — `equal_weight`
$$
w_i=\frac1N .
$$
The robust $1/N$ baseline that beat optimization out-of-sample in `examples/` 05–06.

### 9.2 Inverse volatility (risk-parity-lite) — `inverse_vol`
Each holding contributes $\approx$ equal **risk**, not equal dollars:
$$
\tilde w_i=\frac1{\sigma_i},\qquad
w_i=\frac{\tilde w_i}{\sum_j \tilde w_j},
$$
$\sigma_i=$ std of daily returns over the lookback (default $252$). Tickers with $\sigma_i\le 0$ or
non-finite, or fewer than 2 returns, are dropped (no risk signal). The $1/\sigma$ ratio is
scale-free, so daily-vs-annualized vol cancels in the normalization.

### 9.3 Cap enforcement (water-filling) — `apply_caps`
Project onto $\{w : w_i\le \text{cap},\ \sum w_i=1\}$ by iterated redistribution. Normalize, then repeat
(to a fixed point, $\le N$ passes): clip every $w_i>\text{cap}$ to $\text{cap}$, pool the excess
$E=\sum_{i\in\text{over}}(w_i-\text{cap})$, and spread it over the under-cap holdings **proportionally**:
$$
w_j \mathrel{+}= E\cdot\frac{w_j}{\sum_{k\in\text{under}} w_k}\qquad (j\in\text{under}) .
$$
Feasibility requirement: $\text{cap}\cdot N\ge 1$.

---

## 10. Rebalancing suggestions — `strategy.py`

Universe = held $\cup$ target tickers with a positive price. $V_i$ = current market value of $i$, $V_{\text{tot}}=\sum_i V_i$.

$$
w_i^{\text{cur}}=\frac{V_i}{V_{\text{tot}}},\qquad
\text{drift } \delta_i=w_i^{\text{cur}}-w_i^{\text{tgt}},\qquad
\text{trade } \Delta_i=w_i^{\text{tgt}}\cdot \text{base}-V_i,\qquad
\text{shares}=\frac{|\Delta_i|}{p_i}.
$$

$\Delta_i>0$ buy, $<0$ sell. Base capital per mode: `to_total` $=V_{\text{tot}}+\text{cash}$; `bands` $=V_{\text{tot}}$.

### 10.1 5/25 band — `_to_target`
No-trade region = the **smaller** of an absolute band and a relative fraction of the target (Swedroe's 5/25):
$$
\theta_i=\min\bigl(\text{band},\ \text{band\_rel}\cdot w_i^{\text{tgt}}\bigr),
\qquad \text{hold } i \iff |\delta_i|\le \theta_i .
$$
A target of $0$ gives $\theta_i=0$ → the position is always exited; a small sleeve isn't handed a band many times its size.

### 10.2 Cash-flow-only (proportional shortfall) — `_cash_flow_only`
Deploy new cash into underweights in proportion to each shortfall; never sell:
$$
h_i=\max\!\bigl(0,\ w_i^{\text{tgt}}\,V_{\text{post}}-V_i\bigr),\quad V_{\text{post}}=V_{\text{tot}}+\text{cash},
\qquad
b_i=\text{cash}\cdot\frac{h_i}{\sum_j h_j}.
$$
If nothing is underweight, fall back to the target mix $b_i=\text{cash}\cdot w_i^{\text{tgt}}$.

### 10.3 Fixed DCA — `_fixed_dca`
$$
b_i=\text{cash}\cdot w_i^{\text{tgt}}\qquad\text{(buy the target mix, ignore drift).}
$$

---

## 11. Candidate screen — `screen.py`

Candidate daily returns $r^{c}(d)=P^{c}(d)/P^{c}(d-1)-1$; portfolio returns $r^{p}$; inner-joined on common dates.

### 11.1 Diversifier — `_check_diversifier`
**Pearson correlation** over the full overlap (the gate):
$$
\rho=\frac{\operatorname{cov}(r^{c},r^{p})}{\sigma_{c}\,\sigma_{p}} .
$$
**Downside (conditional) correlation** on red days $\{d : r^{p}(d)<0\}$ (needs $\ge 20$ such days),
and **return through your worst drawdown window** $[\,d_{\text{peak}},d_{\text{trough}}\,]$:
$$
\rho_{\text{down}}=\rho\big|_{\,r^{p}<0},
\qquad
R^{c}_{\text{DD}}=\prod_{d\in[\,\text{peak},\,\text{trough}\,]}\!\bigl(1+r^{c}(d)\bigr)-1 .
$$
Zero-variance series ($\sigma=0$) → the check is **n/a**, never a fabricated correlation.
Decision: fail if $\rho>0.85$, warn if $\rho>0.60$; escalate a pass→warn if
$\rho_{\text{down}}-\rho>0.15$ (diversification that vanishes in stress).

**$\rho$ is a point estimate over one window, and carries no confidence interval —
deliberately.** Every metric in §5 is bootstrapped; this one is not, and the reason is that a
band here would answer the wrong question. A CI measures how precisely *this window's*
correlation was estimated. But correlation is not a fixed quantity measured imprecisely — it
moves. Measured on a real 2.15-year book, splitting the window in half: SCHD $+0.514
\rightarrow +0.236$, VIG $+0.757 \rightarrow +0.614$, VXF $+0.794 \rightarrow +0.698$. Those
shifts are a third to a full multiple of the corresponding bootstrap widths (0.22–0.47 over
538 days). A tight interval would invite *more* trust in a number whose dominant uncertainty
is that it does not stay put. So the figure is reported bare, the thresholds are applied to
it as a gate (never as a ranking), and the reader is told what it is: how this fund moved
against this book over this window — not a forecast, and not a stable property of the pair.

The bar for revisiting this is a **ranker**. Gating on a threshold tolerates an imprecise
estimate; ordering candidates by $\rho$ would treat $0.71$ vs $0.74$ as a real difference,
which at these interval widths it is not. If the screen ever sorts rather than filters, the
uncertainty has to become explicit — or the ranking has to be refused.

### 11.2 Holdings overlap (near-equivalent collapse) — `holdings_overlap`
Top-10 **overlap coefficient** (sum of shared minimums); None if either fund has no look-through:
$$
\mathrm{ov}(a,b)=\sum_{k\in a\cap b}\min(a_k,\,b_k) .
$$
Top-10 only, so it **understates** true overlap — a floor, decisive when high. Decision: fail $\ge 0.70$
(near-duplicate), warn $\ge 0.40$; physical-commodity trusts (no holdings) fall back to a category match.

### 11.3 Concentration — `_check_concentration`
$$
\text{top10}=\sum_{k\in\text{top 10}} w_k,\qquad \text{warn if } >0.50 .
$$

### 11.4 Fund age — `age_years` (`metadata.py`)
$$
\text{age}=\frac{\max\!\bigl((\text{asof}-\text{inception}),\,0\bigr)}{365.25}\ \text{years}.
$$

---

## 12. Walk-forward role check — `backtest.py`

The edge gate's **evidence**: does carving a sleeve for the candidate actually improve drawdown,
*out of sample*? Judged on a held-out window only.

### 12.1 Sleeve construction — `role_check`
$$
w_i'=w_i\,(1-s)\ \ \text{for target tickers},\qquad w_{\text{cand}}'=s,\qquad s=0.05 .
$$

### 12.2 Notional simulation — `simulate`
Allocate round capital $K=\$10{,}000$ at the target weights and walk closes forward; rebalancing is
value-preserving (no cash in/out):
$$
s_k(t_0)=\frac{w_k\,K}{P_k(t_0)},
\qquad\text{on a rebalance day: } s_k(t)=\frac{w_k\,V(t)}{P_k(t)} .
$$
Weights are renormalized over priced tickers, $w_k\leftarrow w_k/\sum_j w_j$.

**Return basis.** $P_k$ here is the **total-return** close (split- *and* dividend-adjusted), not
the split-adjusted-only close $P_k$ denotes everywhere else in this document. The distinction is
forced: the portfolio path (§2) takes dividends from the transaction log, so an adjusted close
would double-count them, while `simulate` holds funds with no log — on a raw close every coupon
and dividend becomes a permanent capital loss. Measured over 2023-01→2026-07 on the same cache,
the raw basis reports BND at $-0.12\%$/yr against $+3.51\%$ (Ulcer $3.35\%$ vs $2.15\%$) and BIL —
whose entire return is coupon — at $+0.02\%$/yr against $+4.58\%$. The benchmark references are
40–55% bonds (`60-40` is 40% BND; `permanent` is 25% BIL + 25% TLT), so this is not a rounding
concern: it moved the `all-weather` leg from $+5.61\%$/yr to $+8.50\%$ over the demo window, and
cut the preset's apparent Ulcer advantage over it from $0.70$pp to $0.43$pp. The bases are cached
in separate
files (`<T>_series.parquet` vs `<T>_series_tr.parquet`) so neither can be served for the other.

### 12.3 In-sample / out-of-sample split
Over the common priced window of $n$ days (the candidate's history is usually binding):
$$
n_{\text{oos}}=\lfloor n\cdot 0.30\rfloor,\qquad n_{\text{is}}=n-n_{\text{oos}},
$$
require $n_{\text{is}}\ge 60$ **and** $n_{\text{oos}}\ge 60$ else the verdict is **insufficient**. Fresh
capital each window; legs date-aligned by inner join (never positional truncation).

**Selection sits inside the split (v2.12.1).** The *evaluation* was always clean: nothing is
tuned, and the statistics are computed on the held-out window alone. Selection was not. On the
`--discover` path the candidate is chosen upstream by `screen.py`, and its correlation and
drawdown-window checks read the **full** history — including the window this split then holds
out. A fund that happened to behave well there was likelier to be the one you were shown, so the
verdict about it was not independent of the data it was measured on.

`screen_candidates` now separates the two jobs. When a role check will follow, the return-bearing
checks — Pearson $\rho$, red-day $\rho$, drawdown-window return, and the candidate's own drawdown —
are computed on the in-sample window only. The cut is **per candidate** and takes the tighter
of two boundaries: that candidate's own `RoleCheck` in-sample window end, and
`backtest.in_sample_end` over the book's return index. Taking the candidate's own window is
what makes the guarantee hold — `role_check` splits its OWN common window (candidate ∩ target
price history), so a candidate whose series ends before the book's (delisted, halted, or a
staler cache entry) has an EARLIER real split, and a book-derived cutoff alone would read past
it. The minimum can only ever cut more, never less. Every figure so computed is labelled
`[in-sample through YYYY-MM-DD]`: what gates is what is shown. The structural checks — expense
ratio, liquidity, fund age, concentration, holdings overlap, leveraged/inverse rejection — carry no
return information, so nothing can leak through them and they keep reading full history, where they
are most informative. Naming the ticker yourself (`--screen SCHD`) has no selection step, so nothing
is cut.

The cut has a price, and it is paid in the honest direction: with ~30% less history the candidate
has less room to fall, so a fund near the two-year `own-drawdown` floor now returns **n/a** where it
previously gave a verdict. On a book with under ~3 years of history that silences the check for every
candidate — measured on a 2.15y book, 5 of 5 candidates returned n/a where a 3.55y book judged 4 of 5.
Refusing to judge is the correct answer when the only way to judge would be to look at the held-out
window, but the cost is concentrated on exactly the young books that can least afford it.

Two figures the cut does NOT move, worth knowing before reading a diff of it: the drawdown-window
return is unchanged whenever the book's worst drawdown falls inside the in-sample side (the common
case — a recent worst drawdown is the exception), and the structural checks are untouched by
construction. On one real 2.15y book the cut shifted $\rho$ by 0.02–0.06 and flipped **no** verdicts.

### 12.4 The own-drawdown peer bar

A candidate's own worst fall is compared against the **deepest-falling fund currently held**, not
against the portfolio's own worst fall. The portfolio is a blend and a blend's drawdown is shallower
than its components' by construction, so "deeper than your book" is nearly tautological for a single
equity fund: on the bundled example (book worst $-9.8\%$) it flags VOO ($-19.0\%$), IAU ($-26.4\%$)
and VEA ($-14.4\%$) — three of the four funds that book holds. The peer bar asks the answerable
question instead: does this fall harder than anything already in the book? A fixed equity-scale bar
($\le -30\%$) warns independently of any peer, and the blended figure is still reported as context
when no peer is available.

This still is not **walk-forward**, and the output still says *held-out recent-window*. Walk-forward
means repeated re-fitting across rolling origins; this is one split, judged once. What changed is
that the one split is now honest on both sides of the line.

### 12.4 Window statistics — `_window_stats`
Per leg (with-candidate, without), over the held-out window: **Ulcer index** (§5.4 — the verdict
statistic) and **CDaR** (§5.5 — the tail check), plus max-drawdown depth (§5.2), annualized vol
(§7), and true TWR (§2.5) as DESCRIPTIVE context. Gains (positive = the tested leg carried less
pain):
$$
\text{ulcergain}=\mathrm{UI}_{\text{without}}-\mathrm{UI}_{\text{with}},
\qquad
\text{cdargain}=\mathrm{CDaR}_{\text{without}}-\mathrm{CDaR}_{\text{with}} .
$$

### 12.5 Verdict — Ulcer-first, three gates
The verdict is judged on the **Ulcer index** (whole-window drawdown pain), **not** max-drawdown
depth: a single worst drop is an extreme-value statistic whose bootstrap CI almost always straddles
zero on a short history (the chronic "inconclusive"). Max drawdown and volatility are reported but
do **not** vote — Ulcer already carries the drawdown-producing (downside) volatility a drawdown-first
verdict cares about; upside volatility is deliberately not penalized. With $\varepsilon_{U}=0.0025$
(Ulcer noise margin) and $\varepsilon_{C}=0.005$ (CDaR contradiction slack, at CDaR's larger tail scale):
$$
\text{verdict}=
\begin{cases}
\textbf{inconclusive} & |\text{ulcergain}|<\varepsilon_{U} & (\texttt{noise\_margin})\\
\textbf{inconclusive} & \operatorname{sign}(\text{ulcergain})\cdot\text{cdargain}<-\varepsilon_{C} & (\texttt{cdar\_contradicts})\\
\textbf{improved} & \text{ulcergain}>0\ \text{and}\ \mathrm{CI}_{\text{low}}>0\\
\textbf{worsened} & \text{ulcergain}<0\ \text{and}\ \mathrm{CI}_{\text{high}}<0\\
\textbf{inconclusive} & \text{otherwise} & (\texttt{bootstrap\_unconfirmed})
\end{cases}
$$
$\mathrm{CI}$ is the paired-bootstrap 95% interval of the Ulcer gain (§12.6). Gate 2 lets CDaR **tie**
but not **contradict** the Ulcer direction — it catches an Ulcer win bought with a deeper worst tail.
The blocking gate is returned as a structured `cause` for consumers to branch on.

### 12.6 Paired moving-block bootstrap of the Ulcer gain — `_paired_ulcer_gain_ci`
The honesty gate. Both legs ride the same market, so resample them with the **same** block indices —
shared market moves cancel in the difference, leaving only the tested leg's marginal effect (the
variance-reduction-by-pairing principle, as in a paired $t$-test). For $j=1\dots B$ ($B=1000$),
drawing one index set $I_j$ from §8.1 with block $b=\max(2,\operatorname{round}\sqrt n)$
(`risk.path_block`, the same √n rule as the point CIs):
$$
\Delta_j=\mathrm{UI}\!\bigl(r^{\text{without}}[I_j]\bigr)-\mathrm{UI}\!\bigl(r^{\text{with}}[I_j]\bigr),
\qquad
\mathrm{UI}(\mathbf r)=\sqrt{\tfrac1N\textstyle\sum_d DD_d^2},\ \ DD=\tfrac{\Pi}{\operatorname{cummax}\Pi}-1,\ \Pi=\operatorname{cumprod}(1+\mathbf r).
$$
$$
\mathrm{CI}_{95}=\bigl[\,Q_{2.5}(\{\Delta_j\}),\ Q_{97.5}(\{\Delta_j\})\,\bigr].
$$
Non-finite resamples are dropped (as in the risk-panel bootstrap). An improved/worsened verdict
stands only if the CI **confirms the direction** ($\mathrm{CI}_{\text{low}}>0$ for improved,
$\mathrm{CI}_{\text{high}}<0$ for worsened); otherwise inconclusive. Needs $n\ge 10$ aligned days
(else no interval → `window_too_short`). Decisive on a real effect, honestly inconclusive inside the
noise band — the property that keeps the gate from passing overfit noise.

---

## Appendix — parameter table

| Parameter | Value | Where | Role |
|---|---|---|---|
| trading days / year | $252$ | returns, risk, backtest | annualization basis |
| calendar days / year | $365.25$ | returns, metadata | CAGR, fund age |
| risk-free rate | $0$ | risk | Sharpe/Sortino |
| min to annualize | $183$ d (MWR/MD), $126$ obs (TWR) — one duration, §12.4 | returns | refuse short windows |
| noisy threshold | $504$ days | risk | flag thin samples |
| bootstrap resamples $B$ | $1000$ | risk, backtest | CI precision |
| CI level $c$ | $0.95$ | risk, backtest | band width |
| block length | $n^{1/3}$ (ratios), $\sqrt n$ (path/paired) | risk, backtest | preserve autocorrelation |
| candidate sleeve $s$ | $0.05$ | backtest | role-check displacement |
| OOS fraction | $0.30$ | backtest | held-out share |
| min window | $60$ days each side | backtest | else "insufficient" |
| verdict margins | $\varepsilon_U=0.0025$ (Ulcer), $\varepsilon_C=0.005$ (CDaR slack) | backtest | signal vs noise |
| corr thresholds | warn $0.60$, fail $0.85$, downside escalate $+0.15$ | screen | diversifier verdict |
| overlap thresholds | warn $0.40$, fail $0.70$ | screen | near-duplicate verdict |
| concentration warn | $0.50$ | screen | top-10 weight |
| cost tiers | $\le 0.20\%$ pass, $\le 0.50\%$ warn | screen | expense ratio |
| liquidity floors | \$100M AUM, 100K sh/day | screen | tradability |
| age tiers | $<1$y fail, $<3$y warn | screen | closure risk |
| split-mismatch factor $\phi$ | $2$ | returns | unhandled-split flag |
