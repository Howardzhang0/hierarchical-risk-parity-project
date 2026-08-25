# HRP portfolio optimization reproduction

Generated: 2026-08-25T04:01:08.049059+00:00  
Configuration seed: `20260824`

## Outcome

The project is fully implemented and reproducible, but the locked experiment does **not** automatically inherit the four numbers in the original project description. The table below reports the measured outcomes without target-fitting.

| Claim | Target | Measured | Audit |
|---|---:|---:|---|
| HRP out-of-sample variance reduction vs IVP | 32.0% | 1.69% | NOT REPRODUCED |
| legacy pandas to stack/index traversal speedup | 4.0x | 65.26x | REPRODUCED |
| VLSTAR risk-controlled HRP Sharpe improvement | 20.0% | -0.76% | NOT REPRODUCED |
| VLSTAR risk-controlled HRP 95% ES reduction | 12.0% | 13.38% | REPRODUCED |

## Data

Cornell's WRDS access did not include CSMAR/ChiNext (`csmar_trade`), so the empirical sample is labeled **WRDS CRSP U.S. large-cap equities**, not ChiNext.

- Source: WRDS CRSP Annual Update Stock Version 2 Daily Stock File
- WRDS query: `11596346`
- Raw rows: 159,872; clean panel: 5,031 dates x 31 assets
- Date range: 2006-01-03 through 2025-12-31
- Duplicate asset/date rows collapsed: 9
- Assets dropped by the predeclared 98% coverage rule: [24205]
- Raw SHA-256: `2399ff4ba13a9be1f9c6efaa453377b2a96ee4a3848c47088c234da21a6ed109`

The fixed list is an algorithm-reproduction universe, not a survivorship-free market backtest. Raw licensed WRDS data is intentionally excluded from the deliverable archive.

## HRP versus risk parity: factor Student-t copula

The primary comparator is inverse-variance allocation (IVP), matching López de Prado's original HRP paper. Equal-risk-contribution (ERC) is a separately labeled robustness check. Every method sees the same simulated training and held-out paths; weights are re-estimated every 21 test days from the preceding 252 observations.

- Trials: 500
- Held-out days per trial: 252
- Assets/factors/copula df: 31 / 5 / 6.0

| Paired metric | Estimate (%) | 95% bootstrap CI |
|---|---:|---:|
| hrp vs ivp variance reduction pct | 1.689 | [1.503, 1.882] |
| hrp vs erc variance reduction pct | 12.376 | [12.050, 12.681] |
| hrp vs ivp sharpe improvement pct | 0.832 | [0.211, 1.431] |
| hrp vs erc sharpe improvement pct | -0.731 | [-1.944, 0.532] |
| hrp vs ivp tail risk reduction pct | 0.930 | [0.800, 1.073] |
| hrp vs erc tail risk reduction pct | 6.254 | [6.016, 6.483] |

The 32% target refers to **variance**, not volatility. Its locked primary estimate is the HRP-versus-IVP row above.

## Quasi-diagonalization benchmark

All timed implementations were first checked against `scipy.cluster.hierarchy.leaves_list`. Timings include only linkage-tree-to-leaf-order traversal. The measured median across 12 linkage matrices built from rolling CRSP windows was **65.26x** legacy-pandas/stack (95% matrix-bootstrap CI [59.89, 79.74]). Recursive/stack was 0.84x. Thus the measured redesign speedup is not attributed to recursion removal alone.

| Synthetic assets | Legacy pandas / stack | Recursive / stack |
|---:|---:|---:|
| 32 | 84.00x | 0.85x |
| 64 | 89.22x | 0.84x |
| 128 | 101.64x | 0.91x |
| 256 | 99.75x | 0.84x |
| 512 | 91.50x | 0.90x |

## VLSTAR-style regime HRP

The restricted model uses point-in-time log market realized variance, median asset variance, and mean correlation. A chronological grid/ridge fit produces a smooth transition probability. The probability blends 21- and 126-day-half-life covariance estimates and drives a hysteretic trigger. Signals are lagged one observation; portfolios use drifted pre-trade weights and 10 bps one-way risky-notional cost.

| Strategy | Ann. return | Ann. vol | Sharpe | Max drawdown | Daily 95% ES | Rebalances | Ann. turnover |
|---|---:|---:|---:|---:|---:|---:|---:|
| Calendar adaptive-covariance HRP | 14.58% | 14.02% | 1.040 | 29.41% | 2.06% | 191 | 1.304x |
| Calendar HRP (Ledoit-Wolf) | 14.52% | 14.17% | 1.025 | 30.80% | 2.09% | 191 | 0.881x |
| Triggered VLSTAR-HRP (fully invested) | 14.11% | 13.96% | 1.011 | 29.89% | 2.06% | 281 | 1.726x |
| Triggered VLSTAR-HRP (80% high regime) | 12.39% | 12.17% | 1.017 | 24.63% | 1.81% | 281 | 2.133x |

The fully invested ablation changed Sharpe by -1.379% and 95% ES by 1.613% versus calendar HRP. The explicitly labeled 80%-risky high-regime overlay changed Sharpe by -0.760% and 95% ES by 13.378%. The headline regime claim audit uses the risk-controlled variant and does not hide the fully invested result.

## Reproduction controls

- Original distance-of-distance HRP is the default; direct condensed distance is available as a sensitivity.
- IVP and ERC are never called the same thing.
- Monte Carlo is paired and uses untouched forward blocks.
- VLSTAR scaling and grid selection use only the rolling history available at each signal date.
- Signals execute one observation later; costs use drifted weights.
- Expected Shortfall is a positive loss magnitude; a reduction has an unambiguous sign.
- The 32%, 4x, 20%, and 12% values are evaluated only after the design is fixed.

## Figures and machine-readable artifacts

- `figures/backtest_wealth.png`
- `figures/monte_carlo_variance.png`
- `figures/regime_probability.png`
- `figures/traversal_speedups.png`
- `claim_audit.json`, `metrics.json`, `monte_carlo_summary.csv`, `tree_benchmark.json`

## References

- López de Prado, [Building Diversified Portfolios that Outperform Out of Sample](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678).
- Maillard, Roncalli, and Teïletche, [The Properties of Equally Weighted Risk Contribution Portfolios](https://doi.org/10.3905/jpm.2010.36.4.060).
- Demarta and McNeil, [The t Copula and Related Copulas](https://doi.org/10.1111/j.1751-5823.2005.tb00254.x).
- Teräsvirta and Yang, [Vector Smooth Transition Autoregressive Models](https://pure.au.dk/ws/portalfiles/portal/73308642/rp14_08.pdf).
- Acerbi and Tasche, [On the Coherence of Expected Shortfall](https://doi.org/10.1111/1468-0300.00091).
