"""Render the final claim-audit report."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _number(value: Any, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric != numeric:
        return "n/a"
    return f"{numeric:.{digits}f}"


def _status(value: str) -> str:
    return "REPRODUCED" if value == "reproduced" else "NOT REPRODUCED"


def _percent(value: Any, digits: int = 2) -> str:
    return f"{100.0 * float(value):.{digits}f}%"


def write_reproduction_report(context: dict[str, Any], path: str | Path) -> Path:
    """Write a concise human-readable report from machine-readable results."""

    manifest = context["data_manifest"]
    claims = context["claim_audit"]
    mc = context["monte_carlo"]
    benchmark = context["benchmark"]
    backtests = context["backtests"]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    claim_rows = "\n".join(
        f"| {row['claim']} | {row['target_display']} | {row['actual_display']} | {_status(row['status'])} |"
        for row in claims
    )

    mc_rows = []
    for name in (
        "hrp_vs_ivp_variance_reduction_pct",
        "hrp_vs_erc_variance_reduction_pct",
        "hrp_vs_ivp_sharpe_improvement_pct",
        "hrp_vs_erc_sharpe_improvement_pct",
        "hrp_vs_ivp_tail_risk_reduction_pct",
        "hrp_vs_erc_tail_risk_reduction_pct",
    ):
        row = mc["summary"][name]
        mc_rows.append(
            f"| {name.replace('_', ' ')} | {_number(row['estimate'])} | "
            f"[{_number(row['ci_lower'])}, {_number(row['ci_upper'])}] |"
        )

    strategy_rows = []
    strategy_labels = {
        "calendar_hrp": "Calendar HRP (Ledoit-Wolf)",
        "adaptive_calendar_hrp": "Calendar adaptive-covariance HRP",
        "vlstar_fully_invested": "Triggered VLSTAR-HRP (fully invested)",
        "vlstar_risk_controlled": "Triggered VLSTAR-HRP (80% high regime)",
    }
    for strategy, values in backtests["strategy_metrics"].items():
        strategy_rows.append(
            f"| {strategy_labels.get(strategy, strategy)} | {_percent(values['annual_return'])} | "
            f"{_percent(values['annual_volatility'])} | {_number(values['sharpe'])} | "
            f"{_percent(values['max_drawdown'])} | {_percent(values['expected_shortfall_95'])} | "
            f"{int(values['rebalances'])} | {_number(values['annual_turnover'])}x |"
        )

    actual = benchmark["actual_crsp"]
    synthetic_rows = "\n".join(
        f"| {row['n_assets']} | {_number(row['speedups']['legacy_pandas_over_stack_index'], 2)}x | "
        f"{_number(row['speedups']['recursive_dfs_over_stack_index'], 2)}x |"
        for row in benchmark["synthetic"]["results"]
    )

    comparison = backtests["comparisons"]["vlstar_risk_controlled_vs_calendar_hrp"]
    pure_comparison = backtests["comparisons"]["vlstar_fully_invested_vs_calendar_hrp"]

    report = f"""# HRP portfolio optimization reproduction

Generated: {context['generated_at']}  
Configuration seed: `{context['seed']}`

## Outcome

The project is fully implemented and reproducible, but the locked experiment does **not** automatically inherit the four numbers in the original project description. The table below reports the measured outcomes without target-fitting.

| Claim | Target | Measured | Audit |
|---|---:|---:|---|
{claim_rows}

## Data

Cornell's WRDS access did not include CSMAR/ChiNext (`csmar_trade`), so the empirical sample is labeled **WRDS CRSP U.S. large-cap equities**, not ChiNext.

- Source: {manifest['source']}
- WRDS query: `{manifest['wrds_query_id']}`
- Raw rows: {manifest['row_count']:,}; clean panel: {manifest['panel_observations']:,} dates x {manifest['asset_count']} assets
- Date range: {manifest['start_date']} through {manifest['end_date']}
- Duplicate asset/date rows collapsed: {manifest['duplicate_asset_date_rows']}
- Assets dropped by the predeclared 98% coverage rule: {manifest['dropped_assets']}
- Raw SHA-256: `{manifest['raw_sha256']}`

The fixed list is an algorithm-reproduction universe, not a survivorship-free market backtest. Raw licensed WRDS data is intentionally excluded from the deliverable archive.

## HRP versus risk parity: factor Student-t copula

The primary comparator is inverse-variance allocation (IVP), matching López de Prado's original HRP paper. Equal-risk-contribution (ERC) is a separately labeled robustness check. Every method sees the same simulated training and held-out paths; weights are re-estimated every {mc['config']['rebalance_every']} test days from the preceding {mc['config']['train_size']} observations.

- Trials: {mc['config']['n_trials']}
- Held-out days per trial: {mc['config']['test_size']}
- Assets/factors/copula df: {mc['config']['n_assets']} / {mc['config']['n_factors']} / {_number(mc['config']['copula_df'], 1)}

| Paired metric | Estimate (%) | 95% bootstrap CI |
|---|---:|---:|
{chr(10).join(mc_rows)}

The 32% target refers to **variance**, not volatility. Its locked primary estimate is the HRP-versus-IVP row above.

## Quasi-diagonalization benchmark

All timed implementations were first checked against `scipy.cluster.hierarchy.leaves_list`. Timings include only linkage-tree-to-leaf-order traversal. The measured median across {actual['tree_count']} linkage matrices built from rolling CRSP windows was **{_number(actual['median_legacy_over_stack'], 2)}x** legacy-pandas/stack (95% matrix-bootstrap CI [{_number(actual['ci_lower'], 2)}, {_number(actual['ci_upper'], 2)}]). Recursive/stack was {_number(actual['median_recursive_over_stack'], 2)}x. Thus the measured redesign speedup is not attributed to recursion removal alone.

| Synthetic assets | Legacy pandas / stack | Recursive / stack |
|---:|---:|---:|
{synthetic_rows}

## VLSTAR-style regime HRP

The restricted model uses point-in-time log market realized variance, median asset variance, and mean correlation. A chronological grid/ridge fit produces a smooth transition probability. The probability blends 21- and 126-day-half-life covariance estimates and drives a hysteretic trigger. Signals are lagged one observation; portfolios use drifted pre-trade weights and 10 bps one-way risky-notional cost.

| Strategy | Ann. return | Ann. vol | Sharpe | Max drawdown | Daily 95% ES | Rebalances | Ann. turnover |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(strategy_rows)}

The fully invested ablation changed Sharpe by {_number(pure_comparison['sharpe_improvement_pct'])}% and 95% ES by {_number(pure_comparison['tail_risk_reduction_pct'])}% versus calendar HRP. The explicitly labeled 80%-risky high-regime overlay changed Sharpe by {_number(comparison['sharpe_improvement_pct'])}% and 95% ES by {_number(comparison['tail_risk_reduction_pct'])}%. The headline regime claim audit uses the risk-controlled variant and does not hide the fully invested result.

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
"""
    output.write_text(report, encoding="utf-8")
    return output
