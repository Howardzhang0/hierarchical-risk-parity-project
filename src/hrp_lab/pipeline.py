"""End-to-end, artifact-producing HRP reproduction pipeline."""

from __future__ import annotations

import gzip
import json
import math
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from hrp_lab.backtest import (
    BacktestResult,
    run_calendar_hrp_backtest,
    run_regime_hrp_backtest,
)
from hrp_lab.clustering.linkage import linkage_from_covariance
from hrp_lab.config import load_config
from hrp_lab.data import build_manifest, load_crsp_export, write_manifest
from hrp_lab.evaluation.benchmark import benchmark_traversal_suite, benchmark_traversals
from hrp_lab.evaluation.metrics import compare_strategies, performance_metrics
from hrp_lab.regimes import blend_ewma_covariances
from hrp_lab.reporting import (
    plot_backtest_wealth,
    plot_monte_carlo_variance,
    plot_regime_probability,
    plot_traversal_speedups,
    write_reproduction_report,
)
from hrp_lab.risk.covariance import estimate_covariance
from hrp_lab.simulation import PairedMonteCarloResult, run_paired_monte_carlo


@dataclass(frozen=True)
class LoadedResearchData:
    panel: Any
    manifest: dict[str, Any]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Index):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(value: Any, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def make_output_dir(path: str | Path | None = None) -> Path:
    if path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path("outputs") / f"reproduction_{stamp}"
    output = Path(path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    return output


def load_research_data(config: dict[str, Any]) -> LoadedResearchData:
    data_config = config["data"]
    raw_path = Path(data_config["path"])
    panel = load_crsp_export(
        raw_path,
        minimum_coverage=float(data_config.get("minimum_coverage", 0.98)),
    )
    with gzip.open(raw_path, "rt", encoding="utf-8", errors="replace") as handle:
        raw_rows = max(sum(1 for _ in handle) - 1, 0)
    manifest = build_manifest(
        raw_path,
        row_count=raw_rows,
        asset_count=panel.returns.shape[1],
        start_date=panel.returns.index.min().date().isoformat(),
        end_date=panel.returns.index.max().date().isoformat(),
        source=data_config.get("source", "WRDS CRSP Daily Stock File"),
        query_id=data_config.get("wrds_query_id"),
        duplicate_rows=panel.duplicate_rows,
    )
    manifest.update(
        {
            "panel_observations": int(panel.returns.shape[0]),
            "clean_long_rows": int(len(panel.long)),
            "dropped_assets": [int(asset) for asset in panel.dropped_assets],
            "asset_labels": {str(key): value for key, value in panel.labels.items()},
            "minimum_coverage": float(data_config.get("minimum_coverage", 0.98)),
        }
    )
    return LoadedResearchData(panel=panel, manifest=manifest)


def run_monte_carlo_phase(
    returns: pd.DataFrame,
    config: dict[str, Any],
) -> PairedMonteCarloResult:
    settings = config["monte_carlo"]
    backtest = config["backtest"]
    return run_paired_monte_carlo(
        returns,
        n_trials=int(settings["replications"]),
        train_size=int(settings["train_observations"]),
        test_size=int(settings["test_observations"]),
        rebalance_every=int(settings.get("rebalance_every", 21)),
        n_factors=int(settings["factors"]),
        copula_df=float(settings["copula_df"]),
        linkage_method=config["hrp"]["linkage_method"],
        annualization=float(backtest["annualization"]),
        risk_free_rate=float(backtest["risk_free_rate"]),
        n_bootstrap=int(settings["bootstrap_replications"]),
        random_state=int(config["seed"]),
    )


def _bootstrap_median_ci(values: np.ndarray, seed: int, draws: int = 2_000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    medians = np.empty(draws, dtype=float)
    for draw in range(draws):
        medians[draw] = np.median(rng.choice(values, size=values.size, replace=True))
    return float(np.median(values)), float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def run_benchmark_phase(returns: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    settings = config["benchmark"]
    lookback = int(config["backtest"]["covariance_lookback"])
    valid_positions = np.arange(lookback - 1, len(returns), dtype=int)
    endpoints = np.unique(np.linspace(valid_positions[0], valid_positions[-1], 12, dtype=int))
    actual_results: list[dict[str, Any]] = []
    for number, endpoint in enumerate(endpoints):
        window = returns.iloc[endpoint - lookback + 1 : endpoint + 1]
        covariance = estimate_covariance(
            window.to_numpy(), method=config["covariance"]["estimator"]
        )
        linkage = linkage_from_covariance(
            covariance,
            method=config["hrp"]["linkage_method"],
            distance_mode=config["hrp"]["distance_mode"],
        )
        result = benchmark_traversals(
            linkage,
            seed=int(config["seed"]) + number,
            repetitions=int(settings["repetitions"]),
            warmups=int(settings["warmups"]),
        )
        result["window_end"] = window.index[-1].isoformat()
        actual_results.append(result)

    legacy = np.array(
        [row["speedups"]["legacy_pandas_over_stack_index"] for row in actual_results]
    )
    recursive = np.array(
        [row["speedups"]["recursive_dfs_over_stack_index"] for row in actual_results]
    )
    median_speedup, lower, upper = _bootstrap_median_ci(legacy, int(config["seed"]))
    synthetic = benchmark_traversal_suite(
        settings["asset_counts"],
        seed=int(config["seed"]),
        repetitions=max(50, int(settings["repetitions"]) // 2),
        warmups=int(settings["warmups"]),
    )
    return {
        "methodology": "linkage to leaf order only; exact scipy equality checked before timing",
        "actual_crsp": {
            "tree_count": len(actual_results),
            "n_assets": int(returns.shape[1]),
            "median_legacy_over_stack": median_speedup,
            "ci_lower": lower,
            "ci_upper": upper,
            "median_recursive_over_stack": float(np.median(recursive)),
            "results": actual_results,
        },
        "synthetic": synthetic,
    }


def _ledoit_wolf_estimator(window: pd.DataFrame) -> np.ndarray:
    return estimate_covariance(window.to_numpy(), method="ledoit_wolf")


def run_backtest_phase(returns: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    backtest = config["backtest"]
    vlstar = config["vlstar"]
    allocation_kwargs = {
        "linkage_method": config["hrp"]["linkage_method"],
        "distance_mode": config["hrp"]["distance_mode"],
    }
    common = {
        "returns": returns,
        "transaction_cost_bps": float(backtest["transaction_cost_bps"]),
        "start": backtest["start"],
        "allocation_kwargs": allocation_kwargs,
    }
    calendar = run_calendar_hrp_backtest(
        **common,
        covariance_lookback=int(backtest["covariance_lookback"]),
        rebalance_frequency=backtest["rebalance_frequency"],
        covariance_estimator=_ledoit_wolf_estimator,
    )

    regime_common = {
        **common,
        "feature_volatility_window": int(vlstar["feature_volatility_window"]),
        "feature_correlation_window": int(vlstar["feature_correlation_window"]),
        "model_lookback": 756,
        "minimum_model_observations": 126,
        "lags": int(vlstar["lags"]),
        "ridge": float(vlstar["ridge"]),
        "gamma_grid": vlstar["gamma_grid"],
        "threshold_quantiles": vlstar["threshold_quantiles"],
        "refit_interval": int(vlstar["refit_interval"]),
        "covariance_lookback": int(vlstar["covariance_window"]),
        "short_half_life": 21.0,
        "long_half_life": 126.0,
        "low_probability": float(vlstar["probability_bands"][0]),
        "high_probability": float(vlstar["probability_bands"][1]),
        "probability_change": float(vlstar["probability_change_trigger"]),
        "minimum_rebalance_gap": int(vlstar["minimum_rebalance_gap"]),
        "force_rebalance_after": 21,
    }
    fully_invested = run_regime_hrp_backtest(
        **regime_common,
        low_regime_risk_scale=1.0,
        high_regime_risk_scale=1.0,
        allow_cash=False,
    )
    supplied_probabilities = fully_invested.probabilities.reindex(returns.index)
    # BacktestResult begins on the first execution date, so restore the first
    # signal-date probability from the execution audit trail before reusing the
    # exact path for the risk-scaling ablation.
    for _, execution in fully_invested.executions.iterrows():
        signal_date = pd.Timestamp(execution["signal_date"])
        probability = execution.get("high_regime_probability", np.nan)
        if signal_date in supplied_probabilities.index and np.isfinite(probability):
            supplied_probabilities.loc[signal_date] = float(probability)
    risk_controlled = run_regime_hrp_backtest(
        **regime_common,
        probabilities=supplied_probabilities,
        low_regime_risk_scale=1.0,
        high_regime_risk_scale=float(vlstar["high_regime_risk_scale"]),
        allow_cash=True,
    )

    probability_ffill = supplied_probabilities.ffill().fillna(0.0)

    def adaptive_covariance(window: pd.DataFrame):
        probability = float(probability_ffill.loc[window.index[-1]])
        return blend_ewma_covariances(
            window,
            probability,
            short_half_life=21.0,
            long_half_life=126.0,
        ).blended

    adaptive_calendar = run_calendar_hrp_backtest(
        **common,
        covariance_lookback=int(vlstar["covariance_window"]),
        rebalance_frequency=backtest["rebalance_frequency"],
        covariance_estimator=adaptive_covariance,
    )
    return {
        "calendar_hrp": calendar,
        "adaptive_calendar_hrp": adaptive_calendar,
        "vlstar_fully_invested": fully_invested,
        "vlstar_risk_controlled": risk_controlled,
    }


def summarize_backtests(
    strategies: dict[str, BacktestResult],
    config: dict[str, Any],
) -> dict[str, Any]:
    common_index = None
    for result in strategies.values():
        common_index = result.returns.index if common_index is None else common_index.intersection(result.returns.index)
    assert common_index is not None
    annualization = int(config["backtest"]["annualization"])
    risk_free = float(config["backtest"]["risk_free_rate"])
    metrics: dict[str, Any] = {}
    for name, result in strategies.items():
        values = performance_metrics(
            result.returns.loc[common_index],
            annualization=annualization,
            risk_free_rate=risk_free,
        )
        years = len(common_index) / annualization
        values.update(
            {
                "gross_sharpe": performance_metrics(
                    result.gross_returns.loc[common_index],
                    annualization=annualization,
                    risk_free_rate=risk_free,
                )["sharpe"],
                "rebalances": int(len(result.executions)),
                "total_cost": float(result.costs.loc[common_index].sum()),
                "annual_turnover": float(result.turnover.loc[common_index].sum() / years),
                "average_risky_weight": float(result.weights.loc[common_index].sum(axis=1).mean()),
            }
        )
        metrics[name] = values

    comparisons: dict[str, Any] = {}
    baseline = strategies["calendar_hrp"].returns.loc[common_index]
    for name in ("adaptive_calendar_hrp", "vlstar_fully_invested", "vlstar_risk_controlled"):
        comparisons[f"{name}_vs_calendar_hrp"] = compare_strategies(
            strategies[name].returns.loc[common_index],
            baseline,
            annualization=annualization,
            risk_free_rate=risk_free,
        )
    return {
        "common_start": common_index.min().isoformat(),
        "common_end": common_index.max().isoformat(),
        "common_observations": int(len(common_index)),
        "strategy_metrics": metrics,
        "comparisons": comparisons,
    }


def _summary_frame_to_dict(summary: pd.DataFrame) -> dict[str, Any]:
    return {
        str(index): {
            key: (None if pd.isna(value) else value)
            for key, value in row.items()
        }
        for index, row in summary.iterrows()
    }


def build_claim_audit(
    monte_carlo: dict[str, Any],
    benchmark: dict[str, Any],
    backtests: dict[str, Any],
) -> list[dict[str, Any]]:
    variance = float(monte_carlo["summary"]["hrp_vs_ivp_variance_reduction_pct"]["estimate"])
    speedup = float(benchmark["actual_crsp"]["median_legacy_over_stack"])
    comparison = backtests["comparisons"]["vlstar_risk_controlled_vs_calendar_hrp"]
    sharpe = float(comparison["sharpe_improvement_pct"])
    tail = float(comparison["tail_risk_reduction_pct"])
    rows = [
        ("HRP out-of-sample variance reduction vs IVP", 32.0, variance, "%"),
        ("legacy pandas to stack/index traversal speedup", 4.0, speedup, "x"),
        ("VLSTAR risk-controlled HRP Sharpe improvement", 20.0, sharpe, "%"),
        ("VLSTAR risk-controlled HRP 95% ES reduction", 12.0, tail, "%"),
    ]
    return [
        {
            "claim": claim,
            "target": target,
            "actual": actual,
            "unit": unit,
            "target_display": f"{target:.1f}{unit}",
            "actual_display": f"{actual:.2f}{unit}",
            "status": "reproduced" if np.isfinite(actual) and actual >= target else "not_reproduced",
        }
        for claim, target, actual, unit in rows
    ]


def save_monte_carlo(result: PairedMonteCarloResult, output: Path) -> dict[str, Any]:
    result.trials.to_csv(output / "monte_carlo_trials.csv.gz", index=False, compression="gzip")
    result.summary.to_csv(output / "monte_carlo_summary.csv")
    np.savez_compressed(output / "monte_carlo_weights.npz", **result.weights)
    payload = {"config": result.config, "summary": _summary_frame_to_dict(result.summary)}
    write_json(payload, output / "monte_carlo_summary.json")
    return payload


def save_backtests(strategies: dict[str, BacktestResult], output: Path) -> None:
    returns = pd.DataFrame({name: result.returns for name, result in strategies.items()})
    returns.to_csv(output / "backtest_returns.csv.gz", compression="gzip")
    for name, result in strategies.items():
        result.weights.to_csv(output / f"weights_{name}.csv.gz", compression="gzip")
        result.executions.to_csv(output / f"executions_{name}.csv")


def run_reproduction(
    config_path: str | Path = "configs/crsp_reproduction.yaml",
    *,
    output_dir: str | Path | None = None,
    monte_carlo_trials: int | None = None,
) -> Path:
    config = load_config(config_path)
    if monte_carlo_trials is not None:
        config["monte_carlo"]["replications"] = int(monte_carlo_trials)
    output = make_output_dir(output_dir)
    (output / "figures").mkdir(exist_ok=True)
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    loaded = load_research_data(config)
    write_manifest(loaded.manifest, output / "data_manifest.json")
    monte_carlo_result = run_monte_carlo_phase(loaded.panel.returns, config)
    monte_carlo_payload = save_monte_carlo(monte_carlo_result, output)
    benchmark = run_benchmark_phase(loaded.panel.returns, config)
    write_json(benchmark, output / "tree_benchmark.json")
    strategies = run_backtest_phase(loaded.panel.returns, config)
    save_backtests(strategies, output)
    backtest_summary = summarize_backtests(strategies, config)
    write_json(backtest_summary, output / "metrics.json")
    claims = build_claim_audit(monte_carlo_payload, benchmark, backtest_summary)
    write_json(claims, output / "claim_audit.json")

    common_index = None
    for result in strategies.values():
        common_index = result.returns.index if common_index is None else common_index.intersection(result.returns.index)
    wealth_returns = pd.DataFrame(
        {name: result.returns.loc[common_index] for name, result in strategies.items()}
    )
    plot_backtest_wealth(wealth_returns, output / "figures" / "backtest_wealth.png")
    plot_monte_carlo_variance(
        monte_carlo_result.trials, output / "figures" / "monte_carlo_variance.png"
    )
    plot_regime_probability(
        strategies["vlstar_fully_invested"].probabilities,
        output / "figures" / "regime_probability.png",
        low=float(config["vlstar"]["probability_bands"][0]),
        high=float(config["vlstar"]["probability_bands"][1]),
    )
    plot_traversal_speedups(benchmark, output / "figures" / "traversal_speedups.png")

    context = {
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": int(config["seed"]),
        "data_manifest": loaded.manifest,
        "monte_carlo": monte_carlo_payload,
        "benchmark": benchmark,
        "backtests": backtest_summary,
        "claim_audit": claims,
    }
    write_reproduction_report(context, output / "report.md")
    write_json(
        {
            "generated_at": context["generated_at"],
            "status": "complete",
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "files": sorted(str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()),
        },
        output / "run_manifest.json",
    )
    return output


def create_source_archive(project_root: str | Path, output_path: str | Path) -> Path:
    """Create a source-only archive; licensed data and generated outputs stay out."""

    root = Path(project_root).resolve()
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    base = destination.with_suffix("")
    archive = shutil.make_archive(
        str(base),
        "zip",
        root_dir=root,
        base_dir=".",
        logger=None,
    )
    # shutil cannot express exclusions; rewrite a compact filtered archive.
    import zipfile

    temporary = Path(archive).with_name(Path(archive).stem + "_filtered.zip")
    with zipfile.ZipFile(archive, "r") as source, zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as target:
        for info in source.infolist():
            name = info.filename.lstrip("./")
            if name.startswith(("data/raw/", "data/processed/", "outputs/", "work/")):
                continue
            if "__pycache__" in name or name.endswith((".pyc", ".DS_Store")):
                continue
            target.writestr(info, source.read(info.filename))
    Path(archive).unlink()
    temporary.replace(destination)
    return destination
