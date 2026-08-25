"""Paired Monte Carlo comparisons of HRP, IVP, and ERC portfolios."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .copula import FactorStudentTCopula
from .metrics import paired_bootstrap_ci, portfolio_statistics


FloatArray = NDArray[np.float64]
METHODS = ("hrp", "ivp", "erc")
BENCHMARKS = ("ivp", "erc")


@dataclass
class PairedMonteCarloResult:
    """Outputs from :func:`run_paired_monte_carlo`.

    ``trials`` has one row per simulated train/test path. Each method-specific
    column is therefore paired by construction. ``weights`` maps method names
    to arrays shaped ``(n_trials, n_rebalances, n_assets)``.
    """

    trials: pd.DataFrame
    summary: pd.DataFrame
    weights: dict[str, FloatArray]
    config: dict[str, Any]
    copula: FactorStudentTCopula = field(repr=False)


def _estimate_covariance(
    returns: FloatArray,
    *,
    shrinkage: float,
    min_variance: float,
) -> FloatArray:
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("covariance_shrinkage must lie in [0, 1]")
    if min_variance <= 0.0:
        raise ValueError("min_variance must be positive")

    covariance = np.asarray(np.cov(returns, rowvar=False, ddof=1), dtype=float)
    covariance = np.atleast_2d(covariance)
    diagonal = np.maximum(np.diag(covariance), min_variance)
    sample = covariance.copy()
    np.fill_diagonal(sample, diagonal)
    target = np.diag(diagonal)
    shrunk = (1.0 - shrinkage) * sample + shrinkage * target
    shrunk = (shrunk + shrunk.T) / 2.0

    eigenvalues, eigenvectors = np.linalg.eigh(shrunk)
    eigenvalues = np.maximum(eigenvalues, min_variance)
    positive = (eigenvectors * eigenvalues) @ eigenvectors.T
    return (positive + positive.T) / 2.0


def _normalize_weights(weights: ArrayLike, n_assets: int, method: str) -> FloatArray:
    array = np.asarray(weights, dtype=float).reshape(-1)
    if array.shape != (n_assets,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{method} returned invalid weights")
    if np.any(array < -1e-8):
        raise ValueError(f"{method} returned materially negative weights")
    array = np.maximum(array, 0.0)
    total = float(np.sum(array))
    if total <= np.finfo(float).eps:
        raise ValueError(f"{method} returned weights with zero total")
    return array / total


def _allocation_weights(
    covariance: FloatArray,
    *,
    linkage_method: str,
    min_variance: float,
) -> dict[str, FloatArray]:
    # Imports are intentionally local. Copula fitting and summary utilities
    # remain usable without importing the allocation optimization stack.
    from hrp_lab.allocation.erc import erc_weights
    from hrp_lab.allocation.hrp import hrp_weights
    from hrp_lab.allocation.ivp import ivp_weights

    n_assets = covariance.shape[0]
    raw = {
        "hrp": hrp_weights(
            covariance,
            linkage_method=linkage_method,
            min_variance=min_variance,
        ),
        "ivp": ivp_weights(covariance, min_variance=min_variance),
        "erc": erc_weights(covariance),
    }
    return {
        method: _normalize_weights(weights, n_assets, method)
        for method, weights in raw.items()
    }


def _safe_percent_change(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or abs(denominator) <= np.finfo(float).eps:
        return np.nan
    return 100.0 * numerator / abs(denominator)


def _relative_reduction(primary: float, benchmark: float) -> float:
    if not np.isfinite(benchmark) or abs(benchmark) <= np.finfo(float).eps:
        return np.nan
    return 100.0 * (1.0 - primary / benchmark)


def summarize_paired_trials(
    trials: pd.DataFrame,
    *,
    n_bootstrap: int = 2_000,
    confidence_level: float = 0.95,
    random_state: int | None = 0,
) -> pd.DataFrame:
    """Summarize method levels and paired improvements with bootstrap CIs.

    The headline comparison is HRP versus IVP, following the original HRP
    Monte Carlo experiment's inverse-variance "traditional risk parity"
    benchmark. HRP versus ERC is reported as a secondary robustness check.
    Reduction percentages use a ratio of paired-trial means, rather than the
    less stable mean of trial-level ratios.
    """

    required = {
        f"{method}_{metric}"
        for method in METHODS
        for metric in ("variance", "sharpe", "expected_shortfall")
    }
    missing = sorted(required.difference(trials.columns))
    if missing:
        raise ValueError(f"trials is missing required columns: {missing}")

    seed_generator = np.random.default_rng(random_state)
    rows: list[dict[str, Any]] = []

    def add_row(
        name: str,
        data: FloatArray,
        statistic: Any,
        *,
        unit: str,
        comparison: str,
    ) -> None:
        seed = int(seed_generator.integers(0, np.iinfo(np.uint32).max))
        estimate, lower, upper = paired_bootstrap_ci(
            data,
            statistic,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_state=seed,
        )
        rows.append(
            {
                "metric": name,
                "comparison": comparison,
                "estimate": estimate,
                "ci_lower": lower,
                "ci_upper": upper,
                "unit": unit,
                "n_trials": int(data.shape[0]),
            }
        )

    for method in METHODS:
        for metric, unit in (
            ("variance", "return_squared"),
            ("sharpe", "ratio"),
            ("expected_shortfall", "return_loss"),
        ):
            data = trials[f"{method}_{metric}"].to_numpy(dtype=float)
            add_row(
                f"{method}_{metric}",
                data,
                lambda sample: float(np.mean(sample)),
                unit=unit,
                comparison=method,
            )

    for benchmark in BENCHMARKS:
        comparison = f"hrp_vs_{benchmark}"

        variance_data = trials[
            ["hrp_variance", f"{benchmark}_variance"]
        ].to_numpy(dtype=float)
        add_row(
            f"{comparison}_variance_reduction_pct",
            variance_data,
            lambda sample: _relative_reduction(
                float(np.mean(sample[:, 0])), float(np.mean(sample[:, 1]))
            ),
            unit="percent",
            comparison=comparison,
        )

        sharpe_data = trials[
            ["hrp_sharpe", f"{benchmark}_sharpe"]
        ].to_numpy(dtype=float)
        add_row(
            f"{comparison}_sharpe_difference",
            sharpe_data,
            lambda sample: float(np.mean(sample[:, 0]) - np.mean(sample[:, 1])),
            unit="ratio_points",
            comparison=comparison,
        )
        add_row(
            f"{comparison}_sharpe_improvement_pct",
            sharpe_data,
            lambda sample: _safe_percent_change(
                float(np.mean(sample[:, 0]) - np.mean(sample[:, 1])),
                float(np.mean(sample[:, 1])),
            ),
            unit="percent",
            comparison=comparison,
        )

        tail_data = trials[
            ["hrp_expected_shortfall", f"{benchmark}_expected_shortfall"]
        ].to_numpy(dtype=float)
        add_row(
            f"{comparison}_tail_risk_reduction_pct",
            tail_data,
            lambda sample: _relative_reduction(
                float(np.mean(sample[:, 0])), float(np.mean(sample[:, 1]))
            ),
            unit="percent",
            comparison=comparison,
        )

    summary = pd.DataFrame(rows).set_index("metric")
    summary.attrs["headline_metric"] = "hrp_vs_ivp_variance_reduction_pct"
    summary.attrs["confidence_level"] = float(confidence_level)
    summary.attrs["bootstrap_method"] = "paired_percentile"
    return summary


def run_paired_monte_carlo(
    historical_returns: ArrayLike | pd.DataFrame,
    *,
    n_trials: int = 250,
    train_size: int = 252,
    test_size: int = 63,
    rebalance_every: int | None = 21,
    n_factors: int = 3,
    copula_df: float = 6.0,
    copula_correlation_shrinkage: float = 0.05,
    covariance_shrinkage: float = 0.05,
    linkage_method: str = "single",
    min_variance: float = 1e-12,
    annualization: float = 252.0,
    risk_free_rate: float = 0.0,
    tail_probability: float = 0.05,
    n_bootstrap: int = 2_000,
    confidence_level: float = 0.95,
    random_state: int | None = 0,
) -> PairedMonteCarloResult:
    """Run a deterministic, paired train/test portfolio experiment.

    Each trial simulates one return block. By default, all three portfolios are
    re-estimated every 21 test observations from the immediately preceding
    ``train_size`` observations, and are then evaluated only on the next
    untouched block. Set ``rebalance_every=None`` for a fixed-weight diagnostic
    fitted once at the train/test boundary. All methods always use the same
    training windows and held-out paths, removing simulation-path noise from
    method-to-method differences.
    """

    if not isinstance(n_trials, (int, np.integer)) or n_trials < 1:
        raise ValueError("n_trials must be a positive integer")
    if not isinstance(train_size, (int, np.integer)) or train_size < 3:
        raise ValueError("train_size must be an integer of at least 3")
    if not isinstance(test_size, (int, np.integer)) or test_size < 2:
        raise ValueError("test_size must be an integer of at least 2")
    if rebalance_every is not None and (
        not isinstance(rebalance_every, (int, np.integer)) or rebalance_every < 1
    ):
        raise ValueError("rebalance_every must be a positive integer or None")
    if not 0.0 < tail_probability < 1.0:
        raise ValueError("tail_probability must lie strictly between 0 and 1")

    copula = FactorStudentTCopula(
        n_factors=n_factors,
        df=copula_df,
        correlation_shrinkage=copula_correlation_shrinkage,
        random_state=random_state,
    ).fit(historical_returns)
    n_assets = copula.n_features_in_
    block_size = int(test_size) if rebalance_every is None else int(rebalance_every)
    n_rebalances = int(np.ceil(int(test_size) / block_size))

    rng = np.random.default_rng(random_state)
    weight_history = {
        method: np.empty((int(n_trials), n_rebalances, n_assets), dtype=float)
        for method in METHODS
    }
    trial_rows: list[dict[str, float | int]] = []

    for trial in range(int(n_trials)):
        draw_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
        simulated = copula.simulate(
            int(train_size) + int(test_size), random_state=draw_seed
        )
        simulated_array = np.asarray(simulated, dtype=float)
        realized_returns = {
            method: np.empty(int(test_size), dtype=float) for method in METHODS
        }
        for rebalance, relative_start in enumerate(
            range(0, int(test_size), block_size)
        ):
            absolute_start = int(train_size) + relative_start
            relative_stop = min(relative_start + block_size, int(test_size))
            absolute_stop = int(train_size) + relative_stop
            training = simulated_array[
                absolute_start - int(train_size) : absolute_start
            ]
            testing_block = simulated_array[absolute_start:absolute_stop]
            covariance = _estimate_covariance(
                training,
                shrinkage=covariance_shrinkage,
                min_variance=min_variance,
            )
            weights = _allocation_weights(
                covariance,
                linkage_method=linkage_method,
                min_variance=min_variance,
            )
            for method in METHODS:
                weight_history[method][trial, rebalance] = weights[method]
                realized_returns[method][relative_start:relative_stop] = (
                    testing_block @ weights[method]
                )

        row: dict[str, float | int] = {
            "trial": trial,
            "draw_seed": draw_seed,
            "n_rebalances": n_rebalances,
        }
        for method in METHODS:
            statistics = portfolio_statistics(
                realized_returns[method],
                annualization=annualization,
                risk_free_rate=risk_free_rate,
                tail_probability=tail_probability,
            )
            for metric, value in statistics.items():
                row[f"{method}_{metric}"] = value

        for benchmark in BENCHMARKS:
            comparison = f"hrp_vs_{benchmark}"
            row[f"{comparison}_variance_reduction_pct"] = _relative_reduction(
                float(row["hrp_variance"]),
                float(row[f"{benchmark}_variance"]),
            )
            row[f"{comparison}_sharpe_difference"] = float(row["hrp_sharpe"]) - float(
                row[f"{benchmark}_sharpe"]
            )
            row[f"{comparison}_tail_risk_reduction_pct"] = _relative_reduction(
                float(row["hrp_expected_shortfall"]),
                float(row[f"{benchmark}_expected_shortfall"]),
            )
        trial_rows.append(row)

    trials = pd.DataFrame(trial_rows).set_index("trial", drop=False)
    bootstrap_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
    summary = summarize_paired_trials(
        trials,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=bootstrap_seed,
    )
    asset_names = (
        list(copula.feature_names_in_)
        if copula.feature_names_in_ is not None
        else list(range(n_assets))
    )
    config: dict[str, Any] = {
        "n_trials": int(n_trials),
        "train_size": int(train_size),
        "test_size": int(test_size),
        "rebalance_every": None if rebalance_every is None else int(rebalance_every),
        "n_rebalances": n_rebalances,
        "weight_mode": "fixed" if rebalance_every is None else "rolling",
        "n_assets": int(n_assets),
        "asset_names": asset_names,
        "n_factors": int(copula.n_factors_),
        "copula_df": float(copula_df),
        "copula_correlation_shrinkage": float(copula_correlation_shrinkage),
        "covariance_shrinkage": float(covariance_shrinkage),
        "linkage_method": linkage_method,
        "min_variance": float(min_variance),
        "annualization": float(annualization),
        "risk_free_rate": float(risk_free_rate),
        "tail_probability": float(tail_probability),
        "n_bootstrap": int(n_bootstrap),
        "confidence_level": float(confidence_level),
        "random_state": random_state,
        "primary_benchmark": "ivp",
        "robustness_benchmark": "erc",
    }
    return PairedMonteCarloResult(
        trials=trials,
        summary=summary,
        weights=weight_history,
        config=config,
        copula=copula,
    )


# Descriptive alias used by callers that do not need to mention Monte Carlo.
run_paired_experiment = run_paired_monte_carlo
