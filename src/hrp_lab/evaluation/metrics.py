"""Portfolio metrics with explicit sign conventions for losses and tail risk."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def expected_shortfall(returns: pd.Series | np.ndarray, alpha: float = 0.95) -> float:
    """Return positive historical expected loss beyond the alpha quantile."""

    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    cutoff = np.quantile(values, 1.0 - alpha)
    tail = values[values <= cutoff]
    return float(-tail.mean()) if tail.size else 0.0


def max_drawdown(returns: pd.Series | np.ndarray) -> float:
    values = np.asarray(returns, dtype=float)
    wealth = np.cumprod(1.0 + np.nan_to_num(values, nan=0.0))
    peaks = np.maximum.accumulate(np.r_[1.0, wealth])[:-1]
    drawdowns = 1.0 - wealth / peaks
    return float(np.max(drawdowns)) if drawdowns.size else float("nan")


def performance_metrics(
    returns: pd.Series | np.ndarray,
    *,
    annualization: int = 252,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {key: float("nan") for key in (
            "observations", "annual_return", "annual_volatility", "sharpe",
            "max_drawdown", "var_95", "expected_shortfall_95",
        )}
    annual_return = float(np.mean(values) * annualization)
    annual_volatility = float(np.std(values, ddof=1) * np.sqrt(annualization))
    sharpe = (
        (annual_return - risk_free_rate) / annual_volatility
        if annual_volatility > 0
        else float("nan")
    )
    return {
        "observations": int(values.size),
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown(values),
        "var_95": float(-np.quantile(values, 0.05)),
        "expected_shortfall_95": expected_shortfall(values, 0.95),
    }


def _percent_change(new: float, baseline: float, *, improvement_when_lower: bool) -> float:
    if not np.isfinite(new) or not np.isfinite(baseline) or baseline == 0:
        return float("nan")
    numerator = baseline - new if improvement_when_lower else new - baseline
    return float(100.0 * numerator / abs(baseline))


def compare_strategies(
    candidate: pd.Series | np.ndarray,
    baseline: pd.Series | np.ndarray,
    *,
    annualization: int = 252,
    risk_free_rate: float = 0.0,
) -> dict[str, Any]:
    candidate_metrics = performance_metrics(
        candidate, annualization=annualization, risk_free_rate=risk_free_rate
    )
    baseline_metrics = performance_metrics(
        baseline, annualization=annualization, risk_free_rate=risk_free_rate
    )
    return {
        "candidate": candidate_metrics,
        "baseline": baseline_metrics,
        "sharpe_improvement_pct": _percent_change(
            candidate_metrics["sharpe"], baseline_metrics["sharpe"],
            improvement_when_lower=False,
        ),
        "tail_risk_reduction_pct": _percent_change(
            candidate_metrics["expected_shortfall_95"],
            baseline_metrics["expected_shortfall_95"],
            improvement_when_lower=True,
        ),
        "volatility_reduction_pct": _percent_change(
            candidate_metrics["annual_volatility"],
            baseline_metrics["annual_volatility"],
            improvement_when_lower=True,
        ),
    }

