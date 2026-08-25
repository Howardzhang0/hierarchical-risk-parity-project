"""Equal- and general-risk-budget portfolios solved with SciPy."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from hrp_lab.risk.covariance import (
    estimate_covariance,
    risk_contributions,
    sanitize_covariance,
)


FloatArray = NDArray[np.float64]


def erc_weights(
    covariance: ArrayLike,
    *,
    budgets: ArrayLike | None = None,
    tol: float = 1.0e-10,
    maxiter: int = 1_000,
) -> FloatArray:
    """Return a long-only risk-budget portfolio.

    The implementation solves the convex Spinu formulation

    ``0.5 * x.T @ covariance @ x - budgets.T @ log(x)``

    with analytic gradient and positive bounds, then normalizes ``x``.  Equal
    budgets give the conventional Equal Risk Contribution portfolio.
    """

    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and strictly positive")
    if not isinstance(maxiter, int) or maxiter <= 0:
        raise ValueError("maxiter must be a positive integer")

    matrix = sanitize_covariance(covariance)
    n_assets = matrix.shape[0]
    if n_assets == 1:
        return np.ones(1, dtype=np.float64)

    if budgets is None:
        target = np.full(n_assets, 1.0 / n_assets, dtype=np.float64)
    else:
        target = np.asarray(budgets, dtype=np.float64)
        if target.ndim != 1 or target.shape[0] != n_assets:
            raise ValueError("budgets length must match covariance dimensions")
        if not np.all(np.isfinite(target)) or np.any(target <= 0.0):
            raise ValueError("budgets must be finite and strictly positive")
        target = target / np.sum(target)

    # Scalar covariance rescaling leaves normalized risk-budget weights
    # unchanged and greatly improves optimizer conditioning for daily returns.
    covariance_scale = float(np.mean(np.diag(matrix)))
    scaled = matrix / covariance_scale
    initial = np.sqrt(target / np.maximum(np.diag(scaled), np.finfo(float).tiny))
    lower_bound = np.sqrt(np.finfo(np.float64).tiny)

    def objective(x: FloatArray) -> float:
        return float(0.5 * x @ scaled @ x - target @ np.log(x))

    def gradient(x: FloatArray) -> FloatArray:
        return np.asarray(scaled @ x - target / x, dtype=np.float64)

    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="L-BFGS-B",
        bounds=[(lower_bound, None)] * n_assets,
        options={
            # Risk-budget accuracy is controlled by the gradient.  A looser
            # function-decrease tolerance can stop L-BFGS-B while component
            # contributions are still a few parts per million from target.
            "ftol": min(tol, 10.0 * np.finfo(np.float64).eps),
            "gtol": tol,
            "maxiter": maxiter,
            "maxls": 50,
        },
    )
    solution = np.asarray(result.x, dtype=np.float64)
    if not np.all(np.isfinite(solution)) or np.any(solution <= 0.0):
        raise RuntimeError(f"ERC optimization failed: {result.message}")

    weights = solution / np.sum(solution)
    achieved = risk_contributions(weights, matrix, normalize=True)
    maximum_error = float(np.max(np.abs(achieved - target)))
    convergence_threshold = max(1.0e-6, 100.0 * tol)
    if maximum_error > convergence_threshold:
        raise RuntimeError(
            "ERC optimization did not reach the requested risk budgets; "
            f"maximum contribution error={maximum_error:.3e}; {result.message}"
        )
    return np.asarray(weights, dtype=np.float64)


def erc_weights_from_returns(
    returns: ArrayLike,
    *,
    covariance_method: str = "ledoit_wolf",
    budgets: ArrayLike | None = None,
    tol: float = 1.0e-10,
    maxiter: int = 1_000,
) -> FloatArray:
    """Estimate covariance from returns and compute risk-budget weights."""

    covariance = estimate_covariance(returns, method=covariance_method)  # type: ignore[arg-type]
    return erc_weights(
        covariance,
        budgets=budgets,
        tol=tol,
        maxiter=maxiter,
    )
