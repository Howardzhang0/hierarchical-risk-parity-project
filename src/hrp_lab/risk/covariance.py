"""Covariance estimation, validation, and portfolio-risk helpers.

All public functions return plain NumPy arrays/scalars.  Labels belong at the
data and reporting boundaries; keeping this module index based makes the hot
allocation path both predictable and fast.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _square_float_matrix(matrix: ArrayLike, *, name: str) -> FloatArray:
    """Return *matrix* as a finite, non-empty, square float64 array."""

    result = np.asarray(matrix, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] != result.shape[1]:
        raise ValueError(f"{name} must be a non-empty square matrix")
    if result.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty square matrix")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return result


def sanitize_covariance(
    covariance: ArrayLike,
    *,
    eigenvalue_floor: float = 1.0e-10,
) -> FloatArray:
    """Symmetrize a covariance matrix and project it to positive definite.

    ``eigenvalue_floor`` is relative to the matrix's spectral scale.  This
    preserves the units of daily, weekly, and annual covariance matrices while
    repairing small negative eigenvalues caused by estimation or roundoff.
    """

    if not np.isfinite(eigenvalue_floor) or eigenvalue_floor <= 0.0:
        raise ValueError("eigenvalue_floor must be finite and strictly positive")

    matrix = _square_float_matrix(covariance, name="covariance")
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)

    largest_variance = float(np.max(np.diag(symmetric)))
    spectral_scale = max(float(np.max(np.abs(eigenvalues))), largest_variance)
    if not np.isfinite(spectral_scale) or spectral_scale <= 0.0:
        raise ValueError("covariance must contain at least one positive variance")

    floor = eigenvalue_floor * spectral_scale
    clipped = np.maximum(eigenvalues, floor)
    repaired = (eigenvectors * clipped) @ eigenvectors.T
    repaired = 0.5 * (repaired + repaired.T)
    return np.asarray(repaired, dtype=np.float64)


def covariance_to_correlation(
    covariance: ArrayLike,
    *,
    eigenvalue_floor: float = 1.0e-10,
) -> FloatArray:
    """Convert a covariance matrix to a sanitized correlation matrix."""

    matrix = sanitize_covariance(
        covariance,
        eigenvalue_floor=eigenvalue_floor,
    )
    standard_deviations = np.sqrt(np.diag(matrix))
    denominator = np.outer(standard_deviations, standard_deviations)
    correlation = matrix / denominator
    correlation = np.clip(0.5 * (correlation + correlation.T), -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    return np.asarray(correlation, dtype=np.float64)


def estimate_covariance(
    returns: ArrayLike,
    *,
    method: Literal["sample", "ledoit_wolf"] = "ledoit_wolf",
    ddof: int = 1,
    eigenvalue_floor: float = 1.0e-10,
) -> FloatArray:
    """Estimate a covariance matrix from rows of observations.

    Missing observations are rejected deliberately.  The data layer must make
    its missing-data policy explicit rather than silently turning a pairwise
    covariance matrix into a potentially indefinite one.
    """

    observations = np.asarray(returns, dtype=np.float64)
    if observations.ndim == 1:
        observations = observations.reshape(-1, 1)
    if observations.ndim != 2 or observations.shape[1] == 0:
        raise ValueError("returns must be a two-dimensional observation matrix")
    if observations.shape[0] < 2:
        raise ValueError("at least two return observations are required")
    if not np.all(np.isfinite(observations)):
        raise ValueError("returns contains NaN or infinite values")

    if method == "sample":
        if not isinstance(ddof, int) or ddof < 0 or ddof >= observations.shape[0]:
            raise ValueError("ddof must be an integer in [0, n_observations)")
        covariance = np.cov(observations, rowvar=False, ddof=ddof)
        covariance = np.atleast_2d(covariance)
    elif method == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf

        covariance = LedoitWolf(assume_centered=False).fit(observations).covariance_
    else:
        raise ValueError("method must be 'sample' or 'ledoit_wolf'")

    return sanitize_covariance(
        covariance,
        eigenvalue_floor=eigenvalue_floor,
    )


def portfolio_variance(weights: ArrayLike, covariance: ArrayLike) -> float:
    """Return ``w.T @ covariance @ w`` after strict shape validation."""

    matrix = _square_float_matrix(covariance, name="covariance")
    vector = np.asarray(weights, dtype=np.float64)
    if vector.ndim != 1 or vector.shape[0] != matrix.shape[0]:
        raise ValueError("weights length must match covariance dimensions")
    if not np.all(np.isfinite(vector)):
        raise ValueError("weights contains NaN or infinite values")
    return float(vector @ matrix @ vector)


def risk_contributions(
    weights: ArrayLike,
    covariance: ArrayLike,
    *,
    normalize: bool = True,
) -> FloatArray:
    """Return component variance contributions, optionally as total shares."""

    matrix = _square_float_matrix(covariance, name="covariance")
    vector = np.asarray(weights, dtype=np.float64)
    if vector.ndim != 1 or vector.shape[0] != matrix.shape[0]:
        raise ValueError("weights length must match covariance dimensions")
    if not np.all(np.isfinite(vector)):
        raise ValueError("weights contains NaN or infinite values")

    contributions = vector * (matrix @ vector)
    if not normalize:
        return np.asarray(contributions, dtype=np.float64)

    total = float(np.sum(contributions))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("portfolio variance must be positive")
    return np.asarray(contributions / total, dtype=np.float64)
