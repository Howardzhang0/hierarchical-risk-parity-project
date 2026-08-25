"""Correlation distances and hierarchical-linkage construction."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.spatial.distance import pdist, squareform

from hrp_lab.risk.covariance import covariance_to_correlation


FloatArray = NDArray[np.float64]
DistanceMode = Literal["original", "direct"]


def _sanitize_correlation(correlation: ArrayLike) -> FloatArray:
    matrix = np.asarray(correlation, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("correlation must be a non-empty square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("correlation contains NaN or infinite values")
    matrix = np.clip(0.5 * (matrix + matrix.T), -1.0, 1.0)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def correlation_distance(correlation: ArrayLike) -> FloatArray:
    """Return the standard HRP distance ``sqrt((1-correlation)/2)``."""

    matrix = _sanitize_correlation(correlation)
    distance = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - matrix)))
    distance = 0.5 * (distance + distance.T)
    np.fill_diagonal(distance, 0.0)
    return np.asarray(distance, dtype=np.float64)


def linkage_from_correlation(
    correlation: ArrayLike,
    *,
    method: str = "single",
    distance_mode: DistanceMode = "original",
    optimal_ordering: bool = False,
) -> FloatArray:
    """Build a SciPy linkage matrix from asset correlations.

    ``distance_mode='original'`` reproduces the original HRP implementation:
    the rows of the correlation-distance matrix are treated as observations and
    Euclidean distances are computed between those row vectors.

    ``distance_mode='direct'`` instead supplies the upper triangle of the HRP
    distance matrix directly to hierarchical clustering.  It is a common
    modern variant, but is intentionally not the default because it can produce
    a different tree.
    """

    distance = correlation_distance(correlation)
    n_assets = distance.shape[0]
    if n_assets == 1:
        return np.empty((0, 4), dtype=np.float64)

    if distance_mode == "original":
        condensed = pdist(distance, metric="euclidean")
    elif distance_mode == "direct":
        condensed = squareform(distance, checks=False)
    else:
        raise ValueError("distance_mode must be 'original' or 'direct'")

    result = scipy_linkage(
        condensed,
        method=method,
        optimal_ordering=optimal_ordering,
    )
    return np.asarray(result, dtype=np.float64)


def linkage_from_covariance(
    covariance: ArrayLike,
    *,
    method: str = "single",
    distance_mode: DistanceMode = "original",
    optimal_ordering: bool = False,
    eigenvalue_floor: float = 1.0e-10,
) -> FloatArray:
    """Convert covariance to correlation and return its linkage matrix."""

    correlation = covariance_to_correlation(
        covariance,
        eigenvalue_floor=eigenvalue_floor,
    )
    return linkage_from_correlation(
        correlation,
        method=method,
        distance_mode=distance_mode,
        optimal_ordering=optimal_ordering,
    )
