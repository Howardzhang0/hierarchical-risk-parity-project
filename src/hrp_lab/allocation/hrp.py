"""Hierarchical Risk Parity allocation."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hrp_lab.clustering.linkage import DistanceMode, linkage_from_covariance
from hrp_lab.clustering.traversal import leaf_order_recursive, leaf_order_stack
from hrp_lab.risk.covariance import estimate_covariance, sanitize_covariance


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]


def cluster_variance(
    covariance: ArrayLike,
    indices: ArrayLike,
    *,
    min_variance: float = 1.0e-12,
) -> float:
    """Return inverse-variance portfolio variance for one asset cluster."""

    if not np.isfinite(min_variance) or min_variance <= 0.0:
        raise ValueError("min_variance must be finite and strictly positive")
    matrix = sanitize_covariance(covariance)
    positions = np.asarray(indices, dtype=np.intp)
    if positions.ndim != 1 or positions.size == 0:
        raise ValueError("indices must be a non-empty one-dimensional array")
    if np.any(positions < 0) or np.any(positions >= matrix.shape[0]):
        raise ValueError("cluster index is outside covariance dimensions")
    if np.unique(positions).size != positions.size:
        raise ValueError("cluster indices must be unique")

    submatrix = matrix[np.ix_(positions, positions)]
    inverse_variances = 1.0 / np.maximum(np.diag(submatrix), min_variance)
    inverse_variances /= np.sum(inverse_variances)
    variance = float(inverse_variances @ submatrix @ inverse_variances)
    return max(variance, min_variance)


def _cluster_variance_sanitized(
    covariance: FloatArray,
    indices: IntArray,
    min_variance: float,
) -> float:
    submatrix = covariance[np.ix_(indices, indices)]
    inverse_variances = 1.0 / np.maximum(np.diag(submatrix), min_variance)
    inverse_variances /= np.sum(inverse_variances)
    variance = float(inverse_variances @ submatrix @ inverse_variances)
    return max(variance, min_variance)


def recursive_bisection(
    covariance: ArrayLike,
    ordered_indices: ArrayLike,
    *,
    min_variance: float = 1.0e-12,
) -> FloatArray:
    """Allocate ordered assets by recursively comparing cluster variances."""

    if not np.isfinite(min_variance) or min_variance <= 0.0:
        raise ValueError("min_variance must be finite and strictly positive")
    matrix = sanitize_covariance(covariance)
    order = np.asarray(ordered_indices, dtype=np.intp)
    n_assets = matrix.shape[0]
    if order.shape != (n_assets,) or not np.array_equal(
        np.sort(order),
        np.arange(n_assets, dtype=np.intp),
    ):
        raise ValueError("ordered_indices must be a permutation of all assets")

    weights = np.ones(n_assets, dtype=np.float64)
    stack: list[IntArray] = [order]
    while stack:
        cluster = stack.pop()
        if cluster.size <= 1:
            continue
        split = cluster.size // 2
        left = cluster[:split]
        right = cluster[split:]
        left_variance = _cluster_variance_sanitized(matrix, left, min_variance)
        right_variance = _cluster_variance_sanitized(matrix, right, min_variance)
        left_fraction = right_variance / (left_variance + right_variance)
        weights[left] *= left_fraction
        weights[right] *= 1.0 - left_fraction
        if right.size > 1:
            stack.append(right)
        if left.size > 1:
            stack.append(left)

    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("HRP produced invalid weights")
    weights /= total
    return weights


def hrp_weights(
    covariance: ArrayLike,
    *,
    linkage_method: str = "single",
    distance_mode: DistanceMode = "original",
    ordering: Literal["stack", "recursive"] = "stack",
    optimal_ordering: bool = False,
    min_variance: float = 1.0e-12,
) -> FloatArray:
    """Compute long-only HRP weights in the covariance's original order."""

    matrix = sanitize_covariance(covariance)
    if matrix.shape[0] == 1:
        return np.ones(1, dtype=np.float64)

    linkage_matrix = linkage_from_covariance(
        matrix,
        method=linkage_method,
        distance_mode=distance_mode,
        optimal_ordering=optimal_ordering,
    )
    if ordering == "stack":
        order = leaf_order_stack(linkage_matrix)
    elif ordering == "recursive":
        order = leaf_order_recursive(linkage_matrix)
    else:
        raise ValueError("ordering must be 'stack' or 'recursive'")
    return recursive_bisection(
        matrix,
        order,
        min_variance=min_variance,
    )


def hrp_weights_from_returns(
    returns: ArrayLike,
    *,
    covariance_method: str = "ledoit_wolf",
    linkage_method: str = "single",
    distance_mode: DistanceMode = "original",
    ordering: Literal["stack", "recursive"] = "stack",
    optimal_ordering: bool = False,
    min_variance: float = 1.0e-12,
) -> FloatArray:
    """Estimate covariance from returns and compute HRP weights."""

    covariance = estimate_covariance(returns, method=covariance_method)  # type: ignore[arg-type]
    return hrp_weights(
        covariance,
        linkage_method=linkage_method,
        distance_mode=distance_mode,
        ordering=ordering,
        optimal_ordering=optimal_ordering,
        min_variance=min_variance,
    )
