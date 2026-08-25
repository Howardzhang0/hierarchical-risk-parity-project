"""Inverse-variance portfolio baseline."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hrp_lab.risk.covariance import estimate_covariance, sanitize_covariance


FloatArray = NDArray[np.float64]


def ivp_weights(
    covariance: ArrayLike,
    *,
    min_variance: float = 1.0e-12,
) -> FloatArray:
    """Return long-only weights proportional to inverse asset variance."""

    if not np.isfinite(min_variance) or min_variance <= 0.0:
        raise ValueError("min_variance must be finite and strictly positive")
    matrix = sanitize_covariance(covariance)
    inverse_variances = 1.0 / np.maximum(np.diag(matrix), min_variance)
    total = float(np.sum(inverse_variances))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("could not construct inverse-variance weights")
    return np.asarray(inverse_variances / total, dtype=np.float64)


def ivp_weights_from_returns(
    returns: ArrayLike,
    *,
    covariance_method: str = "ledoit_wolf",
    min_variance: float = 1.0e-12,
) -> FloatArray:
    """Estimate covariance from returns and compute inverse-variance weights."""

    covariance = estimate_covariance(returns, method=covariance_method)  # type: ignore[arg-type]
    return ivp_weights(covariance, min_variance=min_variance)
