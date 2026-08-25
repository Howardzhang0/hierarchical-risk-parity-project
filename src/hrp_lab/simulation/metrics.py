"""Portfolio and paired-bootstrap metrics used by simulation experiments."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _finite_vector(values: ArrayLike, *, minimum_size: int = 1) -> FloatArray:
    array = np.asarray(values, dtype=float).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size < minimum_size:
        raise ValueError(f"at least {minimum_size} finite observations are required")
    return array


def expected_shortfall(returns: ArrayLike, *, tail_probability: float = 0.05) -> float:
    """Return the positive loss associated with the worst return tail.

    The result is ``-mean(returns[returns <= alpha-quantile])``. Consequently,
    a more severe negative tail produces a larger positive number.
    """

    if not 0.0 < tail_probability < 1.0:
        raise ValueError("tail_probability must lie strictly between 0 and 1")
    values = _finite_vector(returns)
    cutoff = np.quantile(values, tail_probability)
    tail = values[values <= cutoff]
    return float(-np.mean(tail))


def portfolio_statistics(
    returns: ArrayLike,
    *,
    annualization: float = 252.0,
    risk_free_rate: float = 0.0,
    tail_probability: float = 0.05,
) -> dict[str, float]:
    """Compute realized mean, variance, volatility, Sharpe, and tail loss."""

    if not np.isfinite(annualization) or annualization <= 0.0:
        raise ValueError("annualization must be positive")
    values = _finite_vector(returns, minimum_size=2)
    mean_return = float(np.mean(values))
    variance = float(np.var(values, ddof=1))
    volatility = float(np.sqrt(max(variance, 0.0)))
    if volatility <= np.finfo(float).eps:
        sharpe = np.nan
    else:
        sharpe = float(
            (mean_return - risk_free_rate) / volatility * np.sqrt(annualization)
        )
    return {
        "mean_return": mean_return,
        "variance": variance,
        "volatility": volatility,
        "annualized_return": mean_return * annualization,
        "annualized_volatility": volatility * np.sqrt(annualization),
        "sharpe": sharpe,
        "expected_shortfall": expected_shortfall(
            values, tail_probability=tail_probability
        ),
    }


def paired_bootstrap_ci(
    values: ArrayLike,
    statistic: Callable[[FloatArray], float],
    *,
    n_bootstrap: int = 2_000,
    confidence_level: float = 0.95,
    random_state: int | None = 0,
) -> tuple[float, float, float]:
    """Estimate a statistic and percentile CI by resampling paired rows.

    For a matrix input, every bootstrap draw resamples entire rows. This is the
    key property needed for valid HRP-versus-benchmark paired comparisons.
    """

    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1")

    data = np.asarray(values, dtype=float)
    if data.ndim == 1:
        data = data[np.isfinite(data)]
    elif data.ndim == 2:
        data = data[np.all(np.isfinite(data), axis=1)]
    else:
        raise ValueError("values must be one- or two-dimensional")
    if data.shape[0] == 0:
        raise ValueError("values contain no complete finite rows")

    estimate = float(statistic(data))
    rng = np.random.default_rng(random_state)
    bootstrapped = np.empty(int(n_bootstrap), dtype=float)
    for draw in range(int(n_bootstrap)):
        indices = rng.integers(0, data.shape[0], size=data.shape[0])
        bootstrapped[draw] = float(statistic(data[indices]))
    bootstrapped = bootstrapped[np.isfinite(bootstrapped)]
    if bootstrapped.size == 0:
        return estimate, np.nan, np.nan

    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(bootstrapped, [alpha, 1.0 - alpha])
    return estimate, float(lower), float(upper)

