"""Leakage-safe exponentially weighted regime covariance estimates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


Covariance = pd.DataFrame | np.ndarray


def _returns_array(
    returns: pd.DataFrame | np.ndarray,
) -> tuple[np.ndarray, pd.Index | None]:
    if isinstance(returns, pd.DataFrame):
        values = returns.to_numpy(dtype=float)
        columns: pd.Index | None = returns.columns
    else:
        values = np.asarray(returns, dtype=float)
        columns = None
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("returns must be a two-dimensional asset-return matrix")
    complete = values[np.isfinite(values).all(axis=1)]
    if complete.shape[0] < 2:
        raise ValueError("at least two complete return observations are required")
    return complete, columns


def _restore_covariance(values: np.ndarray, columns: pd.Index | None) -> Covariance:
    if columns is None:
        return values
    return pd.DataFrame(values, index=columns, columns=columns)


def _floor_eigenvalues(covariance: np.ndarray, floor: float) -> np.ndarray:
    covariance = (covariance + covariance.T) / 2.0
    if floor <= 0:
        return covariance
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = max(float(np.max(eigenvalues)), 1.0)
    eigenvalues = np.maximum(eigenvalues, floor * scale)
    repaired = (eigenvectors * eigenvalues) @ eigenvectors.T
    return (repaired + repaired.T) / 2.0


def ewma_covariance(
    returns: pd.DataFrame | np.ndarray,
    *,
    half_life: float,
    demean: bool = True,
    eigenvalue_floor: float = 1e-10,
) -> Covariance:
    """Estimate a covariance with exponentially decaying observation weights."""

    if half_life <= 0:
        raise ValueError("half_life must be positive")
    if eigenvalue_floor < 0:
        raise ValueError("eigenvalue_floor must be non-negative")
    values, columns = _returns_array(returns)
    age = np.arange(values.shape[0] - 1, -1, -1, dtype=float)
    weights = np.exp(np.log(0.5) * age / half_life)
    weights /= np.sum(weights)
    center = np.sum(values * weights[:, None], axis=0) if demean else 0.0
    centered = values - center
    denominator = 1.0 - float(weights @ weights)
    if denominator <= 0:
        raise ValueError("effective EWMA sample size is too small")
    covariance = (centered * weights[:, None]).T @ centered / denominator
    covariance = _floor_eigenvalues(covariance, eigenvalue_floor)
    return _restore_covariance(covariance, columns)


@dataclass(frozen=True)
class CovarianceBlend:
    """Short, long, and probability-weighted covariance matrices."""

    short: Covariance
    long: Covariance
    blended: Covariance
    high_probability: float


def blend_ewma_covariances(
    returns: pd.DataFrame | np.ndarray,
    high_probability: float,
    *,
    short_half_life: float = 21.0,
    long_half_life: float = 126.0,
    lookback: int | None = None,
    demean: bool = True,
    eigenvalue_floor: float = 1e-10,
) -> CovarianceBlend:
    """Blend short- and long-horizon EWMA covariance estimates.

    ``high_probability=1`` selects the short estimate; zero selects the long
    estimate.  A convex blend preserves positive semidefiniteness.
    """

    probability = float(high_probability)
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("high_probability must lie in [0, 1]")
    if lookback is not None:
        if lookback < 2:
            raise ValueError("lookback must be at least two")
        returns = (
            returns[-lookback:]
            if isinstance(returns, np.ndarray)
            else returns.iloc[-lookback:]
        )

    short = ewma_covariance(
        returns,
        half_life=short_half_life,
        demean=demean,
        eigenvalue_floor=eigenvalue_floor,
    )
    long = ewma_covariance(
        returns,
        half_life=long_half_life,
        demean=demean,
        eigenvalue_floor=eigenvalue_floor,
    )
    short_values = np.asarray(short, dtype=float)
    long_values = np.asarray(long, dtype=float)
    blended_values = probability * short_values + (1.0 - probability) * long_values
    blended_values = _floor_eigenvalues(blended_values, eigenvalue_floor)
    columns = short.columns if isinstance(short, pd.DataFrame) else None
    blended = _restore_covariance(blended_values, columns)
    return CovarianceBlend(
        short=short,
        long=long,
        blended=blended,
        high_probability=probability,
    )
