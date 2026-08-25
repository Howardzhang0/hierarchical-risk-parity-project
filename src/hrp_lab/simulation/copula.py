"""Factor Student-t copula simulation with empirical return marginals.

The implementation deliberately separates dependence and marginal estimation:

* rank-transformed observations estimate a Student-t copula dependence model;
* a low-rank eigenfactor model makes that dependence scalable; and
* simulated uniforms are mapped through each asset's empirical quantile
  function, so no Gaussian assumption is imposed on marginal returns.

The random seed is explicit and fit itself contains no random operations.  A
model constructed with the default ``random_state=0`` therefore produces the
same sample on repeated calls to :meth:`FactorStudentTCopula.simulate`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.stats import rankdata, t


FloatArray = NDArray[np.float64]


def _validate_probabilities(probabilities: ArrayLike) -> FloatArray:
    probs = np.asarray(probabilities, dtype=float)
    if not np.all(np.isfinite(probs)):
        raise ValueError("probabilities must be finite")
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    return probs


def _one_dimensional_empirical_inverse(
    observations: FloatArray, probabilities: FloatArray
) -> FloatArray:
    finite = observations[np.isfinite(observations)]
    if finite.size < 2:
        raise ValueError("each empirical marginal needs at least two observations")
    ordered = np.sort(finite)
    grid = np.linspace(0.0, 1.0, ordered.size)
    return np.interp(probabilities, grid, ordered)


def empirical_inverse_cdf(
    observations: ArrayLike, probabilities: ArrayLike
) -> FloatArray:
    """Invert one or more empirical marginal distributions.

    Parameters
    ----------
    observations:
        A one-dimensional sample, or a two-dimensional matrix whose columns
        are asset return samples. Non-finite values are omitted separately
        from each marginal.
    probabilities:
        Values in ``[0, 1]``. For matrix observations, the final dimension
        must equal the number of asset columns.

    Returns
    -------
    numpy.ndarray
        Linearly interpolated empirical quantiles with the same shape as
        ``probabilities``.
    """

    sample = np.asarray(observations, dtype=float)
    probs = _validate_probabilities(probabilities)

    if sample.ndim == 1:
        return _one_dimensional_empirical_inverse(sample, probs)
    if sample.ndim != 2:
        raise ValueError("observations must be one- or two-dimensional")
    if probs.ndim == 0 or probs.shape[-1] != sample.shape[1]:
        raise ValueError(
            "the final probabilities dimension must match the number of marginals"
        )

    result = np.empty_like(probs, dtype=float)
    for column in range(sample.shape[1]):
        result[..., column] = _one_dimensional_empirical_inverse(
            sample[:, column], probs[..., column]
        )
    return result


def _nearest_correlation(matrix: FloatArray, min_eigenvalue: float) -> FloatArray:
    """Return a symmetric positive-definite correlation approximation."""

    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
    positive = (eigenvectors * eigenvalues) @ eigenvectors.T
    scales = np.sqrt(np.maximum(np.diag(positive), min_eigenvalue))
    correlation = positive / np.outer(scales, scales)
    correlation = (correlation + correlation.T) / 2.0
    np.fill_diagonal(correlation, 1.0)
    return correlation


class FactorStudentTCopula:
    """Low-rank Student-t copula with nonparametric marginal distributions.

    ``n_factors`` leading eigenvectors of the latent rank correlation form the
    common component. The remaining diagonal variance is simulated as
    idiosyncratic noise. A shared inverse-chi-square scale then turns the
    Gaussian factor draw into a multivariate Student-t draw.
    """

    def __init__(
        self,
        *,
        n_factors: int = 3,
        df: float = 6.0,
        correlation_shrinkage: float = 0.05,
        min_eigenvalue: float = 1e-8,
        min_idiosyncratic_variance: float = 1e-8,
        random_state: int | None = 0,
    ) -> None:
        if not isinstance(n_factors, (int, np.integer)) or n_factors < 1:
            raise ValueError("n_factors must be a positive integer")
        if not np.isfinite(df) or df <= 2.0:
            raise ValueError("df must be finite and greater than 2")
        if not 0.0 <= correlation_shrinkage <= 1.0:
            raise ValueError("correlation_shrinkage must lie in [0, 1]")
        if min_eigenvalue <= 0.0 or min_idiosyncratic_variance <= 0.0:
            raise ValueError("variance floors must be positive")

        self.n_factors = int(n_factors)
        self.df = float(df)
        self.correlation_shrinkage = float(correlation_shrinkage)
        self.min_eigenvalue = float(min_eigenvalue)
        self.min_idiosyncratic_variance = float(min_idiosyncratic_variance)
        self.random_state = random_state

    def fit(self, returns: ArrayLike | pd.DataFrame) -> "FactorStudentTCopula":
        """Fit dependence and marginal distributions to historical returns."""

        is_dataframe = isinstance(returns, pd.DataFrame)
        if is_dataframe:
            feature_names: pd.Index[Any] | None = returns.columns.copy()
            values = returns.to_numpy(dtype=float, copy=True)
        else:
            feature_names = None
            values = np.asarray(returns, dtype=float)

        if values.ndim != 2:
            raise ValueError("returns must be a two-dimensional matrix")
        if values.shape[1] < 2:
            raise ValueError("at least two assets are required")

        complete = values[np.all(np.isfinite(values), axis=1)]
        minimum_observations = max(10, complete.shape[1] + 2)
        if complete.shape[0] < minimum_observations:
            raise ValueError(
                f"at least {minimum_observations} complete observations are required"
            )
        if np.any(np.std(complete, axis=0, ddof=1) <= np.finfo(float).eps):
            raise ValueError("return columns must have positive variance")

        n_observations, n_assets = complete.shape
        ranks = np.column_stack(
            [rankdata(complete[:, column], method="average") for column in range(n_assets)]
        )
        uniforms = ranks / (n_observations + 1.0)
        latent = t.ppf(uniforms, df=self.df)
        latent -= latent.mean(axis=0)
        latent /= latent.std(axis=0, ddof=1)

        latent_correlation = np.corrcoef(latent, rowvar=False)
        identity = np.eye(n_assets)
        latent_correlation = (
            (1.0 - self.correlation_shrinkage) * latent_correlation
            + self.correlation_shrinkage * identity
        )
        latent_correlation = _nearest_correlation(
            latent_correlation, self.min_eigenvalue
        )

        eigenvalues, eigenvectors = np.linalg.eigh(latent_correlation)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        fitted_factors = min(self.n_factors, n_assets)

        loadings = eigenvectors[:, :fitted_factors] * np.sqrt(
            np.maximum(eigenvalues[:fitted_factors], self.min_eigenvalue)
        )
        # Eigenvector signs are arbitrary. Canonicalizing them makes fitted
        # parameters reproducible across linear-algebra implementations.
        for factor in range(fitted_factors):
            pivot = int(np.argmax(np.abs(loadings[:, factor])))
            if loadings[pivot, factor] < 0.0:
                loadings[:, factor] *= -1.0

        communalities = np.sum(loadings * loadings, axis=1)
        residual = np.maximum(
            1.0 - communalities, self.min_idiosyncratic_variance
        )
        # Numerical floors can make the diagonal slightly larger than one.
        # Row scaling restores unit marginal variance exactly.
        total_variance = communalities + residual
        loadings /= np.sqrt(total_variance)[:, None]
        residual /= total_variance

        fitted_correlation = loadings @ loadings.T + np.diag(residual)
        fitted_correlation = (fitted_correlation + fitted_correlation.T) / 2.0
        np.fill_diagonal(fitted_correlation, 1.0)

        self.factor_loadings_ = np.asarray(loadings, dtype=float)
        self.idiosyncratic_variance_ = np.asarray(residual, dtype=float)
        self.correlation_ = np.asarray(fitted_correlation, dtype=float)
        self.latent_correlation_ = np.asarray(latent_correlation, dtype=float)
        self.marginals_ = np.sort(complete, axis=0)
        self.n_features_in_ = n_assets
        self.n_observations_ = n_observations
        self.n_factors_ = fitted_factors
        self.feature_names_in_ = feature_names
        self._return_dataframe = is_dataframe
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "factor_loadings_"):
            raise RuntimeError("fit must be called before simulate")

    def _simulate_array(
        self, n_samples: int, random_state: int | None = None
    ) -> FloatArray:
        self._check_fitted()
        if not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer")

        seed = self.random_state if random_state is None else random_state
        rng = np.random.default_rng(seed)
        common_factors = rng.standard_normal((n_samples, self.n_factors_))
        idiosyncratic = rng.standard_normal((n_samples, self.n_features_in_))
        gaussian = (
            common_factors @ self.factor_loadings_.T
            + idiosyncratic * np.sqrt(self.idiosyncratic_variance_)
        )
        scale = np.sqrt(rng.chisquare(self.df, size=n_samples) / self.df)
        latent_t = gaussian / scale[:, None]
        uniforms = t.cdf(latent_t, df=self.df)
        return empirical_inverse_cdf(self.marginals_, uniforms)

    def simulate(
        self, n_samples: int, *, random_state: int | None = None
    ) -> FloatArray | pd.DataFrame:
        """Simulate a return matrix from the fitted copula.

        Passing the same ``random_state`` produces bitwise-identical draws.
        If it is omitted, the seed supplied at construction is reused.
        """

        simulated = self._simulate_array(n_samples, random_state=random_state)
        if self._return_dataframe:
            return pd.DataFrame(simulated, columns=self.feature_names_in_)
        return simulated

    def simulate_paths(
        self,
        n_paths: int,
        path_length: int,
        *,
        random_state: int | None = None,
    ) -> FloatArray:
        """Simulate independent paths as ``(path, time, asset)`` arrays."""

        if not isinstance(n_paths, (int, np.integer)) or n_paths < 1:
            raise ValueError("n_paths must be a positive integer")
        if not isinstance(path_length, (int, np.integer)) or path_length < 1:
            raise ValueError("path_length must be a positive integer")
        flat = self._simulate_array(
            int(n_paths) * int(path_length), random_state=random_state
        )
        return flat.reshape(int(n_paths), int(path_length), self.n_features_in_)


# Short alias convenient in notebooks and configuration-driven code.
FactorTCopula = FactorStudentTCopula

