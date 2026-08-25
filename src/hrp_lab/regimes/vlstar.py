"""A practical restricted VLSTAR-style volatility regime estimator.

The nonlinear parameters are selected on a chronological holdout.  Conditional
on a logistic transition function, the two-regime vector autoregression is a
multi-output ridge regression, which makes rolling refits stable and cheap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


FEATURE_COLUMNS = (
    "log_market_rv",
    "log_median_asset_rv",
    "fisher_mean_correlation",
)


def _as_feature_array(
    features: pd.DataFrame | np.ndarray,
    *,
    minimum_observations: int = 4,
) -> tuple[np.ndarray, pd.Index | None]:
    if isinstance(features, pd.DataFrame):
        values = features.to_numpy(dtype=float)
        index: pd.Index | None = features.index
    else:
        values = np.asarray(features, dtype=float)
        index = None
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("restricted VLSTAR expects exactly three feature columns")
    if values.shape[0] < minimum_observations:
        raise ValueError(
            f"at least {minimum_observations} complete feature observations are required"
        )
    if not np.isfinite(values).all():
        raise ValueError("features must contain only finite values")
    return values, index


def _positive_scale(values: np.ndarray) -> np.ndarray:
    scale = np.std(values, axis=0, ddof=1)
    return np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)


def _lagged_design(values: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    if lags < 1:
        raise ValueError("lags must be at least one")
    if values.shape[0] <= lags:
        raise ValueError("feature history is shorter than the requested lag order")
    lagged = np.concatenate(
        [values[lags - lag : values.shape[0] - lag] for lag in range(1, lags + 1)],
        axis=1,
    )
    return lagged, values[lags:]


def _logistic(value: np.ndarray, gamma: float, threshold: float) -> np.ndarray:
    argument = np.clip(gamma * (value - threshold), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-argument))


def _ridge_fit(design: np.ndarray, targets: np.ndarray, ridge: float) -> np.ndarray:
    penalty = np.eye(design.shape[1], dtype=float) * ridge
    # Do not shrink the low-regime or transition-regime intercepts.
    penalty[0, 0] = 0.0
    penalty[design.shape[1] // 2, design.shape[1] // 2] = 0.0
    system = design.T @ design + penalty
    rhs = design.T @ targets
    try:
        return np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(system) @ rhs


def _regime_design(lagged: np.ndarray, probability: np.ndarray) -> np.ndarray:
    base = np.column_stack([np.ones(lagged.shape[0]), lagged])
    return np.column_stack([base, probability[:, None] * base])


@dataclass(frozen=True)
class RestrictedVLSTAR:
    """Fitted common-transition two-regime VLSTAR model.

    ``coefficients`` maps ``[1, lags, G, G*lags]`` into standardized target
    features.  ``threshold`` is expressed in standardized transition units.
    """

    coefficients: np.ndarray
    gamma: float
    threshold: float
    threshold_quantile: float
    lags: int
    ridge: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    transition_mean: float
    transition_scale: float
    validation_mse: float
    observations: int
    feature_names: tuple[str, str, str] = FEATURE_COLUMNS

    def probability(self, transition_value: float | np.ndarray) -> float | np.ndarray:
        """Return the smooth high-volatility regime probability."""

        raw = np.asarray(transition_value, dtype=float)
        standardized = (raw - self.transition_mean) / self.transition_scale
        probability = _logistic(standardized, self.gamma, self.threshold)
        if probability.ndim == 0:
            return float(probability)
        return probability

    def predict_next(self, feature_history: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Forecast the next raw feature vector from information in ``history``."""

        values, _ = _as_feature_array(
            feature_history, minimum_observations=self.lags
        )
        if values.shape[0] < self.lags:
            raise ValueError("insufficient feature history for model lag order")
        normalized = (values - self.feature_mean) / self.feature_scale
        lagged = np.concatenate(
            [normalized[-lag] for lag in range(1, self.lags + 1)]
        )[None, :]
        probability = np.asarray([self.probability(values[-1, 0])])
        prediction = _regime_design(lagged, probability) @ self.coefficients
        return prediction[0] * self.feature_scale + self.feature_mean


def fit_restricted_vlstar(
    features: pd.DataFrame | np.ndarray,
    *,
    lags: int = 1,
    ridge: float = 1e-4,
    gamma_grid: Iterable[float] = (0.5, 1.0, 2.0, 5.0, 10.0),
    threshold_quantiles: Iterable[float] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
    validation_fraction: float = 0.2,
) -> RestrictedVLSTAR:
    """Fit a common-transition, two-regime VLSTAR using grid search and ridge.

    Grid candidates are scored on the last ``validation_fraction`` of the
    rolling window.  All data supplied to this function may be used for the
    final refit; callers must therefore pass only information available at the
    signal date.
    """

    values, _ = _as_feature_array(features)
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    if not 0.0 <= validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in [0, 0.5)")

    gammas = tuple(float(value) for value in gamma_grid)
    quantiles = tuple(float(value) for value in threshold_quantiles)
    if not gammas or any(value <= 0 for value in gammas):
        raise ValueError("gamma_grid must contain positive values")
    if not quantiles or any(not 0.0 < value < 1.0 for value in quantiles):
        raise ValueError("threshold_quantiles must lie strictly between zero and one")

    effective = values.shape[0] - lags
    if effective < max(12, 3 * lags):
        raise ValueError("insufficient observations for a stable restricted VLSTAR fit")
    validation_size = int(np.floor(effective * validation_fraction))
    if validation_fraction > 0 and validation_size < 2:
        validation_size = 2
    train_effective = effective - validation_size
    if train_effective < max(8, 2 * lags):
        raise ValueError("chronological training split is too short")

    # Scaling is estimated on the chronological fitting portion only while
    # nonlinear candidates are selected.
    train_raw_end = lags + train_effective
    selection_mean = np.mean(values[:train_raw_end], axis=0)
    selection_scale = _positive_scale(values[:train_raw_end])
    selected_values = (values - selection_mean) / selection_scale
    lagged, targets = _lagged_design(selected_values, lags)
    transition_lag = selected_values[lags - 1 : -1, 0]

    best: tuple[float, float, float] | None = None
    for gamma in gammas:
        for quantile in quantiles:
            threshold = float(np.quantile(transition_lag[:train_effective], quantile))
            probability = _logistic(transition_lag, gamma, threshold)
            design = _regime_design(lagged, probability)
            coefficients = _ridge_fit(
                design[:train_effective], targets[:train_effective], ridge
            )
            if validation_size:
                residual = (
                    targets[train_effective:]
                    - design[train_effective:] @ coefficients
                )
            else:
                residual = targets - design @ coefficients
            score = float(np.mean(np.square(residual)))
            candidate = (score, gamma, quantile)
            if best is None or candidate < best:
                best = candidate

    assert best is not None
    validation_mse, gamma, quantile = best

    # Refit the selected specification on the complete, currently available
    # rolling window.  This is safe because the caller executes one day later.
    feature_mean = np.mean(values, axis=0)
    feature_scale = _positive_scale(values)
    normalized = (values - feature_mean) / feature_scale
    lagged, targets = _lagged_design(normalized, lags)
    transition_lag = normalized[lags - 1 : -1, 0]
    threshold = float(np.quantile(transition_lag, quantile))
    probability = _logistic(transition_lag, gamma, threshold)
    design = _regime_design(lagged, probability)
    coefficients = _ridge_fit(design, targets, ridge)

    return RestrictedVLSTAR(
        coefficients=coefficients,
        gamma=gamma,
        threshold=threshold,
        threshold_quantile=quantile,
        lags=lags,
        ridge=ridge,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        transition_mean=float(feature_mean[0]),
        transition_scale=float(feature_scale[0]),
        validation_mse=validation_mse,
        observations=values.shape[0],
    )


def transition_probability(
    model: RestrictedVLSTAR,
    transition_value: float | np.ndarray | pd.Series,
) -> float | np.ndarray | pd.Series:
    """Evaluate the fitted logistic transition while preserving a Series index."""

    if isinstance(transition_value, pd.Series):
        values = np.asarray(model.probability(transition_value.to_numpy(dtype=float)))
        return pd.Series(values, index=transition_value.index, name="high_regime_probability")
    return model.probability(transition_value)


def build_volatility_features(
    returns: pd.DataFrame,
    *,
    market_returns: pd.Series | None = None,
    volatility_window: int = 21,
    correlation_window: int = 63,
    epsilon: float = 1e-12,
    dropna: bool = True,
) -> pd.DataFrame:
    """Construct the three observable volatility-state features.

    Every feature at date ``t`` uses returns no later than ``t``.  The caller
    must lag signals by one observation before crediting portfolio returns.
    """

    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns must be a non-empty DataFrame")
    if not returns.index.is_monotonic_increasing or returns.index.has_duplicates:
        raise ValueError("returns index must be unique and sorted")
    if volatility_window < 2 or correlation_window < 2:
        raise ValueError("feature windows must be at least two observations")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    numeric = returns.astype(float)
    if market_returns is None:
        market = numeric.mean(axis=1, skipna=True)
    else:
        market = market_returns.reindex(numeric.index).astype(float)

    market_rv = market.pow(2).rolling(
        volatility_window, min_periods=volatility_window
    ).mean()
    asset_rv = numeric.pow(2).rolling(
        volatility_window, min_periods=volatility_window
    ).mean()
    median_asset_rv = asset_rv.median(axis=1, skipna=True)

    mean_correlations = np.full(numeric.shape[0], np.nan, dtype=float)
    values = numeric.to_numpy(dtype=float)
    for end in range(correlation_window - 1, values.shape[0]):
        window = values[end - correlation_window + 1 : end + 1]
        valid_assets = np.isfinite(window).all(axis=0)
        window = window[:, valid_assets]
        if window.shape[1] < 2:
            continue
        standard_deviation = np.std(window, axis=0, ddof=1)
        window = window[:, standard_deviation > 1e-12]
        if window.shape[1] < 2:
            continue
        correlation = np.corrcoef(window, rowvar=False)
        upper = correlation[np.triu_indices(correlation.shape[0], k=1)]
        upper = upper[np.isfinite(upper)]
        if upper.size:
            mean_correlations[end] = float(np.mean(upper))

    clipped = np.clip(mean_correlations, -0.999999, 0.999999)
    features = pd.DataFrame(
        {
            FEATURE_COLUMNS[0]: np.log(np.maximum(market_rv.to_numpy(), epsilon)),
            FEATURE_COLUMNS[1]: np.log(
                np.maximum(median_asset_rv.to_numpy(), epsilon)
            ),
            FEATURE_COLUMNS[2]: np.arctanh(clipped),
        },
        index=numeric.index,
    )
    return features.dropna() if dropna else features
