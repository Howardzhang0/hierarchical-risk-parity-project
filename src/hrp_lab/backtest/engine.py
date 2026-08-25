"""Leakage-controlled HRP and regime-HRP backtest engines.

A target formed with information through observation ``t`` is executed before
the return at ``t+1``.  Target weights are never credited with the same return
used to estimate them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from hrp_lab.regimes import (
    RegimeTrigger,
    RestrictedVLSTAR,
    TriggerConfig,
    blend_ewma_covariances,
    build_volatility_features,
    fit_restricted_vlstar,
    transition_probability,
)


Allocator = Callable[..., np.ndarray | pd.Series]
CovarianceEstimator = Callable[[pd.DataFrame], pd.DataFrame | np.ndarray]


@dataclass(frozen=True)
class BacktestResult:
    """Aligned return, weight, and execution audit trails."""

    returns: pd.Series
    gross_returns: pd.Series
    costs: pd.Series
    turnover: pd.Series
    traded_notional: pd.Series
    weights: pd.DataFrame
    cash_weights: pd.Series
    executions: pd.DataFrame
    probabilities: pd.Series | None = None
    regimes: pd.Series | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def net_returns(self) -> pd.Series:
        return self.returns

    @property
    def wealth(self) -> pd.Series:
        wealth = (1.0 + self.returns).cumprod()
        wealth.name = "wealth"
        return wealth


def _validate_returns(returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns must be a non-empty DataFrame")
    if returns.shape[1] < 1:
        raise ValueError("returns must contain at least one asset")
    if not returns.index.is_monotonic_increasing or returns.index.has_duplicates:
        raise ValueError("returns index must be unique and sorted")
    if returns.columns.has_duplicates:
        raise ValueError("asset columns must be unique")
    numeric = returns.astype(float)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("backtest returns must be finite; clean the point-in-time panel first")
    if (numeric.to_numpy() <= -1.0).any():
        raise ValueError("simple asset returns must be greater than -100%")
    return numeric


def _cash_return_series(
    cash_returns: float | pd.Series,
    index: pd.Index,
) -> pd.Series:
    if np.isscalar(cash_returns):
        values = pd.Series(float(cash_returns), index=index, name="cash_return")
    elif isinstance(cash_returns, pd.Series):
        values = cash_returns.reindex(index).astype(float)
    else:
        raise TypeError("cash_returns must be a scalar or Series")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("cash return series must cover the full return index")
    if (values <= -1.0).any():
        raise ValueError("simple cash returns must be greater than -100%")
    return values


def _validate_target(target: np.ndarray, assets: pd.Index) -> np.ndarray:
    values = np.asarray(target, dtype=float).reshape(-1)
    if values.size != assets.size:
        raise ValueError("allocator returned the wrong number of weights")
    if not np.isfinite(values).all():
        raise ValueError("target weights must be finite")
    if (values < -1e-12).any():
        raise ValueError("backtests support long-only risky weights")
    values = np.maximum(values, 0.0)
    total = float(np.sum(values))
    if total > 1.0 + 1e-10:
        raise ValueError("risky weights may not exceed 100% without explicit leverage")
    if total < 0:
        raise ValueError("invalid risky-weight total")
    return values


def run_weight_backtest(
    returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    *,
    transaction_cost_bps: float = 0.0,
    cash_returns: float | pd.Series = 0.0,
    signal_metadata: pd.DataFrame | None = None,
    probabilities: pd.Series | None = None,
    regimes: pd.Series | None = None,
) -> BacktestResult:
    """Execute dated targets with a strict one-observation signal lag.

    Transaction cost is ``bps * sum(abs(delta risky weight))``.  ``turnover``
    includes the matching cash change, while ``traded_notional`` contains only
    risky buys and sells to which the one-way cost is applied.
    """

    clean_returns = _validate_returns(returns)
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be non-negative")
    if not isinstance(target_weights, pd.DataFrame) or target_weights.empty:
        raise ValueError("target_weights must be a non-empty DataFrame")
    if not target_weights.index.is_monotonic_increasing or target_weights.index.has_duplicates:
        raise ValueError("target-weight signal index must be unique and sorted")
    if set(target_weights.columns) != set(clean_returns.columns):
        raise ValueError("target-weight columns must match return columns")
    targets = target_weights.reindex(columns=clean_returns.columns).astype(float)
    if not np.isfinite(targets.to_numpy()).all():
        raise ValueError("target weights must be finite")

    cash = _cash_return_series(cash_returns, clean_returns.index)
    positions = clean_returns.index.get_indexer(targets.index)
    if (positions < 0).any():
        raise ValueError("every signal date must appear in the return index")
    execution_map: dict[int, tuple[Any, np.ndarray]] = {}
    for signal_date, position, row in zip(targets.index, positions, targets.to_numpy()):
        execution_position = int(position) + 1
        if execution_position < len(clean_returns.index):
            execution_map[execution_position] = (
                signal_date,
                _validate_target(row, clean_returns.columns),
            )
    if not execution_map:
        raise ValueError("no target has a following observation on which it can execute")

    asset_count = clean_returns.shape[1]
    risky_weights = np.zeros(asset_count, dtype=float)
    cash_weight = 1.0
    gross = np.zeros(len(clean_returns), dtype=float)
    costs = np.zeros(len(clean_returns), dtype=float)
    turnover = np.zeros(len(clean_returns), dtype=float)
    traded = np.zeros(len(clean_returns), dtype=float)
    applied = np.zeros((len(clean_returns), asset_count), dtype=float)
    applied_cash = np.ones(len(clean_returns), dtype=float)
    execution_rows: list[dict[str, Any]] = []
    cost_rate = float(transaction_cost_bps) * 1e-4

    for position, date in enumerate(clean_returns.index):
        if position in execution_map:
            signal_date, target = execution_map[position]
            target_cash = 1.0 - float(np.sum(target))
            risky_trade = float(np.sum(np.abs(target - risky_weights)))
            cash_trade = abs(target_cash - cash_weight)
            traded[position] = risky_trade
            turnover[position] = 0.5 * (risky_trade + cash_trade)
            costs[position] = cost_rate * risky_trade
            risky_weights = target.copy()
            cash_weight = target_cash

            execution: dict[str, Any] = {
                "signal_date": signal_date,
                "execution_date": date,
                "turnover": turnover[position],
                "traded_notional": traded[position],
                "cost": costs[position],
                "target_cash_weight": target_cash,
            }
            if signal_metadata is not None and signal_date in signal_metadata.index:
                metadata_row = signal_metadata.loc[signal_date]
                if isinstance(metadata_row, pd.DataFrame):
                    raise ValueError("signal_metadata index must be unique")
                execution.update(metadata_row.to_dict())
            execution_rows.append(execution)

        applied[position] = risky_weights
        applied_cash[position] = cash_weight
        asset_returns = clean_returns.iloc[position].to_numpy(dtype=float)
        gross[position] = float(risky_weights @ asset_returns + cash_weight * cash.iloc[position])
        gross_factor = 1.0 + gross[position]
        if gross_factor <= 0:
            raise ValueError("portfolio lost all capital; weights cannot be drifted")
        risky_weights = risky_weights * (1.0 + asset_returns) / gross_factor
        cash_weight = cash_weight * (1.0 + cash.iloc[position]) / gross_factor

    first_execution = min(execution_map)
    result_index = clean_returns.index[first_execution:]
    net = gross - costs
    execution_frame = pd.DataFrame(execution_rows)
    if not execution_frame.empty:
        execution_frame = execution_frame.set_index("execution_date", drop=False)

    def _slice_optional(values: pd.Series | None, name: str) -> pd.Series | None:
        if values is None:
            return None
        result = values.reindex(clean_returns.index).iloc[first_execution:].copy()
        result.name = name
        return result

    return BacktestResult(
        returns=pd.Series(net[first_execution:], index=result_index, name="net_return"),
        gross_returns=pd.Series(
            gross[first_execution:], index=result_index, name="gross_return"
        ),
        costs=pd.Series(costs[first_execution:], index=result_index, name="cost"),
        turnover=pd.Series(
            turnover[first_execution:], index=result_index, name="turnover"
        ),
        traded_notional=pd.Series(
            traded[first_execution:], index=result_index, name="traded_notional"
        ),
        weights=pd.DataFrame(
            applied[first_execution:], index=result_index, columns=clean_returns.columns
        ),
        cash_weights=pd.Series(
            applied_cash[first_execution:], index=result_index, name="cash_weight"
        ),
        executions=execution_frame,
        probabilities=_slice_optional(probabilities, "high_regime_probability"),
        regimes=_slice_optional(regimes, "regime"),
        metadata={
            "signal_lag_observations": 1,
            "transaction_cost_bps": float(transaction_cost_bps),
            "cost_convention": "one-way bps times absolute risky-weight trades",
        },
    )


def _resolve_allocator(allocator: Allocator | None) -> Allocator:
    if allocator is not None:
        return allocator
    # Lazy import keeps the accounting engine usable in isolation and avoids a
    # circular import while the allocation package imports shared utilities.
    from hrp_lab.allocation import hrp_weights

    return hrp_weights


def _allocate(
    covariance: pd.DataFrame | np.ndarray,
    *,
    allocator: Allocator | None,
    allocation_kwargs: Mapping[str, Any] | None,
) -> np.ndarray:
    function = _resolve_allocator(allocator)
    weights = function(covariance, **dict(allocation_kwargs or {}))
    if isinstance(weights, pd.Series) and isinstance(covariance, pd.DataFrame):
        weights = weights.reindex(covariance.columns).to_numpy(dtype=float)
    values = np.asarray(weights, dtype=float).reshape(-1)
    if not np.isfinite(values).all() or (values < -1e-12).any():
        raise ValueError("allocator must return finite long-only weights")
    values = np.maximum(values, 0.0)
    total = float(np.sum(values))
    if total <= 0:
        raise ValueError("allocator returned zero total risky weight")
    return values / total


def _sample_covariance(window: pd.DataFrame) -> pd.DataFrame:
    covariance = window.cov()
    if not np.isfinite(covariance.to_numpy()).all():
        raise ValueError("sample covariance contains non-finite values")
    return covariance


def _eligible_calendar_positions(
    index: pd.Index,
    first_position: int,
    frequency: str | int | None,
) -> list[int]:
    if first_position >= len(index):
        return []
    if frequency is None or frequency == "static":
        return [first_position]
    if isinstance(frequency, int):
        if frequency < 1:
            raise ValueError("integer rebalance frequency must be positive")
        return list(range(first_position, len(index), frequency))
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("named calendar frequencies require a DatetimeIndex")
    normalized = frequency.lower()
    if normalized == "daily":
        return list(range(first_position, len(index)))
    aliases = {
        "weekly": "W-FRI",
        "monthly": "M",
        "quarterly": "Q",
        "yearly": "Y",
        "annual": "Y",
    }
    if normalized not in aliases:
        raise ValueError(f"unsupported rebalance frequency: {frequency!r}")
    periods = index.to_period(aliases[normalized])
    positions: list[int] = []
    for position in range(first_position, len(index)):
        is_period_end = position == len(index) - 1 or periods[position] != periods[position + 1]
        if is_period_end:
            positions.append(position)
    return positions


def _first_signal_position(
    index: pd.Index,
    covariance_lookback: int,
    start: str | pd.Timestamp | None,
) -> int:
    if covariance_lookback < 2:
        raise ValueError("covariance_lookback must be at least two")
    position = covariance_lookback - 1
    if start is not None:
        start_position = int(index.searchsorted(pd.Timestamp(start), side="left"))
        position = max(position, start_position)
    return position


def run_calendar_hrp_backtest(
    returns: pd.DataFrame,
    *,
    covariance_lookback: int = 252,
    rebalance_frequency: str | int = "monthly",
    transaction_cost_bps: float = 10.0,
    cash_returns: float | pd.Series = 0.0,
    start: str | pd.Timestamp | None = None,
    allocator: Allocator | None = None,
    allocation_kwargs: Mapping[str, Any] | None = None,
    covariance_estimator: CovarianceEstimator | None = None,
) -> BacktestResult:
    """Run periodically re-estimated, fully invested HRP."""

    clean_returns = _validate_returns(returns)
    first = _first_signal_position(clean_returns.index, covariance_lookback, start)
    positions = _eligible_calendar_positions(
        clean_returns.index, first, rebalance_frequency
    )
    estimate = covariance_estimator or _sample_covariance
    targets: list[np.ndarray] = []
    dates: list[Any] = []
    metadata_rows: list[dict[str, Any]] = []
    for position in positions:
        window = clean_returns.iloc[position - covariance_lookback + 1 : position + 1]
        covariance = estimate(window)
        weights = _allocate(
            covariance, allocator=allocator, allocation_kwargs=allocation_kwargs
        )
        targets.append(weights)
        dates.append(clean_returns.index[position])
        metadata_rows.append(
            {
                "reason": "calendar",
                "regime": "static",
                "high_regime_probability": np.nan,
                "risk_scale": 1.0,
                "estimation_start": window.index[0],
                "estimation_end": window.index[-1],
            }
        )
    if not targets:
        raise ValueError("no calendar rebalance date has sufficient covariance history")
    target_frame = pd.DataFrame(targets, index=dates, columns=clean_returns.columns)
    signal_metadata = pd.DataFrame(metadata_rows, index=dates)
    result = run_weight_backtest(
        clean_returns,
        target_frame,
        transaction_cost_bps=transaction_cost_bps,
        cash_returns=cash_returns,
        signal_metadata=signal_metadata,
    )
    return BacktestResult(
        **{
            **result.__dict__,
            "metadata": {
                **result.metadata,
                "strategy": "calendar_hrp",
                "covariance_lookback": covariance_lookback,
                "rebalance_frequency": rebalance_frequency,
            },
        }
    )


def run_static_hrp_backtest(
    returns: pd.DataFrame,
    *,
    covariance_lookback: int = 252,
    transaction_cost_bps: float = 10.0,
    cash_returns: float | pd.Series = 0.0,
    start: str | pd.Timestamp | None = None,
    allocator: Allocator | None = None,
    allocation_kwargs: Mapping[str, Any] | None = None,
    covariance_estimator: CovarianceEstimator | None = None,
) -> BacktestResult:
    """Estimate HRP once, then buy and hold with natural weight drift."""

    result = run_calendar_hrp_backtest(
        returns,
        covariance_lookback=covariance_lookback,
        rebalance_frequency="static",
        transaction_cost_bps=transaction_cost_bps,
        cash_returns=cash_returns,
        start=start,
        allocator=allocator,
        allocation_kwargs=allocation_kwargs,
        covariance_estimator=covariance_estimator,
    )
    return BacktestResult(
        **{
            **result.__dict__,
            "metadata": {**result.metadata, "strategy": "static_hrp"},
        }
    )


def run_regime_hrp_backtest(
    returns: pd.DataFrame,
    *,
    features: pd.DataFrame | None = None,
    probabilities: pd.Series | None = None,
    market_returns: pd.Series | None = None,
    feature_volatility_window: int = 21,
    feature_correlation_window: int = 63,
    model_lookback: int = 756,
    minimum_model_observations: int = 126,
    lags: int = 1,
    ridge: float = 1e-4,
    gamma_grid: Iterable[float] = (0.5, 1.0, 2.0, 5.0, 10.0),
    threshold_quantiles: Iterable[float] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
    validation_fraction: float = 0.2,
    refit_interval: int = 63,
    covariance_lookback: int = 504,
    short_half_life: float = 21.0,
    long_half_life: float = 126.0,
    trigger_config: TriggerConfig | None = None,
    low_probability: float = 0.35,
    high_probability: float = 0.65,
    probability_change: float | None = 0.20,
    minimum_rebalance_gap: int = 5,
    force_rebalance_after: int | None = 21,
    low_regime_risk_scale: float = 1.0,
    high_regime_risk_scale: float = 0.80,
    allow_cash: bool = True,
    transaction_cost_bps: float = 10.0,
    cash_returns: float | pd.Series = 0.0,
    start: str | pd.Timestamp | None = None,
    allocator: Allocator | None = None,
    allocation_kwargs: Mapping[str, Any] | None = None,
    eigenvalue_floor: float = 1e-10,
) -> BacktestResult:
    """Run triggered HRP using point-in-time VLSTAR probabilities.

    Passing ``probabilities`` bypasses model fitting and is useful for auditing
    a precomputed signal.  Otherwise the restricted VLSTAR is refit only from
    features available through each signal date.
    """

    clean_returns = _validate_returns(returns)
    if covariance_lookback < 2:
        raise ValueError("covariance_lookback must be at least two")
    if model_lookback < 4 or minimum_model_observations < 4:
        raise ValueError("VLSTAR lookbacks must be at least four")
    if minimum_model_observations > model_lookback:
        raise ValueError("minimum_model_observations may not exceed model_lookback")
    if refit_interval < 1:
        raise ValueError("refit_interval must be positive")
    for name, scale in (
        ("low_regime_risk_scale", low_regime_risk_scale),
        ("high_regime_risk_scale", high_regime_risk_scale),
    ):
        if not 0.0 <= scale <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
    if not allow_cash and (
        low_regime_risk_scale != 1.0 or high_regime_risk_scale != 1.0
    ):
        raise ValueError("risk scaling below one requires allow_cash=True")

    if probabilities is not None:
        if not isinstance(probabilities, pd.Series):
            raise TypeError("probabilities must be a Series")
        if probabilities.index.has_duplicates:
            raise ValueError("probability index must be unique")
        probability_series = probabilities.reindex(clean_returns.index).astype(float)
        feature_frame = None
    else:
        if features is None:
            feature_frame = build_volatility_features(
                clean_returns,
                market_returns=market_returns,
                volatility_window=feature_volatility_window,
                correlation_window=feature_correlation_window,
                dropna=True,
            )
        else:
            if not isinstance(features, pd.DataFrame):
                raise TypeError("features must be a DataFrame")
            feature_frame = features.sort_index().dropna().astype(float)
            if feature_frame.shape[1] != 3:
                raise ValueError("features must contain exactly three columns")
            if feature_frame.index.has_duplicates:
                raise ValueError("feature index must be unique")
        probability_series = pd.Series(
            np.nan, index=clean_returns.index, name="high_regime_probability"
        )

    config = trigger_config or TriggerConfig(
        low_probability=low_probability,
        high_probability=high_probability,
        probability_change=probability_change,
        minimum_gap=minimum_rebalance_gap,
        force_after=force_rebalance_after,
    )
    trigger = RegimeTrigger(config)
    regimes = pd.Series(index=clean_returns.index, dtype=object, name="regime")
    model: RestrictedVLSTAR | None = None
    last_refit_feature_count: int | None = None
    start_position = _first_signal_position(
        clean_returns.index, covariance_lookback, start
    )

    target_dates: list[Any] = []
    targets: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []
    gamma_values = tuple(float(value) for value in gamma_grid)
    quantile_values = tuple(float(value) for value in threshold_quantiles)

    for position, date in enumerate(clean_returns.index):
        probability = float("nan")
        if probabilities is not None:
            raw_probability = probability_series.iloc[position]
            if np.isfinite(raw_probability):
                probability = float(raw_probability)
        elif feature_frame is not None and date in feature_frame.index:
            feature_count = int(feature_frame.index.searchsorted(date, side="right"))
            history = feature_frame.iloc[
                max(0, feature_count - model_lookback) : feature_count
            ]
            if len(history) >= minimum_model_observations and (
                model is None
                or last_refit_feature_count is None
                or feature_count - last_refit_feature_count >= refit_interval
            ):
                model = fit_restricted_vlstar(
                    history,
                    lags=lags,
                    ridge=ridge,
                    gamma_grid=gamma_values,
                    threshold_quantiles=quantile_values,
                    validation_fraction=validation_fraction,
                )
                last_refit_feature_count = feature_count
            if model is not None:
                probability = float(
                    transition_probability(model, float(feature_frame.loc[date].iloc[0]))
                )
                probability_series.loc[date] = probability

        if not np.isfinite(probability) or position < start_position:
            continue
        if not 0.0 <= probability <= 1.0:
            raise ValueError("regime probability must lie in [0, 1]")

        decision = trigger.step(probability, position)
        regimes.loc[date] = decision.regime
        if not decision.rebalance:
            continue

        window = clean_returns.iloc[position - covariance_lookback + 1 : position + 1]
        covariance = blend_ewma_covariances(
            window,
            probability,
            short_half_life=short_half_life,
            long_half_life=long_half_life,
            eigenvalue_floor=eigenvalue_floor,
        )
        fully_invested = _allocate(
            covariance.blended,
            allocator=allocator,
            allocation_kwargs=allocation_kwargs,
        )
        if allow_cash:
            risk_scale = (
                high_regime_risk_scale
                if decision.regime == "high"
                else low_regime_risk_scale
            )
        else:
            risk_scale = 1.0
        target_dates.append(date)
        targets.append(fully_invested * risk_scale)
        metadata_rows.append(
            {
                "reason": "+".join(decision.reasons),
                "regime": decision.regime,
                "high_regime_probability": probability,
                "risk_scale": risk_scale,
                "estimation_start": window.index[0],
                "estimation_end": window.index[-1],
                "model_gamma": model.gamma if model is not None else np.nan,
                "model_threshold": model.threshold if model is not None else np.nan,
            }
        )

    if not targets:
        raise ValueError("no regime signal had sufficient point-in-time history")
    target_frame = pd.DataFrame(
        targets, index=target_dates, columns=clean_returns.columns
    )
    signal_metadata = pd.DataFrame(metadata_rows, index=target_dates)
    # Carry the most recently observed regime forward for the daily audit trail;
    # probabilities themselves are not imputed.
    regimes = regimes.ffill()
    result = run_weight_backtest(
        clean_returns,
        target_frame,
        transaction_cost_bps=transaction_cost_bps,
        cash_returns=cash_returns,
        signal_metadata=signal_metadata,
        probabilities=probability_series,
        regimes=regimes,
    )
    return BacktestResult(
        **{
            **result.__dict__,
            "metadata": {
                **result.metadata,
                "strategy": "regime_hrp",
                "covariance_lookback": covariance_lookback,
                "short_half_life": short_half_life,
                "long_half_life": long_half_life,
                "allow_cash": allow_cash,
                "low_regime_risk_scale": low_regime_risk_scale,
                "high_regime_risk_scale": high_regime_risk_scale,
                "probability_source": (
                    "supplied" if probabilities is not None else "restricted_vlstar"
                ),
            },
        }
    )
