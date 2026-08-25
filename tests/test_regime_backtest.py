from __future__ import annotations

import numpy as np
import pandas as pd

from hrp_lab.backtest import (
    run_regime_hrp_backtest,
    run_static_hrp_backtest,
    run_weight_backtest,
)
from hrp_lab.regimes import (
    RegimeTrigger,
    TriggerConfig,
    blend_ewma_covariances,
    build_volatility_features,
    fit_restricted_vlstar,
    transition_probability,
)


def _equal_allocator(covariance: pd.DataFrame | np.ndarray, **_: object) -> np.ndarray:
    asset_count = np.asarray(covariance).shape[0]
    return np.full(asset_count, 1.0 / asset_count)


def test_volatility_features_build_from_real_pandas_series() -> None:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2022-01-03", periods=90, freq="B")
    returns = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(90, 4)), index=dates, columns=list("ABCD")
    )

    features = build_volatility_features(
        returns, volatility_window=10, correlation_window=20
    )

    assert list(features.columns) == [
        "log_market_rv",
        "log_median_asset_rv",
        "fisher_mean_correlation",
    ]
    assert not features.empty
    assert np.isfinite(features.to_numpy()).all()


def test_restricted_vlstar_grid_fit_and_probability_are_well_formed() -> None:
    rng = np.random.default_rng(91)
    observations = 180
    transition = np.zeros(observations)
    transition[0] = -1.0
    for position in range(1, observations):
        transition[position] = 0.92 * transition[position - 1] + rng.normal(0.0, 0.25)
    high = 1.0 / (1.0 + np.exp(-3.0 * transition))
    feature_2 = 0.5 * transition + 0.7 * high + rng.normal(0.0, 0.08, observations)
    feature_3 = -0.2 * transition + 0.4 * high + rng.normal(0.0, 0.08, observations)
    features = pd.DataFrame(
        np.column_stack([transition, feature_2, feature_3]),
        columns=["market", "cross_section", "correlation"],
    )

    model = fit_restricted_vlstar(
        features,
        gamma_grid=(0.5, 1.0, 3.0),
        threshold_quantiles=(0.3, 0.5, 0.7),
        validation_fraction=0.2,
    )
    probabilities = transition_probability(
        model, pd.Series([-2.0, 0.0, 2.0], index=list("abc"))
    )

    assert model.gamma in {0.5, 1.0, 3.0}
    assert model.coefficients.shape == (8, 3)
    assert model.observations == observations
    assert isinstance(probabilities, pd.Series)
    assert probabilities.index.tolist() == list("abc")
    assert 0.0 < probabilities.iloc[0] < probabilities.iloc[1] < probabilities.iloc[2] < 1.0
    assert model.predict_next(features.iloc[-3:]).shape == (3,)


def test_ewma_probability_blend_has_exact_endpoints_and_is_psd() -> None:
    rng = np.random.default_rng(17)
    returns = pd.DataFrame(
        rng.normal(size=(80, 3)) * np.linspace(0.01, 0.04, 80)[:, None],
        columns=list("ABC"),
    )
    low = blend_ewma_covariances(
        returns, 0.0, short_half_life=4, long_half_life=30
    )
    high = blend_ewma_covariances(
        returns, 1.0, short_half_life=4, long_half_life=30
    )
    middle = blend_ewma_covariances(
        returns, 0.25, short_half_life=4, long_half_life=30
    )

    np.testing.assert_allclose(low.blended, low.long, atol=1e-12)
    np.testing.assert_allclose(high.blended, high.short, atol=1e-12)
    np.testing.assert_allclose(
        middle.blended,
        0.25 * np.asarray(middle.short) + 0.75 * np.asarray(middle.long),
        atol=1e-12,
    )
    assert np.linalg.eigvalsh(np.asarray(middle.blended)).min() >= -1e-12


def test_trigger_hysteresis_respects_minimum_gap_and_force_timeout() -> None:
    trigger = RegimeTrigger(
        TriggerConfig(
            low_probability=0.30,
            high_probability=0.70,
            probability_change=None,
            minimum_gap=2,
            force_after=4,
        )
    )

    initial = trigger.step(0.20, 0)
    blocked_transition = trigger.step(0.80, 1)
    delayed_transition = trigger.step(0.60, 2)
    assert initial.rebalance and initial.reasons == ("initial",)
    assert blocked_transition.regime == "high" and not blocked_transition.rebalance
    assert delayed_transition.rebalance
    assert "regime_change" in delayed_transition.reasons

    trigger.step(0.50, 3)
    low_transition = trigger.step(0.20, 4)
    assert low_transition.rebalance and low_transition.regime == "low"
    for position in (5, 6, 7):
        assert not trigger.step(0.20, position).rebalance
    forced = trigger.step(0.20, 8)
    assert forced.rebalance and forced.reasons == ("forced",)


def test_weight_engine_lags_signals_and_uses_drifted_pretrade_weights() -> None:
    dates = pd.date_range("2021-01-04", periods=4, freq="B")
    returns = pd.DataFrame(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        index=dates,
        columns=["A", "B"],
    )
    targets = pd.DataFrame(
        [[0.5, 0.5], [0.5, 0.5]],
        index=dates[:2],
        columns=returns.columns,
    )

    result = run_weight_backtest(
        returns, targets, transaction_cost_bps=100.0
    )

    # The day-zero signal first earns the day-one return, never day zero.
    assert result.returns.index[0] == dates[1]
    np.testing.assert_allclose(result.weights.loc[dates[1]], [0.5, 0.5])
    assert result.gross_returns.loc[dates[1]] == 0.5
    assert result.costs.loc[dates[1]] == 0.01

    # After A doubles, pre-trade weights are 2/3 and 1/3. Restoring 50/50
    # trades 1/3 of risky notional before the next return.
    assert np.isclose(result.traded_notional.loc[dates[2]], 1.0 / 3.0)
    assert np.isclose(result.costs.loc[dates[2]], 0.01 / 3.0)
    assert result.executions.loc[dates[2], "signal_date"] == dates[1]


def test_static_hrp_wrapper_uses_only_trailing_covariance_and_one_day_lag() -> None:
    dates = pd.date_range("2022-01-03", periods=5, freq="B")
    returns = pd.DataFrame(
        [[0.0, 0.0], [0.50, 0.0], [0.10, 0.0], [0.0, 0.0], [0.0, 0.0]],
        index=dates,
        columns=["A", "B"],
    )

    def all_a(covariance: pd.DataFrame, **_: object) -> np.ndarray:
        assert covariance.shape == (2, 2)
        return np.array([1.0, 0.0])

    result = run_static_hrp_backtest(
        returns,
        covariance_lookback=2,
        transaction_cost_bps=0.0,
        allocator=all_a,
    )

    assert result.executions.iloc[0]["signal_date"] == dates[1]
    assert result.executions.iloc[0]["execution_date"] == dates[2]
    assert result.gross_returns.iloc[0] == 0.10
    assert dates[1] not in result.returns.index


def test_regime_backtest_applies_high_state_risk_scale_on_next_day() -> None:
    dates = pd.date_range("2020-01-02", periods=12, freq="B")
    returns = pd.DataFrame(
        np.zeros((12, 2)), index=dates, columns=["A", "B"]
    )
    probabilities = pd.Series(0.10, index=dates)
    probabilities.iloc[5:] = 0.90
    config = TriggerConfig(
        low_probability=0.30,
        high_probability=0.70,
        probability_change=None,
        minimum_gap=1,
        force_after=100,
    )

    result = run_regime_hrp_backtest(
        returns,
        probabilities=probabilities,
        covariance_lookback=3,
        short_half_life=2,
        long_half_life=5,
        trigger_config=config,
        high_regime_risk_scale=0.50,
        low_regime_risk_scale=1.0,
        transaction_cost_bps=0.0,
        allocator=_equal_allocator,
    )

    # Initial low-state signal at position 2 executes at position 3.
    assert np.isclose(result.weights.loc[dates[3]].sum(), 1.0)
    # High-state signal occurs at position 5 and is not applied until position 6.
    assert np.isclose(result.weights.loc[dates[5]].sum(), 1.0)
    assert np.isclose(result.weights.loc[dates[6]].sum(), 0.5)
    assert np.isclose(result.cash_weights.loc[dates[6]], 0.5)
    assert result.executions.loc[dates[6], "signal_date"] == dates[5]
    assert result.executions.loc[dates[6], "regime"] == "high"


def test_regime_backtest_can_fit_vlstar_point_in_time() -> None:
    rng = np.random.default_rng(123)
    dates = pd.date_range("2019-01-02", periods=90, freq="B")
    returns = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(90, 3)), index=dates, columns=list("ABC")
    )
    state = np.sin(np.linspace(-4.0, 5.0, 90))
    features = pd.DataFrame(
        {
            "market": state + rng.normal(0.0, 0.05, 90),
            "cross": 0.5 * state + rng.normal(0.0, 0.08, 90),
            "corr": -0.3 * state + rng.normal(0.0, 0.08, 90),
        },
        index=dates,
    )

    result = run_regime_hrp_backtest(
        returns,
        features=features,
        model_lookback=40,
        minimum_model_observations=20,
        refit_interval=10,
        covariance_lookback=10,
        short_half_life=3,
        long_half_life=10,
        gamma_grid=(1.0, 3.0),
        threshold_quantiles=(0.4, 0.6),
        minimum_rebalance_gap=2,
        force_rebalance_after=10,
        transaction_cost_bps=0.0,
        allocator=_equal_allocator,
    )

    assert not result.returns.empty
    assert result.probabilities is not None
    assert result.probabilities.notna().any()
    assert result.metadata["probability_source"] == "restricted_vlstar"
    assert (result.executions["execution_date"] > result.executions["signal_date"]).all()
