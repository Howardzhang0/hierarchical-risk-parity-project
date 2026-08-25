from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hrp_lab.simulation import (
    FactorStudentTCopula,
    empirical_inverse_cdf,
    run_paired_monte_carlo,
    summarize_paired_trials,
)


def _dependent_returns(n_observations: int = 600, n_assets: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(712)
    common = rng.standard_t(df=5, size=(n_observations, 2))
    loadings = np.linspace(0.25, 0.75, n_assets)[:, None] * np.array([[1.0, -0.35]])
    noise = rng.standard_t(df=7, size=(n_observations, n_assets))
    values = 0.012 * (common @ loadings.T + 0.65 * noise)
    return pd.DataFrame(values, columns=[f"asset_{index}" for index in range(n_assets)])


def test_empirical_inverse_cdf_interpolates_and_supports_columns() -> None:
    observations = np.array([0.0, 10.0, 20.0])
    probabilities = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(
        empirical_inverse_cdf(observations, probabilities),
        np.array([0.0, 5.0, 10.0, 15.0, 20.0]),
    )

    matrix = np.column_stack([observations, observations + 100.0])
    matrix_probabilities = np.array([[0.25, 0.75], [1.0, 0.0]])
    np.testing.assert_allclose(
        empirical_inverse_cdf(matrix, matrix_probabilities),
        np.array([[5.0, 115.0], [20.0, 100.0]]),
    )

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        empirical_inverse_cdf(observations, [-0.1])


def test_factor_student_t_copula_is_deterministic_and_preserves_marginal_range() -> None:
    historical = _dependent_returns()
    model = FactorStudentTCopula(
        n_factors=2,
        df=5.5,
        correlation_shrinkage=0.02,
        random_state=99,
    ).fit(historical)

    first = model.simulate(500)
    second = model.simulate(500)
    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == list(historical.columns)
    assert first.shape == (500, historical.shape[1])
    assert np.isfinite(first.to_numpy()).all()
    assert model.factor_loadings_.shape == (historical.shape[1], 2)
    np.testing.assert_allclose(np.diag(model.correlation_), 1.0)
    assert np.linalg.eigvalsh(model.correlation_).min() > 0.0

    assert np.all(first.min().to_numpy() >= historical.min().to_numpy())
    assert np.all(first.max().to_numpy() <= historical.max().to_numpy())
    # The common factors should survive the copula and marginal transforms.
    assert first.corr().to_numpy()[0, -1] > 0.1


def test_simulated_paths_have_requested_shape_and_seed_control() -> None:
    model = FactorStudentTCopula(n_factors=2, random_state=3).fit(
        _dependent_returns(n_observations=250, n_assets=4)
    )
    first = model.simulate_paths(3, 20, random_state=17)
    second = model.simulate_paths(3, 20, random_state=17)
    assert first.shape == (3, 20, 4)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, model.simulate_paths(3, 20, random_state=18))


def test_paired_summary_uses_ivp_as_headline_and_has_deterministic_ci() -> None:
    n_trials = 12
    trials = pd.DataFrame(
        {
            "hrp_variance": np.full(n_trials, 0.8),
            "ivp_variance": np.full(n_trials, 1.0),
            "erc_variance": np.full(n_trials, 0.9),
            "hrp_sharpe": np.full(n_trials, 1.2),
            "ivp_sharpe": np.full(n_trials, 1.0),
            "erc_sharpe": np.full(n_trials, 1.1),
            "hrp_expected_shortfall": np.full(n_trials, 0.8),
            "ivp_expected_shortfall": np.full(n_trials, 1.0),
            "erc_expected_shortfall": np.full(n_trials, 0.9),
        }
    )
    first = summarize_paired_trials(
        trials, n_bootstrap=100, confidence_level=0.9, random_state=8
    )
    second = summarize_paired_trials(
        trials, n_bootstrap=100, confidence_level=0.9, random_state=8
    )
    pd.testing.assert_frame_equal(first, second)
    assert first.attrs["headline_metric"] == "hrp_vs_ivp_variance_reduction_pct"
    headline = first.loc["hrp_vs_ivp_variance_reduction_pct"]
    assert headline["estimate"] == pytest.approx(20.0)
    assert headline["ci_lower"] == pytest.approx(20.0)
    assert headline["ci_upper"] == pytest.approx(20.0)
    assert first.loc["hrp_vs_erc_variance_reduction_pct", "estimate"] == pytest.approx(
        100.0 * (1.0 - 0.8 / 0.9)
    )


def test_paired_monte_carlo_reproducible_and_weights_are_valid() -> None:
    historical = _dependent_returns(n_observations=450, n_assets=4)
    kwargs = dict(
        n_trials=5,
        train_size=80,
        test_size=40,
        rebalance_every=21,
        n_factors=2,
        copula_df=6.0,
        covariance_shrinkage=0.1,
        n_bootstrap=80,
        confidence_level=0.9,
        random_state=1234,
    )
    first = run_paired_monte_carlo(historical, **kwargs)
    second = run_paired_monte_carlo(historical, **kwargs)

    pd.testing.assert_frame_equal(first.trials, second.trials)
    pd.testing.assert_frame_equal(first.summary, second.summary)
    assert first.config["primary_benchmark"] == "ivp"
    assert first.config["robustness_benchmark"] == "erc"
    assert first.config["weight_mode"] == "rolling"
    assert first.config["rebalance_every"] == 21
    assert first.config["n_rebalances"] == 2
    assert first.trials.shape[0] == kwargs["n_trials"]
    assert "hrp_vs_ivp_variance_reduction_pct" in first.summary.index
    assert "hrp_vs_erc_variance_reduction_pct" in first.summary.index
    assert not np.allclose(
        first.weights["ivp"][:, 0, :], first.weights["ivp"][:, 1, :]
    )

    for method in ("hrp", "ivp", "erc"):
        assert first.weights[method].shape == (
            kwargs["n_trials"],
            2,
            historical.shape[1],
        )
        assert np.all(first.weights[method] >= 0.0)
        np.testing.assert_allclose(first.weights[method].sum(axis=2), 1.0)
        np.testing.assert_array_equal(first.weights[method], second.weights[method])

    for metric in first.summary.itertuples():
        if np.isfinite(metric.estimate):
            assert metric.ci_lower <= metric.estimate <= metric.ci_upper


def test_fixed_weight_mode_has_one_rebalance() -> None:
    result = run_paired_monte_carlo(
        _dependent_returns(n_observations=250, n_assets=3),
        n_trials=1,
        train_size=60,
        test_size=20,
        rebalance_every=None,
        n_factors=2,
        n_bootstrap=10,
        random_state=41,
    )
    assert result.config["weight_mode"] == "fixed"
    assert result.config["rebalance_every"] is None
    assert result.config["n_rebalances"] == 1
    for weights in result.weights.values():
        assert weights.shape == (1, 1, 3)
