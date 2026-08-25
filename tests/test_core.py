"""Deterministic correctness tests for the numerical portfolio core."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.cluster.hierarchy import leaves_list, linkage as scipy_linkage
from scipy.spatial.distance import pdist, squareform


# Keep this focused test runnable before the package is installed editable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hrp_lab.allocation import (  # noqa: E402
    erc_weights,
    erc_weights_from_returns,
    hrp_weights,
    hrp_weights_from_returns,
    ivp_weights,
    ivp_weights_from_returns,
)
from hrp_lab.clustering import (  # noqa: E402
    correlation_distance,
    leaf_order_recursive,
    leaf_order_stack,
    linkage_from_correlation,
    linkage_from_covariance,
)
from hrp_lab.risk import (  # noqa: E402
    covariance_to_correlation,
    estimate_covariance,
    risk_contributions,
    sanitize_covariance,
)


def random_covariance(seed: int, n_assets: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    loadings = rng.normal(size=(n_assets, max(2, n_assets // 2)))
    covariance = loadings @ loadings.T
    covariance += np.diag(rng.uniform(0.2, 1.0, size=n_assets))
    return covariance


def test_sanitize_covariance_is_symmetric_and_positive_definite() -> None:
    indefinite_asymmetric = np.array(
        [
            [1.0, 1.20, -0.10],
            [1.10, 1.0, 0.40],
            [-0.20, 0.30, 0.05],
        ]
    )
    repaired = sanitize_covariance(indefinite_asymmetric)

    np.testing.assert_allclose(repaired, repaired.T, atol=1.0e-14)
    assert np.min(np.linalg.eigvalsh(repaired)) > 0.0
    assert np.all(np.diag(repaired) > 0.0)


@pytest.mark.parametrize(
    "invalid",
    [
        np.array([1.0, 2.0]),
        np.ones((2, 3)),
        np.array([[1.0, np.nan], [0.0, 1.0]]),
        np.zeros((2, 2)),
    ],
)
def test_sanitize_covariance_rejects_invalid_inputs(invalid: np.ndarray) -> None:
    with pytest.raises(ValueError):
        sanitize_covariance(invalid)


def test_covariance_to_correlation_has_required_invariants() -> None:
    covariance = np.array(
        [
            [4.0, 1.2, -0.6],
            [1.2, 9.0, 0.3],
            [-0.6, 0.3, 1.0],
        ]
    )
    correlation = covariance_to_correlation(covariance)

    np.testing.assert_allclose(np.diag(correlation), 1.0, atol=1.0e-14)
    np.testing.assert_allclose(correlation, correlation.T, atol=1.0e-14)
    assert np.max(correlation) <= 1.0
    assert np.min(correlation) >= -1.0
    np.testing.assert_allclose(correlation[0, 1], 0.2, atol=1.0e-10)


def test_covariance_estimators_match_contract() -> None:
    rng = np.random.default_rng(20260824)
    returns = rng.normal(size=(300, 5)) @ np.diag([0.01, 0.02, 0.03, 0.015, 0.025])

    sample = estimate_covariance(returns, method="sample")
    expected = np.cov(returns, rowvar=False, ddof=1)
    np.testing.assert_allclose(sample, expected, rtol=1.0e-10, atol=1.0e-14)

    shrunk = estimate_covariance(returns, method="ledoit_wolf")
    assert shrunk.shape == (5, 5)
    assert np.min(np.linalg.eigvalsh(shrunk)) > 0.0

    with pytest.raises(ValueError, match="NaN"):
        estimate_covariance(np.array([[0.1, np.nan], [0.2, 0.3]]))


def test_correlation_distance_exact_values() -> None:
    correlation = np.array(
        [
            [1.0, 1.0, 0.0, -1.0],
            [1.0, 1.0, 0.5, 0.0],
            [0.0, 0.5, 1.0, 0.25],
            [-1.0, 0.0, 0.25, 1.0],
        ]
    )
    distance = correlation_distance(correlation)

    np.testing.assert_allclose(np.diag(distance), 0.0)
    np.testing.assert_allclose(distance, distance.T)
    assert distance[0, 1] == pytest.approx(0.0)
    assert distance[0, 2] == pytest.approx(np.sqrt(0.5))
    assert distance[0, 3] == pytest.approx(1.0)


def test_original_and_direct_linkage_modes_are_pinned() -> None:
    correlation = np.eye(3)
    distance = correlation_distance(correlation)

    original = linkage_from_correlation(correlation, distance_mode="original")
    expected_original = scipy_linkage(pdist(distance), method="single")
    np.testing.assert_allclose(original, expected_original)
    np.testing.assert_allclose(original[:, 2], 1.0)

    direct = linkage_from_correlation(correlation, distance_mode="direct")
    expected_direct = scipy_linkage(squareform(distance), method="single")
    np.testing.assert_allclose(direct, expected_direct)
    np.testing.assert_allclose(direct[:, 2], np.sqrt(0.5))

    with pytest.raises(ValueError, match="distance_mode"):
        linkage_from_correlation(correlation, distance_mode="not-a-mode")  # type: ignore[arg-type]


@pytest.mark.parametrize("method", ["single", "complete", "average", "ward"])
@pytest.mark.parametrize("n_assets", [2, 3, 8, 31])
def test_stack_and_recursive_leaf_orders_equal_scipy(
    method: str,
    n_assets: int,
) -> None:
    rng = np.random.default_rng(10_000 + n_assets)
    observations = rng.normal(size=(n_assets, 6))
    linkage_matrix = scipy_linkage(pdist(observations), method=method)
    expected = leaves_list(linkage_matrix).astype(np.intp)

    np.testing.assert_array_equal(leaf_order_recursive(linkage_matrix), expected)
    np.testing.assert_array_equal(leaf_order_stack(linkage_matrix), expected)


def test_leaf_orders_handle_single_asset_and_reject_malformed_tree() -> None:
    empty = np.empty((0, 4))
    np.testing.assert_array_equal(leaf_order_recursive(empty), [0])
    np.testing.assert_array_equal(leaf_order_stack(empty), [0])

    # Leaf 1 is used twice and leaf 2 never appears.
    malformed = np.array([[0, 1, 0.5, 2], [3, 1, 0.7, 3]], dtype=float)
    with pytest.raises(ValueError, match="each leaf once"):
        leaf_order_recursive(malformed)
    with pytest.raises(ValueError, match="each leaf once"):
        leaf_order_stack(malformed)


def test_linkage_from_covariance_returns_valid_tree() -> None:
    covariance = random_covariance(101, 10)
    for distance_mode in ("original", "direct"):
        linkage_matrix = linkage_from_covariance(
            covariance,
            distance_mode=distance_mode,
        )
        assert linkage_matrix.shape == (9, 4)
        np.testing.assert_array_equal(
            leaf_order_stack(linkage_matrix),
            leaves_list(linkage_matrix),
        )


def test_inverse_variance_weights_are_exact() -> None:
    covariance = np.diag([1.0, 4.0, 16.0])
    expected = np.array([1.0, 0.25, 0.0625])
    expected /= expected.sum()
    weights = ivp_weights(covariance)

    np.testing.assert_allclose(weights, expected, rtol=1.0e-12, atol=1.0e-12)
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(weights >= 0.0)


@pytest.mark.parametrize("distance_mode", ["original", "direct"])
def test_hrp_stack_and_recursive_allocations_are_identical(distance_mode: str) -> None:
    covariance = random_covariance(82, 19)
    optimized = hrp_weights(
        covariance,
        distance_mode=distance_mode,  # type: ignore[arg-type]
        ordering="stack",
    )
    reference = hrp_weights(
        covariance,
        distance_mode=distance_mode,  # type: ignore[arg-type]
        ordering="recursive",
    )

    np.testing.assert_allclose(optimized, reference, rtol=0.0, atol=0.0)
    assert optimized.sum() == pytest.approx(1.0)
    assert np.all(np.isfinite(optimized))
    assert np.all(optimized >= 0.0)


def test_hrp_on_diagonal_covariance_matches_ivp() -> None:
    covariance = np.diag([0.01, 0.04, 0.09, 0.16, 0.25, 0.36, 0.49, 0.64])
    np.testing.assert_allclose(
        hrp_weights(covariance),
        ivp_weights(covariance),
        rtol=1.0e-11,
        atol=1.0e-12,
    )


def test_erc_diagonal_case_and_equal_risk_contributions() -> None:
    covariance = np.diag([1.0, 4.0, 9.0, 16.0])
    expected = np.array([1.0, 0.5, 1.0 / 3.0, 0.25])
    expected /= expected.sum()
    weights = erc_weights(covariance, tol=1.0e-12)

    np.testing.assert_allclose(weights, expected, rtol=1.0e-7, atol=1.0e-9)
    np.testing.assert_allclose(
        risk_contributions(weights, covariance),
        np.full(4, 0.25),
        rtol=0.0,
        atol=2.0e-7,
    )


def test_erc_random_covariance_hits_equal_risk_budget() -> None:
    covariance = random_covariance(777, 12)
    weights = erc_weights(covariance, tol=1.0e-12, maxiter=3_000)
    shares = risk_contributions(weights, covariance)

    assert weights.sum() == pytest.approx(1.0)
    assert np.all(weights > 0.0)
    np.testing.assert_allclose(shares, np.full(12, 1.0 / 12.0), atol=8.0e-7)


def test_erc_custom_risk_budgets() -> None:
    covariance = random_covariance(909, 6)
    target = np.arange(1.0, 7.0)
    target /= target.sum()
    weights = erc_weights(covariance, budgets=target, tol=1.0e-12)

    np.testing.assert_allclose(
        risk_contributions(weights, covariance),
        target,
        rtol=0.0,
        atol=8.0e-7,
    )


def test_return_wrappers_are_deterministic_and_well_formed() -> None:
    rng = np.random.default_rng(44)
    factors = rng.standard_t(df=7, size=(500, 3))
    loadings = rng.normal(scale=0.01, size=(3, 7))
    returns = factors @ loadings + rng.normal(scale=0.005, size=(500, 7))

    allocations = [
        hrp_weights_from_returns(returns),
        ivp_weights_from_returns(returns),
        erc_weights_from_returns(returns),
    ]
    for weights in allocations:
        assert weights.shape == (7,)
        assert weights.sum() == pytest.approx(1.0)
        assert np.all(np.isfinite(weights))
        assert np.all(weights >= 0.0)

    np.testing.assert_array_equal(
        hrp_weights_from_returns(returns),
        hrp_weights_from_returns(returns),
    )


def test_all_allocators_handle_one_asset() -> None:
    covariance = np.array([[0.04]])
    np.testing.assert_array_equal(hrp_weights(covariance), [1.0])
    np.testing.assert_array_equal(ivp_weights(covariance), [1.0])
    np.testing.assert_array_equal(erc_weights(covariance), [1.0])
