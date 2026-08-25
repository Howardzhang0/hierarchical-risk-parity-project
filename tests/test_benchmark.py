"""Fast deterministic tests for the traversal benchmark harness."""

from __future__ import annotations

import gc

import numpy as np
import pytest
from scipy.cluster.hierarchy import leaves_list, linkage as scipy_linkage
from scipy.spatial.distance import pdist

from hrp_lab.evaluation.benchmark import (
    benchmark_traversal_suite,
    benchmark_traversals,
    generate_synthetic_linkage,
    legacy_pandas_leaf_order,
)


@pytest.mark.parametrize("method", ["single", "complete", "average", "ward"])
@pytest.mark.parametrize("n_assets", [2, 5, 17])
def test_legacy_pandas_order_equals_scipy(method: str, n_assets: int) -> None:
    rng = np.random.default_rng(2_000 + n_assets)
    observations = rng.normal(size=(n_assets, 5))
    matrix = scipy_linkage(pdist(observations), method=method)

    np.testing.assert_array_equal(
        legacy_pandas_leaf_order(matrix),
        leaves_list(matrix),
    )


def test_legacy_pandas_order_pins_repeated_expansion() -> None:
    # Nodes 4 and 5 must be expanded across separate iterations before root 6
    # becomes the SciPy leaf order [0, 1, 2, 3].
    matrix = np.array(
        [
            [0, 1, 0.10, 2],
            [2, 3, 0.20, 2],
            [4, 5, 0.30, 4],
        ],
        dtype=float,
    )
    np.testing.assert_array_equal(legacy_pandas_leaf_order(matrix), [0, 1, 2, 3])
    np.testing.assert_array_equal(legacy_pandas_leaf_order(np.empty((0, 4))), [0])


def test_seeded_synthetic_linkage_is_reproducible() -> None:
    first = generate_synthetic_linkage(20, seed=123, n_features=4)
    second = generate_synthetic_linkage(20, seed=123, n_features=4)
    different = generate_synthetic_linkage(20, seed=124, n_features=4)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)
    assert first.shape == (19, 4)


def test_benchmark_accepts_provided_linkage_and_reports_required_fields() -> None:
    matrix = generate_synthetic_linkage(24, seed=88)
    gc_enabled_before = gc.isenabled()
    report = benchmark_traversals(
        matrix,
        seed=777,
        repetitions=5,
        warmups=2,
        include_samples=True,
    )

    assert gc.isenabled() is gc_enabled_before
    assert report["source"] == "provided"
    assert report["n_assets"] == 24
    assert report["leaf_order_equality_verified"] is True
    assert report["repetitions"] == 5
    assert report["warmups"] == 2
    assert report["methodology"]["method_order"].startswith("seeded random")
    assert report["methodology"]["garbage_collection_disabled_equally"] is True

    expected_names = {"legacy_pandas", "recursive_dfs", "stack_index_dfs"}
    assert set(report["implementations"]) == expected_names
    for summary in report["implementations"].values():
        assert summary["median_ns"] > 0.0
        assert summary["p95_ns"] >= summary["median_ns"]
        assert summary["median_us"] == pytest.approx(summary["median_ns"] / 1_000.0)
        assert summary["p95_us"] == pytest.approx(summary["p95_ns"] / 1_000.0)
        assert len(summary["samples_ns"]) == 5

    stack_median = report["implementations"]["stack_index_dfs"]["median_ns"]
    assert report["speedups"]["legacy_pandas_over_stack_index"] == pytest.approx(
        report["implementations"]["legacy_pandas"]["median_ns"] / stack_median
    )
    assert report["speedups"]["recursive_dfs_over_stack_index"] == pytest.approx(
        report["implementations"]["recursive_dfs"]["median_ns"] / stack_median
    )

    environment = report["environment"]
    for key in ("python", "platform", "machine", "numpy", "pandas", "scipy", "timer"):
        assert environment[key]
    assert environment["garbage_collection_during_warmup_and_timing"] == "disabled"


def test_benchmark_synthetic_defaults_and_omits_samples() -> None:
    report = benchmark_traversals(
        n_assets=12,
        seed=42,
        repetitions=3,
        warmups=1,
    )

    assert report["source"] == "synthetic"
    assert report["n_assets"] == 12
    assert report["seed"] == 42
    for summary in report["implementations"].values():
        assert "samples_ns" not in summary


def test_benchmark_suite_is_size_stable_and_seeded() -> None:
    first = benchmark_traversal_suite(
        [4, 7],
        seed=91,
        repetitions=2,
        warmups=0,
    )
    second = benchmark_traversal_suite(
        [7, 4],
        seed=91,
        repetitions=2,
        warmups=0,
    )

    assert first["asset_counts"] == [4, 7]
    assert [item["n_assets"] for item in first["results"]] == [4, 7]
    first_seeds = {item["n_assets"]: item["seed"] for item in first["results"]}
    second_seeds = {item["n_assets"]: item["seed"] for item in second["results"]}
    assert first_seeds == second_seeds
    assert first["speedup_definition"].startswith("baseline median_ns")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"repetitions": 0}, "repetitions"),
        ({"warmups": -1}, "warmups"),
        ({"n_assets": 0}, "n_assets"),
    ],
)
def test_benchmark_rejects_invalid_settings(
    kwargs: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        benchmark_traversals(**kwargs)


def test_provided_linkage_size_must_match() -> None:
    matrix = generate_synthetic_linkage(8, seed=5)
    with pytest.raises(ValueError, match="does not match"):
        benchmark_traversals(
            matrix,
            n_assets=9,
            repetitions=1,
            warmups=0,
        )
