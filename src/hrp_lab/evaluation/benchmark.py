"""Reproducible benchmarks for HRP tree quasi-diagonalization.

The benchmark intentionally keeps the paper-style pandas implementation as a
faithful baseline.  It repeatedly expands internal linkage-node identifiers,
inserts their right children, sorts the sparse index, and resets that index.
The two comparison implementations are the recursive and preallocated
stack/index traversals used by the numerical core.
"""

from __future__ import annotations

import gc
import platform
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.cluster.hierarchy import (
    is_valid_linkage,
    leaves_list,
    linkage as scipy_linkage,
)

from hrp_lab.clustering.traversal import leaf_order_recursive, leaf_order_stack


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]
Traversal = Callable[[ArrayLike], IntArray]


def _linkage_array(linkage_matrix: ArrayLike) -> FloatArray:
    matrix = np.asarray(linkage_matrix, dtype=np.float64)
    if matrix.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 4:
        raise ValueError("linkage_matrix must have shape (n_assets - 1, 4)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("linkage_matrix contains NaN or infinite values")
    # SciPy's validator checks child identifiers, node reuse, and cluster sizes.
    is_valid_linkage(matrix, throw=True, name="linkage_matrix")
    return np.array(matrix, dtype=np.float64, copy=True)


def legacy_pandas_leaf_order(linkage_matrix: ArrayLike) -> IntArray:
    """Faithfully reproduce the paper-style pandas expansion/sort traversal."""

    matrix = _linkage_array(linkage_matrix)
    if matrix.shape[0] == 0:
        return np.array([0], dtype=np.intp)

    # The original implementation casts the complete linkage matrix to int and
    # obtains n from the root cluster count in the final column.
    integer_linkage = matrix.astype(np.int64)
    n_assets = integer_linkage.shape[0] + 1
    if int(integer_linkage[-1, 3]) != n_assets:
        raise ValueError("root linkage cluster count does not equal n_assets")

    ordered = pd.Series(
        [integer_linkage[-1, 0], integer_linkage[-1, 1]],
        dtype="int64",
    )
    while int(ordered.max()) >= n_assets:
        # Create gaps after every current node, replace internal nodes by their
        # left children, insert right children into the adjacent gaps, sort, and
        # compact.  These repeated pandas operations are the legacy overhead the
        # index-stack implementation removes.
        ordered.index = np.arange(0, ordered.shape[0] * 2, 2)
        internal = ordered[ordered >= n_assets]
        positions = internal.index.to_numpy(dtype=np.int64)
        linkage_rows = internal.to_numpy(dtype=np.int64) - n_assets
        ordered.loc[positions] = integer_linkage[linkage_rows, 0]
        right_children = pd.Series(
            integer_linkage[linkage_rows, 1],
            index=positions + 1,
            dtype="int64",
        )
        ordered = pd.concat((ordered, right_children)).sort_index()
        ordered.index = pd.RangeIndex(ordered.shape[0])

    result = ordered.to_numpy(dtype=np.intp, copy=True)
    if result.shape != (n_assets,) or not np.array_equal(
        np.sort(result),
        np.arange(n_assets, dtype=np.intp),
    ):
        raise ValueError("linkage does not define each leaf exactly once")
    return result


def generate_synthetic_linkage(
    n_assets: int,
    *,
    seed: int = 20260824,
    n_features: int = 8,
    linkage_method: str = "single",
) -> FloatArray:
    """Generate a deterministic continuous-data linkage matrix for benchmarking."""

    if not isinstance(n_assets, int) or n_assets <= 0:
        raise ValueError("n_assets must be a positive integer")
    if not isinstance(n_features, int) or n_features <= 0:
        raise ValueError("n_features must be a positive integer")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if n_assets == 1:
        return np.empty((0, 4), dtype=np.float64)

    generator = np.random.default_rng(seed)
    observations = generator.standard_normal((n_assets, n_features))
    return np.asarray(
        scipy_linkage(observations, method=linkage_method, metric="euclidean"),
        dtype=np.float64,
    )


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def _environment_metadata() -> dict[str, Any]:
    return {
        "python": sys.version.splitlines()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "numpy": _package_version("numpy"),
        "pandas": _package_version("pandas"),
        "scipy": _package_version("scipy"),
        "timer": "time.perf_counter_ns",
        "garbage_collection_during_warmup_and_timing": "disabled",
    }


def _timing_summary(samples_ns: list[int], *, include_samples: bool) -> dict[str, Any]:
    samples = np.asarray(samples_ns, dtype=np.float64)
    summary: dict[str, Any] = {
        "median_ns": float(np.median(samples)),
        "p95_ns": float(np.percentile(samples, 95.0)),
        "median_us": float(np.median(samples) / 1_000.0),
        "p95_us": float(np.percentile(samples, 95.0) / 1_000.0),
    }
    if include_samples:
        summary["samples_ns"] = [int(value) for value in samples_ns]
    return summary


def benchmark_traversals(
    linkage_matrix: ArrayLike | None = None,
    *,
    n_assets: int | None = None,
    seed: int = 20260824,
    n_features: int = 8,
    linkage_method: str = "single",
    repetitions: int = 200,
    warmups: int = 20,
    include_samples: bool = False,
) -> dict[str, Any]:
    """Benchmark legacy, recursive, and stack/index leaf traversals.

    A provided linkage matrix is copied and validated.  With no matrix, a
    deterministic synthetic one is generated.  Exact agreement with SciPy's
    ``leaves_list`` is checked before any warm-up or timed invocation.

    Method order is independently shuffled for every warm-up and measured
    repetition.  Garbage collection is disabled for the entire warm-up/timing
    region and restored to its entry state afterward.
    """

    if not isinstance(repetitions, int) or repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    if not isinstance(warmups, int) or warmups < 0:
        raise ValueError("warmups must be a non-negative integer")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")

    if linkage_matrix is None:
        generated_n_assets = 256 if n_assets is None else n_assets
        matrix = generate_synthetic_linkage(
            generated_n_assets,
            seed=int(seed),
            n_features=n_features,
            linkage_method=linkage_method,
        )
        source = "synthetic"
    else:
        matrix = _linkage_array(linkage_matrix)
        inferred_n_assets = matrix.shape[0] + 1
        if n_assets is not None and n_assets != inferred_n_assets:
            raise ValueError("n_assets does not match the provided linkage matrix")
        generated_n_assets = inferred_n_assets
        source = "provided"

    implementations: dict[str, Traversal] = {
        "legacy_pandas": legacy_pandas_leaf_order,
        "recursive_dfs": leaf_order_recursive,
        "stack_index_dfs": leaf_order_stack,
    }

    if matrix.shape[0] == 0:
        expected = np.array([0], dtype=np.intp)
    else:
        expected = leaves_list(matrix).astype(np.intp, copy=False)
    verified_orders: dict[str, IntArray] = {
        name: function(matrix) for name, function in implementations.items()
    }
    mismatched = [
        name
        for name, order in verified_orders.items()
        if not np.array_equal(order, expected)
    ]
    if mismatched:
        raise AssertionError(
            "leaf-order implementations disagree with scipy.leaves_list: "
            + ", ".join(mismatched)
        )

    timing_samples: dict[str, list[int]] = {
        name: [] for name in implementations
    }
    names = np.array(tuple(implementations), dtype=object)
    # A separate derived stream makes the timing schedule deterministic without
    # coupling it to synthetic-linkage random-number consumption.
    schedule_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 1]))
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(warmups):
            for name in schedule_rng.permutation(names):
                implementations[str(name)](matrix)

        for _ in range(repetitions):
            for name in schedule_rng.permutation(names):
                method_name = str(name)
                start = time.perf_counter_ns()
                implementations[method_name](matrix)
                elapsed = time.perf_counter_ns() - start
                timing_samples[method_name].append(elapsed)
    finally:
        if gc_was_enabled:
            gc.enable()

    summaries = {
        name: _timing_summary(samples, include_samples=include_samples)
        for name, samples in timing_samples.items()
    }
    stack_median = summaries["stack_index_dfs"]["median_ns"]
    speedups = {
        # A value greater than one means the stack/index implementation is
        # faster by that factor.
        "legacy_pandas_over_stack_index": (
            summaries["legacy_pandas"]["median_ns"] / stack_median
        ),
        "recursive_dfs_over_stack_index": (
            summaries["recursive_dfs"]["median_ns"] / stack_median
        ),
    }

    return {
        "schema_version": 1,
        "source": source,
        "seed": int(seed),
        "n_assets": int(generated_n_assets),
        "n_features": int(n_features),
        "linkage_method": linkage_method,
        "repetitions": repetitions,
        "warmups": warmups,
        "leaf_order_equality_verified": True,
        "methodology": {
            "equality_reference": "scipy.cluster.hierarchy.leaves_list",
            "method_order": "seeded random permutation each round",
            "warmups_per_implementation": warmups,
            "timed_samples_per_implementation": repetitions,
            "garbage_collection_disabled_equally": True,
            "speedup_definition": "baseline median_ns / stack_index_dfs median_ns",
        },
        "implementations": summaries,
        "speedups": speedups,
        "environment": _environment_metadata(),
    }


def benchmark_traversal_suite(
    asset_counts: Iterable[int],
    *,
    seed: int = 20260824,
    n_features: int = 8,
    linkage_method: str = "single",
    repetitions: int = 200,
    warmups: int = 20,
    include_samples: bool = False,
) -> dict[str, Any]:
    """Run ``benchmark_traversals`` for a deterministic sequence of sizes."""

    counts = [int(value) for value in asset_counts]
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("asset_counts must contain positive integers")

    results = []
    for n_assets in counts:
        child_seed = int(
            np.random.SeedSequence([int(seed), n_assets]).generate_state(
                1,
                dtype=np.uint64,
            )[0]
        )
        results.append(
            benchmark_traversals(
                n_assets=n_assets,
                seed=child_seed,
                n_features=n_features,
                linkage_method=linkage_method,
                repetitions=repetitions,
                warmups=warmups,
                include_samples=include_samples,
            )
        )

    return {
        "schema_version": 1,
        "seed": int(seed),
        "asset_counts": counts,
        "results": results,
        "speedup_definition": "baseline median_ns / stack_index_dfs median_ns",
    }


# A singular alias reads naturally at call sites benchmarking one tree.
benchmark_traversal = benchmark_traversals
