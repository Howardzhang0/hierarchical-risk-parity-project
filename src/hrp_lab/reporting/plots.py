"""Small, deterministic figures used by the reproduction report."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

_mpl_cache = Path(tempfile.gettempdir()) / "hrp_lab_matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _finish(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=170, bbox_inches="tight")
    plt.close()
    return output


def plot_backtest_wealth(returns: pd.DataFrame, path: str | Path) -> Path:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    plt.figure(figsize=(10, 5.5))
    for column in wealth:
        plt.plot(wealth.index, wealth[column], label=str(column), linewidth=1.5)
    plt.yscale("log")
    plt.title("Out-of-sample portfolio wealth (net of costs)")
    plt.ylabel("Growth of $1 (log scale)")
    plt.xlabel("")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, fontsize=8)
    return _finish(path)


def plot_monte_carlo_variance(trials: pd.DataFrame, path: str | Path) -> Path:
    plt.figure(figsize=(8, 5))
    data = [trials[f"{method}_variance"].to_numpy() for method in ("hrp", "ivp", "erc")]
    plt.boxplot(data, tick_labels=["HRP", "IVP", "ERC"], showfliers=False)
    plt.title("Paired held-out variance across Monte Carlo trials")
    plt.ylabel("Daily realized variance")
    plt.grid(axis="y", alpha=0.25)
    return _finish(path)


def plot_regime_probability(
    probability: pd.Series,
    path: str | Path,
    *,
    low: float = 0.35,
    high: float = 0.65,
) -> Path:
    values = probability.dropna()
    plt.figure(figsize=(10, 4.5))
    plt.plot(values.index, values, color="#6a3d9a", linewidth=1.0)
    plt.axhline(low, color="#1b9e77", linestyle="--", linewidth=1, label="exit band")
    plt.axhline(high, color="#d95f02", linestyle="--", linewidth=1, label="entry band")
    plt.fill_between(values.index, high, values.to_numpy(), where=values.to_numpy() >= high, color="#d95f02", alpha=0.15)
    plt.ylim(0, 1)
    plt.title("Point-in-time VLSTAR-style high-volatility probability")
    plt.ylabel("Probability")
    plt.xlabel("")
    plt.grid(alpha=0.2)
    plt.legend(frameon=False)
    return _finish(path)


def plot_traversal_speedups(benchmark: dict[str, Any], path: str | Path) -> Path:
    synthetic = benchmark["synthetic"]["results"]
    sizes = [row["n_assets"] for row in synthetic]
    legacy = [row["speedups"]["legacy_pandas_over_stack_index"] for row in synthetic]
    recursive = [row["speedups"]["recursive_dfs_over_stack_index"] for row in synthetic]
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, legacy, marker="o", label="legacy pandas / stack")
    plt.plot(sizes, recursive, marker="s", label="recursive / stack")
    plt.axhline(4.0, color="black", linestyle="--", linewidth=1, label="4x target")
    plt.xscale("log", base=2)
    plt.title("Quasi-diagonalization traversal speedup")
    plt.xlabel("Number of assets")
    plt.ylabel("Median speedup (x)")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    return _finish(path)
