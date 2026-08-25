"""Plots and Markdown reporting for complete reproduction runs."""

from .plots import (
    plot_backtest_wealth,
    plot_monte_carlo_variance,
    plot_regime_probability,
    plot_traversal_speedups,
)
from .report import write_reproduction_report

__all__ = [
    "plot_backtest_wealth",
    "plot_monte_carlo_variance",
    "plot_regime_probability",
    "plot_traversal_speedups",
    "write_reproduction_report",
]

