"""Simulation tools for reproducible HRP portfolio experiments."""

from .copula import (
    FactorStudentTCopula,
    FactorTCopula,
    empirical_inverse_cdf,
)
from .experiment import (
    PairedMonteCarloResult,
    run_paired_experiment,
    run_paired_monte_carlo,
    summarize_paired_trials,
)
from .metrics import expected_shortfall, paired_bootstrap_ci, portfolio_statistics

__all__ = [
    "FactorStudentTCopula",
    "FactorTCopula",
    "PairedMonteCarloResult",
    "empirical_inverse_cdf",
    "expected_shortfall",
    "paired_bootstrap_ci",
    "portfolio_statistics",
    "run_paired_experiment",
    "run_paired_monte_carlo",
    "summarize_paired_trials",
]

