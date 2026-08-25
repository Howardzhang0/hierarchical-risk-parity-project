"""Configuration loading with stable, serializable defaults."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 20260824,
    "data": {
        "path": "data/raw/crsp_daily_2006_2025.csv.gz",
        "date_column": "dlycaldt",
        "id_column": "permno",
        "label_column": "ticker",
        "return_column": "dlyret",
        "market_cap_column": "dlycap",
        "minimum_coverage": 0.98,
    },
    "covariance": {"estimator": "ledoit_wolf", "eigenvalue_floor": 1e-10},
    "hrp": {"linkage_method": "single", "distance_mode": "original"},
    "monte_carlo": {
        "factors": 5,
        "copula_df": 6,
        "train_observations": 252,
        "test_observations": 252,
        "rebalance_every": 21,
        "replications": 500,
        "bootstrap_replications": 2000,
    },
    "benchmark": {
        "asset_counts": [32, 64, 128, 256, 512],
        "repetitions": 200,
        "warmups": 20,
    },
    "vlstar": {
        "feature_volatility_window": 21,
        "feature_correlation_window": 63,
        "covariance_window": 504,
        "lags": 1,
        "ridge": 1e-4,
        "gamma_grid": [0.5, 1.0, 2.0, 5.0, 10.0],
        "threshold_quantiles": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "probability_bands": [0.35, 0.65],
        "probability_change_trigger": 0.20,
        "minimum_rebalance_gap": 5,
        "refit_interval": 63,
        "high_regime_risk_scale": 0.80,
    },
    "backtest": {
        "start": "2010-01-01",
        "covariance_lookback": 252,
        "rebalance_frequency": "monthly",
        "transaction_cost_bps": 10.0,
        "annualization": 252,
        "risk_free_rate": 0.0,
    },
}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config and merge it recursively over documented defaults."""

    config = deepcopy(DEFAULT_CONFIG)
    if path is None:
        return config
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        supplied = yaml.safe_load(handle) or {}
    if not isinstance(supplied, dict):
        raise ValueError("Configuration root must be a mapping")
    return _deep_update(config, supplied)
