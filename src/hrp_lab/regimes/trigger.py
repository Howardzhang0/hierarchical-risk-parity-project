"""Hysteretic, rate-limited volatility-regime rebalance triggers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TriggerConfig:
    low_probability: float = 0.35
    high_probability: float = 0.65
    probability_change: float | None = 0.20
    minimum_gap: int = 5
    force_after: int | None = 21
    initial_regime: str = "low"

    def __post_init__(self) -> None:
        if not 0.0 <= self.low_probability < self.high_probability <= 1.0:
            raise ValueError("probability bands must satisfy 0 <= low < high <= 1")
        if self.probability_change is not None and not 0.0 < self.probability_change <= 1.0:
            raise ValueError("probability_change must lie in (0, 1]")
        if self.minimum_gap < 0:
            raise ValueError("minimum_gap must be non-negative")
        if self.force_after is not None and self.force_after < max(1, self.minimum_gap):
            raise ValueError("force_after must be at least the minimum gap")
        if self.initial_regime not in {"low", "high"}:
            raise ValueError("initial_regime must be 'low' or 'high'")


@dataclass(frozen=True)
class TriggerDecision:
    rebalance: bool
    regime: str
    probability: float
    reasons: tuple[str, ...]
    observations_since_rebalance: int | None
    state_changed: bool


class RegimeTrigger:
    """Stateful trigger with hysteresis, a minimum gap, and a force timeout."""

    def __init__(self, config: TriggerConfig | None = None) -> None:
        self.config = config or TriggerConfig()
        self.regime = self.config.initial_regime
        self.last_rebalance_position: int | None = None
        self.last_rebalance_probability: float | None = None
        self.last_rebalance_regime: str | None = None
        self._last_position: int | None = None

    def step(self, probability: float, position: int) -> TriggerDecision:
        probability = float(probability)
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("probability must lie in [0, 1]")
        if self._last_position is not None and position <= self._last_position:
            raise ValueError("trigger positions must be strictly increasing")
        self._last_position = int(position)

        previous_regime = self.regime
        if self.regime == "low" and probability >= self.config.high_probability:
            self.regime = "high"
        elif self.regime == "high" and probability <= self.config.low_probability:
            self.regime = "low"
        state_changed = self.regime != previous_regime

        if self.last_rebalance_position is None:
            gap: int | None = None
            eligible = True
        else:
            gap = position - self.last_rebalance_position
            eligible = gap >= self.config.minimum_gap

        reasons: list[str] = []
        if self.last_rebalance_position is None:
            reasons.append("initial")
        elif eligible:
            if self.regime != self.last_rebalance_regime:
                reasons.append("regime_change")
            if (
                self.config.probability_change is not None
                and self.last_rebalance_probability is not None
                and abs(probability - self.last_rebalance_probability)
                >= self.config.probability_change
            ):
                reasons.append("probability_change")
            if (
                self.config.force_after is not None
                and gap is not None
                and gap >= self.config.force_after
            ):
                reasons.append("forced")

        rebalance = bool(reasons) and eligible
        if rebalance:
            self.last_rebalance_position = int(position)
            self.last_rebalance_probability = probability
            self.last_rebalance_regime = self.regime

        return TriggerDecision(
            rebalance=rebalance,
            regime=self.regime,
            probability=probability,
            reasons=tuple(reasons) if rebalance else (),
            observations_since_rebalance=gap,
            state_changed=state_changed,
        )


def generate_rebalance_schedule(
    probabilities: pd.Series,
    *,
    config: TriggerConfig | None = None,
) -> pd.DataFrame:
    """Apply :class:`RegimeTrigger` to a dated probability series."""

    if not isinstance(probabilities, pd.Series) or probabilities.empty:
        raise ValueError("probabilities must be a non-empty Series")
    if not probabilities.index.is_monotonic_increasing or probabilities.index.has_duplicates:
        raise ValueError("probability index must be unique and sorted")
    trigger = RegimeTrigger(config)
    rows: list[dict[str, object]] = []
    for position, (date, probability) in enumerate(probabilities.items()):
        if not np.isfinite(probability):
            continue
        decision = trigger.step(float(probability), position)
        rows.append(
            {
                "date": date,
                "probability": decision.probability,
                "regime": decision.regime,
                "rebalance": decision.rebalance,
                "reason": "+".join(decision.reasons),
                "state_changed": decision.state_changed,
                "observations_since_rebalance": decision.observations_since_rebalance,
            }
        )
    return pd.DataFrame(rows).set_index("date")
