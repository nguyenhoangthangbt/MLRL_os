"""Reward function protocol and built-in reward shapers for RL training.

Each reward function implements the ``RewardFunction`` protocol: given the
previous state, the action taken, the resulting state, and a done flag, it
returns a scalar reward.  Built-in shapers can be combined via
``CompositeReward`` for multi-objective optimization.
"""

from __future__ import annotations

from typing import Any, Protocol


class RewardFunction(Protocol):
    """Protocol for all reward functions."""

    def compute(
        self,
        prev_state: dict[str, Any],
        action: int,
        curr_state: dict[str, Any],
        done: bool,
    ) -> float: ...


class ThroughputReward:
    """Positive reward proportional to entity completions since the last step."""

    def compute(
        self,
        prev_state: dict[str, Any],
        action: int,
        curr_state: dict[str, Any],
        done: bool,
    ) -> float:
        prev_completed = prev_state.get("completed_count", 0)
        curr_completed = curr_state.get("completed_count", 0)
        return float(curr_completed - prev_completed)


class SLAComplianceReward:
    """Penalises SLA breaches and optionally rewards compliant completions."""

    def __init__(
        self,
        breach_penalty: float = -10.0,
        compliance_bonus: float = 1.0,
    ) -> None:
        self.breach_penalty = breach_penalty
        self.compliance_bonus = compliance_bonus

    def compute(
        self,
        prev_state: dict[str, Any],
        action: int,
        curr_state: dict[str, Any],
        done: bool,
    ) -> float:
        breaches = curr_state.get("sla_breaches", 0) - prev_state.get("sla_breaches", 0)
        if breaches > 0:
            return self.breach_penalty * breaches
        if done and curr_state.get("completed_count", 0) > prev_state.get("completed_count", 0):
            return self.compliance_bonus
        return 0.0


class CostMinimizationReward:
    """Negative reward equal to the cost incurred in the current step."""

    def compute(
        self,
        prev_state: dict[str, Any],
        action: int,
        curr_state: dict[str, Any],
        done: bool,
    ) -> float:
        prev_cost = prev_state.get("total_cost", 0.0)
        curr_cost = curr_state.get("total_cost", 0.0)
        return -(curr_cost - prev_cost)


class CompositeReward:
    """Weighted combination of multiple reward functions."""

    def __init__(self, components: list[tuple[RewardFunction, float]]) -> None:
        self.components = components

    def compute(
        self,
        prev_state: dict[str, Any],
        action: int,
        curr_state: dict[str, Any],
        done: bool,
    ) -> float:
        return sum(
            weight * reward_fn.compute(prev_state, action, curr_state, done)
            for reward_fn, weight in self.components
        )
