"""Observation and Action space specifications for RL environments.

Defines lightweight dataclasses that describe the shape and bounds of
observation and action spaces without depending on Gymnasium.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class ObservationSpec:
    """Describes observation space shape and bounds."""

    dim: int
    names: list[str]
    low: np.ndarray
    high: np.ndarray

    def contains(self, obs: np.ndarray) -> bool:
        """Check if observation is within bounds."""
        return bool(np.all(obs >= self.low) and np.all(obs <= self.high))


@dataclass
class ActionSpec:
    """Describes action space (discrete or continuous)."""

    type: Literal["discrete", "continuous"]
    n: int | None = None
    shape: tuple[int, ...] | None = None
    low: np.ndarray | None = None
    high: np.ndarray | None = None
    labels: list[str] | None = None


# -- Feature name constants --------------------------------------------------

ENTITY_FEATURES: list[str] = [
    "entity_priority",
    "entity_elapsed_time",
    "entity_cumulative_wait",
    "entity_cumulative_processing",
    "entity_wait_ratio",
    "entity_steps_completed",
    "entity_cumulative_cost",
    "entity_cumulative_transport_cost",
    "entity_cumulative_setup_time",
    "entity_source_index",
]

NODE_FEATURES: list[str] = [
    "node_utilization",
    "node_avg_queue_depth",
    "node_max_queue_depth",
    "node_avg_wait_time",
    "node_avg_processing_time",
    "node_throughput_per_hour",
    "node_concurrency",
    "node_setup_count",
    "node_mutation_count",
]

SYSTEM_FEATURES: list[str] = [
    "sys_utilization",
    "sys_throughput_per_hour",
    "sys_bottleneck_utilization",
    "sys_node_count",
    "sys_total_capacity",
]


def build_specs_from_template(
    metadata: dict,
) -> tuple[ObservationSpec, ActionSpec]:
    """Auto-construct observation and action specs from SimOS template metadata.

    The observation space is built from the canonical feature groups:
    entity_state (10) + node_state (9) + system_state (5) = 24 dimensions.

    The action space is discrete with one action per routing node,
    with a minimum of 2 actions.

    Args:
        metadata: Dictionary with optional ``node_names`` and ``resource_names`` keys.

    Returns:
        Tuple of (ObservationSpec, ActionSpec).
    """
    node_names: list[str] = metadata.get("node_names", [])

    all_names = ENTITY_FEATURES + NODE_FEATURES + SYSTEM_FEATURES
    dim = len(all_names)

    obs_spec = ObservationSpec(
        dim=dim,
        names=all_names,
        low=np.full(dim, -np.inf, dtype=np.float32),
        high=np.full(dim, np.inf, dtype=np.float32),
    )

    act_spec = ActionSpec(
        type="discrete",
        n=max(len(node_names), 2),
        labels=node_names if node_names else None,
    )

    return obs_spec, act_spec
