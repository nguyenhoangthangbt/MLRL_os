"""RL algorithm protocol -- interface all RL algorithms must implement.

Mirrors the supervised Algorithm protocol in models/algorithms/protocol.py,
adapted for the RL setting: select_action, train_step, save/load.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class RLAlgorithm(Protocol):
    """Interface all RL algorithms must implement.

    Algorithms are registered by name in the RLAlgorithmRegistry and
    instantiated lazily so that heavy dependencies (PyTorch) are only
    imported when actually used.
    """

    @property
    def name(self) -> str:
        """Unique algorithm identifier (e.g. 'dqn', 'ppo')."""
        ...

    def select_action(self, obs: np.ndarray, explore: bool = True) -> int:
        """Select an action given an observation.

        Args:
            obs: Observation vector (obs_dim,).
            explore: If True, use exploration strategy (e.g. epsilon-greedy).
                     If False, use greedy/deterministic policy.

        Returns:
            Integer action index.
        """
        ...

    def train_step(self, batch: object) -> dict[str, float]:
        """Perform a single training update on a batch of experience.

        Args:
            batch: Experience batch (ExperienceBatch for DQN,
                   RolloutBatch for PPO).

        Returns:
            Dictionary of loss/metric values for logging.
        """
        ...

    def save(self, path: Path) -> None:
        """Save algorithm state (networks, optimizer, config) to disk.

        Args:
            path: File path to save checkpoint.
        """
        ...

    def load(self, path: Path) -> None:
        """Load algorithm state from a checkpoint on disk.

        Args:
            path: File path to load checkpoint from.
        """
        ...
