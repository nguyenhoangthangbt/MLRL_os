"""Experience replay buffers for off-policy RL training.

Provides both a uniform ``ReplayBuffer`` and a ``PrioritizedReplayBuffer``
that samples transitions proportionally to their stored priority.
"""

from __future__ import annotations

import random as stdlib_random
from dataclasses import dataclass

import numpy as np


@dataclass
class ExperienceBatch:
    """A batch of transitions sampled from a replay buffer."""

    states: np.ndarray  # (batch, obs_dim)
    actions: np.ndarray  # (batch,)
    rewards: np.ndarray  # (batch,)
    next_states: np.ndarray  # (batch, obs_dim)
    dones: np.ndarray  # (batch,) bool


class ReplayBuffer:
    """Fixed-capacity ring buffer with uniform sampling."""

    def __init__(self, capacity: int, seed: int) -> None:
        self._capacity = capacity
        self._rng = stdlib_random.Random(seed)
        self._buffer: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self._pos = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition, overwriting the oldest if at capacity."""
        transition = (
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        )
        if len(self._buffer) < self._capacity:
            self._buffer.append(transition)
        else:
            self._buffer[self._pos] = transition
        self._pos = (self._pos + 1) % self._capacity

    def sample(self, batch_size: int) -> ExperienceBatch:
        """Sample a uniformly random batch of transitions."""
        k = min(batch_size, len(self._buffer))
        batch = self._rng.sample(self._buffer, k)
        states, actions, rewards, next_states, dones = zip(*batch, strict=True)
        return ExperienceBatch(
            states=np.array(states),
            actions=np.array(actions),
            rewards=np.array(rewards),
            next_states=np.array(next_states),
            dones=np.array(dones),
        )

    def __len__(self) -> int:
        return len(self._buffer)


class PrioritizedReplayBuffer:
    """Proportional prioritized experience replay.

    Transitions are sampled with probability proportional to
    ``priority ** alpha``.  When ``alpha=0`` sampling is uniform;
    when ``alpha=1`` sampling is fully proportional to priority.
    """

    def __init__(self, capacity: int, seed: int, alpha: float = 0.6) -> None:
        self._capacity = capacity
        self._rng = np.random.default_rng(seed)
        self._buffer: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self._priorities: list[float] = []
        self._pos = 0
        self._alpha = alpha

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: float = 1.0,
    ) -> None:
        """Store a transition with an associated priority."""
        transition = (
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        )
        scaled_priority = priority**self._alpha
        if len(self._buffer) < self._capacity:
            self._buffer.append(transition)
            self._priorities.append(scaled_priority)
        else:
            self._buffer[self._pos] = transition
            self._priorities[self._pos] = scaled_priority
        self._pos = (self._pos + 1) % self._capacity

    def sample(self, batch_size: int) -> ExperienceBatch:
        """Sample a batch of transitions weighted by priority."""
        k = min(batch_size, len(self._buffer))
        probs = np.array(self._priorities)
        probs = probs / probs.sum()
        indices = self._rng.choice(len(self._buffer), size=k, p=probs, replace=False)
        batch = [self._buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch, strict=True)
        return ExperienceBatch(
            states=np.array(states),
            actions=np.array(actions),
            rewards=np.array(rewards),
            next_states=np.array(next_states),
            dones=np.array(dones),
        )

    def __len__(self) -> int:
        return len(self._buffer)
