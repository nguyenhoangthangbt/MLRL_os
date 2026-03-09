"""Tests for mlrl_os.rl.replay_buffer — Experience replay storage."""

from __future__ import annotations

import numpy as np

from mlrl_os.rl.replay_buffer import (
    ExperienceBatch,
    PrioritizedReplayBuffer,
    ReplayBuffer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _push_n(buf: ReplayBuffer | PrioritizedReplayBuffer, n: int) -> None:
    """Push *n* trivial transitions into a buffer."""
    for i in range(n):
        state = np.array([float(i), 0.0], dtype=np.float32)
        next_state = np.array([float(i + 1), 0.0], dtype=np.float32)
        buf.push(state, action=i % 3, reward=float(i), next_state=next_state, done=(i == n - 1))


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------


class TestReplayBuffer:
    """Tests for the uniform ReplayBuffer."""

    def test_push_and_length(self) -> None:
        buf = ReplayBuffer(capacity=100, seed=42)
        assert len(buf) == 0
        _push_n(buf, 5)
        assert len(buf) == 5

    def test_capacity_wraps(self) -> None:
        buf = ReplayBuffer(capacity=4, seed=42)
        _push_n(buf, 10)
        assert len(buf) == 4

    def test_sample_returns_experience_batch(self) -> None:
        buf = ReplayBuffer(capacity=100, seed=42)
        _push_n(buf, 20)
        batch = buf.sample(batch_size=8)

        assert isinstance(batch, ExperienceBatch)
        assert batch.states.shape == (8, 2)
        assert batch.actions.shape == (8,)
        assert batch.rewards.shape == (8,)
        assert batch.next_states.shape == (8, 2)
        assert batch.dones.shape == (8,)

    def test_sample_clamps_to_buffer_size(self) -> None:
        buf = ReplayBuffer(capacity=100, seed=42)
        _push_n(buf, 3)
        batch = buf.sample(batch_size=10)
        assert batch.states.shape[0] == 3

    def test_deterministic_sampling_with_same_seed(self) -> None:
        buf1 = ReplayBuffer(capacity=100, seed=99)
        buf2 = ReplayBuffer(capacity=100, seed=99)
        _push_n(buf1, 50)
        _push_n(buf2, 50)

        b1 = buf1.sample(10)
        b2 = buf2.sample(10)
        np.testing.assert_array_equal(b1.actions, b2.actions)
        np.testing.assert_array_equal(b1.rewards, b2.rewards)

    def test_different_seeds_differ(self) -> None:
        buf1 = ReplayBuffer(capacity=100, seed=1)
        buf2 = ReplayBuffer(capacity=100, seed=2)
        _push_n(buf1, 50)
        _push_n(buf2, 50)

        b1 = buf1.sample(20)
        b2 = buf2.sample(20)
        # Very unlikely to be identical with different seeds
        assert not np.array_equal(b1.rewards, b2.rewards)

    def test_dtypes(self) -> None:
        buf = ReplayBuffer(capacity=100, seed=42)
        _push_n(buf, 5)
        batch = buf.sample(3)

        assert batch.states.dtype == np.float32
        assert batch.next_states.dtype == np.float32
        assert batch.dones.dtype == bool


# ---------------------------------------------------------------------------
# PrioritizedReplayBuffer
# ---------------------------------------------------------------------------


class TestPrioritizedReplayBuffer:
    """Tests for the prioritized experience replay buffer."""

    def test_push_and_length(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=100, seed=42)
        assert len(buf) == 0
        _push_n(buf, 5)
        assert len(buf) == 5

    def test_capacity_wraps(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=4, seed=42)
        _push_n(buf, 10)
        assert len(buf) == 4

    def test_sample_returns_experience_batch(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=100, seed=42)
        _push_n(buf, 20)
        batch = buf.sample(batch_size=8)

        assert isinstance(batch, ExperienceBatch)
        assert batch.states.shape == (8, 2)
        assert batch.actions.shape == (8,)

    def test_high_priority_sampled_more_often(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=100, seed=42, alpha=1.0)

        # Push one low-priority and one high-priority transition
        low_state = np.array([0.0, 0.0], dtype=np.float32)
        high_state = np.array([1.0, 1.0], dtype=np.float32)
        buf.push(low_state, 0, 0.0, low_state, False, priority=0.01)
        buf.push(high_state, 1, 1.0, high_state, False, priority=100.0)

        # Sample many times and count how often the high-priority item appears
        high_count = 0
        for _ in range(200):
            batch = buf.sample(1)
            if batch.actions[0] == 1:
                high_count += 1

        # High-priority item should dominate
        assert high_count > 150

    def test_sample_clamps_to_buffer_size(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=100, seed=42)
        _push_n(buf, 3)
        batch = buf.sample(batch_size=10)
        assert batch.states.shape[0] == 3

    def test_custom_alpha(self) -> None:
        # alpha=0 means uniform sampling regardless of priority
        buf = PrioritizedReplayBuffer(capacity=100, seed=42, alpha=0.0)
        low = np.array([0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0], dtype=np.float32)
        buf.push(low, 0, 0.0, low, False, priority=0.01)
        buf.push(high, 1, 1.0, high, False, priority=100.0)

        counts = {0: 0, 1: 0}
        for _ in range(200):
            batch = buf.sample(1)
            counts[int(batch.actions[0])] += 1

        # With alpha=0, both should be sampled roughly equally
        assert counts[0] > 60
        assert counts[1] > 60
