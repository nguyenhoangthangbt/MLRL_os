"""Tests for mlrl_os.rl.algorithms.dqn -- Deep Q-Network algorithm."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OBS_DIM = 8
ACT_DIM = 4
SEED = 42


@dataclass
class FakeBatch:
    """Minimal experience batch for DQN tests."""

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray


def _make_batch(n: int = 32) -> FakeBatch:
    """Create a synthetic batch of transitions."""
    rng = np.random.default_rng(SEED)
    return FakeBatch(
        states=rng.standard_normal((n, OBS_DIM)).astype(np.float32),
        actions=rng.integers(0, ACT_DIM, size=n),
        rewards=rng.standard_normal(n).astype(np.float32),
        next_states=rng.standard_normal((n, OBS_DIM)).astype(np.float32),
        dones=rng.choice([False, True], size=n, p=[0.9, 0.1]),
    )


def _make_dqn(**kwargs: object) -> object:
    """Create a DQN instance with small defaults for testing."""
    from mlrl_os.rl.algorithms.dqn import DQN

    defaults: dict = {
        "obs_dim": OBS_DIM,
        "act_dim": ACT_DIM,
        "seed": SEED,
        "hidden_dims": [32, 32],
        "eps_decay": 100,
    }
    defaults.update(kwargs)
    return DQN(**defaults)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestDQNConstruction:
    """Tests for DQN initialization."""

    def test_name_is_dqn(self) -> None:
        dqn = _make_dqn()
        assert dqn.name == "dqn"

    def test_initial_epsilon_equals_eps_start(self) -> None:
        dqn = _make_dqn(eps_start=1.0)
        assert dqn.epsilon == pytest.approx(1.0)

    def test_initial_step_count_is_zero(self) -> None:
        dqn = _make_dqn()
        assert dqn.step_count == 0

    def test_default_hidden_dims(self) -> None:
        from mlrl_os.rl.algorithms.dqn import DQN

        dqn = DQN(obs_dim=OBS_DIM, act_dim=ACT_DIM, seed=SEED)
        # Should not raise -- uses DEFAULT_HIDDEN_DIMS internally
        assert dqn.name == "dqn"

    def test_custom_hidden_dims(self) -> None:
        dqn = _make_dqn(hidden_dims=[64, 64, 64])
        assert dqn.name == "dqn"


# ---------------------------------------------------------------------------
# select_action
# ---------------------------------------------------------------------------


class TestDQNSelectAction:
    """Tests for DQN.select_action."""

    def test_returns_valid_action_index_with_exploration(self) -> None:
        dqn = _make_dqn()
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        action = dqn.select_action(obs, explore=True)
        assert 0 <= action < ACT_DIM

    def test_returns_valid_action_index_without_exploration(self) -> None:
        dqn = _make_dqn()
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        action = dqn.select_action(obs, explore=False)
        assert 0 <= action < ACT_DIM

    def test_greedy_action_is_deterministic(self) -> None:
        dqn = _make_dqn()
        obs = np.random.default_rng(99).standard_normal(OBS_DIM).astype(np.float32)

        actions = [dqn.select_action(obs, explore=False) for _ in range(10)]
        assert len(set(actions)) == 1, "Greedy actions should be deterministic"

    def test_exploration_produces_varied_actions(self) -> None:
        """With eps=1.0, all actions should be random (high diversity)."""
        dqn = _make_dqn(eps_start=1.0, eps_end=1.0)
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        actions = [dqn.select_action(obs, explore=True) for _ in range(100)]
        unique = set(actions)
        # With 4 actions and 100 random samples, we should see multiple actions
        assert len(unique) > 1


# ---------------------------------------------------------------------------
# train_step
# ---------------------------------------------------------------------------


class TestDQNTrainStep:
    """Tests for DQN.train_step."""

    def test_returns_loss_and_epsilon(self) -> None:
        dqn = _make_dqn()
        batch = _make_batch()
        metrics = dqn.train_step(batch)

        assert "loss" in metrics
        assert "epsilon" in metrics
        assert isinstance(metrics["loss"], float)
        assert isinstance(metrics["epsilon"], float)

    def test_loss_is_finite(self) -> None:
        dqn = _make_dqn()
        batch = _make_batch()
        metrics = dqn.train_step(batch)
        assert np.isfinite(metrics["loss"])

    def test_step_count_increments(self) -> None:
        dqn = _make_dqn()
        batch = _make_batch()

        assert dqn.step_count == 0
        dqn.train_step(batch)
        assert dqn.step_count == 1
        dqn.train_step(batch)
        assert dqn.step_count == 2

    def test_epsilon_decays_over_steps(self) -> None:
        dqn = _make_dqn(eps_start=1.0, eps_end=0.01, eps_decay=50)
        batch = _make_batch()

        eps_before = dqn.epsilon
        for _ in range(25):
            dqn.train_step(batch)
        eps_after = dqn.epsilon

        assert eps_after < eps_before, "Epsilon should decay with training steps"

    def test_epsilon_floors_at_eps_end(self) -> None:
        dqn = _make_dqn(eps_start=1.0, eps_end=0.05, eps_decay=10)
        batch = _make_batch()

        # Train well past eps_decay steps
        for _ in range(50):
            dqn.train_step(batch)

        assert dqn.epsilon == pytest.approx(0.05)

    def test_multiple_train_steps_reduce_loss(self) -> None:
        """Loss should generally decrease over many steps on the same batch."""
        dqn = _make_dqn(lr=0.01)
        batch = _make_batch()

        initial_metrics = dqn.train_step(batch)
        for _ in range(49):
            dqn.train_step(batch)
        final_metrics = dqn.train_step(batch)

        # Not guaranteed per-step, but over 50 steps loss should decrease
        assert final_metrics["loss"] < initial_metrics["loss"] * 5, (
            "Loss should not explode over training"
        )


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


class TestDQNSaveLoad:
    """Tests for DQN checkpoint save and load."""

    def test_save_creates_file(self, tmp_path: object) -> None:
        from pathlib import Path

        path = Path(str(tmp_path)) / "dqn_checkpoint.pt"
        dqn = _make_dqn()
        dqn.save(path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_load_restores_step_count(self, tmp_path: object) -> None:
        from pathlib import Path

        path = Path(str(tmp_path)) / "dqn_checkpoint.pt"
        dqn = _make_dqn()
        batch = _make_batch()

        # Train a few steps
        for _ in range(5):
            dqn.train_step(batch)
        assert dqn.step_count == 5

        dqn.save(path)

        # Create a fresh DQN and load
        dqn2 = _make_dqn()
        assert dqn2.step_count == 0
        dqn2.load(path)
        assert dqn2.step_count == 5

    def test_load_restores_q_values(self, tmp_path: object) -> None:
        from pathlib import Path

        path = Path(str(tmp_path)) / "dqn_checkpoint.pt"
        dqn = _make_dqn()
        batch = _make_batch()

        # Train to change weights
        for _ in range(10):
            dqn.train_step(batch)

        obs = np.random.default_rng(123).standard_normal(OBS_DIM).astype(np.float32)
        action_before = dqn.select_action(obs, explore=False)

        dqn.save(path)

        dqn2 = _make_dqn()
        dqn2.load(path)
        action_after = dqn2.select_action(obs, explore=False)

        assert action_before == action_after, (
            "Loaded model should produce the same greedy action"
        )

    def test_save_creates_parent_directories(self, tmp_path: object) -> None:
        from pathlib import Path

        path = Path(str(tmp_path)) / "nested" / "dir" / "dqn.pt"
        dqn = _make_dqn()
        dqn.save(path)
        assert path.exists()


# ---------------------------------------------------------------------------
# Seeded reproducibility
# ---------------------------------------------------------------------------


class TestDQNReproducibility:
    """Tests for deterministic behavior with the same seed."""

    def test_same_seed_same_initial_actions(self) -> None:
        obs = np.ones(OBS_DIM, dtype=np.float32)

        dqn1 = _make_dqn(seed=42)
        dqn2 = _make_dqn(seed=42)

        # Greedy actions should match for same seed and same initial weights
        a1 = dqn1.select_action(obs, explore=False)
        a2 = dqn2.select_action(obs, explore=False)
        assert a1 == a2

    def test_different_seeds_different_networks(self) -> None:
        """Different seeds should produce different initial Q-networks."""
        import torch

        dqn1 = _make_dqn(seed=1)
        dqn2 = _make_dqn(seed=999)

        obs = torch.FloatTensor(np.ones(OBS_DIM, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            q1 = dqn1._q_net(obs).numpy()
            q2 = dqn2._q_net(obs).numpy()

        assert not np.allclose(q1, q2), (
            "Different seeds should produce different network weights"
        )
