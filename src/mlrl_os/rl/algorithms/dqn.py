"""Deep Q-Network (DQN) algorithm for discrete action RL.

Features:
- MLP Q-network with configurable hidden layers
- Target network with soft (Polyak) updates
- Epsilon-greedy exploration with linear decay
- Huber loss (SmoothL1Loss)
- Deterministic seeding via seed_hash
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mlrl_os.experiment.seed import seed_hash

DEFAULT_HIDDEN_DIMS: list[int] = [128, 128]


@dataclass
class DQNConfig:
    """All hyperparameters for DQN, gathered in one place."""

    obs_dim: int
    act_dim: int
    seed: int = 42
    lr: float = 0.001
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay: int = 10_000
    tau: float = 0.005
    hidden_dims: list[int] | None = None


def _build_mlp(input_dim: int, output_dim: int, hidden_dims: list[int]) -> Any:
    """Build a simple MLP with ReLU activations using PyTorch.

    Args:
        input_dim: Input feature dimension.
        output_dim: Output dimension.
        hidden_dims: List of hidden layer widths.

    Returns:
        A torch.nn.Sequential MLP module.
    """
    import torch.nn as nn

    layers: list[nn.Module] = []
    prev_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, h_dim))
        layers.append(nn.ReLU())
        prev_dim = h_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class DQN:
    """Deep Q-Network with target network and epsilon-greedy exploration.

    Implements the standard DQN algorithm with:
    - Separate Q-network and target network
    - Soft target updates (Polyak averaging)
    - Linear epsilon decay for exploration
    - Huber loss for robust Q-value regression
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        seed: int = 42,
        lr: float = 0.001,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay: int = 10_000,
        tau: float = 0.005,
        hidden_dims: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        import torch
        import torch.nn as nn

        self._config = DQNConfig(
            obs_dim=obs_dim,
            act_dim=act_dim,
            seed=seed,
            lr=lr,
            gamma=gamma,
            eps_start=eps_start,
            eps_end=eps_end,
            eps_decay=eps_decay,
            tau=tau,
            hidden_dims=hidden_dims,
        )

        self._seed = seed_hash("dqn", seed)
        torch.manual_seed(self._seed)
        self._rng = np.random.default_rng(self._seed)

        dims = hidden_dims if hidden_dims is not None else DEFAULT_HIDDEN_DIMS
        self._q_net = _build_mlp(obs_dim, act_dim, dims)
        self._target_net = _build_mlp(obs_dim, act_dim, dims)
        self._target_net.load_state_dict(self._q_net.state_dict())

        self._optimizer = torch.optim.Adam(self._q_net.parameters(), lr=lr)
        self._loss_fn = nn.SmoothL1Loss()
        self._step_count = 0

    @property
    def name(self) -> str:
        return "dqn"

    @property
    def epsilon(self) -> float:
        """Current exploration epsilon (linearly decayed)."""
        cfg = self._config
        progress = min(self._step_count / max(cfg.eps_decay, 1), 1.0)
        return cfg.eps_end + (cfg.eps_start - cfg.eps_end) * (1.0 - progress)

    @property
    def step_count(self) -> int:
        """Number of training steps completed."""
        return self._step_count

    def select_action(self, obs: np.ndarray, explore: bool = True) -> int:
        """Select an action using epsilon-greedy policy.

        Args:
            obs: Observation vector (obs_dim,).
            explore: If True, use epsilon-greedy. If False, use greedy.

        Returns:
            Integer action index.
        """
        import torch

        if explore and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self._config.act_dim))

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            q_values = self._q_net(obs_tensor)
            return int(q_values.argmax(dim=1).item())

    def train_step(self, batch: object) -> dict[str, float]:
        """Perform one DQN training step on a batch of transitions.

        Expects a batch object with attributes: states, actions, rewards,
        next_states, dones (all numpy arrays).

        Args:
            batch: Experience batch with transition data.

        Returns:
            Dictionary with 'loss' and 'epsilon' metrics.
        """
        import torch

        self._step_count += 1

        states = torch.FloatTensor(batch.states)  # type: ignore[union-attr]
        actions = torch.LongTensor(batch.actions).unsqueeze(1)  # type: ignore[union-attr]
        rewards = torch.FloatTensor(batch.rewards)  # type: ignore[union-attr]
        next_states = torch.FloatTensor(batch.next_states)  # type: ignore[union-attr]
        dones = torch.BoolTensor(batch.dones)  # type: ignore[union-attr]

        # Current Q-values for selected actions
        q_values = self._q_net(states).gather(1, actions).squeeze(1)

        # Target Q-values via target network
        with torch.no_grad():
            next_q_max = self._target_net(next_states).max(dim=1)[0]
            target = rewards + self._config.gamma * next_q_max * (~dones).float()

        loss = self._loss_fn(q_values, target)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # Soft update target network (Polyak averaging)
        self._soft_update_target()

        return {"loss": loss.item(), "epsilon": self.epsilon}

    def _soft_update_target(self) -> None:
        """Polyak-average Q-net parameters into target network."""
        tau = self._config.tau
        for param, target_param in zip(
            self._q_net.parameters(), self._target_net.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1.0 - tau) * target_param.data
            )

    def save(self, path: Path) -> None:
        """Save DQN checkpoint to disk.

        Args:
            path: File path for the checkpoint.
        """
        import torch

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_net": self._q_net.state_dict(),
                "target_net": self._target_net.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "step_count": self._step_count,
                "config": {
                    "obs_dim": self._config.obs_dim,
                    "act_dim": self._config.act_dim,
                    "seed": self._config.seed,
                    "lr": self._config.lr,
                    "gamma": self._config.gamma,
                    "eps_start": self._config.eps_start,
                    "eps_end": self._config.eps_end,
                    "eps_decay": self._config.eps_decay,
                    "tau": self._config.tau,
                    "hidden_dims": self._config.hidden_dims,
                },
            },
            path,
        )

    def load(self, path: Path) -> None:
        """Load DQN checkpoint from disk.

        Args:
            path: File path to the checkpoint.
        """
        import torch

        checkpoint = torch.load(path, weights_only=False)
        self._q_net.load_state_dict(checkpoint["q_net"])
        self._target_net.load_state_dict(checkpoint["target_net"])
        self._optimizer.load_state_dict(checkpoint["optimizer"])
        self._step_count = checkpoint["step_count"]
