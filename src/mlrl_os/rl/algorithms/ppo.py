"""Proximal Policy Optimization (PPO) algorithm for discrete action RL.

Features:
- Actor-Critic network with shared or separate hidden layers
- Clipped surrogate objective
- Generalized Advantage Estimation (GAE)
- Mini-batch updates with multiple epochs
- Entropy bonus for exploration
- Gradient clipping
- Deterministic seeding via seed_hash
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mlrl_os.experiment.seed import seed_hash

DEFAULT_HIDDEN_DIMS: list[int] = [128, 128]


@dataclass
class PPOConfig:
    """All hyperparameters for PPO, gathered in one place."""

    obs_dim: int
    act_dim: int
    seed: int = 42
    lr: float = 0.0003
    gamma: float = 0.99
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    mini_batch_size: int = 64
    hidden_dims: list[int] | None = None


@dataclass
class RolloutBatch:
    """A batch of experience collected from a PPO rollout.

    Contains full rollout data plus computed advantages and returns.
    """

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray
    log_probs: np.ndarray
    advantages: np.ndarray | None = None
    returns: np.ndarray | None = None


def _build_actor_critic(
    obs_dim: int, act_dim: int, hidden_dims: list[int]
) -> Any:
    """Build an Actor-Critic network for PPO.

    The actor outputs a categorical distribution over discrete actions.
    The critic outputs a scalar state value.

    Args:
        obs_dim: Observation dimension.
        act_dim: Number of discrete actions.
        hidden_dims: Hidden layer widths (shared backbone).

    Returns:
        An ActorCritic nn.Module.
    """
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical

    class ActorCritic(nn.Module):
        """Combined actor-critic with a shared feature backbone."""

        def __init__(self) -> None:
            super().__init__()

            # Shared feature backbone
            backbone_layers: list[nn.Module] = []
            prev_dim = obs_dim
            for h_dim in hidden_dims:
                backbone_layers.append(nn.Linear(prev_dim, h_dim))
                backbone_layers.append(nn.ReLU())
                prev_dim = h_dim
            self.backbone = nn.Sequential(*backbone_layers)

            # Actor head: logits -> Categorical distribution
            self.actor_head = nn.Linear(prev_dim, act_dim)

            # Critic head: scalar value
            self.critic_head = nn.Linear(prev_dim, 1)

        def forward(
            self, obs: torch.Tensor
        ) -> tuple[Categorical, torch.Tensor]:
            """Forward pass returning action distribution and state value.

            Args:
                obs: Observation tensor (batch, obs_dim).

            Returns:
                Tuple of (Categorical distribution, value tensor (batch, 1)).
            """
            features = self.backbone(obs)
            logits = self.actor_head(features)
            value = self.critic_head(features)
            return Categorical(logits=logits), value

    return ActorCritic()


class PPO:
    """Proximal Policy Optimization with clipped surrogate and GAE.

    Implements PPO-Clip with:
    - Shared actor-critic backbone
    - Clipped surrogate objective
    - Generalized Advantage Estimation
    - Mini-batch updates over multiple epochs
    - Entropy bonus for exploration
    - Gradient clipping for stability
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        seed: int = 42,
        lr: float = 0.0003,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        gae_lambda: float = 0.95,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        mini_batch_size: int = 64,
        hidden_dims: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        import torch

        self._config = PPOConfig(
            obs_dim=obs_dim,
            act_dim=act_dim,
            seed=seed,
            lr=lr,
            gamma=gamma,
            clip_eps=clip_eps,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            gae_lambda=gae_lambda,
            max_grad_norm=max_grad_norm,
            n_epochs=n_epochs,
            mini_batch_size=mini_batch_size,
            hidden_dims=hidden_dims,
        )

        self._seed = seed_hash("ppo", seed)
        torch.manual_seed(self._seed)
        self._rng = np.random.default_rng(self._seed)

        dims = hidden_dims if hidden_dims is not None else DEFAULT_HIDDEN_DIMS
        self._ac = _build_actor_critic(obs_dim, act_dim, dims)
        self._optimizer = torch.optim.Adam(self._ac.parameters(), lr=lr)

    @property
    def name(self) -> str:
        return "ppo"

    def select_action(self, obs: np.ndarray, explore: bool = True) -> int:
        """Select an action using the current policy.

        Args:
            obs: Observation vector (obs_dim,).
            explore: If True, sample from the policy distribution.
                     If False, return the most probable action.

        Returns:
            Integer action index.
        """
        import torch

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            dist, _value = self._ac(obs_tensor)

            if explore:
                action = dist.sample()
            else:
                action = dist.probs.argmax(dim=1)

            return int(action.item())

    def get_value_and_log_prob(
        self, obs: np.ndarray, action: int
    ) -> tuple[float, float]:
        """Get the value estimate and log-probability for a state-action pair.

        Used during rollout collection to store data needed for PPO updates.

        Args:
            obs: Observation vector (obs_dim,).
            action: The action that was taken.

        Returns:
            Tuple of (value estimate, log probability).
        """
        import torch

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            dist, value = self._ac(obs_tensor)
            log_prob = dist.log_prob(torch.tensor([action]))
            return float(value.item()), float(log_prob.item())

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation.

        Args:
            rewards: Reward array (n_steps,).
            values: Value estimates (n_steps,).
            dones: Done flags (n_steps,).
            last_value: Bootstrap value for the final state.

        Returns:
            Tuple of (advantages, returns) arrays, each (n_steps,).
        """
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        gamma = self._config.gamma
        lam = self._config.gae_lambda

        for t in reversed(range(n)):
            next_value = last_value if t == n - 1 else values[t + 1]
            next_non_terminal = 1.0 - float(dones[t])
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def collect_rollout_batch(
        self,
        states: np.ndarray | list[np.ndarray],
        actions: np.ndarray | list[int],
        rewards: np.ndarray | list[float],
        dones: np.ndarray | list[bool],
        values: np.ndarray | list[float],
        log_probs: np.ndarray | list[float],
        last_value: float = 0.0,
    ) -> RolloutBatch:
        """Create a RolloutBatch with computed GAE advantages and returns.

        Args:
            states: Observation arrays.
            actions: Action indices.
            rewards: Reward values.
            dones: Episode termination flags.
            values: Value estimates from the critic.
            log_probs: Log probabilities from the actor.
            last_value: Bootstrap value for the final state.

        Returns:
            RolloutBatch with advantages and returns computed via GAE.
        """
        rewards_arr = np.asarray(rewards, dtype=np.float32)
        values_arr = np.asarray(values, dtype=np.float32)
        dones_arr = np.asarray(dones, dtype=bool)

        advantages, returns = self.compute_gae(
            rewards_arr, values_arr, dones_arr, last_value
        )

        return RolloutBatch(
            states=np.asarray(states, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.int64),
            rewards=rewards_arr,
            dones=dones_arr,
            values=values_arr,
            log_probs=np.asarray(log_probs, dtype=np.float32),
            advantages=advantages,
            returns=returns,
        )

    def train_step(self, batch: object) -> dict[str, float]:
        """Perform PPO update on a rollout batch.

        Runs multiple epochs of mini-batch updates with the clipped
        surrogate objective, value loss, and entropy bonus.

        Args:
            batch: A RolloutBatch with advantages and returns, or any
                   object with states, actions, log_probs, advantages,
                   and returns attributes.

        Returns:
            Dictionary with policy_loss, value_loss, and entropy metrics
            (averaged over all mini-batch updates).
        """
        import torch
        import torch.nn as nn

        # Extract batch data
        states = torch.FloatTensor(batch.states)  # type: ignore[union-attr]
        actions = torch.LongTensor(batch.actions)  # type: ignore[union-attr]
        old_log_probs = torch.FloatTensor(batch.log_probs)  # type: ignore[union-attr]

        # Use precomputed advantages/returns if available
        if hasattr(batch, "advantages") and batch.advantages is not None:  # type: ignore[union-attr]
            advantages_arr = batch.advantages  # type: ignore[union-attr]
            returns_arr = batch.returns  # type: ignore[union-attr]
        else:
            # Fallback: center rewards as crude advantages
            rewards = batch.rewards  # type: ignore[union-attr]
            advantages_arr = rewards - rewards.mean()
            returns_arr = rewards

        advantages_t = torch.FloatTensor(advantages_arr)
        returns_t = torch.FloatTensor(returns_arr)

        # Normalize advantages for stability
        advantages_t = (advantages_t - advantages_t.mean()) / (
            advantages_t.std() + 1e-8
        )

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        n_samples = len(states)

        for _epoch in range(self._config.n_epochs):
            indices = self._rng.permutation(n_samples)

            for start in range(0, n_samples, self._config.mini_batch_size):
                end = min(start + self._config.mini_batch_size, n_samples)
                mb_idx = indices[start:end]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages_t[mb_idx]
                mb_returns = returns_t[mb_idx]

                dist, values = self._ac(mb_states)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # Clipped surrogate objective
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0 - self._config.clip_eps,
                        1.0 + self._config.clip_eps,
                    )
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (values.squeeze() - mb_returns).pow(2).mean()

                loss = (
                    policy_loss
                    + self._config.value_coef * value_loss
                    - self._config.entropy_coef * entropy
                )

                self._optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self._ac.parameters(), self._config.max_grad_norm
                )
                self._optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        divisor = max(n_updates, 1)
        return {
            "policy_loss": total_policy_loss / divisor,
            "value_loss": total_value_loss / divisor,
            "entropy": total_entropy / divisor,
        }

    def save(self, path: Path) -> None:
        """Save PPO checkpoint to disk.

        Args:
            path: File path for the checkpoint.
        """
        import torch

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "ac": self._ac.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "config": {
                    "obs_dim": self._config.obs_dim,
                    "act_dim": self._config.act_dim,
                    "seed": self._config.seed,
                    "lr": self._config.lr,
                    "gamma": self._config.gamma,
                    "clip_eps": self._config.clip_eps,
                    "entropy_coef": self._config.entropy_coef,
                    "value_coef": self._config.value_coef,
                    "gae_lambda": self._config.gae_lambda,
                    "max_grad_norm": self._config.max_grad_norm,
                    "n_epochs": self._config.n_epochs,
                    "mini_batch_size": self._config.mini_batch_size,
                    "hidden_dims": self._config.hidden_dims,
                },
            },
            path,
        )

    def load(self, path: Path) -> None:
        """Load PPO checkpoint from disk.

        Args:
            path: File path to the checkpoint.
        """
        import torch

        checkpoint = torch.load(path, weights_only=False)
        self._ac.load_state_dict(checkpoint["ac"])
        self._optimizer.load_state_dict(checkpoint["optimizer"])
