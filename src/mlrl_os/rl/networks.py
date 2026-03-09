"""Neural network building blocks for RL agents.

PyTorch is **lazy-imported** to keep it an optional dependency.  Importing
this module does not fail when torch is absent -- only constructing a network
will raise ``ModuleNotFoundError``.
"""

from __future__ import annotations

from typing import Any


def _import_torch() -> tuple[Any, Any]:
    """Lazy-import torch and torch.nn, raising a clear error if absent."""
    import torch
    import torch.nn as nn

    return torch, nn


class MLP:
    """Configurable multi-layer perceptron.

    Args:
        input_dim: Dimensionality of inputs.
        output_dim: Dimensionality of outputs.
        hidden_dims: Sizes of hidden layers (default ``[64, 64]``).
        activation: ``"relu"`` or ``"tanh"`` (default ``"relu"``).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
    ) -> None:
        torch, nn = _import_torch()
        hidden_dims = hidden_dims or [64, 64]

        act_cls = nn.ReLU if activation == "relu" else nn.Tanh

        layers: list[Any] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(act_cls())
            prev = h
        layers.append(nn.Linear(prev, output_dim))

        self._net = nn.Sequential(*layers)

    def __call__(self, x: Any) -> Any:
        return self._net(x)

    @property
    def parameters(self) -> Any:
        """Return an iterator over network parameters."""
        return self._net.parameters()

    def state_dict(self) -> dict[str, Any]:
        """Return the network's state dictionary."""
        return self._net.state_dict()

    def load_state_dict(self, d: dict[str, Any]) -> None:
        """Load a state dictionary into the network."""
        self._net.load_state_dict(d)


class ActorCritic:
    """Shared-backbone actor-critic architecture for PPO.

    A common MLP backbone feeds into two separate heads:
    - **Actor head**: produces logits over discrete actions.
    - **Critic head**: produces a scalar state-value estimate.

    Args:
        obs_dim: Observation dimensionality.
        act_dim: Number of discrete actions.
        hidden_dims: Sizes of shared backbone layers (default ``[64, 64]``).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: list[int] | None = None,
    ) -> None:
        torch, nn = _import_torch()
        hidden_dims = hidden_dims or [64, 64]

        backbone_layers: list[Any] = []
        prev = obs_dim
        for h in hidden_dims:
            backbone_layers.append(nn.Linear(prev, h))
            backbone_layers.append(nn.ReLU())
            prev = h

        self._backbone = nn.Sequential(*backbone_layers)
        self._actor_head = nn.Linear(prev, act_dim)
        self._critic_head = nn.Linear(prev, 1)

        self._all_params = (
            list(self._backbone.parameters())
            + list(self._actor_head.parameters())
            + list(self._critic_head.parameters())
        )

    def __call__(self, obs: Any) -> tuple[Any, Any]:
        """Forward pass returning ``(action_distribution, state_value)``."""
        torch, _ = _import_torch()
        features = self._backbone(obs)
        logits = self._actor_head(features)
        dist = torch.distributions.Categorical(logits=logits)
        value = self._critic_head(features)
        return dist, value

    @property
    def parameters(self) -> list[Any]:
        """Return all trainable parameters across backbone, actor, and critic."""
        return self._all_params

    def state_dict(self) -> dict[str, Any]:
        """Return a composite state dictionary."""
        return {
            "backbone": self._backbone.state_dict(),
            "actor": self._actor_head.state_dict(),
            "critic": self._critic_head.state_dict(),
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        """Load a composite state dictionary."""
        self._backbone.load_state_dict(d["backbone"])
        self._actor_head.load_state_dict(d["actor"])
        self._critic_head.load_state_dict(d["critic"])
