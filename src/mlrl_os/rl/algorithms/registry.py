"""RL algorithm registry -- lazy registration and instantiation of RL algorithms.

Same pattern as models/algorithms/registry.py: stores algorithm *classes*
(not instances) so that PyTorch is only imported when an algorithm is
actually instantiated via get().
"""

from __future__ import annotations

from typing import Any

from mlrl_os.rl.algorithms.protocol import RLAlgorithm

KNOWN_RL_ALGORITHMS: list[str] = [
    "dqn",
    "ppo",
]


class RLAlgorithmRegistry:
    """Registry for RL algorithm classes.

    Stores algorithm *classes* (not instances) so that heavy dependencies
    (PyTorch) are only imported when an algorithm is actually instantiated
    via get().
    """

    def __init__(self) -> None:
        self._algorithms: dict[str, type] = {}

    def register(self, name: str, cls: type) -> None:
        """Register an RL algorithm class by name.

        Args:
            name: Unique algorithm identifier (e.g. "dqn").
            cls: The algorithm class (not an instance).

        Raises:
            ValueError: If an algorithm with this name is already registered.
        """
        if name in self._algorithms:
            msg = f"RL algorithm {name!r} is already registered."
            raise ValueError(msg)
        self._algorithms[name] = cls

    def get(self, name: str, **kwargs: Any) -> RLAlgorithm:
        """Instantiate and return an RL algorithm by name.

        Args:
            name: Algorithm identifier.
            **kwargs: Constructor arguments forwarded to the algorithm class.

        Returns:
            A fresh instance of the requested RL algorithm.

        Raises:
            KeyError: If no algorithm is registered under this name.
        """
        if name not in self._algorithms:
            available = ", ".join(sorted(self._algorithms)) or "(none)"
            msg = f"Unknown RL algorithm {name!r}. Available: {available}"
            raise KeyError(msg)
        cls = self._algorithms[name]
        return cls(**kwargs)  # type: ignore[return-value]

    def list_available(self) -> list[str]:
        """Return sorted list of registered RL algorithm names."""
        return sorted(self._algorithms)

    def has(self, name: str) -> bool:
        """Check whether an RL algorithm is registered."""
        return name in self._algorithms


def default_rl_registry() -> RLAlgorithmRegistry:
    """Create and return a registry pre-loaded with all built-in RL algorithms.

    Algorithm classes are referenced but NOT instantiated -- PyTorch
    is only imported when the algorithm is actually used via registry.get().
    """
    from mlrl_os.rl.algorithms.dqn import DQN
    from mlrl_os.rl.algorithms.ppo import PPO

    registry = RLAlgorithmRegistry()
    registry.register("dqn", DQN)
    registry.register("ppo", PPO)
    return registry
