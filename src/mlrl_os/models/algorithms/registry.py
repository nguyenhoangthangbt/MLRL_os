"""Algorithm registry — lazy registration and instantiation of ML algorithms."""

from __future__ import annotations

from mlrl_os.models.algorithms.protocol import Algorithm

KNOWN_ALGORITHMS: list[str] = [
    "lightgbm",
    "xgboost",
    "random_forest",
    "extra_trees",
    "linear",
    "lstm",
]


class AlgorithmRegistry:
    """Registry for ML algorithm classes.

    Stores algorithm *classes* (not instances) so that heavy dependencies
    are only imported when an algorithm is actually instantiated via get().
    """

    def __init__(self) -> None:
        self._algorithms: dict[str, type] = {}

    def register(self, name: str, cls: type) -> None:
        """Register an algorithm class by name.

        Args:
            name: Unique algorithm identifier (e.g. "lightgbm").
            cls: The algorithm class (not an instance).

        Raises:
            ValueError: If an algorithm with this name is already registered.
        """
        if name in self._algorithms:
            msg = f"Algorithm {name!r} is already registered."
            raise ValueError(msg)
        self._algorithms[name] = cls

    def get(self, name: str) -> Algorithm:
        """Instantiate and return an algorithm by name.

        Args:
            name: Algorithm identifier.

        Returns:
            A fresh instance of the requested algorithm.

        Raises:
            KeyError: If no algorithm is registered under this name.
        """
        if name not in self._algorithms:
            available = ", ".join(sorted(self._algorithms)) or "(none)"
            msg = f"Unknown algorithm {name!r}. Available: {available}"
            raise KeyError(msg)
        cls = self._algorithms[name]
        return cls()  # type: ignore[return-value]

    def list_available(self) -> list[str]:
        """Return sorted list of registered algorithm names."""
        return sorted(self._algorithms)

    def has(self, name: str) -> bool:
        """Check whether an algorithm is registered."""
        return name in self._algorithms


def default_registry() -> AlgorithmRegistry:
    """Create and return a registry pre-loaded with all built-in algorithms.

    Algorithm classes are referenced but NOT instantiated — heavy
    dependencies (LightGBM, XGBoost) are only imported when the
    algorithm is actually used via registry.get().
    """
    from mlrl_os.models.algorithms.extra_trees import ExtraTreesAlgorithm
    from mlrl_os.models.algorithms.lightgbm import LightGBMAlgorithm
    from mlrl_os.models.algorithms.linear import LinearAlgorithm
    from mlrl_os.models.algorithms.lstm import LSTMAlgorithm
    from mlrl_os.models.algorithms.random_forest import RandomForestAlgorithm
    from mlrl_os.models.algorithms.xgboost import XGBoostAlgorithm

    registry = AlgorithmRegistry()
    registry.register("lightgbm", LightGBMAlgorithm)
    registry.register("xgboost", XGBoostAlgorithm)
    registry.register("random_forest", RandomForestAlgorithm)
    registry.register("extra_trees", ExtraTreesAlgorithm)
    registry.register("linear", LinearAlgorithm)
    registry.register("lstm", LSTMAlgorithm)
    return registry
