"""Algorithm protocol — interface all ML algorithms must implement."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class TrainedModel:
    """Container for a trained model artifact.

    Wraps the underlying library-specific model object alongside metadata
    needed by the evaluation and prediction pipeline.
    """

    def __init__(
        self,
        model: Any,
        algorithm_name: str,
        task: str,
        feature_names: list[str],
    ) -> None:
        self.model = model
        self.algorithm_name = algorithm_name
        self.task = task  # "regression" | "classification"
        self.feature_names = feature_names

    def __repr__(self) -> str:
        return (
            f"TrainedModel(algorithm={self.algorithm_name!r}, "
            f"task={self.task!r}, features={len(self.feature_names)})"
        )


class Algorithm(Protocol):
    """Interface all ML algorithms must implement.

    Algorithms are registered by name in the AlgorithmRegistry and
    instantiated lazily so that heavy dependencies (LightGBM, XGBoost)
    are only imported when actually used.
    """

    @property
    def name(self) -> str:
        """Unique algorithm identifier (e.g. 'lightgbm', 'xgboost')."""
        ...

    @property
    def supports_regression(self) -> bool:
        """Whether this algorithm can handle regression tasks."""
        ...

    @property
    def supports_classification(self) -> bool:
        """Whether this algorithm can handle classification tasks."""
        ...

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: str,
        seed: int,
        **kwargs: Any,
    ) -> TrainedModel:
        """Train a model on the given data.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target vector (n_samples,).
            task: Either "regression" or "classification".
            seed: Random seed for reproducibility.
            **kwargs: Algorithm-specific overrides.

        Returns:
            A TrainedModel wrapping the fitted model.
        """
        ...

    def predict(self, model: TrainedModel, X: np.ndarray) -> np.ndarray:
        """Generate predictions for the given feature matrix.

        Args:
            model: A TrainedModel produced by this algorithm's train().
            X: Feature matrix (n_samples, n_features).

        Returns:
            Predictions array (n_samples,).
        """
        ...

    def predict_proba(
        self, model: TrainedModel, X: np.ndarray
    ) -> np.ndarray | None:
        """Generate class probability predictions.

        Args:
            model: A TrainedModel produced by this algorithm's train().
            X: Feature matrix (n_samples, n_features).

        Returns:
            Probability matrix (n_samples, n_classes) for classifiers,
            None for regressors.
        """
        ...

    def feature_importance(
        self, model: TrainedModel
    ) -> dict[str, float] | None:
        """Extract feature importance scores from a trained model.

        Args:
            model: A TrainedModel produced by this algorithm's train().

        Returns:
            Dict mapping feature name to importance (normalized to sum=1),
            or None if the algorithm does not support feature importance.
        """
        ...
