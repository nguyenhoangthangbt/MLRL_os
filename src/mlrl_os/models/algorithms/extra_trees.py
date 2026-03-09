"""Extra Trees algorithm wrapper for ML/RL OS."""

from __future__ import annotations

from typing import Any

import numpy as np

from mlrl_os.models.algorithms.protocol import TrainedModel

REGRESSION_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "n_jobs": -1,
}

CLASSIFICATION_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "n_jobs": -1,
}


class ExtraTreesAlgorithm:
    """Scikit-learn Extra Trees (Extremely Randomized Trees) algorithm.

    Similar to Random Forest but with additional randomization in split
    thresholds. Supports both regression and classification.
    """

    @property
    def name(self) -> str:
        return "extra_trees"

    @property
    def supports_regression(self) -> bool:
        return True

    @property
    def supports_classification(self) -> bool:
        return True

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: str,
        seed: int,
        **kwargs: Any,
    ) -> TrainedModel:
        """Train an Extra Trees model.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target vector (n_samples,).
            task: "regression" or "classification".
            seed: Random seed for reproducibility.
            **kwargs: Parameter overrides merged on top of defaults.

        Returns:
            TrainedModel wrapping the fitted ExtraTrees estimator.

        Raises:
            ValueError: If the task is not supported.
        """
        from sklearn.ensemble import (
            ExtraTreesClassifier,
            ExtraTreesRegressor,
        )

        feature_names = kwargs.pop("feature_names", [f"f{i}" for i in range(X.shape[1])])
        handle_imbalance = kwargs.pop("handle_imbalance", False)

        if task == "regression":
            params = {**REGRESSION_PARAMS, **kwargs, "random_state": seed}
            model = ExtraTreesRegressor(**params)
        elif task == "classification":
            params = {**CLASSIFICATION_PARAMS, **kwargs, "random_state": seed}
            if handle_imbalance:
                params["class_weight"] = "balanced"
            model = ExtraTreesClassifier(**params)
        else:
            msg = f"Unsupported task: {task!r}. Expected 'regression' or 'classification'."
            raise ValueError(msg)

        model.fit(X, y)
        return TrainedModel(
            model=model,
            algorithm_name=self.name,
            task=task,
            feature_names=feature_names,
        )

    def predict(self, model: TrainedModel, X: np.ndarray) -> np.ndarray:
        """Generate predictions using the trained Extra Trees model."""
        return np.asarray(model.model.predict(X))

    def predict_proba(
        self, model: TrainedModel, X: np.ndarray
    ) -> np.ndarray | None:
        """Return class probabilities for classification, None for regression."""
        if model.task == "regression":
            return None
        return np.asarray(model.model.predict_proba(X))

    def feature_importance(
        self, model: TrainedModel
    ) -> dict[str, float] | None:
        """Return normalized feature importance scores."""
        raw = model.model.feature_importances_
        total = raw.sum()
        if total == 0:
            return dict.fromkeys(model.feature_names, 0.0)
        normalized = raw / total
        return dict(zip(model.feature_names, normalized.tolist()))
