"""Linear algorithm wrapper for ML/RL OS."""

from __future__ import annotations

from typing import Any

import numpy as np

from mlrl_os.models.algorithms.protocol import TrainedModel

REGRESSION_PARAMS: dict[str, Any] = {
    "alpha": 1.0,
}

CLASSIFICATION_PARAMS: dict[str, Any] = {
    "max_iter": 1000,
    "class_weight": "balanced",
}


class LinearAlgorithm:
    """Linear models: Ridge regression and Logistic Regression.

    Uses scikit-learn's Ridge for regression tasks and
    LogisticRegression for classification tasks.
    """

    @property
    def name(self) -> str:
        return "linear"

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
        """Train a linear model.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target vector (n_samples,).
            task: "regression" or "classification".
            seed: Random seed for reproducibility.
            **kwargs: Parameter overrides merged on top of defaults.

        Returns:
            TrainedModel wrapping the fitted Ridge or LogisticRegression.

        Raises:
            ValueError: If the task is not supported.
        """
        from sklearn.linear_model import LogisticRegression, Ridge

        feature_names = kwargs.pop("feature_names", [f"f{i}" for i in range(X.shape[1])])
        handle_imbalance = kwargs.pop("handle_imbalance", False)

        if task == "regression":
            params = {**REGRESSION_PARAMS, **kwargs, "random_state": seed}
            model = Ridge(**params)
        elif task == "classification":
            params = {**CLASSIFICATION_PARAMS, **kwargs, "random_state": seed}
            if handle_imbalance:
                params["class_weight"] = "balanced"
            model = LogisticRegression(**params)
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
        """Generate predictions using the trained linear model."""
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
        """Return normalized absolute coefficient values as importance.

        For multi-class classification, coefficients are averaged across
        classes before taking the absolute value.
        """
        coef = model.model.coef_
        # Multi-class LogisticRegression has shape (n_classes, n_features).
        if coef.ndim == 2:
            coef = np.mean(np.abs(coef), axis=0)
        else:
            coef = np.abs(coef)

        total = coef.sum()
        if total == 0:
            return dict.fromkeys(model.feature_names, 0.0)
        normalized = coef / total
        return dict(zip(model.feature_names, normalized.tolist()))
