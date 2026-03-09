"""LightGBM algorithm wrapper for ML/RL OS."""

from __future__ import annotations

from typing import Any

import numpy as np

from mlrl_os.models.algorithms.protocol import TrainedModel

REGRESSION_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "n_jobs": -1,
}

CLASSIFICATION_PARAMS: dict[str, Any] = {
    "objective": "multiclass",
    "metric": "multi_logloss",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "is_unbalance": True,
    "verbose": -1,
    "n_jobs": -1,
}


class LightGBMAlgorithm:
    """LightGBM gradient boosting algorithm.

    Supports both regression (LGBMRegressor) and classification
    (LGBMClassifier). LightGBM is imported lazily inside train()
    to avoid import errors when the package is not installed.
    """

    @property
    def name(self) -> str:
        return "lightgbm"

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
        """Train a LightGBM model.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target vector (n_samples,).
            task: "regression" or "classification".
            seed: Random seed for reproducibility.
            **kwargs: Parameter overrides merged on top of defaults.

        Returns:
            TrainedModel wrapping the fitted LGBMRegressor/LGBMClassifier.

        Raises:
            ImportError: If lightgbm is not installed.
            ValueError: If the task is not supported.
        """
        import lightgbm as lgb  # noqa: F811 — lazy import

        feature_names = kwargs.pop("feature_names", [f"f{i}" for i in range(X.shape[1])])
        handle_imbalance = kwargs.pop("handle_imbalance", False)

        if task == "regression":
            params = {**REGRESSION_PARAMS, **kwargs, "random_state": seed}
            model = lgb.LGBMRegressor(**params)
        elif task == "classification":
            n_classes = len(np.unique(y))
            params = {**CLASSIFICATION_PARAMS, **kwargs, "random_state": seed}
            if handle_imbalance:
                params["is_unbalance"] = True
            # Use binary objective when there are exactly 2 classes.
            if n_classes == 2:
                params["objective"] = "binary"
                params["metric"] = "binary_logloss"
            model = lgb.LGBMClassifier(**params)
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
        """Generate predictions using the trained LightGBM model."""
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
