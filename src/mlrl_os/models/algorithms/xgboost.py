"""XGBoost algorithm wrapper for ML/RL OS."""

from __future__ import annotations

from typing import Any

import numpy as np

from mlrl_os.models.algorithms.protocol import TrainedModel

REGRESSION_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbosity": 0,
    "n_jobs": -1,
}

CLASSIFICATION_PARAMS: dict[str, Any] = {
    "objective": "multi:softprob",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbosity": 0,
    "n_jobs": -1,
}


class XGBoostAlgorithm:
    """XGBoost gradient boosting algorithm.

    Supports both regression (XGBRegressor) and classification
    (XGBClassifier). XGBoost is imported lazily inside train()
    to avoid import errors when the package is not installed.
    """

    @property
    def name(self) -> str:
        return "xgboost"

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
        """Train an XGBoost model.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target vector (n_samples,).
            task: "regression" or "classification".
            seed: Random seed for reproducibility.
            **kwargs: Parameter overrides merged on top of defaults.

        Returns:
            TrainedModel wrapping the fitted XGBRegressor/XGBClassifier.

        Raises:
            ImportError: If xgboost is not installed.
            ValueError: If the task is not supported.
        """
        import xgboost as xgb  # noqa: F811 — lazy import

        feature_names = kwargs.pop("feature_names", [f"f{i}" for i in range(X.shape[1])])
        handle_imbalance = kwargs.pop("handle_imbalance", False)

        if task == "regression":
            params = {**REGRESSION_PARAMS, **kwargs, "random_state": seed}
            model = xgb.XGBRegressor(**params)
        elif task == "classification":
            n_classes = len(np.unique(y))
            params = {**CLASSIFICATION_PARAMS, **kwargs, "random_state": seed}
            if handle_imbalance:
                # Compute scale_pos_weight for binary classification
                n_pos = np.sum(y == 1)
                n_neg = np.sum(y == 0)
                if n_pos > 0:
                    params["scale_pos_weight"] = float(n_neg / n_pos)
            # Use binary logistic for 2-class problems.
            if n_classes == 2:
                params["objective"] = "binary:logistic"
            model = xgb.XGBClassifier(**params)
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
        """Generate predictions using the trained XGBoost model."""
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
