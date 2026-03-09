"""Tests for mlrl_os.models.algorithms.linear."""

from __future__ import annotations

import numpy as np
import pytest

from mlrl_os.core.types import FeatureMatrix
from mlrl_os.models.algorithms.linear import LinearAlgorithm
from mlrl_os.models.algorithms.protocol import TrainedModel


@pytest.fixture()
def linear_algo() -> LinearAlgorithm:
    """Fresh LinearAlgorithm instance."""
    return LinearAlgorithm()


class TestLinearAlgorithmProperties:
    """Tests for LinearAlgorithm metadata properties."""

    def test_name_is_linear(self, linear_algo: LinearAlgorithm) -> None:
        assert linear_algo.name == "linear"

    def test_supports_regression(self, linear_algo: LinearAlgorithm) -> None:
        assert linear_algo.supports_regression is True

    def test_supports_classification(self, linear_algo: LinearAlgorithm) -> None:
        assert linear_algo.supports_classification is True


class TestLinearAlgorithmTrain:
    """Tests for LinearAlgorithm.train()."""

    def test_train_regression(
        self,
        linear_algo: LinearAlgorithm,
        regression_feature_matrix: FeatureMatrix,
    ) -> None:
        """train() on regression data returns a TrainedModel."""
        fm = regression_feature_matrix
        trained = linear_algo.train(
            X=fm.X,
            y=fm.y,
            task="regression",
            seed=42,
            feature_names=fm.feature_names,
        )

        assert isinstance(trained, TrainedModel)
        assert trained.algorithm_name == "linear"
        assert trained.task == "regression"
        assert trained.feature_names == fm.feature_names

    def test_train_classification(
        self,
        linear_algo: LinearAlgorithm,
        classification_feature_matrix: FeatureMatrix,
    ) -> None:
        """train() on classification data returns a TrainedModel."""
        fm = classification_feature_matrix
        trained = linear_algo.train(
            X=fm.X,
            y=fm.y,
            task="classification",
            seed=42,
            feature_names=fm.feature_names,
        )

        assert isinstance(trained, TrainedModel)
        assert trained.algorithm_name == "linear"
        assert trained.task == "classification"


class TestLinearAlgorithmPredict:
    """Tests for LinearAlgorithm.predict() and predict_proba()."""

    def test_predict_returns_correct_shape(
        self,
        linear_algo: LinearAlgorithm,
        regression_feature_matrix: FeatureMatrix,
    ) -> None:
        """predict() returns an array with shape (n_samples,)."""
        fm = regression_feature_matrix
        trained = linear_algo.train(X=fm.X, y=fm.y, task="regression", seed=42)
        preds = linear_algo.predict(trained, fm.X)

        assert isinstance(preds, np.ndarray)
        assert preds.shape == (fm.X.shape[0],)

    def test_predict_proba_none_for_regression(
        self,
        linear_algo: LinearAlgorithm,
        regression_feature_matrix: FeatureMatrix,
    ) -> None:
        """predict_proba() returns None for regression models."""
        fm = regression_feature_matrix
        trained = linear_algo.train(X=fm.X, y=fm.y, task="regression", seed=42)
        proba = linear_algo.predict_proba(trained, fm.X)

        assert proba is None

    def test_predict_proba_matrix_for_classification(
        self,
        linear_algo: LinearAlgorithm,
        classification_feature_matrix: FeatureMatrix,
    ) -> None:
        """predict_proba() returns a probability matrix for classification."""
        fm = classification_feature_matrix
        trained = linear_algo.train(X=fm.X, y=fm.y, task="classification", seed=42)
        proba = linear_algo.predict_proba(trained, fm.X)

        assert proba is not None
        assert isinstance(proba, np.ndarray)
        assert proba.shape == (fm.X.shape[0], 2)  # binary classification
        # Probabilities should sum to 1 per row.
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


class TestLinearAlgorithmFeatureImportance:
    """Tests for LinearAlgorithm.feature_importance()."""

    def test_feature_importance_returns_dict_with_feature_names(
        self,
        linear_algo: LinearAlgorithm,
        regression_feature_matrix: FeatureMatrix,
    ) -> None:
        """feature_importance() returns a dict keyed by feature names."""
        fm = regression_feature_matrix
        trained = linear_algo.train(
            X=fm.X,
            y=fm.y,
            task="regression",
            seed=42,
            feature_names=fm.feature_names,
        )
        importance = linear_algo.feature_importance(trained)

        assert importance is not None
        assert isinstance(importance, dict)
        assert set(importance.keys()) == set(fm.feature_names)
        # Normalized importances should sum to ~1.
        assert abs(sum(importance.values()) - 1.0) < 1e-6
