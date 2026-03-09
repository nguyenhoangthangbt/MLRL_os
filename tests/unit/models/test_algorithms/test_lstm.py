"""Tests for mlrl_os.models.algorithms.lstm."""

from __future__ import annotations

import pytest

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")

import numpy as np

from mlrl_os.models.algorithms.lstm import LSTMAlgorithm
from mlrl_os.models.algorithms.protocol import TrainedModel


@pytest.fixture()
def lstm_algo() -> LSTMAlgorithm:
    """Fresh LSTMAlgorithm instance."""
    return LSTMAlgorithm()


class TestLSTMAlgorithmProperties:
    """Tests for LSTMAlgorithm metadata properties."""

    def test_name_is_lstm(self, lstm_algo: LSTMAlgorithm) -> None:
        assert lstm_algo.name == "lstm"

    def test_supports_regression(self, lstm_algo: LSTMAlgorithm) -> None:
        assert lstm_algo.supports_regression is True

    def test_supports_classification(self, lstm_algo: LSTMAlgorithm) -> None:
        assert lstm_algo.supports_classification is True


class TestLSTMAlgorithmTrain:
    """Tests for LSTMAlgorithm.train()."""

    def test_train_regression(self, lstm_algo: LSTMAlgorithm) -> None:
        """train() on regression data returns a TrainedModel."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10).astype(np.float32)
        y = rng.randn(100).astype(np.float32)

        trained = lstm_algo.train(
            X=X,
            y=y,
            task="regression",
            seed=42,
            epochs=5,
            feature_names=[f"f{i}" for i in range(10)],
        )

        assert isinstance(trained, TrainedModel)
        assert trained.algorithm_name == "lstm"
        assert trained.task == "regression"
        assert len(trained.feature_names) == 10

    def test_train_classification(self, lstm_algo: LSTMAlgorithm) -> None:
        """train() on classification data returns a TrainedModel."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10).astype(np.float32)
        y = rng.randint(0, 3, 100)

        trained = lstm_algo.train(
            X=X,
            y=y,
            task="classification",
            seed=42,
            epochs=5,
        )

        assert isinstance(trained, TrainedModel)
        assert trained.algorithm_name == "lstm"
        assert trained.task == "classification"

    def test_train_invalid_task(self, lstm_algo: LSTMAlgorithm) -> None:
        """train() raises ValueError for unsupported task."""
        X = np.random.randn(50, 5).astype(np.float32)
        y = np.random.randn(50).astype(np.float32)

        with pytest.raises(ValueError, match="Unsupported task"):
            lstm_algo.train(X=X, y=y, task="unknown", seed=42, epochs=2)


class TestLSTMAlgorithmPredict:
    """Tests for LSTMAlgorithm.predict() and predict_proba()."""

    def test_predict_regression_shape(self, lstm_algo: LSTMAlgorithm) -> None:
        """predict() returns an array with shape (n_samples,) for regression."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10).astype(np.float32)
        y = rng.randn(100).astype(np.float32)

        trained = lstm_algo.train(X=X, y=y, task="regression", seed=42, epochs=5)
        preds = lstm_algo.predict(trained, X)

        assert isinstance(preds, np.ndarray)
        assert preds.shape == (100,)

    def test_predict_classification_shape(self, lstm_algo: LSTMAlgorithm) -> None:
        """predict() returns integer class labels for classification."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10).astype(np.float32)
        y = rng.randint(0, 3, 100)

        trained = lstm_algo.train(X=X, y=y, task="classification", seed=42, epochs=5)
        preds = lstm_algo.predict(trained, X)

        assert preds.shape == (100,)
        assert all(p in (0, 1, 2) for p in preds)

    def test_predict_proba_none_for_regression(
        self, lstm_algo: LSTMAlgorithm
    ) -> None:
        """predict_proba() returns None for regression models."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10).astype(np.float32)
        y = rng.randn(100).astype(np.float32)

        trained = lstm_algo.train(X=X, y=y, task="regression", seed=42, epochs=5)
        proba = lstm_algo.predict_proba(trained, X)

        assert proba is None

    def test_predict_proba_matrix_for_classification(
        self, lstm_algo: LSTMAlgorithm
    ) -> None:
        """predict_proba() returns a probability matrix for classification."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10).astype(np.float32)
        y = rng.randint(0, 3, 100)

        trained = lstm_algo.train(X=X, y=y, task="classification", seed=42, epochs=5)
        proba = lstm_algo.predict_proba(trained, X)

        assert proba is not None
        assert isinstance(proba, np.ndarray)
        assert proba.shape == (100, 3)
        # Probabilities should sum to 1 per row.
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


class TestLSTMAlgorithmFeatureImportance:
    """Tests for LSTMAlgorithm.feature_importance()."""

    def test_feature_importance_returns_none(
        self, lstm_algo: LSTMAlgorithm
    ) -> None:
        """LSTM does not have built-in feature importance."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 5).astype(np.float32)
        y = rng.randn(50).astype(np.float32)

        trained = lstm_algo.train(X=X, y=y, task="regression", seed=42, epochs=3)
        fi = lstm_algo.feature_importance(trained)

        assert fi is None


class TestLSTMInRegistry:
    """Tests for LSTM presence in the algorithm registry."""

    def test_lstm_in_default_registry(self) -> None:
        """default_registry() should include 'lstm'."""
        from mlrl_os.models.algorithms.registry import default_registry

        reg = default_registry()
        assert reg.has("lstm")

    def test_lstm_in_known_algorithms(self) -> None:
        """KNOWN_ALGORITHMS in registry module should include 'lstm'."""
        from mlrl_os.models.algorithms.registry import KNOWN_ALGORITHMS

        assert "lstm" in KNOWN_ALGORITHMS

    def test_lstm_in_gate_known_algorithms(self) -> None:
        """KNOWN_ALGORITHMS in validation gate should include 'lstm'."""
        from mlrl_os.validation.gate import KNOWN_ALGORITHMS as GATE_KNOWN

        assert "lstm" in GATE_KNOWN
