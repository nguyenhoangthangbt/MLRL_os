"""Tests for mlrl_os.models.engine — ModelEngine training and tuning."""

from __future__ import annotations

import numpy as np
import pytest

from mlrl_os.core.types import CVStrategy, FeatureMatrix, ProblemType, TaskType
from mlrl_os.models.engine import ModelEngine, TrainingResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _classification_matrix(n: int = 200, p: int = 5) -> FeatureMatrix:
    rng = np.random.RandomState(42)
    X = rng.randn(n, p)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return FeatureMatrix(
        X=X,
        y=y,
        feature_names=[f"f{i}" for i in range(p)],
        problem_type=ProblemType.ENTITY_CLASSIFICATION,
        target_name="label",
        task_type=TaskType.CLASSIFICATION,
        class_names=["0", "1"],
    )


def _regression_matrix(n: int = 200, p: int = 5) -> FeatureMatrix:
    rng = np.random.RandomState(42)
    X = rng.randn(n, p)
    y = X[:, 0] * 2 + X[:, 1] + rng.randn(n) * 0.1
    return FeatureMatrix(
        X=X,
        y=y,
        feature_names=[f"f{i}" for i in range(p)],
        problem_type=ProblemType.TIME_SERIES,
        target_name="value",
        task_type=TaskType.REGRESSION,
    )


# ---------------------------------------------------------------------------
# Basic train_and_evaluate (no tuning)
# ---------------------------------------------------------------------------


class TestTrainAndEvaluate:
    def test_classification_with_linear(self) -> None:
        fm = _classification_matrix()
        engine = ModelEngine()
        result = engine.train_and_evaluate(
            feature_matrix=fm,
            algorithm_names=["linear"],
            metric_names=["accuracy", "f1_weighted"],
            cv_strategy=CVStrategy.STRATIFIED_KFOLD,
            cv_folds=3,
            seed=42,
        )
        assert isinstance(result, TrainingResult)
        assert result.best_algorithm == "linear"
        assert "accuracy" in result.best_metrics
        assert result.best_metrics["accuracy"] > 0.5
        assert result.y_pred.shape == fm.y.shape
        assert result.y_proba is not None
        assert len(result.all_scores) == 1

    def test_regression_with_linear(self) -> None:
        fm = _regression_matrix()
        engine = ModelEngine()
        result = engine.train_and_evaluate(
            feature_matrix=fm,
            algorithm_names=["linear"],
            metric_names=["rmse", "r2"],
            cv_strategy=CVStrategy.KFOLD,
            cv_folds=3,
            seed=42,
        )
        assert result.best_algorithm == "linear"
        assert result.best_metrics["rmse"] >= 0
        assert result.y_proba is None

    def test_multiple_algorithms(self) -> None:
        fm = _classification_matrix()
        engine = ModelEngine()
        result = engine.train_and_evaluate(
            feature_matrix=fm,
            algorithm_names=["linear", "random_forest"],
            metric_names=["accuracy"],
            cv_strategy=CVStrategy.STRATIFIED_KFOLD,
            cv_folds=3,
            seed=42,
        )
        assert len(result.all_scores) == 2
        # Best should be rank 1
        assert result.all_scores[0].rank == 1

    def test_all_algorithms_fail_raises_runtime_error(self) -> None:
        fm = _classification_matrix()
        engine = ModelEngine()
        with pytest.raises(RuntimeError, match="All algorithms failed"):
            engine.train_and_evaluate(
                feature_matrix=fm,
                algorithm_names=["nonexistent_algo"],
                metric_names=["accuracy"],
                cv_strategy=CVStrategy.STRATIFIED_KFOLD,
                cv_folds=3,
                seed=42,
            )

    def test_feature_importance_populated(self) -> None:
        fm = _regression_matrix()
        engine = ModelEngine()
        result = engine.train_and_evaluate(
            feature_matrix=fm,
            algorithm_names=["random_forest"],
            metric_names=["rmse"],
            cv_strategy=CVStrategy.KFOLD,
            cv_folds=3,
            seed=42,
        )
        assert len(result.feature_importance) > 0
        assert result.feature_importance[0].rank == 1

    def test_temporal_cv_for_regression(self) -> None:
        fm = _regression_matrix()
        engine = ModelEngine()
        result = engine.train_and_evaluate(
            feature_matrix=fm,
            algorithm_names=["linear"],
            metric_names=["rmse"],
            cv_strategy=CVStrategy.TEMPORAL,
            cv_folds=3,
            seed=42,
        )
        assert result.best_algorithm == "linear"

    def test_handle_imbalance_flag(self) -> None:
        fm = _classification_matrix()
        engine = ModelEngine()
        result = engine.train_and_evaluate(
            feature_matrix=fm,
            algorithm_names=["linear"],
            metric_names=["accuracy"],
            cv_strategy=CVStrategy.STRATIFIED_KFOLD,
            cv_folds=3,
            seed=42,
            handle_imbalance=True,
        )
        assert result.best_algorithm == "linear"


# ---------------------------------------------------------------------------
# Hyperparameter tuning (with Optuna)
# ---------------------------------------------------------------------------


class TestHyperparameterTuning:
    """Test ModelEngine with hyperparameter_tuning=True.

    Uses linear algorithm (fast) and only 3 trials to keep tests quick.
    """

    def test_tuning_classification(self) -> None:
        fm = _classification_matrix()
        engine = ModelEngine()
        result = engine.train_and_evaluate(
            feature_matrix=fm,
            algorithm_names=["linear"],
            metric_names=["accuracy"],
            cv_strategy=CVStrategy.STRATIFIED_KFOLD,
            cv_folds=3,
            seed=42,
            hyperparameter_tuning=True,
        )
        assert result.best_algorithm == "linear"
        assert result.best_metrics["accuracy"] > 0.5

    def test_tuning_regression(self) -> None:
        fm = _regression_matrix()
        engine = ModelEngine()
        result = engine.train_and_evaluate(
            feature_matrix=fm,
            algorithm_names=["linear"],
            metric_names=["rmse"],
            cv_strategy=CVStrategy.KFOLD,
            cv_folds=3,
            seed=42,
            hyperparameter_tuning=True,
        )
        assert result.best_algorithm == "linear"
        assert result.best_metrics["rmse"] >= 0

    def test_tuning_with_random_forest(self) -> None:
        fm = _classification_matrix(n=100, p=3)
        engine = ModelEngine()
        result = engine.train_and_evaluate(
            feature_matrix=fm,
            algorithm_names=["random_forest"],
            metric_names=["accuracy"],
            cv_strategy=CVStrategy.STRATIFIED_KFOLD,
            cv_folds=3,
            seed=42,
            hyperparameter_tuning=True,
        )
        assert result.best_algorithm == "random_forest"
        assert len(result.feature_importance) > 0

    def test_tuning_deterministic(self) -> None:
        """Same seed + data → same result."""
        fm = _classification_matrix(n=80, p=3)
        engine = ModelEngine()
        kwargs = dict(
            feature_matrix=fm,
            algorithm_names=["linear"],
            metric_names=["accuracy"],
            cv_strategy=CVStrategy.STRATIFIED_KFOLD,
            cv_folds=3,
            seed=42,
            hyperparameter_tuning=True,
        )
        r1 = engine.train_and_evaluate(**kwargs)
        r2 = engine.train_and_evaluate(**kwargs)
        assert r1.best_metrics["accuracy"] == r2.best_metrics["accuracy"]
