"""Tests for mlrl_os.core.experiment."""

from __future__ import annotations

from mlrl_os.core.experiment import (
    AlgorithmScore,
    ExperimentResult,
    FeatureImportanceEntry,
)
from mlrl_os.core.types import ExperimentStatus, ProblemType


# ---------------------------------------------------------------------------
# AlgorithmScore
# ---------------------------------------------------------------------------


class TestAlgorithmScore:
    def test_creation(self) -> None:
        score = AlgorithmScore(
            algorithm="lightgbm",
            metrics={"rmse": 0.25, "r2": 0.91},
            metrics_std={"rmse": 0.03, "r2": 0.02},
            rank=1,
        )
        assert score.algorithm == "lightgbm"
        assert score.metrics["rmse"] == 0.25
        assert score.rank == 1

    def test_defaults(self) -> None:
        score = AlgorithmScore(algorithm="xgboost")
        assert score.metrics == {}
        assert score.metrics_std == {}
        assert score.rank == 0


# ---------------------------------------------------------------------------
# FeatureImportanceEntry
# ---------------------------------------------------------------------------


class TestFeatureImportanceEntry:
    def test_creation(self) -> None:
        entry = FeatureImportanceEntry(
            feature="throughput_lag_1",
            importance=0.42,
            rank=1,
        )
        assert entry.feature == "throughput_lag_1"
        assert entry.importance == 0.42
        assert entry.rank == 1

    def test_default_rank(self) -> None:
        entry = FeatureImportanceEntry(feature="wip", importance=0.05)
        assert entry.rank == 0


# ---------------------------------------------------------------------------
# ExperimentResult
# ---------------------------------------------------------------------------


class TestExperimentResult:
    def test_creation_all_fields(self) -> None:
        scores = [
            AlgorithmScore(algorithm="lgbm", metrics={"rmse": 0.2}, rank=1),
            AlgorithmScore(algorithm="xgb", metrics={"rmse": 0.3}, rank=2),
        ]
        importance = [
            FeatureImportanceEntry(feature="f0", importance=0.6, rank=1),
        ]
        result = ExperimentResult(
            experiment_id="exp_001",
            name="Throughput forecast",
            status=ExperimentStatus.COMPLETED,
            experiment_type=ProblemType.TIME_SERIES,
            created_at="2025-06-01T10:00:00Z",
            completed_at="2025-06-01T10:05:00Z",
            duration_seconds=300.0,
            best_algorithm="lgbm",
            metrics={"rmse": 0.2, "r2": 0.95},
            all_algorithm_scores=scores,
            feature_importance=importance,
            model_id="model_001",
            sample_count=500,
            feature_count=10,
            resolved_config={"seed": 42},
        )
        assert result.experiment_id == "exp_001"
        assert result.status == ExperimentStatus.COMPLETED
        assert result.best_algorithm == "lgbm"
        assert len(result.all_algorithm_scores) == 2
        assert result.duration_seconds == 300.0

    def test_creation_minimal_defaults(self) -> None:
        result = ExperimentResult(
            experiment_id="exp_002",
            name="Quick test",
            status=ExperimentStatus.PENDING,
            experiment_type=ProblemType.ENTITY_CLASSIFICATION,
            created_at="2025-06-01T10:00:00Z",
        )
        assert result.completed_at is None
        assert result.duration_seconds is None
        assert result.best_algorithm is None
        assert result.metrics is None
        assert result.all_algorithm_scores is None
        assert result.feature_importance is None
        assert result.model_id is None
        assert result.sample_count is None
        assert result.feature_count is None
        assert result.resolved_config is None
        assert result.error_message is None

    def test_serialization_round_trip(self) -> None:
        original = ExperimentResult(
            experiment_id="exp_003",
            name="Round-trip test",
            status=ExperimentStatus.FAILED,
            experiment_type=ProblemType.TIME_SERIES,
            created_at="2025-06-01T12:00:00Z",
            error_message="Validation failed",
            best_algorithm="xgboost",
            metrics={"rmse": 1.5},
            all_algorithm_scores=[
                AlgorithmScore(algorithm="xgboost", metrics={"rmse": 1.5}, rank=1),
            ],
            feature_importance=[
                FeatureImportanceEntry(feature="lag_1", importance=0.8, rank=1),
            ],
        )
        data = original.model_dump()
        restored = ExperimentResult.model_validate(data)
        assert restored.experiment_id == original.experiment_id
        assert restored.status == original.status
        assert restored.error_message == original.error_message
        assert restored.best_algorithm == original.best_algorithm
        assert restored.metrics == original.metrics
        assert len(restored.all_algorithm_scores) == 1
        assert restored.all_algorithm_scores[0].algorithm == "xgboost"
        assert len(restored.feature_importance) == 1
        assert restored.feature_importance[0].feature == "lag_1"
