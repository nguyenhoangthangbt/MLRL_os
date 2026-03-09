"""Tests for mlrl_os.core.types."""

from __future__ import annotations

import numpy as np
import pytest

from mlrl_os.core.types import (
    AvailableTargets,
    ColumnInfo,
    CVStrategy,
    ExperimentStatus,
    FeatureMatrix,
    ObservationPoint,
    ProblemType,
    TargetInfo,
    TaskType,
)


# ---------------------------------------------------------------------------
# ProblemType
# ---------------------------------------------------------------------------


class TestProblemType:
    def test_time_series_value(self) -> None:
        assert ProblemType.TIME_SERIES.value == "time_series"

    def test_entity_classification_value(self) -> None:
        assert ProblemType.ENTITY_CLASSIFICATION.value == "entity_classification"

    def test_reinforcement_learning_value(self) -> None:
        assert ProblemType.REINFORCEMENT_LEARNING.value == "reinforcement_learning"

    def test_members_count(self) -> None:
        assert len(ProblemType) == 3


# ---------------------------------------------------------------------------
# TaskType
# ---------------------------------------------------------------------------


class TestTaskType:
    def test_regression_value(self) -> None:
        assert TaskType.REGRESSION.value == "regression"

    def test_classification_value(self) -> None:
        assert TaskType.CLASSIFICATION.value == "classification"

    def test_members_count(self) -> None:
        assert len(TaskType) == 2


# ---------------------------------------------------------------------------
# ExperimentStatus
# ---------------------------------------------------------------------------


class TestExperimentStatus:
    def test_pending_value(self) -> None:
        assert ExperimentStatus.PENDING.value == "pending"

    def test_running_value(self) -> None:
        assert ExperimentStatus.RUNNING.value == "running"

    def test_completed_value(self) -> None:
        assert ExperimentStatus.COMPLETED.value == "completed"

    def test_failed_value(self) -> None:
        assert ExperimentStatus.FAILED.value == "failed"

    def test_members_count(self) -> None:
        assert len(ExperimentStatus) == 4


# ---------------------------------------------------------------------------
# ObservationPoint
# ---------------------------------------------------------------------------


class TestObservationPoint:
    def test_all_steps_value(self) -> None:
        assert ObservationPoint.ALL_STEPS.value == "all_steps"

    def test_entry_only_value(self) -> None:
        assert ObservationPoint.ENTRY_ONLY.value == "entry_only"

    def test_midpoint_value(self) -> None:
        assert ObservationPoint.MIDPOINT.value == "midpoint"

    def test_members_count(self) -> None:
        assert len(ObservationPoint) == 3


# ---------------------------------------------------------------------------
# CVStrategy
# ---------------------------------------------------------------------------


class TestCVStrategy:
    def test_temporal_value(self) -> None:
        assert CVStrategy.TEMPORAL.value == "temporal"

    def test_stratified_kfold_value(self) -> None:
        assert CVStrategy.STRATIFIED_KFOLD.value == "stratified_kfold"

    def test_kfold_value(self) -> None:
        assert CVStrategy.KFOLD.value == "kfold"

    def test_members_count(self) -> None:
        assert len(CVStrategy) == 3


# ---------------------------------------------------------------------------
# ColumnInfo
# ---------------------------------------------------------------------------


class TestColumnInfo:
    def test_creation_with_defaults(self) -> None:
        info = ColumnInfo(name="col_a", dtype="Float64")
        assert info.name == "col_a"
        assert info.dtype == "Float64"
        assert info.null_count == 0
        assert info.is_numeric is False
        assert info.categories is None

    def test_creation_numeric(self) -> None:
        info = ColumnInfo(
            name="price",
            dtype="Float64",
            is_numeric=True,
            mean=10.5,
            std=2.1,
            min=1.0,
            max=20.0,
        )
        assert info.mean == 10.5
        assert info.min == 1.0

    def test_serialization_round_trip(self) -> None:
        info = ColumnInfo(
            name="status",
            dtype="Utf8",
            is_categorical=True,
            categories=["ok", "fail"],
            category_counts={"ok": 8, "fail": 2},
        )
        data = info.model_dump()
        restored = ColumnInfo.model_validate(data)
        assert restored == info


# ---------------------------------------------------------------------------
# TargetInfo
# ---------------------------------------------------------------------------


class TestTargetInfo:
    def test_creation_regression(self) -> None:
        target = TargetInfo(
            column="throughput",
            task_type=TaskType.REGRESSION,
            mean=5.5,
            std=1.2,
        )
        assert target.column == "throughput"
        assert target.task_type == TaskType.REGRESSION
        assert target.is_default is False

    def test_creation_classification(self) -> None:
        target = TargetInfo(
            column="status",
            task_type=TaskType.CLASSIFICATION,
            is_default=True,
            classes=["completed", "failed"],
            class_balance={"completed": 0.8, "failed": 0.2},
        )
        assert target.is_default is True
        assert target.classes == ["completed", "failed"]


# ---------------------------------------------------------------------------
# AvailableTargets
# ---------------------------------------------------------------------------


class TestAvailableTargets:
    def test_creation_empty(self) -> None:
        at = AvailableTargets(dataset_id="ds_001")
        assert at.dataset_id == "ds_001"
        assert at.time_series_targets == []
        assert at.entity_targets == []

    def test_creation_with_targets(self) -> None:
        ts_target = TargetInfo(column="throughput", task_type=TaskType.REGRESSION)
        ent_target = TargetInfo(column="status", task_type=TaskType.CLASSIFICATION)
        at = AvailableTargets(
            dataset_id="ds_002",
            time_series_targets=[ts_target],
            entity_targets=[ent_target],
        )
        assert len(at.time_series_targets) == 1
        assert len(at.entity_targets) == 1


# ---------------------------------------------------------------------------
# FeatureMatrix
# ---------------------------------------------------------------------------


class TestFeatureMatrix:
    def test_valid_construction(self, regression_feature_matrix: FeatureMatrix) -> None:
        fm = regression_feature_matrix
        assert fm.X.shape == (200, 5)
        assert fm.y.shape == (200,)
        assert fm.problem_type == ProblemType.TIME_SERIES
        assert fm.task_type == TaskType.REGRESSION
        assert fm.target_name == "target"
        assert len(fm.feature_names) == 5

    def test_feature_matrix_rejects_1d_input(self) -> None:
        with pytest.raises(ValueError, match="X must be 2D"):
            FeatureMatrix(
                X=np.array([1.0, 2.0, 3.0]),
                y=np.array([1.0, 2.0, 3.0]),
                feature_names=["f0"],
                problem_type=ProblemType.TIME_SERIES,
                target_name="t",
                task_type=TaskType.REGRESSION,
            )

    def test_feature_matrix_rejects_row_mismatch(self) -> None:
        with pytest.raises(ValueError, match="X rows.*!= y rows"):
            FeatureMatrix(
                X=np.zeros((10, 3)),
                y=np.zeros(5),
                feature_names=["a", "b", "c"],
                problem_type=ProblemType.TIME_SERIES,
                target_name="t",
                task_type=TaskType.REGRESSION,
            )

    def test_feature_matrix_rejects_column_name_mismatch(self) -> None:
        with pytest.raises(ValueError, match="X columns.*!= feature_names length"):
            FeatureMatrix(
                X=np.zeros((10, 3)),
                y=np.zeros(10),
                feature_names=["a", "b"],
                problem_type=ProblemType.TIME_SERIES,
                target_name="t",
                task_type=TaskType.REGRESSION,
            )

    def test_feature_matrix_rejects_temporal_index_mismatch(self) -> None:
        with pytest.raises(ValueError, match="temporal_index length.*!= X rows"):
            FeatureMatrix(
                X=np.zeros((10, 2)),
                y=np.zeros(10),
                feature_names=["a", "b"],
                problem_type=ProblemType.TIME_SERIES,
                target_name="t",
                task_type=TaskType.REGRESSION,
                temporal_index=np.arange(5, dtype=np.float64),
            )

    def test_feature_matrix_rejects_entity_ids_mismatch(self) -> None:
        with pytest.raises(ValueError, match="entity_ids length.*!= X rows"):
            FeatureMatrix(
                X=np.zeros((10, 2)),
                y=np.zeros(10),
                feature_names=["a", "b"],
                problem_type=ProblemType.ENTITY_CLASSIFICATION,
                target_name="t",
                task_type=TaskType.CLASSIFICATION,
                entity_ids=np.array(["e1", "e2", "e3"]),
            )

    def test_sample_count_property(self, regression_feature_matrix: FeatureMatrix) -> None:
        assert regression_feature_matrix.sample_count == 200

    def test_feature_count_property(self, regression_feature_matrix: FeatureMatrix) -> None:
        assert regression_feature_matrix.feature_count == 5

    def test_classification_feature_matrix(
        self, classification_feature_matrix: FeatureMatrix
    ) -> None:
        fm = classification_feature_matrix
        assert fm.task_type == TaskType.CLASSIFICATION
        assert fm.problem_type == ProblemType.ENTITY_CLASSIFICATION
        assert fm.class_names == ["0", "1"]
