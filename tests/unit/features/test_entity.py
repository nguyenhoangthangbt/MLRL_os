"""Tests for mlrl_os.features.entity module."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from mlrl_os.core.types import ObservationPoint, ProblemType, TaskType
from mlrl_os.features.entity import EntityFeatureEngine


class TestEntityFeatureEngine:
    """Tests for EntityFeatureEngine.build_features."""

    @pytest.fixture()
    def engine(self) -> EntityFeatureEngine:
        return EntityFeatureEngine()

    def test_returns_entity_classification_problem_type(
        self, engine: EntityFeatureEngine, trajectory_df: pl.DataFrame
    ) -> None:
        fm = engine.build_features(trajectory_df, target="status")
        assert fm.problem_type == ProblemType.ENTITY_CLASSIFICATION

    def test_string_target_produces_classification_with_class_names(
        self, engine: EntityFeatureEngine, trajectory_df: pl.DataFrame
    ) -> None:
        fm = engine.build_features(trajectory_df, target="status")
        assert fm.task_type == TaskType.CLASSIFICATION
        assert fm.class_names is not None
        assert len(fm.class_names) > 0
        # Encoded target values should be integer class indices
        assert fm.y.dtype == np.int64

    def test_derived_features_present(
        self, engine: EntityFeatureEngine, trajectory_df: pl.DataFrame
    ) -> None:
        fm = engine.build_features(
            trajectory_df,
            target="status",
            add_progress_ratio=True,
            add_wait_trend=True,
        )
        names = fm.feature_names
        assert "d_progress_ratio" in names, "Expected d_progress_ratio derived feature"
        assert "d_wait_trend" in names, "Expected d_wait_trend derived feature"
        assert "d_cost_rate" in names, "Expected d_cost_rate derived feature"

    def test_all_steps_uses_all_rows(
        self, engine: EntityFeatureEngine, trajectory_df: pl.DataFrame
    ) -> None:
        fm = engine.build_features(
            trajectory_df,
            target="status",
            observation_point=ObservationPoint.ALL_STEPS,
        )
        # ALL_STEPS should keep all rows (minus any dropped nulls)
        assert fm.X.shape[0] > 0
        # Should be close to the full trajectory length
        assert fm.X.shape[0] >= len(trajectory_df) * 0.9

    def test_entry_only_uses_step_zero(
        self, engine: EntityFeatureEngine, trajectory_df: pl.DataFrame
    ) -> None:
        fm_all = engine.build_features(
            trajectory_df,
            target="status",
            observation_point=ObservationPoint.ALL_STEPS,
        )
        fm_entry = engine.build_features(
            trajectory_df,
            target="status",
            observation_point=ObservationPoint.ENTRY_ONLY,
        )
        # ENTRY_ONLY should have fewer samples than ALL_STEPS
        assert fm_entry.X.shape[0] < fm_all.X.shape[0]
        # Number of entry-only samples should match entities with step==0
        expected = len(trajectory_df.filter(pl.col("step") == 0))
        assert fm_entry.X.shape[0] == expected

    def test_raises_if_target_not_found(
        self, engine: EntityFeatureEngine, trajectory_df: pl.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="Target column.*not found"):
            engine.build_features(trajectory_df, target="nonexistent_col")

    def test_raises_if_no_numeric_features(
        self, engine: EntityFeatureEngine
    ) -> None:
        df = pl.DataFrame({
            "eid": ["e1", "e2", "e3"],
            "step": [0, 0, 0],
            "status": ["ok", "ok", "fail"],
            "node": ["a", "b", "c"],
        })
        with pytest.raises(ValueError, match="No valid numeric feature columns"):
            engine.build_features(df, target="status")

    def test_auto_selected_columns_include_state_groups(
        self, engine: EntityFeatureEngine, trajectory_df: pl.DataFrame
    ) -> None:
        fm = engine.build_features(trajectory_df, target="status")
        names = set(fm.feature_names)
        # Should include entity state columns
        assert "s_priority" in names or "s_elapsed" in names
        # Should include node state columns
        assert "s_node_util" in names or "s_node_avg_queue" in names
        # Should include system state columns
        assert "s_sys_util" in names or "s_sys_throughput" in names

    def test_exclude_columns_works(
        self, engine: EntityFeatureEngine, trajectory_df: pl.DataFrame
    ) -> None:
        excluded = ["s_priority", "s_elapsed"]
        fm = engine.build_features(
            trajectory_df,
            target="status",
            exclude_columns=excluded,
        )
        for col in excluded:
            assert col not in fm.feature_names
