"""Tests for mlrl_os.data.discovery."""

from __future__ import annotations

import polars as pl
import pytest

from mlrl_os.core.dataset import DatasetMeta, compute_column_info
from mlrl_os.core.types import AvailableTargets, TargetInfo, TaskType
from mlrl_os.data.discovery import TargetDiscovery


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture()
def discovery() -> TargetDiscovery:
    return TargetDiscovery()


@pytest.fixture()
def snapshot_meta(snapshot_df: pl.DataFrame) -> DatasetMeta:
    cols = compute_column_info(snapshot_df)
    return DatasetMeta(
        id="ds_snap",
        name="Snapshots",
        content_hash="hash1",
        source_type="simos",
        source_path="/fake",
        snapshot_columns=cols,
        has_snapshots=True,
    )


@pytest.fixture()
def trajectory_meta(trajectory_df: pl.DataFrame) -> DatasetMeta:
    cols = compute_column_info(trajectory_df)
    return DatasetMeta(
        id="ds_traj",
        name="Trajectories",
        content_hash="hash2",
        source_type="simos",
        source_path="/fake",
        trajectory_columns=cols,
        has_trajectories=True,
    )


@pytest.fixture()
def empty_meta() -> DatasetMeta:
    return DatasetMeta(
        id="ds_empty",
        name="Empty",
        content_hash="hash0",
        source_type="csv",
        source_path="/fake",
    )


# =====================================================================
# Time-series target discovery
# =====================================================================


class TestTimeSeriesTargetDiscovery:
    """Tests for discovering time-series (snapshot) targets."""

    def test_discover_from_live_dataframe(
        self,
        discovery: TargetDiscovery,
        snapshot_meta: DatasetMeta,
        snapshot_df: pl.DataFrame,
    ) -> None:
        result = discovery.discover(snapshot_meta, snapshots_df=snapshot_df)

        assert isinstance(result, AvailableTargets)
        assert result.dataset_id == "ds_snap"
        assert len(result.time_series_targets) > 0

    def test_discover_from_column_info(
        self,
        discovery: TargetDiscovery,
        snapshot_meta: DatasetMeta,
    ) -> None:
        # No DataFrame passed; should use pre-computed column info
        result = discovery.discover(snapshot_meta)

        assert len(result.time_series_targets) > 0

    def test_all_targets_are_regression(
        self,
        discovery: TargetDiscovery,
        snapshot_meta: DatasetMeta,
        snapshot_df: pl.DataFrame,
    ) -> None:
        result = discovery.discover(snapshot_meta, snapshots_df=snapshot_df)

        for target in result.time_series_targets:
            assert target.task_type == TaskType.REGRESSION

    def test_structural_columns_excluded(
        self,
        discovery: TargetDiscovery,
        snapshot_meta: DatasetMeta,
        snapshot_df: pl.DataFrame,
    ) -> None:
        result = discovery.discover(snapshot_meta, snapshots_df=snapshot_df)
        target_names = [t.column for t in result.time_series_targets]

        assert "ts" not in target_names
        assert "bucket_idx" not in target_names

    def test_numeric_columns_included(
        self,
        discovery: TargetDiscovery,
        snapshot_meta: DatasetMeta,
        snapshot_df: pl.DataFrame,
    ) -> None:
        result = discovery.discover(snapshot_meta, snapshots_df=snapshot_df)
        target_names = [t.column for t in result.time_series_targets]

        # These are numeric columns from the snapshot fixture
        assert "wip" in target_names
        assert "throughput" in target_names
        assert "avg_wait" in target_names

    def test_default_target_is_avg_wait(
        self,
        discovery: TargetDiscovery,
        snapshot_meta: DatasetMeta,
        snapshot_df: pl.DataFrame,
    ) -> None:
        result = discovery.discover(snapshot_meta, snapshots_df=snapshot_df)

        defaults = [t for t in result.time_series_targets if t.is_default]
        assert len(defaults) == 1
        assert defaults[0].column == "avg_wait"

    def test_fallback_default_when_avg_wait_absent(
        self, discovery: TargetDiscovery
    ) -> None:
        df = pl.DataFrame({
            "ts": [1.0, 2.0, 3.0],
            "metric_a": [10, 20, 30],
            "metric_b": [1.1, 2.2, 3.3],
        })
        meta = DatasetMeta(
            id="ds_no_avg_wait",
            name="No avg_wait",
            content_hash="hash",
            source_type="csv",
            source_path="/fake",
        )
        result = discovery.discover(meta, snapshots_df=df)

        defaults = [t for t in result.time_series_targets if t.is_default]
        assert len(defaults) == 1
        # First numeric non-excluded column becomes default
        assert defaults[0].column == "metric_a"

    def test_target_info_has_stats(
        self,
        discovery: TargetDiscovery,
        snapshot_meta: DatasetMeta,
        snapshot_df: pl.DataFrame,
    ) -> None:
        result = discovery.discover(snapshot_meta, snapshots_df=snapshot_df)

        for target in result.time_series_targets:
            assert isinstance(target, TargetInfo)
            # Numeric targets should have stats
            assert target.mean is not None or target.unique_count is not None


# =====================================================================
# Entity target discovery
# =====================================================================


class TestEntityTargetDiscovery:
    """Tests for discovering entity (trajectory) targets."""

    def test_discover_from_live_dataframe(
        self,
        discovery: TargetDiscovery,
        trajectory_meta: DatasetMeta,
        trajectory_df: pl.DataFrame,
    ) -> None:
        result = discovery.discover(
            trajectory_meta, trajectories_df=trajectory_df
        )

        assert len(result.entity_targets) > 0

    def test_status_is_default_target(
        self,
        discovery: TargetDiscovery,
        trajectory_meta: DatasetMeta,
        trajectory_df: pl.DataFrame,
    ) -> None:
        result = discovery.discover(
            trajectory_meta, trajectories_df=trajectory_df
        )

        defaults = [t for t in result.entity_targets if t.is_default]
        assert len(defaults) == 1
        assert defaults[0].column == "status"
        assert defaults[0].task_type == TaskType.CLASSIFICATION

    def test_status_target_has_classes(
        self,
        discovery: TargetDiscovery,
        trajectory_meta: DatasetMeta,
        trajectory_df: pl.DataFrame,
    ) -> None:
        result = discovery.discover(
            trajectory_meta, trajectories_df=trajectory_df
        )

        status_target = next(
            t for t in result.entity_targets if t.column == "status"
        )
        assert status_target.classes is not None
        assert len(status_target.classes) > 0

    def test_status_target_has_class_balance(
        self,
        discovery: TargetDiscovery,
        trajectory_meta: DatasetMeta,
        trajectory_df: pl.DataFrame,
    ) -> None:
        result = discovery.discover(
            trajectory_meta, trajectories_df=trajectory_df
        )

        status_target = next(
            t for t in result.entity_targets if t.column == "status"
        )
        assert status_target.class_balance is not None
        # Balance values should sum to ~1.0
        total = sum(status_target.class_balance.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_sla_breach_derived_when_applicable(
        self,
        discovery: TargetDiscovery,
        trajectory_meta: DatasetMeta,
        trajectory_df: pl.DataFrame,
    ) -> None:
        # trajectory_df has s_cum_wait, so sla_breach should be derivable
        result = discovery.discover(
            trajectory_meta, trajectories_df=trajectory_df
        )

        target_names = [t.column for t in result.entity_targets]
        assert "sla_breach" in target_names

        sla_target = next(
            t for t in result.entity_targets if t.column == "sla_breach"
        )
        assert sla_target.task_type == TaskType.CLASSIFICATION
        assert sla_target.is_default is False

    def test_delay_severity_derived_when_total_time_present(
        self,
        discovery: TargetDiscovery,
        trajectory_meta: DatasetMeta,
        trajectory_df: pl.DataFrame,
    ) -> None:
        # trajectory_df has total_time column
        result = discovery.discover(
            trajectory_meta, trajectories_df=trajectory_df
        )

        target_names = [t.column for t in result.entity_targets]
        assert "delay_severity" in target_names

        severity_target = next(
            t for t in result.entity_targets if t.column == "delay_severity"
        )
        assert severity_target.classes == ["low", "medium", "high"]

    def test_discover_from_column_info(
        self,
        discovery: TargetDiscovery,
        trajectory_meta: DatasetMeta,
    ) -> None:
        # No DataFrame passed; should use pre-computed column info
        result = discovery.discover(trajectory_meta)

        # At minimum, status should be found from column info
        target_names = [t.column for t in result.entity_targets]
        assert "status" in target_names

    def test_no_entity_targets_without_status_column(
        self, discovery: TargetDiscovery
    ) -> None:
        """DataFrame without status/categorical columns yields no entity targets."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0],
        })
        meta = DatasetMeta(
            id="ds_no_status",
            name="No status",
            content_hash="hash",
            source_type="csv",
            source_path="/fake",
        )
        result = discovery.discover(meta, trajectories_df=df)
        assert len(result.entity_targets) == 0


# =====================================================================
# Combined discovery
# =====================================================================


class TestCombinedDiscovery:
    """Tests for datasets with both snapshots and trajectories."""

    def test_discovers_both_target_types(
        self,
        discovery: TargetDiscovery,
        snapshot_df: pl.DataFrame,
        trajectory_df: pl.DataFrame,
    ) -> None:
        meta = DatasetMeta(
            id="ds_combined",
            name="Combined",
            content_hash="hash",
            source_type="simos",
            source_path="/fake",
            has_snapshots=True,
            has_trajectories=True,
        )
        result = discovery.discover(
            meta,
            snapshots_df=snapshot_df,
            trajectories_df=trajectory_df,
        )

        assert len(result.time_series_targets) > 0
        assert len(result.entity_targets) > 0


# =====================================================================
# Edge cases
# =====================================================================


class TestEdgeCases:
    """Tests for edge cases in target discovery."""

    def test_empty_meta_no_targets(
        self, discovery: TargetDiscovery, empty_meta: DatasetMeta
    ) -> None:
        result = discovery.discover(empty_meta)
        assert len(result.time_series_targets) == 0
        assert len(result.entity_targets) == 0

    def test_result_has_dataset_id(
        self,
        discovery: TargetDiscovery,
        snapshot_meta: DatasetMeta,
        snapshot_df: pl.DataFrame,
    ) -> None:
        result = discovery.discover(snapshot_meta, snapshots_df=snapshot_df)
        assert result.dataset_id == snapshot_meta.id

    def test_result_is_pydantic_model(
        self,
        discovery: TargetDiscovery,
        snapshot_meta: DatasetMeta,
        snapshot_df: pl.DataFrame,
    ) -> None:
        result = discovery.discover(snapshot_meta, snapshots_df=snapshot_df)
        d = result.model_dump()
        assert "time_series_targets" in d
        assert "entity_targets" in d
