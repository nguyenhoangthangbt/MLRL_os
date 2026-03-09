"""Tests for mlrl_os.data.simos_loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from mlrl_os.core.dataset import RawDataset
from mlrl_os.data.simos_loader import SimosLoader, SimosSchemaAdapter


# =====================================================================
# SimosSchemaAdapter tests
# =====================================================================


class TestSimosSchemaAdapterMapSnapshotField:
    """Tests for SimosSchemaAdapter.map_snapshot_field."""

    def setup_method(self) -> None:
        self.adapter = SimosSchemaAdapter()

    @pytest.mark.parametrize(
        ("simos_name", "expected"),
        [
            ("timestamp", "ts"),
            ("bucket_index", "bucket_idx"),
            ("sys_wip", "wip"),
            ("sys_total_in_queue", "in_queue"),
            ("sys_total_busy", "busy"),
            ("sys_throughput_per_hour", "throughput"),
            ("sys_avg_wait_time", "avg_wait"),
            ("sys_avg_processing_time", "avg_processing"),
            ("sys_arrival_rate_per_hour", "arrival_rate"),
            ("sys_wip_ratio", "wip_ratio"),
            ("sys_sla_compliance", "sla_compliance"),
        ],
    )
    def test_maps_known_system_fields(
        self, simos_name: str, expected: str
    ) -> None:
        assert self.adapter.map_snapshot_field(simos_name) == expected

    @pytest.mark.parametrize(
        ("simos_name", "expected"),
        [
            ("node_intake_queue", "n_intake_queue"),
            ("node_intake_utilization", "n_intake_util"),
            ("node_assembly_busy", "n_assembly_busy"),
            ("node_qc_throughput_per_hour", "n_qc_throughput"),
            ("node_intake_avg_wait_time", "n_intake_avg_wait"),
            ("node_assembly_cumulative_processed", "n_assembly_cum_processed"),
        ],
    )
    def test_maps_node_fields(self, simos_name: str, expected: str) -> None:
        assert self.adapter.map_snapshot_field(simos_name) == expected

    @pytest.mark.parametrize(
        ("simos_name", "expected"),
        [
            ("resource_workers_utilization", "r_workers_util"),
            ("resource_workers_capacity", "r_workers_capacity"),
        ],
    )
    def test_maps_resource_fields(
        self, simos_name: str, expected: str
    ) -> None:
        assert self.adapter.map_snapshot_field(simos_name) == expected

    def test_returns_none_for_unknown_field(self) -> None:
        assert self.adapter.map_snapshot_field("totally_unknown_field") is None

    def test_returns_none_for_unknown_node_metric(self) -> None:
        assert self.adapter.map_snapshot_field("node_intake_bogus") is None

    def test_returns_none_for_unknown_resource_metric(self) -> None:
        assert (
            self.adapter.map_snapshot_field("resource_workers_bogus") is None
        )


class TestSimosSchemaAdapterMapTrajectoryTopLevel:
    """Tests for SimosSchemaAdapter.map_trajectory_top_level."""

    def setup_method(self) -> None:
        self.adapter = SimosSchemaAdapter()

    @pytest.mark.parametrize(
        ("simos_name", "expected"),
        [
            ("entity_id", "eid"),
            ("entity_type", "etype"),
            ("entity_priority", "epriority"),
            ("step_index", "step"),
            ("node_name", "node"),
            ("sim_time_enter", "t_enter"),
            ("sim_time_complete", "t_complete"),
            ("wait_time", "wait"),
            ("processing_time", "processing"),
            ("episode_done", "done"),
            ("episode_status", "status"),
            ("episode_total_time", "total_time"),
        ],
    )
    def test_maps_known_fields(
        self, simos_name: str, expected: str
    ) -> None:
        assert self.adapter.map_trajectory_top_level(simos_name) == expected

    def test_returns_none_for_unknown(self) -> None:
        assert self.adapter.map_trajectory_top_level("unknown_field") is None


class TestSimosSchemaAdapterMapStateField:
    """Tests for SimosSchemaAdapter.map_state_field."""

    def setup_method(self) -> None:
        self.adapter = SimosSchemaAdapter()

    @pytest.mark.parametrize(
        ("simos_name", "expected"),
        [
            ("entity_priority", "s_priority"),
            ("entity_elapsed_time", "s_elapsed"),
            ("entity_cumulative_wait", "s_cum_wait"),
            ("entity_wait_ratio", "s_wait_ratio"),
            ("entity_steps_completed", "s_steps_done"),
            ("node_utilization", "s_node_util"),
            ("node_avg_queue_depth", "s_node_avg_queue"),
            ("sys_utilization", "s_sys_util"),
            ("sys_throughput_per_hour", "s_sys_throughput"),
            ("sys_bottleneck_utilization", "s_sys_bottleneck_util"),
        ],
    )
    def test_maps_static_state_fields(
        self, simos_name: str, expected: str
    ) -> None:
        assert self.adapter.map_state_field(simos_name) == expected

    @pytest.mark.parametrize(
        ("simos_name", "expected"),
        [
            ("resource_workers_utilization", "s_r_workers_util"),
            ("resource_workers_capacity", "s_r_workers_capacity"),
        ],
    )
    def test_maps_dynamic_resource_fields(
        self, simos_name: str, expected: str
    ) -> None:
        assert self.adapter.map_state_field(simos_name) == expected

    @pytest.mark.parametrize(
        ("simos_name", "expected"),
        [
            ("attr_color", "s_attr_color"),
            ("attr_weight", "s_attr_weight"),
            ("attr_priority_class", "s_attr_priority_class"),
        ],
    )
    def test_maps_attr_fields(
        self, simos_name: str, expected: str
    ) -> None:
        assert self.adapter.map_state_field(simos_name) == expected

    def test_returns_none_for_unknown(self) -> None:
        assert self.adapter.map_state_field("unknown_state_field") is None


class TestSimosSchemaAdapterMapRewardField:
    """Tests for SimosSchemaAdapter.map_reward_field."""

    def setup_method(self) -> None:
        self.adapter = SimosSchemaAdapter()

    @pytest.mark.parametrize(
        ("simos_name", "expected"),
        [
            ("time_penalty", "rw_time_penalty"),
            ("cost_penalty", "rw_cost_penalty"),
            ("step_completion", "rw_step_completion"),
            ("sla_penalty", "rw_sla_penalty"),
            ("episode_completion", "rw_episode_completion"),
            ("processing_efficiency", "rw_processing_efficiency"),
        ],
    )
    def test_maps_reward_fields(
        self, simos_name: str, expected: str
    ) -> None:
        assert self.adapter.map_reward_field(simos_name) == expected

    def test_returns_none_for_unknown(self) -> None:
        assert self.adapter.map_reward_field("unknown_reward") is None


class TestSimosSchemaAdapterMapSnapshotRecord:
    """Tests for SimosSchemaAdapter.map_snapshot_record."""

    def test_maps_full_record(self) -> None:
        adapter = SimosSchemaAdapter()
        record: dict[str, Any] = {
            "timestamp": 300,
            "sys_wip": 5,
            "node_intake_queue": 2,
            "resource_workers_utilization": 0.6,
            "unknown_field": 999,
        }
        mapped = adapter.map_snapshot_record(record)

        assert mapped["ts"] == 300
        assert mapped["wip"] == 5
        assert mapped["n_intake_queue"] == 2
        assert mapped["r_workers_util"] == 0.6
        # Unknown field should be dropped
        assert "unknown_field" not in mapped


class TestSimosSchemaAdapterMapTrajectoryRecord:
    """Tests for SimosSchemaAdapter.map_trajectory_record."""

    def test_flattens_state_reward_next_state(self) -> None:
        adapter = SimosSchemaAdapter()
        record: dict[str, Any] = {
            "entity_id": "e_001",
            "step_index": 0,
            "node_name": "intake",
            "episode_done": False,
            "state": {
                "entity_priority": 2,
                "entity_elapsed_time": 100.0,
                "node_utilization": 0.8,
            },
            "reward": {
                "time_penalty": -0.1,
                "step_completion": 1.0,
            },
            "next_state": {
                "entity_priority": 2,
                "entity_elapsed_time": 200.0,
            },
        }
        mapped = adapter.map_trajectory_record(record)

        # Top-level fields
        assert mapped["eid"] == "e_001"
        assert mapped["step"] == 0
        assert mapped["node"] == "intake"
        assert mapped["done"] is False

        # State fields (s_ prefix)
        assert mapped["s_priority"] == 2
        assert mapped["s_elapsed"] == 100.0
        assert mapped["s_node_util"] == 0.8

        # Reward fields (rw_ prefix)
        assert mapped["rw_time_penalty"] == -0.1
        assert mapped["rw_step_completion"] == 1.0

        # Next-state fields (ns_ prefix)
        assert mapped["ns_priority"] == 2
        assert mapped["ns_elapsed"] == 200.0

        # Nested dicts should not appear as keys
        assert "state" not in mapped
        assert "reward" not in mapped
        assert "next_state" not in mapped


# =====================================================================
# SimosLoader tests
# =====================================================================


class TestSimosLoaderLoad:
    """Tests for SimosLoader.load."""

    def test_load_valid_fixture(self, simos_export_small_path: Path) -> None:
        loader = SimosLoader()
        dataset = loader.load(simos_export_small_path)

        assert isinstance(dataset, RawDataset)
        assert dataset.has_snapshots
        assert dataset.has_trajectories
        assert dataset.source_type == "simos"

    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        loader = SimosLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(tmp_path / "nonexistent.json")

    def test_snapshot_df_has_canonical_columns(
        self, simos_export_small_path: Path
    ) -> None:
        loader = SimosLoader()
        dataset = loader.load(simos_export_small_path)
        snap = dataset.snapshots
        assert snap is not None

        # System-level canonical columns
        assert "ts" in snap.columns
        assert "wip" in snap.columns
        assert "in_queue" in snap.columns
        assert "throughput" in snap.columns
        assert "avg_wait" in snap.columns

        # Node-level canonical columns
        assert "n_intake_queue" in snap.columns
        assert "n_intake_util" in snap.columns
        assert "n_assembly_busy" in snap.columns

        # Resource-level canonical columns
        assert "r_workers_util" in snap.columns
        assert "r_workers_capacity" in snap.columns

        # SimOS names should NOT appear
        assert "timestamp" not in snap.columns
        assert "sys_wip" not in snap.columns
        assert "node_intake_queue" not in snap.columns
        assert "resource_workers_utilization" not in snap.columns

    def test_trajectory_df_has_canonical_columns(
        self, simos_export_small_path: Path
    ) -> None:
        loader = SimosLoader()
        dataset = loader.load(simos_export_small_path)
        traj = dataset.trajectories
        assert traj is not None

        # Top-level canonical columns
        assert "eid" in traj.columns
        assert "step" in traj.columns
        assert "node" in traj.columns
        assert "done" in traj.columns
        assert "status" in traj.columns

        # State canonical columns (s_ prefix)
        assert "s_priority" in traj.columns
        assert "s_elapsed" in traj.columns
        assert "s_cum_wait" in traj.columns
        assert "s_node_util" in traj.columns

        # Reward canonical columns (rw_ prefix)
        assert "rw_time_penalty" in traj.columns
        assert "rw_step_completion" in traj.columns

    def test_snapshot_df_sorted_by_ts(
        self, simos_export_small_path: Path
    ) -> None:
        loader = SimosLoader()
        dataset = loader.load(simos_export_small_path)
        snap = dataset.snapshots
        assert snap is not None

        ts_values = snap["ts"].to_list()
        assert ts_values == sorted(ts_values)

    def test_trajectory_df_sorted_by_eid_and_step(
        self, simos_export_small_path: Path
    ) -> None:
        loader = SimosLoader()
        dataset = loader.load(simos_export_small_path)
        traj = dataset.trajectories
        assert traj is not None

        # Verify sorting: extract (eid, step) pairs
        pairs = list(zip(traj["eid"].to_list(), traj["step"].to_list()))
        assert pairs == sorted(pairs)

    def test_column_info_computed(
        self, simos_export_small_path: Path
    ) -> None:
        loader = SimosLoader()
        dataset = loader.load(simos_export_small_path)

        assert dataset.column_info_snapshots is not None
        assert len(dataset.column_info_snapshots) > 0

        assert dataset.column_info_trajectories is not None
        assert len(dataset.column_info_trajectories) > 0

    def test_metadata_preserved(
        self, simos_export_small_path: Path
    ) -> None:
        loader = SimosLoader()
        dataset = loader.load(simos_export_small_path)

        assert dataset.metadata_dict is not None
        assert dataset.metadata_dict.get("export_version") == "3.0"
        assert dataset.summary_dict is not None
        assert "total_entities" in dataset.summary_dict
