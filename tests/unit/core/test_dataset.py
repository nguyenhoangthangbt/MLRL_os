"""Tests for mlrl_os.core.dataset."""

from __future__ import annotations

import polars as pl
import pytest

from mlrl_os.core.dataset import DatasetMeta, RawDataset, compute_column_info
from mlrl_os.core.types import ColumnInfo


# ---------------------------------------------------------------------------
# DatasetMeta
# ---------------------------------------------------------------------------


class TestDatasetMeta:
    def test_creation_with_required_fields(self) -> None:
        meta = DatasetMeta(
            id="ds_001",
            name="Test",
            content_hash="abc123",
            source_type="simos",
            source_path="/path/to/file.json",
        )
        assert meta.id == "ds_001"
        assert meta.version == 1
        assert meta.has_snapshots is False
        assert meta.has_trajectories is False
        assert meta.snapshot_columns is None

    def test_creation_with_snapshot_fields(
        self, snapshot_dataset_meta: DatasetMeta
    ) -> None:
        meta = snapshot_dataset_meta
        assert meta.has_snapshots is True
        assert meta.has_trajectories is False
        assert meta.snapshot_row_count == 100
        assert meta.snapshot_columns is not None
        assert len(meta.snapshot_columns) > 0

    def test_serialization_round_trip(
        self, snapshot_dataset_meta: DatasetMeta
    ) -> None:
        data = snapshot_dataset_meta.model_dump()
        restored = DatasetMeta.model_validate(data)
        assert restored.id == snapshot_dataset_meta.id
        assert restored.name == snapshot_dataset_meta.name
        assert restored.has_snapshots == snapshot_dataset_meta.has_snapshots
        assert len(restored.snapshot_columns) == len(snapshot_dataset_meta.snapshot_columns)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# RawDataset
# ---------------------------------------------------------------------------


class TestRawDataset:
    def test_construction_snapshots_only(
        self, raw_snapshot_dataset: RawDataset
    ) -> None:
        ds = raw_snapshot_dataset
        assert ds.has_snapshots is True
        assert ds.has_trajectories is False
        assert ds.source_type == "simos"

    def test_construction_trajectories_only(
        self, raw_trajectory_dataset: RawDataset
    ) -> None:
        ds = raw_trajectory_dataset
        assert ds.has_snapshots is False
        assert ds.has_trajectories is True

    def test_construction_both(
        self, snapshot_df: pl.DataFrame, trajectory_df: pl.DataFrame
    ) -> None:
        ds = RawDataset(
            source_type="simos",
            source_path="/fake",
            snapshots=snapshot_df,
            trajectories=trajectory_df,
        )
        assert ds.has_snapshots is True
        assert ds.has_trajectories is True

    def test_raises_when_both_none(self) -> None:
        with pytest.raises(ValueError, match="At least one of"):
            RawDataset(
                source_type="simos",
                source_path="/fake",
                snapshots=None,
                trajectories=None,
            )

    def test_has_snapshots_property(
        self, raw_snapshot_dataset: RawDataset
    ) -> None:
        assert raw_snapshot_dataset.has_snapshots is True

    def test_has_trajectories_property(
        self, raw_trajectory_dataset: RawDataset
    ) -> None:
        assert raw_trajectory_dataset.has_trajectories is True


# ---------------------------------------------------------------------------
# compute_column_info
# ---------------------------------------------------------------------------


class TestComputeColumnInfo:
    def test_numeric_columns_have_stats(self, snapshot_df: pl.DataFrame) -> None:
        infos = compute_column_info(snapshot_df)
        # "throughput" is a numeric column
        throughput_info = next(i for i in infos if i.name == "throughput")
        assert throughput_info.is_numeric is True
        assert throughput_info.mean is not None
        assert throughput_info.std is not None
        assert throughput_info.min is not None
        assert throughput_info.max is not None

    def test_categorical_columns_have_categories(self) -> None:
        df = pl.DataFrame({"color": ["red", "blue", "red", "green"]})
        infos = compute_column_info(df)
        assert len(infos) == 1
        info = infos[0]
        assert info.is_categorical is True
        assert info.categories is not None
        assert set(info.categories) == {"red", "blue", "green"}
        assert info.category_counts is not None
        assert info.category_counts["red"] == 2

    def test_null_counts_computed(self) -> None:
        df = pl.DataFrame({"val": [1.0, None, 3.0, None]})
        infos = compute_column_info(df)
        assert infos[0].null_count == 2
        assert infos[0].null_rate == pytest.approx(0.5, abs=1e-5)

    def test_empty_dataframe_returns_empty_list(self) -> None:
        df = pl.DataFrame()
        infos = compute_column_info(df)
        assert infos == []
