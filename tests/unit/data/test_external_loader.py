"""Tests for mlrl_os.data.external_loader."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from mlrl_os.core.dataset import RawDataset
from mlrl_os.data.external_loader import ExternalLoader


@pytest.fixture()
def csv_with_timestamp(tmp_path: Path) -> Path:
    """Create a CSV file with a timestamp column (time-series)."""
    df = pl.DataFrame({
        "timestamp": [1.0, 2.0, 3.0, 4.0, 5.0],
        "value_a": [10, 20, 30, 40, 50],
        "value_b": [1.1, 2.2, 3.3, 4.4, 5.5],
    })
    path = tmp_path / "ts_data.csv"
    df.write_csv(path)
    return path


@pytest.fixture()
def csv_without_timestamp(tmp_path: Path) -> Path:
    """Create a CSV file without a timestamp column (entity data)."""
    df = pl.DataFrame({
        "entity_id": ["a", "b", "c", "d", "e"],
        "score": [0.9, 0.8, 0.7, 0.6, 0.5],
        "label": ["ok", "ok", "fail", "ok", "fail"],
    })
    path = tmp_path / "entity_data.csv"
    df.write_csv(path)
    return path


@pytest.fixture()
def parquet_with_timestamp(tmp_path: Path) -> Path:
    """Create a Parquet file with a time column."""
    df = pl.DataFrame({
        "time": [100.0, 200.0, 300.0],
        "metric": [5, 10, 15],
    })
    path = tmp_path / "ts_data.parquet"
    df.write_parquet(path)
    return path


@pytest.fixture()
def parquet_without_timestamp(tmp_path: Path) -> Path:
    """Create a Parquet file without a time column."""
    df = pl.DataFrame({
        "item_id": [1, 2, 3],
        "weight": [10.0, 20.0, 30.0],
    })
    path = tmp_path / "entity_data.parquet"
    df.write_parquet(path)
    return path


@pytest.fixture()
def empty_csv(tmp_path: Path) -> Path:
    """Create an empty CSV file with headers only."""
    path = tmp_path / "empty.csv"
    path.write_text("col_a,col_b\n")
    return path


class TestExternalLoaderCSV:
    """Tests for ExternalLoader.load_csv."""

    def test_load_csv_returns_raw_dataset(
        self, csv_with_timestamp: Path
    ) -> None:
        loader = ExternalLoader()
        dataset = loader.load_csv(csv_with_timestamp)

        assert isinstance(dataset, RawDataset)
        assert dataset.source_type == "csv"

    def test_csv_with_timestamp_detected_as_snapshots(
        self, csv_with_timestamp: Path
    ) -> None:
        loader = ExternalLoader()
        dataset = loader.load_csv(csv_with_timestamp)

        assert dataset.has_snapshots
        assert not dataset.has_trajectories
        assert dataset.snapshots is not None
        assert len(dataset.snapshots) == 5

    def test_csv_without_timestamp_detected_as_trajectories(
        self, csv_without_timestamp: Path
    ) -> None:
        loader = ExternalLoader()
        dataset = loader.load_csv(csv_without_timestamp)

        assert not dataset.has_snapshots
        assert dataset.has_trajectories
        assert dataset.trajectories is not None
        assert len(dataset.trajectories) == 5

    def test_csv_file_not_found(self, tmp_path: Path) -> None:
        loader = ExternalLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_csv(tmp_path / "nonexistent.csv")

    def test_csv_empty_raises_value_error(self, empty_csv: Path) -> None:
        loader = ExternalLoader()
        with pytest.raises(ValueError, match="empty"):
            loader.load_csv(empty_csv)

    def test_csv_column_info_computed(
        self, csv_with_timestamp: Path
    ) -> None:
        loader = ExternalLoader()
        dataset = loader.load_csv(csv_with_timestamp)

        assert dataset.column_info_snapshots is not None
        col_names = [c.name for c in dataset.column_info_snapshots]
        assert "timestamp" in col_names
        assert "value_a" in col_names
        assert "value_b" in col_names

    def test_csv_source_path_recorded(
        self, csv_with_timestamp: Path
    ) -> None:
        loader = ExternalLoader()
        dataset = loader.load_csv(csv_with_timestamp)
        assert str(csv_with_timestamp) == dataset.source_path


class TestExternalLoaderParquet:
    """Tests for ExternalLoader.load_parquet."""

    def test_load_parquet_returns_raw_dataset(
        self, parquet_with_timestamp: Path
    ) -> None:
        loader = ExternalLoader()
        dataset = loader.load_parquet(parquet_with_timestamp)

        assert isinstance(dataset, RawDataset)
        assert dataset.source_type == "parquet"

    def test_parquet_with_timestamp_detected_as_snapshots(
        self, parquet_with_timestamp: Path
    ) -> None:
        loader = ExternalLoader()
        dataset = loader.load_parquet(parquet_with_timestamp)

        assert dataset.has_snapshots
        assert not dataset.has_trajectories

    def test_parquet_without_timestamp_detected_as_trajectories(
        self, parquet_without_timestamp: Path
    ) -> None:
        loader = ExternalLoader()
        dataset = loader.load_parquet(parquet_without_timestamp)

        assert not dataset.has_snapshots
        assert dataset.has_trajectories

    def test_parquet_file_not_found(self, tmp_path: Path) -> None:
        loader = ExternalLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_parquet(tmp_path / "nonexistent.parquet")

    def test_parquet_column_info_computed(
        self, parquet_with_timestamp: Path
    ) -> None:
        loader = ExternalLoader()
        dataset = loader.load_parquet(parquet_with_timestamp)

        assert dataset.column_info_snapshots is not None
        col_names = [c.name for c in dataset.column_info_snapshots]
        assert "time" in col_names
        assert "metric" in col_names


class TestTimestampDetection:
    """Tests for the time-series vs entity heuristic."""

    @pytest.mark.parametrize(
        "ts_column",
        ["ts", "timestamp", "time", "date", "datetime", "t", "period",
         "bucket_idx", "bucket_index", "time_step", "timestep"],
    )
    def test_recognized_timestamp_columns(
        self, tmp_path: Path, ts_column: str
    ) -> None:
        df = pl.DataFrame({
            ts_column: [1, 2, 3],
            "value": [10, 20, 30],
        })
        path = tmp_path / f"data_{ts_column}.csv"
        df.write_csv(path)

        loader = ExternalLoader()
        dataset = loader.load_csv(path)
        assert dataset.has_snapshots, (
            f"Column '{ts_column}' should be detected as time-series"
        )
