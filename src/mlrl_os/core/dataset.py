"""Dataset-related models for ML/RL OS."""

from __future__ import annotations

from typing import Any

import polars as pl
from pydantic import BaseModel, Field

from mlrl_os.core.types import ColumnInfo


class DatasetMeta(BaseModel):
    """Metadata for a registered dataset. Stored in the Dataset Registry."""

    id: str
    name: str
    version: int = 1
    content_hash: str
    source_type: str  # "simos" | "csv" | "parquet"
    source_path: str

    snapshot_row_count: int | None = None
    trajectory_row_count: int | None = None
    snapshot_column_count: int | None = None
    trajectory_column_count: int | None = None
    snapshot_columns: list[ColumnInfo] | None = None
    trajectory_columns: list[ColumnInfo] | None = None

    has_snapshots: bool = False
    has_trajectories: bool = False

    simos_metadata: dict[str, Any] | None = None
    simos_summary: dict[str, Any] | None = None

    registered_at: str = ""


class RawDataset:
    """Loaded data before registration. Holds Polars DataFrames.

    At least one of snapshots or trajectories must be non-None.
    """

    def __init__(
        self,
        source_type: str,
        source_path: str,
        snapshots: pl.DataFrame | None = None,
        trajectories: pl.DataFrame | None = None,
        metadata_dict: dict[str, Any] | None = None,
        summary_dict: dict[str, Any] | None = None,
        column_info_snapshots: list[ColumnInfo] | None = None,
        column_info_trajectories: list[ColumnInfo] | None = None,
    ) -> None:
        if snapshots is None and trajectories is None:
            msg = "At least one of snapshots or trajectories must be provided"
            raise ValueError(msg)

        self.source_type = source_type
        self.source_path = source_path
        self.snapshots = snapshots
        self.trajectories = trajectories
        self.metadata_dict = metadata_dict or {}
        self.summary_dict = summary_dict or {}
        self.column_info_snapshots = column_info_snapshots
        self.column_info_trajectories = column_info_trajectories

    @property
    def has_snapshots(self) -> bool:
        return self.snapshots is not None

    @property
    def has_trajectories(self) -> bool:
        return self.trajectories is not None


def compute_column_info(df: pl.DataFrame) -> list[ColumnInfo]:
    """Compute column-level statistics for a Polars DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        List of ColumnInfo, one per column.
    """
    result: list[ColumnInfo] = []
    row_count = len(df)

    for col_name in df.columns:
        series = df[col_name]
        null_count = series.null_count()
        null_rate = null_count / row_count if row_count > 0 else 0.0
        unique_count = series.n_unique()

        is_numeric = series.dtype.is_numeric()
        is_categorical = series.dtype == pl.Utf8 or series.dtype == pl.Categorical

        mean_val: float | None = None
        std_val: float | None = None
        min_val: float | None = None
        max_val: float | None = None
        categories: list[str] | None = None
        category_counts: dict[str, int] | None = None

        if is_numeric and row_count > 0:
            non_null = series.drop_nulls()
            if len(non_null) > 0:
                mean_val = float(non_null.mean())  # type: ignore[arg-type]
                std_val = float(non_null.std())  # type: ignore[arg-type]
                min_val = float(non_null.min())  # type: ignore[arg-type]
                max_val = float(non_null.max())  # type: ignore[arg-type]

        if is_categorical and row_count > 0:
            non_null = series.drop_nulls().cast(pl.Utf8)
            cats = non_null.unique().sort().to_list()
            categories = [str(c) for c in cats]
            counts = non_null.value_counts()
            category_counts = {
                str(row[col_name]): int(row["count"])
                for row in counts.iter_rows(named=True)
            }

        info = ColumnInfo(
            name=col_name,
            dtype=str(series.dtype),
            null_count=null_count,
            null_rate=round(null_rate, 6),
            unique_count=unique_count,
            is_numeric=is_numeric,
            is_categorical=is_categorical,
            mean=round(mean_val, 6) if mean_val is not None else None,
            std=round(std_val, 6) if std_val is not None else None,
            min=round(min_val, 6) if min_val is not None else None,
            max=round(max_val, 6) if max_val is not None else None,
            categories=categories,
            category_counts=category_counts,
        )
        result.append(info)

    return result
