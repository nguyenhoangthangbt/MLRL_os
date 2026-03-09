"""External data loader for CSV and Parquet files.

Loads tabular data from common file formats into RawDataset,
auto-detecting whether the data represents time-series snapshots
or entity-level trajectory data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl

from mlrl_os.core.dataset import RawDataset, compute_column_info

logger = logging.getLogger(__name__)

# Column names that suggest a timestamp / time-series index
_TIMESTAMP_HINTS = frozenset({
    "ts",
    "timestamp",
    "time",
    "date",
    "datetime",
    "t",
    "period",
    "bucket_idx",
    "bucket_index",
    "time_step",
    "timestep",
})


def _is_timestamp_column(col_name: str, dtype: pl.DataType) -> bool:
    """Heuristic: is this column likely a time index?"""
    lower = col_name.lower().strip()
    if lower in _TIMESTAMP_HINTS:
        return True
    if dtype in (pl.Date, pl.Datetime, pl.Time, pl.Duration):
        return True
    return False


def _looks_like_time_series(df: pl.DataFrame) -> bool:
    """Heuristic: does the DataFrame look like time-series data?

    Returns True if any column appears to be a temporal index.
    """
    for col_name in df.columns:
        if _is_timestamp_column(col_name, df[col_name].dtype):
            return True
    return False


class ExternalLoader:
    """Load CSV and Parquet files into RawDataset.

    Auto-detects whether data is time-series (snapshots) or entity-level
    (trajectories) based on column names and types.

    Usage::

        loader = ExternalLoader()
        dataset = loader.load_csv(Path("data.csv"))
        dataset = loader.load_parquet(Path("data.parquet"))
    """

    def load_csv(
        self,
        path: Path,
        **kwargs: Any,
    ) -> RawDataset:
        """Load a CSV file into a RawDataset.

        Args:
            path: Path to the CSV file.
            **kwargs: Additional keyword arguments forwarded to
                ``polars.read_csv`` (e.g. ``separator``, ``has_header``,
                ``dtypes``, ``null_values``).

        Returns:
            RawDataset with the data assigned to snapshots or trajectories.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the file is empty or cannot be parsed.
        """
        if not path.exists():
            msg = f"CSV file not found: {path}"
            raise FileNotFoundError(msg)

        logger.info("Loading CSV from %s", path)

        # Let polars infer types; user can override via kwargs
        df = pl.read_csv(path, **kwargs)

        if df.is_empty():
            msg = f"CSV file is empty: {path}"
            raise ValueError(msg)

        return self._build_raw_dataset(df, source_type="csv", source_path=str(path))

    def load_parquet(self, path: Path) -> RawDataset:
        """Load a Parquet file into a RawDataset.

        Args:
            path: Path to the Parquet file.

        Returns:
            RawDataset with the data assigned to snapshots or trajectories.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the file is empty.
        """
        if not path.exists():
            msg = f"Parquet file not found: {path}"
            raise FileNotFoundError(msg)

        logger.info("Loading Parquet from %s", path)

        df = pl.read_parquet(path)

        if df.is_empty():
            msg = f"Parquet file is empty: {path}"
            raise ValueError(msg)

        return self._build_raw_dataset(
            df, source_type="parquet", source_path=str(path)
        )

    def _build_raw_dataset(
        self,
        df: pl.DataFrame,
        source_type: str,
        source_path: str,
    ) -> RawDataset:
        """Classify the DataFrame and wrap it in a RawDataset.

        If the data looks like time-series (has a timestamp-like column),
        it is stored as ``snapshots``. Otherwise it is stored as
        ``trajectories`` (entity-level data).
        """
        col_info = compute_column_info(df)
        is_ts = _looks_like_time_series(df)

        if is_ts:
            logger.info(
                "Detected time-series data (%d rows, %d cols)",
                len(df),
                len(df.columns),
            )
            return RawDataset(
                source_type=source_type,
                source_path=source_path,
                snapshots=df,
                column_info_snapshots=col_info,
            )

        logger.info(
            "Detected entity/tabular data (%d rows, %d cols)",
            len(df),
            len(df.columns),
        )
        return RawDataset(
            source_type=source_type,
            source_path=source_path,
            trajectories=df,
            column_info_trajectories=col_info,
        )
