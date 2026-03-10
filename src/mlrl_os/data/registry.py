"""Dataset versioning and storage registry.

Manages registered datasets on the file system. Datasets are stored as Parquet
files under ``{data_dir}/datasets/{dataset_id}/`` with JSON metadata alongside.

Datasets are immutable — new versions create new entries. Content hashing
enables deduplication.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from mlrl_os.config.defaults import MLRLSettings
from mlrl_os.core.dataset import DatasetMeta, RawDataset, compute_column_info

logger = logging.getLogger(__name__)


def _content_hash(raw: RawDataset) -> str:
    """Compute a SHA-256 content hash from serialised IPC bytes."""
    hasher = hashlib.sha256()
    if raw.snapshots is not None:
        buf = io.BytesIO()
        raw.snapshots.write_ipc(buf)
        hasher.update(buf.getvalue())
    if raw.trajectories is not None:
        buf = io.BytesIO()
        raw.trajectories.write_ipc(buf)
        hasher.update(buf.getvalue())
    return hasher.hexdigest()


def _generate_id(name: str, timestamp: str, content_hash: str) -> str:
    """Generate a short dataset ID (first 12 chars of SHA-256)."""
    source = f"{name}{timestamp}{content_hash}"
    return hashlib.sha256(source.encode()).hexdigest()[:12]


class DatasetRegistry:
    """File-system based dataset registry."""

    def __init__(self, settings: MLRLSettings | None = None) -> None:
        self._settings = settings or MLRLSettings()
        self._base_dir = Path(self._settings.data_dir) / "datasets"
        self._base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        raw: RawDataset,
        name: str = "",
        source_instrument: str | None = None,
        source_job_id: str | None = None,
        source_template: str | None = None,
    ) -> DatasetMeta:
        """Register a RawDataset. Saves data as Parquet and metadata as JSON.

        Returns DatasetMeta with generated ID, content hash, etc.
        """
        now = datetime.now(timezone.utc)
        timestamp = now.isoformat()
        content = _content_hash(raw)
        dataset_id = _generate_id(name, timestamp, content)

        dataset_dir = self._base_dir / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Compute column info if not already provided
        snap_cols = raw.column_info_snapshots
        traj_cols = raw.column_info_trajectories
        if snap_cols is None and raw.snapshots is not None:
            snap_cols = compute_column_info(raw.snapshots)
        if traj_cols is None and raw.trajectories is not None:
            traj_cols = compute_column_info(raw.trajectories)

        # Write Parquet files
        if raw.snapshots is not None:
            raw.snapshots.write_parquet(dataset_dir / "snapshots.parquet")
        if raw.trajectories is not None:
            raw.trajectories.write_parquet(dataset_dir / "trajectories.parquet")

        # Build metadata
        meta = DatasetMeta(
            id=dataset_id,
            name=name or dataset_id,
            version=1,
            content_hash=content,
            source_type=raw.source_type,
            source_path=raw.source_path,
            snapshot_row_count=(
                len(raw.snapshots) if raw.snapshots is not None else None
            ),
            trajectory_row_count=(
                len(raw.trajectories) if raw.trajectories is not None else None
            ),
            snapshot_column_count=(
                len(raw.snapshots.columns) if raw.snapshots is not None else None
            ),
            trajectory_column_count=(
                len(raw.trajectories.columns)
                if raw.trajectories is not None
                else None
            ),
            snapshot_columns=snap_cols,
            trajectory_columns=traj_cols,
            has_snapshots=raw.has_snapshots,
            has_trajectories=raw.has_trajectories,
            simos_metadata=raw.metadata_dict if raw.metadata_dict else None,
            simos_summary=raw.summary_dict if raw.summary_dict else None,
            source_instrument=source_instrument,
            source_job_id=source_job_id,
            source_template=source_template,
            registered_at=timestamp,
        )

        # Write metadata JSON
        meta_path = dataset_dir / "meta.json"
        meta_path.write_text(
            json.dumps(meta.model_dump(), indent=2, default=str),
            encoding="utf-8",
        )

        logger.info(
            "Registered dataset %s (%s) — %s rows snapshots, %s rows trajectories",
            dataset_id,
            name,
            meta.snapshot_row_count,
            meta.trajectory_row_count,
        )

        return meta

    def get_meta(self, dataset_id: str) -> DatasetMeta:
        """Load metadata for a registered dataset.

        Raises:
            KeyError: If the dataset is not registered.
        """
        self._require_exists(dataset_id)
        meta_path = self._base_dir / dataset_id / "meta.json"
        raw_json = json.loads(meta_path.read_text(encoding="utf-8"))
        return DatasetMeta.model_validate(raw_json)

    def get_data(
        self, dataset_id: str, layer: str = "snapshots"
    ) -> pl.DataFrame:
        """Load the actual data (snapshots or trajectories) as a DataFrame.

        Args:
            dataset_id: The dataset identifier.
            layer: ``"snapshots"`` or ``"trajectories"``.

        Raises:
            KeyError: If the dataset is not registered.
            ValueError: If *layer* is invalid or the requested layer does not
                exist for this dataset.
        """
        self._require_exists(dataset_id)

        valid_layers = ("snapshots", "trajectories")
        if layer not in valid_layers:
            msg = (
                f"Invalid layer {layer!r}. Must be one of {valid_layers}"
            )
            raise ValueError(msg)

        parquet_path = self._base_dir / dataset_id / f"{layer}.parquet"
        if not parquet_path.exists():
            msg = (
                f"Dataset {dataset_id!r} does not have a {layer!r} layer"
            )
            raise ValueError(msg)

        return pl.read_parquet(parquet_path)

    def list_datasets(self) -> list[DatasetMeta]:
        """List all registered datasets, sorted by registration date."""
        datasets: list[DatasetMeta] = []

        if not self._base_dir.exists():
            return datasets

        for entry in self._base_dir.iterdir():
            meta_path = entry / "meta.json"
            if entry.is_dir() and meta_path.exists():
                try:
                    raw_json = json.loads(
                        meta_path.read_text(encoding="utf-8")
                    )
                    datasets.append(DatasetMeta.model_validate(raw_json))
                except (json.JSONDecodeError, Exception):
                    logger.warning(
                        "Skipping invalid metadata in %s", entry.name
                    )

        datasets.sort(key=lambda m: m.registered_at)
        return datasets

    def has(self, dataset_id: str) -> bool:
        """Check if a dataset is registered."""
        meta_path = self._base_dir / dataset_id / "meta.json"
        return meta_path.exists()

    def delete(self, dataset_id: str) -> None:
        """Delete a dataset and all its files.

        Raises:
            KeyError: If the dataset is not registered.
        """
        self._require_exists(dataset_id)
        dataset_dir = self._base_dir / dataset_id
        shutil.rmtree(dataset_dir)
        logger.info("Deleted dataset %s", dataset_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_exists(self, dataset_id: str) -> None:
        """Raise KeyError if *dataset_id* is not registered."""
        if not self.has(dataset_id):
            msg = f"Dataset {dataset_id!r} not found in registry"
            raise KeyError(msg)
