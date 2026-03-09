"""PostgreSQL storage backend using asyncpg.

Stores metadata in PostgreSQL (datasets, experiments, models tables) and
keeps Parquet data files and model artifacts on disk. asyncpg is
lazy-imported so that the module can be loaded even when asyncpg is not
installed — a clear error is raised only when connect() is called.

This backend exposes both async methods (``*_async``) and synchronous
wrappers that call ``asyncio.run()`` for use in non-async contexts.
The sync wrappers satisfy the :class:`StorageBackend` protocol.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import polars as pl

from mlrl_os.config.defaults import MLRLSettings
from mlrl_os.core.dataset import DatasetMeta, RawDataset, compute_column_info
from mlrl_os.core.experiment import ExperimentResult
from mlrl_os.core.types import ExperimentStatus
from mlrl_os.evaluation.reports import EvaluationReport
from mlrl_os.models.algorithms.protocol import TrainedModel
from mlrl_os.models.registry import ModelMeta

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger(__name__)

# Lazy import helper — avoids module-level import of asyncpg
asyncpg: Any = None  # type: ignore[no-redef]


def _ensure_asyncpg() -> Any:
    """Lazy-import asyncpg, raising ImportError with a helpful message."""
    global asyncpg  # noqa: PLW0603
    if asyncpg is None:
        import asyncpg as _asyncpg

        asyncpg = _asyncpg
    return asyncpg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    """Generate a short ID (first 12 chars of SHA-256)."""
    source = f"{name}{timestamp}{content_hash}"
    return hashlib.sha256(source.encode()).hexdigest()[:12]


def _run_sync(coro: Any) -> Any:
    """Run an async coroutine synchronously.

    Uses ``asyncio.run()`` when no event loop is running, or
    creates a new loop if the current one is already running.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        return asyncio.run(coro)

    # If an event loop is already running (e.g. in FastAPI), create a new one
    # in a thread. This is a fallback for sync callers in async contexts.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()


# ---------------------------------------------------------------------------
# PostgreSQL Storage Backend
# ---------------------------------------------------------------------------


class PostgresStorageBackend:
    """Async PostgreSQL storage backend.

    Metadata lives in PostgreSQL; Parquet files and model artifacts live on
    disk under the paths configured in :class:`MLRLSettings`.
    """

    def __init__(self, settings: MLRLSettings | None = None) -> None:
        self._settings = settings or MLRLSettings()
        self._conn: Any = None  # asyncpg.Connection, set by connect()
        self._data_dir = Path(self._settings.data_dir) / "datasets"
        self._models_dir = Path(self._settings.models_dir) / "models"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._models_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the asyncpg connection."""
        _ensure_asyncpg()
        self._conn = await asyncpg.connect(self._settings.database_url)
        logger.info("Connected to PostgreSQL: %s", self._settings.database_url)

    async def close(self) -> None:
        """Close the asyncpg connection."""
        if self._conn is not None:
            await self._conn.close()
            logger.info("Closed PostgreSQL connection")

    def _require_conn(self) -> Any:
        """Return the connection, raising RuntimeError if not connected."""
        if self._conn is None:
            msg = "Not connected to PostgreSQL. Call connect() first."
            raise RuntimeError(msg)
        return self._conn

    # ==================================================================
    # ASYNC DATASET OPERATIONS
    # ==================================================================

    async def register_dataset_async(
        self, raw: RawDataset, name: str = ""
    ) -> DatasetMeta:
        """Register a RawDataset: write Parquet to disk, metadata to DB."""
        conn = self._require_conn()

        now = datetime.now(timezone.utc)
        timestamp = now.isoformat()
        content = _content_hash(raw)
        dataset_id = _generate_id(name, timestamp, content)

        # Write Parquet files to disk
        dataset_dir = self._data_dir / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)

        snap_cols = raw.column_info_snapshots
        traj_cols = raw.column_info_trajectories
        if snap_cols is None and raw.snapshots is not None:
            snap_cols = compute_column_info(raw.snapshots)
        if traj_cols is None and raw.trajectories is not None:
            traj_cols = compute_column_info(raw.trajectories)

        if raw.snapshots is not None:
            raw.snapshots.write_parquet(dataset_dir / "snapshots.parquet")
        if raw.trajectories is not None:
            raw.trajectories.write_parquet(dataset_dir / "trajectories.parquet")

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
            registered_at=timestamp,
        )

        await conn.execute(
            """
            INSERT INTO datasets (id, name, version, content_hash, source_type,
                                  source_path, meta, data_path, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            dataset_id,
            meta.name,
            meta.version,
            content,
            raw.source_type,
            raw.source_path,
            json.dumps(meta.model_dump(mode="json"), default=str),
            str(dataset_dir),
            now,
        )

        logger.info("Registered dataset %s (%s) in PostgreSQL", dataset_id, name)
        return meta

    async def get_dataset_meta_async(self, dataset_id: str) -> DatasetMeta:
        """Load dataset metadata from PostgreSQL."""
        conn = self._require_conn()
        row = await conn.fetchrow(
            "SELECT * FROM datasets WHERE id = $1", dataset_id
        )
        if row is None:
            msg = f"Dataset {dataset_id!r} not found in database"
            raise KeyError(msg)

        meta_json = json.loads(row["meta"])
        return DatasetMeta.model_validate(meta_json)

    async def get_dataset_data_async(
        self, dataset_id: str, layer: str = "snapshots"
    ) -> pl.DataFrame:
        """Load dataset data from disk, using DB to find the path."""
        conn = self._require_conn()

        valid_layers = ("snapshots", "trajectories")
        if layer not in valid_layers:
            msg = f"Invalid layer {layer!r}. Must be one of {valid_layers}"
            raise ValueError(msg)

        row = await conn.fetchrow(
            "SELECT data_path FROM datasets WHERE id = $1", dataset_id
        )
        if row is None:
            msg = f"Dataset {dataset_id!r} not found in database"
            raise KeyError(msg)

        parquet_path = Path(row["data_path"]) / f"{layer}.parquet"
        if not parquet_path.exists():
            msg = f"Dataset {dataset_id!r} does not have a {layer!r} layer"
            raise ValueError(msg)

        return pl.read_parquet(parquet_path)

    async def list_datasets_async(self) -> list[DatasetMeta]:
        """List all datasets from PostgreSQL, sorted by created_at."""
        conn = self._require_conn()
        rows = await conn.fetch(
            "SELECT * FROM datasets ORDER BY created_at ASC"
        )
        datasets: list[DatasetMeta] = []
        for row in rows:
            meta_json = json.loads(row["meta"])
            datasets.append(DatasetMeta.model_validate(meta_json))
        return datasets

    async def has_dataset_async(self, dataset_id: str) -> bool:
        """Check if a dataset exists in PostgreSQL."""
        conn = self._require_conn()
        result = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM datasets WHERE id = $1)", dataset_id
        )
        return bool(result)

    async def delete_dataset_async(self, dataset_id: str) -> None:
        """Delete a dataset from DB and disk."""
        conn = self._require_conn()

        exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM datasets WHERE id = $1)", dataset_id
        )
        if not exists:
            msg = f"Dataset {dataset_id!r} not found in database"
            raise KeyError(msg)

        row = await conn.fetchrow(
            "SELECT data_path FROM datasets WHERE id = $1", dataset_id
        )
        if row and row["data_path"]:
            data_path = Path(row["data_path"])
            if data_path.exists():
                shutil.rmtree(data_path)

        await conn.execute("DELETE FROM datasets WHERE id = $1", dataset_id)
        logger.info("Deleted dataset %s from PostgreSQL", dataset_id)

    # ==================================================================
    # ASYNC EXPERIMENT OPERATIONS
    # ==================================================================

    async def create_experiment_async(
        self, experiment_id: str, name: str, config_dict: dict[str, Any]
    ) -> None:
        """Create a new experiment record with PENDING status."""
        conn = self._require_conn()

        exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM experiments WHERE id = $1)",
            experiment_id,
        )
        if exists:
            msg = f"Experiment already exists: {experiment_id}"
            raise FileExistsError(msg)

        now = datetime.now(timezone.utc)
        problem_type = config_dict.get("problem_type", "time_series")

        initial_result = ExperimentResult(
            experiment_id=experiment_id,
            name=name,
            status=ExperimentStatus.PENDING,
            experiment_type=problem_type,
            created_at=now.isoformat(),
            resolved_config=config_dict,
        )

        await conn.execute(
            """
            INSERT INTO experiments (id, name, problem_type, config, result,
                                     status, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            experiment_id,
            name,
            problem_type,
            json.dumps(config_dict),
            json.dumps(initial_result.model_dump(mode="json")),
            ExperimentStatus.PENDING.value,
            now,
        )

        logger.info("Created experiment %s (%s) in PostgreSQL", experiment_id, name)

    async def save_experiment_result_async(
        self, result: ExperimentResult
    ) -> None:
        """Save (or update) the experiment result in DB."""
        conn = self._require_conn()

        completed_at = None
        if result.status in (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED):
            completed_at = datetime.now(timezone.utc)

        await conn.execute(
            """
            UPDATE experiments SET result = $1, status = $2, completed_at = $3
            WHERE id = $4
            """,
            json.dumps(result.model_dump(mode="json")),
            result.status.value,
            completed_at,
            result.experiment_id,
        )

        logger.info("Saved result for experiment %s", result.experiment_id)

    async def save_experiment_report_async(
        self, experiment_id: str, report: EvaluationReport
    ) -> None:
        """Save an evaluation report as part of the experiment result."""
        conn = self._require_conn()

        row = await conn.fetchrow(
            "SELECT result FROM experiments WHERE id = $1", experiment_id
        )
        if row is None:
            msg = f"Experiment {experiment_id!r} not found in database"
            raise KeyError(msg)

        result_data = json.loads(row["result"]) if row["result"] else {}
        result_data["_report"] = report.model_dump(mode="json")

        await conn.execute(
            "UPDATE experiments SET result = $1 WHERE id = $2",
            json.dumps(result_data),
            experiment_id,
        )

        logger.info("Saved report for experiment %s", experiment_id)

    async def get_experiment_result_async(
        self, experiment_id: str
    ) -> ExperimentResult:
        """Load experiment result from PostgreSQL."""
        conn = self._require_conn()
        row = await conn.fetchrow(
            "SELECT * FROM experiments WHERE id = $1", experiment_id
        )
        if row is None:
            msg = f"Experiment {experiment_id!r} not found in database"
            raise KeyError(msg)

        result_json = json.loads(row["result"])
        # Remove embedded report if present (it's stored alongside)
        result_json.pop("_report", None)
        return ExperimentResult.model_validate(result_json)

    async def get_experiment_report_async(
        self, experiment_id: str
    ) -> EvaluationReport:
        """Load evaluation report from PostgreSQL."""
        conn = self._require_conn()
        row = await conn.fetchrow(
            "SELECT result FROM experiments WHERE id = $1", experiment_id
        )
        if row is None:
            msg = f"Experiment {experiment_id!r} not found in database"
            raise KeyError(msg)

        result_json = json.loads(row["result"]) if row["result"] else {}
        report_data = result_json.get("_report")
        if report_data is None:
            msg = f"Report not found for experiment: {experiment_id}"
            raise KeyError(msg)

        return EvaluationReport.model_validate(report_data)

    async def list_experiments_async(self) -> list[ExperimentResult]:
        """List all experiments from PostgreSQL, sorted by created_at descending."""
        conn = self._require_conn()
        rows = await conn.fetch(
            "SELECT * FROM experiments ORDER BY created_at DESC"
        )
        results: list[ExperimentResult] = []
        for row in rows:
            result_json = json.loads(row["result"])
            result_json.pop("_report", None)
            results.append(ExperimentResult.model_validate(result_json))
        return results

    async def has_experiment_async(self, experiment_id: str) -> bool:
        """Check if an experiment exists in PostgreSQL."""
        conn = self._require_conn()
        result = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM experiments WHERE id = $1)",
            experiment_id,
        )
        return bool(result)

    async def update_experiment_status_async(
        self, experiment_id: str, status: ExperimentStatus
    ) -> None:
        """Update experiment status in PostgreSQL."""
        conn = self._require_conn()

        exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM experiments WHERE id = $1)",
            experiment_id,
        )
        if not exists:
            msg = f"Experiment {experiment_id!r} not found in database"
            raise KeyError(msg)

        completed_at = None
        if status in (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED):
            completed_at = datetime.now(timezone.utc)

        await conn.execute(
            """
            UPDATE experiments SET status = $1, completed_at = $2
            WHERE id = $3
            """,
            status.value,
            completed_at,
            experiment_id,
        )

        # Also update the status in the result JSONB
        row = await conn.fetchrow(
            "SELECT result FROM experiments WHERE id = $1", experiment_id
        )
        if row and row["result"]:
            result_data = json.loads(row["result"])
            result_data["status"] = status.value
            if completed_at:
                result_data["completed_at"] = completed_at.isoformat()
            await conn.execute(
                "UPDATE experiments SET result = $1 WHERE id = $2",
                json.dumps(result_data),
                experiment_id,
            )

        logger.info(
            "Updated experiment %s status to %s", experiment_id, status.value
        )

    # ==================================================================
    # ASYNC MODEL OPERATIONS
    # ==================================================================

    async def register_model_async(
        self,
        trained_model: TrainedModel,
        experiment_id: str,
        metrics: dict[str, float] | None = None,
    ) -> ModelMeta:
        """Save a trained model artifact to disk and metadata to DB."""
        conn = self._require_conn()

        now = datetime.now(timezone.utc)
        timestamp = now.isoformat()
        raw = f"{experiment_id}{trained_model.algorithm_name}{timestamp}"
        model_id = hashlib.sha256(raw.encode()).hexdigest()[:12]

        model_dir = self._models_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        artifact_path = model_dir / "model.joblib"
        joblib.dump(trained_model, artifact_path)

        meta = ModelMeta(
            id=model_id,
            experiment_id=experiment_id,
            algorithm_name=trained_model.algorithm_name,
            task=trained_model.task,
            feature_names=trained_model.feature_names,
            metrics=metrics or {},
            created_at=timestamp,
            file_path=str(artifact_path),
        )

        await conn.execute(
            """
            INSERT INTO models (id, experiment_id, algorithm, task,
                                feature_names, metrics, artifact_path, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            model_id,
            experiment_id,
            trained_model.algorithm_name,
            trained_model.task,
            json.dumps(trained_model.feature_names),
            json.dumps(metrics or {}),
            str(artifact_path),
            now,
        )

        logger.info(
            "Registered model %s for experiment %s in PostgreSQL",
            model_id,
            experiment_id,
        )
        return meta

    async def get_model_meta_async(self, model_id: str) -> ModelMeta:
        """Load model metadata from PostgreSQL."""
        conn = self._require_conn()
        row = await conn.fetchrow(
            "SELECT * FROM models WHERE id = $1", model_id
        )
        if row is None:
            msg = f"Model {model_id!r} not found in database"
            raise KeyError(msg)

        feature_names = json.loads(row["feature_names"])
        metrics = json.loads(row["metrics"])

        return ModelMeta(
            id=row["id"],
            experiment_id=row["experiment_id"],
            algorithm_name=row["algorithm"],
            task=row["task"],
            feature_names=feature_names,
            metrics=metrics,
            created_at=row["created_at"].isoformat()
            if isinstance(row["created_at"], datetime)
            else str(row["created_at"]),
            file_path=row["artifact_path"] or "",
        )

    async def load_model_async(self, model_id: str) -> TrainedModel:
        """Load the actual trained model artifact from disk."""
        meta = await self.get_model_meta_async(model_id)
        artifact_path = Path(meta.file_path)

        if not artifact_path.exists():
            msg = f"Model artifact missing for {model_id}: {artifact_path}"
            raise KeyError(msg)

        model: TrainedModel = joblib.load(artifact_path)
        logger.info("Loaded model %s from %s", model_id, artifact_path)
        return model

    async def list_models_async(self) -> list[ModelMeta]:
        """List all models from PostgreSQL, sorted by creation time (newest first)."""
        conn = self._require_conn()
        rows = await conn.fetch(
            "SELECT * FROM models ORDER BY created_at DESC"
        )
        models: list[ModelMeta] = []
        for row in rows:
            feature_names = json.loads(row["feature_names"])
            metrics = json.loads(row["metrics"])
            models.append(
                ModelMeta(
                    id=row["id"],
                    experiment_id=row["experiment_id"],
                    algorithm_name=row["algorithm"],
                    task=row["task"],
                    feature_names=feature_names,
                    metrics=metrics,
                    created_at=row["created_at"].isoformat()
                    if isinstance(row["created_at"], datetime)
                    else str(row["created_at"]),
                    file_path=row["artifact_path"] or "",
                )
            )
        return models

    async def has_model_async(self, model_id: str) -> bool:
        """Check if a model exists in PostgreSQL."""
        conn = self._require_conn()
        result = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM models WHERE id = $1)", model_id
        )
        return bool(result)

    async def delete_model_async(self, model_id: str) -> None:
        """Delete a model from DB and disk."""
        conn = self._require_conn()

        exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM models WHERE id = $1)", model_id
        )
        if not exists:
            msg = f"Model {model_id!r} not found in database"
            raise KeyError(msg)

        row = await conn.fetchrow(
            "SELECT artifact_path FROM models WHERE id = $1", model_id
        )
        if row and row["artifact_path"]:
            artifact_path = Path(row["artifact_path"])
            if artifact_path.parent.exists():
                shutil.rmtree(artifact_path.parent)

        await conn.execute("DELETE FROM models WHERE id = $1", model_id)
        logger.info("Deleted model %s from PostgreSQL", model_id)

    # ==================================================================
    # SYNC WRAPPERS (satisfy StorageBackend protocol)
    # ==================================================================

    def register_dataset(self, raw: RawDataset, name: str = "") -> DatasetMeta:
        """Sync wrapper for register_dataset_async."""
        return _run_sync(self.register_dataset_async(raw, name=name))

    def get_dataset_meta(self, dataset_id: str) -> DatasetMeta:
        """Sync wrapper for get_dataset_meta_async."""
        return _run_sync(self.get_dataset_meta_async(dataset_id))

    def get_dataset_data(
        self, dataset_id: str, layer: str = "snapshots"
    ) -> pl.DataFrame:
        """Sync wrapper for get_dataset_data_async."""
        return _run_sync(self.get_dataset_data_async(dataset_id, layer=layer))

    def list_datasets(self) -> list[DatasetMeta]:
        """Sync wrapper for list_datasets_async."""
        return _run_sync(self.list_datasets_async())

    def has_dataset(self, dataset_id: str) -> bool:
        """Sync wrapper for has_dataset_async."""
        return _run_sync(self.has_dataset_async(dataset_id))

    def delete_dataset(self, dataset_id: str) -> None:
        """Sync wrapper for delete_dataset_async."""
        return _run_sync(self.delete_dataset_async(dataset_id))

    def create_experiment(
        self, experiment_id: str, name: str, config_dict: dict[str, Any]
    ) -> None:
        """Sync wrapper for create_experiment_async."""
        return _run_sync(
            self.create_experiment_async(experiment_id, name, config_dict)
        )

    def save_experiment_result(self, result: ExperimentResult) -> None:
        """Sync wrapper for save_experiment_result_async."""
        return _run_sync(self.save_experiment_result_async(result))

    def save_experiment_report(
        self, experiment_id: str, report: EvaluationReport
    ) -> None:
        """Sync wrapper for save_experiment_report_async."""
        return _run_sync(
            self.save_experiment_report_async(experiment_id, report)
        )

    def get_experiment_result(self, experiment_id: str) -> ExperimentResult:
        """Sync wrapper for get_experiment_result_async."""
        return _run_sync(self.get_experiment_result_async(experiment_id))

    def get_experiment_report(self, experiment_id: str) -> EvaluationReport:
        """Sync wrapper for get_experiment_report_async."""
        return _run_sync(self.get_experiment_report_async(experiment_id))

    def list_experiments(self) -> list[ExperimentResult]:
        """Sync wrapper for list_experiments_async."""
        return _run_sync(self.list_experiments_async())

    def has_experiment(self, experiment_id: str) -> bool:
        """Sync wrapper for has_experiment_async."""
        return _run_sync(self.has_experiment_async(experiment_id))

    def update_experiment_status(
        self, experiment_id: str, status: ExperimentStatus
    ) -> None:
        """Sync wrapper for update_experiment_status_async."""
        return _run_sync(
            self.update_experiment_status_async(experiment_id, status)
        )

    def register_model(
        self,
        trained_model: TrainedModel,
        experiment_id: str,
        metrics: dict[str, float] | None = None,
    ) -> ModelMeta:
        """Sync wrapper for register_model_async."""
        return _run_sync(
            self.register_model_async(trained_model, experiment_id, metrics=metrics)
        )

    def get_model_meta(self, model_id: str) -> ModelMeta:
        """Sync wrapper for get_model_meta_async."""
        return _run_sync(self.get_model_meta_async(model_id))

    def load_model(self, model_id: str) -> TrainedModel:
        """Sync wrapper for load_model_async."""
        return _run_sync(self.load_model_async(model_id))

    def list_models(self) -> list[ModelMeta]:
        """Sync wrapper for list_models_async."""
        return _run_sync(self.list_models_async())

    def has_model(self, model_id: str) -> bool:
        """Sync wrapper for has_model_async."""
        return _run_sync(self.has_model_async(model_id))

    def delete_model(self, model_id: str) -> None:
        """Sync wrapper for delete_model_async."""
        return _run_sync(self.delete_model_async(model_id))
