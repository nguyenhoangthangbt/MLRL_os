"""Integration tests for PostgreSQL storage backend against a live database.

Requires:
    - PostgreSQL running at localhost:5432
    - Database MLRL_os created with tables (run scripts/create_db.py)
    - MLRLOS_DATABASE_URL env var or default credentials

Run with:
    pytest tests/integration/test_postgres_integration.py -v
"""

from __future__ import annotations

import os
import uuid

import polars as pl
import pytest
import pytest_asyncio

# Skip entire module if asyncpg is not installed
asyncpg = pytest.importorskip("asyncpg")

from mlrl_os.config.defaults import MLRLSettings
from mlrl_os.core.dataset import RawDataset, compute_column_info
from mlrl_os.core.experiment import ExperimentResult
from mlrl_os.core.types import ExperimentStatus, ProblemType
from mlrl_os.storage.postgres_backend import PostgresStorageBackend

DATABASE_URL = os.environ.get(
    "MLRLOS_DATABASE_URL",
    "postgresql://mlrlos:mlrlos_dev@localhost:5432/MLRL_os",
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_raw_dataset() -> RawDataset:
    """Build a small synthetic RawDataset for testing."""
    snapshots = pl.DataFrame(
        {
            "ts": [1.0, 2.0, 3.0],
            "wip": [10, 15, 12],
            "throughput": [5.0, 6.0, 5.5],
        }
    )
    trajectories = pl.DataFrame(
        {
            "eid": ["e001", "e001", "e002"],
            "step": [0, 1, 0],
            "done": [False, True, False],
            "total_time": [None, 100.0, None],
            "node": ["A", "B", "A"],
        }
    )
    return RawDataset(
        source_type="test",
        source_path="integration_test",
        snapshots=snapshots,
        trajectories=trajectories,
        column_info_snapshots=compute_column_info(snapshots),
        column_info_trajectories=compute_column_info(trajectories),
    )


def _unique_name(prefix: str = "intg") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def backend(tmp_path):
    """Create a connected PostgresStorageBackend with temp dirs for artifacts."""
    settings = MLRLSettings(
        data_dir=str(tmp_path / "data"),
        models_dir=str(tmp_path / "models"),
        experiments_dir=str(tmp_path / "experiments"),
        database_url=DATABASE_URL,
        storage_backend="postgres",
    )
    be = PostgresStorageBackend(settings)
    await be.connect()
    yield be
    await be.close()


# ── Dataset Tests ────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestDatasetOperations:
    async def test_register_and_retrieve(self, backend: PostgresStorageBackend):
        raw = _make_raw_dataset()
        name = _unique_name("ds")
        meta = await backend.register_dataset_async(raw, name=name)

        assert meta.id
        assert meta.name == name
        assert meta.source_type == "test"
        assert meta.has_snapshots is True
        assert meta.has_trajectories is True

        # Retrieve metadata
        fetched = await backend.get_dataset_meta_async(meta.id)
        assert fetched.id == meta.id
        assert fetched.name == name

        # Retrieve data
        snap_df = await backend.get_dataset_data_async(meta.id, layer="snapshots")
        assert snap_df.shape == (3, 3)

        traj_df = await backend.get_dataset_data_async(meta.id, layer="trajectories")
        assert traj_df.shape == (3, 5)

        # Cleanup
        await backend.delete_dataset_async(meta.id)

    async def test_list_datasets(self, backend: PostgresStorageBackend):
        raw = _make_raw_dataset()
        name = _unique_name("ds_list")
        meta = await backend.register_dataset_async(raw, name=name)

        datasets = await backend.list_datasets_async()
        ids = [d.id for d in datasets]
        assert meta.id in ids

        await backend.delete_dataset_async(meta.id)

    async def test_has_dataset(self, backend: PostgresStorageBackend):
        raw = _make_raw_dataset()
        name = _unique_name("ds_has")
        meta = await backend.register_dataset_async(raw, name=name)

        assert await backend.has_dataset_async(meta.id) is True
        assert await backend.has_dataset_async("nonexistent_id") is False

        await backend.delete_dataset_async(meta.id)

    async def test_delete_dataset(self, backend: PostgresStorageBackend):
        raw = _make_raw_dataset()
        name = _unique_name("ds_del")
        meta = await backend.register_dataset_async(raw, name=name)

        await backend.delete_dataset_async(meta.id)
        assert await backend.has_dataset_async(meta.id) is False

        with pytest.raises(KeyError):
            await backend.get_dataset_meta_async(meta.id)

    async def test_invalid_layer_raises(self, backend: PostgresStorageBackend):
        raw = _make_raw_dataset()
        name = _unique_name("ds_inv")
        meta = await backend.register_dataset_async(raw, name=name)

        with pytest.raises(ValueError, match="Invalid layer"):
            await backend.get_dataset_data_async(meta.id, layer="events")

        await backend.delete_dataset_async(meta.id)

    async def test_missing_dataset_raises(self, backend: PostgresStorageBackend):
        with pytest.raises(KeyError):
            await backend.get_dataset_meta_async("does_not_exist")


# ── Experiment Tests ─────────────────────────────────────────────────


@pytest.mark.asyncio
class TestExperimentOperations:
    async def test_create_and_retrieve(self, backend: PostgresStorageBackend):
        exp_id = _unique_name("exp")
        config = {"problem_type": "entity_classification", "seed": 42}

        await backend.create_experiment_async(exp_id, "test_exp", config)

        assert await backend.has_experiment_async(exp_id) is True

        result = await backend.get_experiment_result_async(exp_id)
        assert result.experiment_id == exp_id
        assert result.status == ExperimentStatus.PENDING

        # Cleanup — delete directly from DB
        conn = backend._require_conn()
        await conn.execute("DELETE FROM experiments WHERE id = $1", exp_id)

    async def test_save_and_update_result(self, backend: PostgresStorageBackend):
        exp_id = _unique_name("exp_upd")
        config = {"problem_type": "time_series", "seed": 42}

        await backend.create_experiment_async(exp_id, "update_test", config)

        # Simulate completion
        completed_result = ExperimentResult(
            experiment_id=exp_id,
            name="update_test",
            status=ExperimentStatus.COMPLETED,
            experiment_type=ProblemType.TIME_SERIES,
            created_at="2026-03-10T00:00:00",
            completed_at="2026-03-10T00:01:00",
            best_algorithm="lightgbm",
            metrics={"rmse": 12.5, "mae": 8.3},
            sample_count=500,
            feature_count=20,
        )

        await backend.save_experiment_result_async(completed_result)

        fetched = await backend.get_experiment_result_async(exp_id)
        assert fetched.status == ExperimentStatus.COMPLETED
        assert fetched.best_algorithm == "lightgbm"
        assert fetched.metrics["rmse"] == 12.5

        # Cleanup
        conn = backend._require_conn()
        await conn.execute("DELETE FROM experiments WHERE id = $1", exp_id)

    async def test_update_status(self, backend: PostgresStorageBackend):
        exp_id = _unique_name("exp_status")
        config = {"problem_type": "entity_classification"}

        await backend.create_experiment_async(exp_id, "status_test", config)

        await backend.update_experiment_status_async(exp_id, ExperimentStatus.RUNNING)
        result = await backend.get_experiment_result_async(exp_id)
        assert result.status == ExperimentStatus.RUNNING

        await backend.update_experiment_status_async(exp_id, ExperimentStatus.FAILED)
        result = await backend.get_experiment_result_async(exp_id)
        assert result.status == ExperimentStatus.FAILED

        # Cleanup
        conn = backend._require_conn()
        await conn.execute("DELETE FROM experiments WHERE id = $1", exp_id)

    async def test_list_experiments(self, backend: PostgresStorageBackend):
        exp_id = _unique_name("exp_list")
        config = {"problem_type": "time_series"}

        await backend.create_experiment_async(exp_id, "list_test", config)

        experiments = await backend.list_experiments_async()
        ids = [e.experiment_id for e in experiments]
        assert exp_id in ids

        # Cleanup
        conn = backend._require_conn()
        await conn.execute("DELETE FROM experiments WHERE id = $1", exp_id)

    async def test_duplicate_experiment_raises(self, backend: PostgresStorageBackend):
        exp_id = _unique_name("exp_dup")
        config = {"problem_type": "time_series"}

        await backend.create_experiment_async(exp_id, "dup_test", config)

        with pytest.raises(FileExistsError):
            await backend.create_experiment_async(exp_id, "dup_test_2", config)

        # Cleanup
        conn = backend._require_conn()
        await conn.execute("DELETE FROM experiments WHERE id = $1", exp_id)


# ── Sync Wrapper Tests ───────────────────────────────────────────────


class TestSyncWrappers:
    """Test sync wrappers work outside async context."""

    def test_register_dataset_sync(self, tmp_path):
        settings = MLRLSettings(
            data_dir=str(tmp_path / "data"),
            models_dir=str(tmp_path / "models"),
            experiments_dir=str(tmp_path / "experiments"),
            database_url=DATABASE_URL,
            storage_backend="postgres",
        )
        be = PostgresStorageBackend(settings)

        import asyncio

        asyncio.run(be.connect())
        try:
            raw = _make_raw_dataset()
            name = _unique_name("sync_ds")

            meta = be.register_dataset(raw, name=name)
            assert meta.id
            assert be.has_dataset(meta.id) is True

            datasets = be.list_datasets()
            assert any(d.id == meta.id for d in datasets)

            be.delete_dataset(meta.id)
            assert be.has_dataset(meta.id) is False
        finally:
            asyncio.run(be.close())

    def test_experiment_sync(self, tmp_path):
        settings = MLRLSettings(
            data_dir=str(tmp_path / "data"),
            models_dir=str(tmp_path / "models"),
            experiments_dir=str(tmp_path / "experiments"),
            database_url=DATABASE_URL,
            storage_backend="postgres",
        )
        be = PostgresStorageBackend(settings)

        import asyncio

        asyncio.run(be.connect())
        try:
            exp_id = _unique_name("sync_exp")
            config = {"problem_type": "time_series"}

            be.create_experiment(exp_id, "sync_test", config)
            assert be.has_experiment(exp_id) is True

            result = be.get_experiment_result(exp_id)
            assert result.experiment_id == exp_id
            assert result.status == ExperimentStatus.PENDING

            # Cleanup
            asyncio.run(
                be._require_conn().execute(
                    "DELETE FROM experiments WHERE id = $1", exp_id
                )
            )
        finally:
            asyncio.run(be.close())


# ── Connection Error Tests ───────────────────────────────────────────


@pytest.mark.asyncio
class TestConnectionErrors:
    async def test_operations_without_connect_raise(self, tmp_path):
        settings = MLRLSettings(
            data_dir=str(tmp_path / "data"),
            models_dir=str(tmp_path / "models"),
            experiments_dir=str(tmp_path / "experiments"),
            database_url=DATABASE_URL,
            storage_backend="postgres",
        )
        be = PostgresStorageBackend(settings)
        # Don't call connect()

        with pytest.raises(RuntimeError, match="Not connected"):
            await be.list_datasets_async()

    async def test_invalid_url_raises(self, tmp_path):
        settings = MLRLSettings(
            data_dir=str(tmp_path / "data"),
            models_dir=str(tmp_path / "models"),
            experiments_dir=str(tmp_path / "experiments"),
            database_url="postgresql://bad:bad@localhost:5432/nonexistent",
            storage_backend="postgres",
        )
        be = PostgresStorageBackend(settings)

        with pytest.raises(Exception):
            await be.connect()
