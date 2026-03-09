"""Tests for the PostgresStorageBackend.

These tests verify the async PostgreSQL backend. Tests are skipped when
asyncpg is not installed or when no test database is available.

The test suite uses a mock-based approach: it patches asyncpg.connect to
return a mock connection, so no real PostgreSQL server is needed for unit
tests. Integration tests against a real database belong in tests/integration/.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import polars as pl
import pytest

# Skip entire module if asyncpg is not installed
asyncpg = pytest.importorskip("asyncpg", reason="asyncpg not installed")

from mlrl_os.config.defaults import MLRLSettings
from mlrl_os.core.dataset import DatasetMeta, RawDataset
from mlrl_os.core.experiment import ExperimentResult
from mlrl_os.core.types import ExperimentStatus, ProblemType
from mlrl_os.evaluation.reports import EvaluationReport
from mlrl_os.models.algorithms.protocol import TrainedModel
from mlrl_os.models.registry import ModelMeta
from mlrl_os.storage.postgres_backend import PostgresStorageBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def settings(tmp_path: Any) -> MLRLSettings:
    """Create settings with temp dirs and a test database URL."""
    return MLRLSettings(
        data_dir=str(tmp_path / "data"),
        models_dir=str(tmp_path / "models"),
        experiments_dir=str(tmp_path / "experiments"),
        storage_backend="postgres",
        database_url="postgresql://test:test@localhost:5432/test_mlrl",
    )


@pytest.fixture()
def mock_conn() -> AsyncMock:
    """Create a mock asyncpg connection."""
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=None)
    conn.fetch = AsyncMock(return_value=[])
    conn.execute = AsyncMock(return_value="INSERT 0 1")
    conn.fetchval = AsyncMock(return_value=None)
    conn.close = AsyncMock()
    return conn


@pytest.fixture()
def backend(settings: MLRLSettings, mock_conn: AsyncMock) -> PostgresStorageBackend:
    """Create a PostgresStorageBackend with a mocked connection."""
    be = PostgresStorageBackend(settings)
    be._conn = mock_conn
    return be


@pytest.fixture()
def sample_raw_dataset() -> RawDataset:
    """Create a minimal RawDataset for testing."""
    trajectories = pl.DataFrame(
        {
            "entity_id": ["e1", "e1", "e2", "e2"],
            "step": [0, 1, 0, 1],
            "wait_time": [1.0, 2.0, 3.0, 4.0],
            "status": ["active", "completed", "active", "completed"],
        }
    )
    return RawDataset(
        source_type="test",
        source_path="/test/data.parquet",
        trajectories=trajectories,
    )


@pytest.fixture()
def sample_dataset_meta() -> DatasetMeta:
    """Create a sample DatasetMeta as returned from DB."""
    return DatasetMeta(
        id="ds-abc123",
        name="test-dataset",
        version=1,
        content_hash="abc123def456",
        source_type="test",
        source_path="/test/data.parquet",
        has_trajectories=True,
        trajectory_row_count=4,
        trajectory_column_count=4,
        registered_at="2026-03-10T00:00:00+00:00",
    )


@pytest.fixture()
def sample_experiment_result() -> ExperimentResult:
    """Create a minimal ExperimentResult for testing."""
    return ExperimentResult(
        experiment_id="exp-001",
        name="Test Experiment",
        status=ExperimentStatus.COMPLETED,
        experiment_type=ProblemType.ENTITY_CLASSIFICATION,
        created_at="2026-03-10T00:00:00+00:00",
        best_algorithm="lightgbm",
        metrics={"f1_weighted": 0.95, "auc_roc": 0.98},
        sample_count=100,
        feature_count=10,
        resolved_config={"problem_type": "entity_classification"},
    )


@pytest.fixture()
def sample_report() -> EvaluationReport:
    """Create a minimal EvaluationReport for testing."""
    return EvaluationReport(
        experiment_id="exp-001",
        experiment_name="Test Experiment",
        experiment_type="entity_classification",
        created_at="2026-03-10T00:00:00+00:00",
        dataset_id="ds-001",
        dataset_name="test-dataset",
        sample_count=100,
        feature_count=10,
        target="status",
        task_type="classification",
        best_algorithm="lightgbm",
        best_metrics={"f1_weighted": 0.95},
    )


@pytest.fixture()
def sample_trained_model() -> TrainedModel:
    """Create a minimal TrainedModel for testing."""
    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier(random_state=42)
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    model.fit(X, y)

    return TrainedModel(
        model=model,
        algorithm_name="decision_tree",
        task="classification",
        feature_names=["feature_a", "feature_b"],
    )


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------


class TestPostgresConnectionManagement:
    """Test connection lifecycle."""

    @pytest.mark.asyncio()
    async def test_connect_creates_connection(
        self, settings: MLRLSettings
    ) -> None:
        """connect() calls asyncpg.connect with the database URL."""
        mock_conn = AsyncMock()
        with patch("mlrl_os.storage.postgres_backend.asyncpg") as mock_asyncpg:
            mock_asyncpg.connect = AsyncMock(return_value=mock_conn)
            be = PostgresStorageBackend(settings)
            await be.connect()
            mock_asyncpg.connect.assert_called_once_with(settings.database_url)
            assert be._conn is mock_conn

    @pytest.mark.asyncio()
    async def test_close_closes_connection(
        self, backend: PostgresStorageBackend, mock_conn: AsyncMock
    ) -> None:
        """close() closes the underlying connection."""
        await backend.close()
        mock_conn.close.assert_called_once()


# ---------------------------------------------------------------------------
# Dataset operations
# ---------------------------------------------------------------------------


class TestPostgresDatasetOperations:
    """Dataset CRUD via PostgresStorageBackend with mocked DB."""

    @pytest.mark.asyncio()
    async def test_register_dataset_inserts_row(
        self,
        backend: PostgresStorageBackend,
        sample_raw_dataset: RawDataset,
        mock_conn: AsyncMock,
    ) -> None:
        """register_dataset writes meta to DB and Parquet to disk."""
        meta = await backend.register_dataset_async(
            sample_raw_dataset, name="test-ds"
        )
        assert meta.id
        assert meta.name == "test-ds"
        assert meta.source_type == "test"
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio()
    async def test_get_dataset_meta_returns_meta(
        self,
        backend: PostgresStorageBackend,
        sample_dataset_meta: DatasetMeta,
        mock_conn: AsyncMock,
    ) -> None:
        """get_dataset_meta queries DB and returns DatasetMeta."""
        row = {
            "id": sample_dataset_meta.id,
            "name": sample_dataset_meta.name,
            "version": sample_dataset_meta.version,
            "content_hash": sample_dataset_meta.content_hash,
            "source_type": sample_dataset_meta.source_type,
            "source_path": sample_dataset_meta.source_path,
            "meta": json.dumps(sample_dataset_meta.model_dump(mode="json")),
            "data_path": "/some/path",
            "created_at": datetime.now(timezone.utc),
        }
        mock_conn.fetchrow.return_value = row

        meta = await backend.get_dataset_meta_async("ds-abc123")
        assert meta.id == "ds-abc123"
        assert meta.name == "test-dataset"

    @pytest.mark.asyncio()
    async def test_get_dataset_meta_not_found_raises(
        self,
        backend: PostgresStorageBackend,
        mock_conn: AsyncMock,
    ) -> None:
        """get_dataset_meta raises KeyError when not found."""
        mock_conn.fetchrow.return_value = None
        with pytest.raises(KeyError, match="not found"):
            await backend.get_dataset_meta_async("nonexistent")

    @pytest.mark.asyncio()
    async def test_list_datasets_returns_list(
        self,
        backend: PostgresStorageBackend,
        sample_dataset_meta: DatasetMeta,
        mock_conn: AsyncMock,
    ) -> None:
        """list_datasets fetches all datasets."""
        row = {
            "id": sample_dataset_meta.id,
            "name": sample_dataset_meta.name,
            "version": sample_dataset_meta.version,
            "content_hash": sample_dataset_meta.content_hash,
            "source_type": sample_dataset_meta.source_type,
            "source_path": sample_dataset_meta.source_path,
            "meta": json.dumps(sample_dataset_meta.model_dump(mode="json")),
            "data_path": "/some/path",
            "created_at": datetime.now(timezone.utc),
        }
        mock_conn.fetch.return_value = [row]

        datasets = await backend.list_datasets_async()
        assert len(datasets) == 1
        assert datasets[0].id == "ds-abc123"

    @pytest.mark.asyncio()
    async def test_has_dataset(
        self,
        backend: PostgresStorageBackend,
        mock_conn: AsyncMock,
    ) -> None:
        """has_dataset returns True when found, False otherwise."""
        mock_conn.fetchval.return_value = True
        assert await backend.has_dataset_async("ds-abc123") is True

        mock_conn.fetchval.return_value = None
        assert await backend.has_dataset_async("nonexistent") is False

    @pytest.mark.asyncio()
    async def test_delete_dataset(
        self,
        backend: PostgresStorageBackend,
        mock_conn: AsyncMock,
    ) -> None:
        """delete_dataset removes from DB."""
        mock_conn.fetchval.return_value = True
        mock_conn.fetchrow.return_value = {"data_path": "/some/path"}
        await backend.delete_dataset_async("ds-abc123")
        # Verify execute was called (DELETE statement)
        assert mock_conn.execute.called

    @pytest.mark.asyncio()
    async def test_delete_dataset_not_found_raises(
        self,
        backend: PostgresStorageBackend,
        mock_conn: AsyncMock,
    ) -> None:
        """delete_dataset raises KeyError when not found."""
        mock_conn.fetchval.return_value = None
        with pytest.raises(KeyError, match="not found"):
            await backend.delete_dataset_async("nonexistent")


# ---------------------------------------------------------------------------
# Experiment operations
# ---------------------------------------------------------------------------


class TestPostgresExperimentOperations:
    """Experiment CRUD via PostgresStorageBackend with mocked DB."""

    @pytest.mark.asyncio()
    async def test_create_experiment(
        self,
        backend: PostgresStorageBackend,
        mock_conn: AsyncMock,
    ) -> None:
        """create_experiment inserts a row."""
        mock_conn.fetchval.return_value = None  # no existing experiment
        config = {"problem_type": "entity_classification", "seed": 42}
        await backend.create_experiment_async("exp-001", "Test Exp", config)
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio()
    async def test_create_experiment_duplicate_raises(
        self,
        backend: PostgresStorageBackend,
        mock_conn: AsyncMock,
    ) -> None:
        """create_experiment raises FileExistsError on duplicate."""
        mock_conn.fetchval.return_value = True  # already exists
        config = {"problem_type": "entity_classification"}
        with pytest.raises(FileExistsError, match="already exists"):
            await backend.create_experiment_async("exp-001", "Test", config)

    @pytest.mark.asyncio()
    async def test_get_experiment_result(
        self,
        backend: PostgresStorageBackend,
        sample_experiment_result: ExperimentResult,
        mock_conn: AsyncMock,
    ) -> None:
        """get_experiment_result deserialises the result from DB."""
        row = {
            "id": "exp-001",
            "name": "Test Experiment",
            "problem_type": "entity_classification",
            "dataset_id": None,
            "config": json.dumps({"problem_type": "entity_classification"}),
            "result": json.dumps(
                sample_experiment_result.model_dump(mode="json")
            ),
            "status": "completed",
            "created_at": datetime.now(timezone.utc),
            "completed_at": None,
        }
        mock_conn.fetchrow.return_value = row

        result = await backend.get_experiment_result_async("exp-001")
        assert result.experiment_id == "exp-001"
        assert result.best_algorithm == "lightgbm"

    @pytest.mark.asyncio()
    async def test_get_experiment_result_not_found_raises(
        self,
        backend: PostgresStorageBackend,
        mock_conn: AsyncMock,
    ) -> None:
        """get_experiment_result raises KeyError when not found."""
        mock_conn.fetchrow.return_value = None
        with pytest.raises(KeyError, match="not found"):
            await backend.get_experiment_result_async("nonexistent")

    @pytest.mark.asyncio()
    async def test_list_experiments(
        self,
        backend: PostgresStorageBackend,
        sample_experiment_result: ExperimentResult,
        mock_conn: AsyncMock,
    ) -> None:
        """list_experiments fetches all experiments."""
        row = {
            "id": "exp-001",
            "name": "Test Experiment",
            "problem_type": "entity_classification",
            "dataset_id": None,
            "config": json.dumps({"problem_type": "entity_classification"}),
            "result": json.dumps(
                sample_experiment_result.model_dump(mode="json")
            ),
            "status": "completed",
            "created_at": datetime.now(timezone.utc),
            "completed_at": None,
        }
        mock_conn.fetch.return_value = [row]

        experiments = await backend.list_experiments_async()
        assert len(experiments) == 1
        assert experiments[0].experiment_id == "exp-001"

    @pytest.mark.asyncio()
    async def test_has_experiment(
        self,
        backend: PostgresStorageBackend,
        mock_conn: AsyncMock,
    ) -> None:
        """has_experiment returns True/False based on DB lookup."""
        mock_conn.fetchval.return_value = True
        assert await backend.has_experiment_async("exp-001") is True

        mock_conn.fetchval.return_value = None
        assert await backend.has_experiment_async("nonexistent") is False

    @pytest.mark.asyncio()
    async def test_update_experiment_status(
        self,
        backend: PostgresStorageBackend,
        mock_conn: AsyncMock,
    ) -> None:
        """update_experiment_status updates the status in DB."""
        mock_conn.fetchval.return_value = True  # experiment exists
        await backend.update_experiment_status_async(
            "exp-001", ExperimentStatus.RUNNING
        )
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio()
    async def test_update_status_not_found_raises(
        self,
        backend: PostgresStorageBackend,
        mock_conn: AsyncMock,
    ) -> None:
        """update_experiment_status raises KeyError when not found."""
        mock_conn.fetchval.return_value = None
        with pytest.raises(KeyError, match="not found"):
            await backend.update_experiment_status_async(
                "nonexistent", ExperimentStatus.RUNNING
            )


# ---------------------------------------------------------------------------
# Model operations
# ---------------------------------------------------------------------------


class TestPostgresModelOperations:
    """Model CRUD via PostgresStorageBackend with mocked DB."""

    @pytest.mark.asyncio()
    async def test_register_model(
        self,
        backend: PostgresStorageBackend,
        sample_trained_model: TrainedModel,
        mock_conn: AsyncMock,
    ) -> None:
        """register_model saves artifact and inserts DB row."""
        meta = await backend.register_model_async(
            sample_trained_model,
            experiment_id="exp-001",
            metrics={"accuracy": 0.95},
        )
        assert meta.id
        assert meta.algorithm_name == "decision_tree"
        assert meta.metrics["accuracy"] == 0.95
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio()
    async def test_get_model_meta(
        self,
        backend: PostgresStorageBackend,
        mock_conn: AsyncMock,
    ) -> None:
        """get_model_meta queries DB and returns ModelMeta."""
        row = {
            "id": "model-abc",
            "experiment_id": "exp-001",
            "algorithm": "decision_tree",
            "task": "classification",
            "feature_names": json.dumps(["feature_a", "feature_b"]),
            "metrics": json.dumps({"accuracy": 0.95}),
            "artifact_path": "/some/model.joblib",
            "created_at": datetime.now(timezone.utc),
        }
        mock_conn.fetchrow.return_value = row

        meta = await backend.get_model_meta_async("model-abc")
        assert meta.id == "model-abc"
        assert meta.algorithm_name == "decision_tree"

    @pytest.mark.asyncio()
    async def test_get_model_meta_not_found_raises(
        self,
        backend: PostgresStorageBackend,
        mock_conn: AsyncMock,
    ) -> None:
        """get_model_meta raises KeyError when not found."""
        mock_conn.fetchrow.return_value = None
        with pytest.raises(KeyError, match="not found"):
            await backend.get_model_meta_async("nonexistent")

    @pytest.mark.asyncio()
    async def test_list_models(
        self,
        backend: PostgresStorageBackend,
        mock_conn: AsyncMock,
    ) -> None:
        """list_models fetches all models from DB."""
        row = {
            "id": "model-abc",
            "experiment_id": "exp-001",
            "algorithm": "decision_tree",
            "task": "classification",
            "feature_names": json.dumps(["feature_a", "feature_b"]),
            "metrics": json.dumps({"accuracy": 0.95}),
            "artifact_path": "/some/model.joblib",
            "created_at": datetime.now(timezone.utc),
        }
        mock_conn.fetch.return_value = [row]

        models = await backend.list_models_async()
        assert len(models) == 1
        assert models[0].id == "model-abc"

    @pytest.mark.asyncio()
    async def test_has_model(
        self,
        backend: PostgresStorageBackend,
        mock_conn: AsyncMock,
    ) -> None:
        """has_model returns True/False based on DB lookup."""
        mock_conn.fetchval.return_value = True
        assert await backend.has_model_async("model-abc") is True

        mock_conn.fetchval.return_value = None
        assert await backend.has_model_async("nonexistent") is False

    @pytest.mark.asyncio()
    async def test_delete_model(
        self,
        backend: PostgresStorageBackend,
        mock_conn: AsyncMock,
    ) -> None:
        """delete_model removes from DB."""
        mock_conn.fetchval.return_value = True
        mock_conn.fetchrow.return_value = {"artifact_path": "/some/model.joblib"}
        await backend.delete_model_async("model-abc")
        assert mock_conn.execute.called

    @pytest.mark.asyncio()
    async def test_delete_model_not_found_raises(
        self,
        backend: PostgresStorageBackend,
        mock_conn: AsyncMock,
    ) -> None:
        """delete_model raises KeyError when not found."""
        mock_conn.fetchval.return_value = None
        with pytest.raises(KeyError, match="not found"):
            await backend.delete_model_async("nonexistent")


# ---------------------------------------------------------------------------
# Sync wrapper tests
# ---------------------------------------------------------------------------


class TestPostgresSyncWrappers:
    """Verify that sync methods delegate to async implementations."""

    def test_sync_methods_exist(
        self, backend: PostgresStorageBackend
    ) -> None:
        """All sync protocol methods exist on the backend."""
        sync_methods = [
            "register_dataset",
            "get_dataset_meta",
            "get_dataset_data",
            "list_datasets",
            "has_dataset",
            "delete_dataset",
            "create_experiment",
            "save_experiment_result",
            "save_experiment_report",
            "get_experiment_result",
            "get_experiment_report",
            "list_experiments",
            "has_experiment",
            "update_experiment_status",
            "register_model",
            "get_model_meta",
            "load_model",
            "list_models",
            "has_model",
            "delete_model",
        ]
        for method_name in sync_methods:
            assert hasattr(backend, method_name), f"Missing sync method: {method_name}"
            assert callable(getattr(backend, method_name)), (
                f"Not callable: {method_name}"
            )
