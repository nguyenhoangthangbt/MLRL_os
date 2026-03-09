"""Tests for the FileStorageBackend.

Verifies that the file-based backend correctly wraps v0.1 registries
(DatasetRegistry, ModelRegistry, ExperimentTracker) behind the
StorageBackend protocol.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest

from mlrl_os.config.defaults import MLRLSettings
from mlrl_os.core.dataset import DatasetMeta, RawDataset
from mlrl_os.core.experiment import ExperimentResult
from mlrl_os.core.types import ExperimentStatus, ProblemType
from mlrl_os.evaluation.reports import EvaluationReport
from mlrl_os.models.algorithms.protocol import TrainedModel
from mlrl_os.models.registry import ModelMeta
from mlrl_os.storage.file_backend import FileStorageBackend
from mlrl_os.storage.protocol import StorageBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def settings(tmp_path: Any) -> MLRLSettings:
    """Create settings pointing to a temp directory."""
    return MLRLSettings(
        data_dir=str(tmp_path / "data"),
        models_dir=str(tmp_path / "models"),
        experiments_dir=str(tmp_path / "experiments"),
    )


@pytest.fixture()
def backend(settings: MLRLSettings) -> FileStorageBackend:
    """Create a FileStorageBackend with temp directories."""
    return FileStorageBackend(settings)


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
# Protocol compliance
# ---------------------------------------------------------------------------


class TestFileBackendProtocol:
    """Verify FileStorageBackend satisfies StorageBackend protocol."""

    def test_is_protocol_instance(self, backend: FileStorageBackend) -> None:
        """FileStorageBackend satisfies the StorageBackend protocol."""
        assert isinstance(backend, StorageBackend)

    def test_can_assign_to_protocol_type(
        self, backend: FileStorageBackend
    ) -> None:
        """FileStorageBackend can be assigned to a StorageBackend variable."""
        storage: StorageBackend = backend
        assert storage is not None


# ---------------------------------------------------------------------------
# Dataset operations
# ---------------------------------------------------------------------------


class TestFileBackendDatasets:
    """Dataset CRUD through FileStorageBackend."""

    def test_register_and_get_meta(
        self,
        backend: FileStorageBackend,
        sample_raw_dataset: RawDataset,
    ) -> None:
        """Register a dataset, then retrieve its metadata."""
        meta = backend.register_dataset(sample_raw_dataset, name="test-ds")
        assert meta.id
        assert meta.name == "test-ds"
        assert meta.source_type == "test"
        assert meta.has_trajectories is True

        loaded = backend.get_dataset_meta(meta.id)
        assert loaded.id == meta.id
        assert loaded.content_hash == meta.content_hash

    def test_get_dataset_data(
        self,
        backend: FileStorageBackend,
        sample_raw_dataset: RawDataset,
    ) -> None:
        """Retrieve actual data after registration."""
        meta = backend.register_dataset(sample_raw_dataset, name="test-ds")
        df = backend.get_dataset_data(meta.id, layer="trajectories")
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 4
        assert "entity_id" in df.columns

    def test_get_dataset_data_invalid_layer(
        self,
        backend: FileStorageBackend,
        sample_raw_dataset: RawDataset,
    ) -> None:
        """Invalid layer raises ValueError."""
        meta = backend.register_dataset(sample_raw_dataset, name="test-ds")
        with pytest.raises(ValueError, match="Invalid layer"):
            backend.get_dataset_data(meta.id, layer="invalid")

    def test_get_dataset_data_missing_layer(
        self,
        backend: FileStorageBackend,
        sample_raw_dataset: RawDataset,
    ) -> None:
        """Requesting a layer that the dataset does not have raises ValueError."""
        meta = backend.register_dataset(sample_raw_dataset, name="test-ds")
        with pytest.raises(ValueError, match="does not have"):
            backend.get_dataset_data(meta.id, layer="snapshots")

    def test_list_datasets_empty(self, backend: FileStorageBackend) -> None:
        """Empty registry returns empty list."""
        assert backend.list_datasets() == []

    def test_list_datasets(
        self,
        backend: FileStorageBackend,
        sample_raw_dataset: RawDataset,
    ) -> None:
        """Registered datasets appear in list."""
        backend.register_dataset(sample_raw_dataset, name="ds-1")
        datasets = backend.list_datasets()
        assert len(datasets) == 1
        assert datasets[0].name == "ds-1"

    def test_has_dataset(
        self,
        backend: FileStorageBackend,
        sample_raw_dataset: RawDataset,
    ) -> None:
        """has_dataset returns True for registered, False for unknown."""
        meta = backend.register_dataset(sample_raw_dataset, name="test-ds")
        assert backend.has_dataset(meta.id) is True
        assert backend.has_dataset("nonexistent") is False

    def test_delete_dataset(
        self,
        backend: FileStorageBackend,
        sample_raw_dataset: RawDataset,
    ) -> None:
        """Delete removes the dataset."""
        meta = backend.register_dataset(sample_raw_dataset, name="test-ds")
        backend.delete_dataset(meta.id)
        assert backend.has_dataset(meta.id) is False

    def test_delete_nonexistent_raises(
        self, backend: FileStorageBackend
    ) -> None:
        """Deleting a nonexistent dataset raises KeyError."""
        with pytest.raises(KeyError):
            backend.delete_dataset("nonexistent")

    def test_get_meta_nonexistent_raises(
        self, backend: FileStorageBackend
    ) -> None:
        """Getting metadata for a nonexistent dataset raises KeyError."""
        with pytest.raises(KeyError):
            backend.get_dataset_meta("nonexistent")


# ---------------------------------------------------------------------------
# Experiment operations
# ---------------------------------------------------------------------------


class TestFileBackendExperiments:
    """Experiment CRUD through FileStorageBackend."""

    def test_create_and_get_result(
        self, backend: FileStorageBackend
    ) -> None:
        """Create an experiment, then retrieve its result."""
        config = {"problem_type": "entity_classification", "seed": 42}
        backend.create_experiment("exp-001", "Test Exp", config)

        result = backend.get_experiment_result("exp-001")
        assert result.experiment_id == "exp-001"
        assert result.name == "Test Exp"
        assert result.status == ExperimentStatus.PENDING

    def test_create_duplicate_raises(
        self, backend: FileStorageBackend
    ) -> None:
        """Creating an experiment with an existing ID raises FileExistsError."""
        config = {"problem_type": "entity_classification"}
        backend.create_experiment("exp-001", "Test", config)
        with pytest.raises(FileExistsError):
            backend.create_experiment("exp-001", "Duplicate", config)

    def test_save_and_get_result(
        self,
        backend: FileStorageBackend,
        sample_experiment_result: ExperimentResult,
    ) -> None:
        """Save a full result and retrieve it."""
        config = {"problem_type": "entity_classification"}
        backend.create_experiment("exp-001", "Test Exp", config)
        backend.save_experiment_result(sample_experiment_result)

        loaded = backend.get_experiment_result("exp-001")
        assert loaded.best_algorithm == "lightgbm"
        assert loaded.metrics is not None
        assert loaded.metrics["f1_weighted"] == 0.95

    def test_save_and_get_report(
        self,
        backend: FileStorageBackend,
        sample_report: EvaluationReport,
    ) -> None:
        """Save and retrieve an evaluation report."""
        config = {"problem_type": "entity_classification"}
        backend.create_experiment("exp-001", "Test Exp", config)

        backend.save_experiment_report("exp-001", sample_report)
        loaded = backend.get_experiment_report("exp-001")
        assert loaded.experiment_id == "exp-001"
        assert loaded.best_algorithm == "lightgbm"

    def test_get_report_missing_raises(
        self, backend: FileStorageBackend
    ) -> None:
        """Getting a report that does not exist raises KeyError."""
        config = {"problem_type": "entity_classification"}
        backend.create_experiment("exp-001", "Test Exp", config)
        with pytest.raises(KeyError):
            backend.get_experiment_report("exp-001")

    def test_list_experiments_empty(
        self, backend: FileStorageBackend
    ) -> None:
        """Empty tracker returns empty list."""
        assert backend.list_experiments() == []

    def test_list_experiments(self, backend: FileStorageBackend) -> None:
        """Created experiments appear in list."""
        config = {"problem_type": "entity_classification"}
        backend.create_experiment("exp-001", "Exp 1", config)
        experiments = backend.list_experiments()
        assert len(experiments) == 1
        assert experiments[0].experiment_id == "exp-001"

    def test_has_experiment(self, backend: FileStorageBackend) -> None:
        """has_experiment returns True for existing, False for unknown."""
        config = {"problem_type": "entity_classification"}
        backend.create_experiment("exp-001", "Test", config)
        assert backend.has_experiment("exp-001") is True
        assert backend.has_experiment("nonexistent") is False

    def test_update_experiment_status(
        self, backend: FileStorageBackend
    ) -> None:
        """Update status changes the persisted status."""
        config = {"problem_type": "entity_classification"}
        backend.create_experiment("exp-001", "Test", config)
        backend.update_experiment_status("exp-001", ExperimentStatus.RUNNING)

        result = backend.get_experiment_result("exp-001")
        assert result.status == ExperimentStatus.RUNNING

    def test_update_status_nonexistent_raises(
        self, backend: FileStorageBackend
    ) -> None:
        """Updating status of nonexistent experiment raises KeyError."""
        with pytest.raises(KeyError):
            backend.update_experiment_status(
                "nonexistent", ExperimentStatus.RUNNING
            )

    def test_get_result_nonexistent_raises(
        self, backend: FileStorageBackend
    ) -> None:
        """Getting result for nonexistent experiment raises KeyError."""
        with pytest.raises(KeyError):
            backend.get_experiment_result("nonexistent")


# ---------------------------------------------------------------------------
# Model operations
# ---------------------------------------------------------------------------


class TestFileBackendModels:
    """Model CRUD through FileStorageBackend."""

    def test_register_and_get_meta(
        self,
        backend: FileStorageBackend,
        sample_trained_model: TrainedModel,
    ) -> None:
        """Register a model, then retrieve its metadata."""
        meta = backend.register_model(
            sample_trained_model,
            experiment_id="exp-001",
            metrics={"accuracy": 0.95},
        )
        assert meta.id
        assert meta.algorithm_name == "decision_tree"
        assert meta.task == "classification"
        assert meta.metrics["accuracy"] == 0.95

        loaded = backend.get_model_meta(meta.id)
        assert loaded.id == meta.id
        assert loaded.experiment_id == "exp-001"

    def test_load_model(
        self,
        backend: FileStorageBackend,
        sample_trained_model: TrainedModel,
    ) -> None:
        """Load the actual trained model artifact."""
        meta = backend.register_model(
            sample_trained_model, experiment_id="exp-001"
        )
        loaded = backend.load_model(meta.id)
        assert loaded.algorithm_name == "decision_tree"
        assert loaded.feature_names == ["feature_a", "feature_b"]

        # Verify the model can predict
        X = np.array([[1, 2]])
        prediction = loaded.model.predict(X)
        assert prediction.shape == (1,)

    def test_list_models_empty(self, backend: FileStorageBackend) -> None:
        """Empty registry returns empty list."""
        assert backend.list_models() == []

    def test_list_models(
        self,
        backend: FileStorageBackend,
        sample_trained_model: TrainedModel,
    ) -> None:
        """Registered models appear in list."""
        backend.register_model(sample_trained_model, experiment_id="exp-001")
        models = backend.list_models()
        assert len(models) == 1

    def test_has_model(
        self,
        backend: FileStorageBackend,
        sample_trained_model: TrainedModel,
    ) -> None:
        """has_model returns True for registered, False for unknown."""
        meta = backend.register_model(
            sample_trained_model, experiment_id="exp-001"
        )
        assert backend.has_model(meta.id) is True
        assert backend.has_model("nonexistent") is False

    def test_delete_model(
        self,
        backend: FileStorageBackend,
        sample_trained_model: TrainedModel,
    ) -> None:
        """Delete removes the model."""
        meta = backend.register_model(
            sample_trained_model, experiment_id="exp-001"
        )
        backend.delete_model(meta.id)
        assert backend.has_model(meta.id) is False

    def test_delete_nonexistent_raises(
        self, backend: FileStorageBackend
    ) -> None:
        """Deleting a nonexistent model raises KeyError."""
        with pytest.raises(KeyError):
            backend.delete_model("nonexistent")

    def test_get_meta_nonexistent_raises(
        self, backend: FileStorageBackend
    ) -> None:
        """Getting metadata for a nonexistent model raises KeyError."""
        with pytest.raises(KeyError):
            backend.get_model_meta("nonexistent")
