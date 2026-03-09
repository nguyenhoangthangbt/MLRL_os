"""File-system storage backend — thin adapter over v0.1 registries.

Wraps DatasetRegistry, ModelRegistry, and ExperimentTracker behind the
StorageBackend protocol so that the rest of the codebase can use a single
unified interface regardless of the underlying storage engine.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from mlrl_os.config.defaults import MLRLSettings
from mlrl_os.core.dataset import DatasetMeta, RawDataset
from mlrl_os.core.experiment import ExperimentResult
from mlrl_os.core.types import ExperimentStatus
from mlrl_os.data.registry import DatasetRegistry
from mlrl_os.evaluation.reports import EvaluationReport
from mlrl_os.experiment.tracker import ExperimentTracker
from mlrl_os.models.algorithms.protocol import TrainedModel
from mlrl_os.models.registry import ModelMeta, ModelRegistry


class FileStorageBackend:
    """File-system storage backend wrapping v0.1 registries.

    All state lives on disk under the directories configured in
    :class:`MLRLSettings`. This backend delegates every operation to the
    corresponding v0.1 registry class.
    """

    def __init__(self, settings: MLRLSettings | None = None) -> None:
        self._settings = settings or MLRLSettings()
        self._datasets = DatasetRegistry(self._settings)
        self._models = ModelRegistry(self._settings)
        self._experiments = ExperimentTracker(self._settings)

    # ------------------------------------------------------------------
    # Dataset operations
    # ------------------------------------------------------------------

    def register_dataset(self, raw: RawDataset, name: str = "") -> DatasetMeta:
        """Register a RawDataset via the underlying DatasetRegistry."""
        return self._datasets.register(raw, name=name)

    def get_dataset_meta(self, dataset_id: str) -> DatasetMeta:
        """Load dataset metadata from the file-system registry."""
        return self._datasets.get_meta(dataset_id)

    def get_dataset_data(
        self, dataset_id: str, layer: str = "snapshots"
    ) -> pl.DataFrame:
        """Load dataset data (snapshots or trajectories) as a Polars DataFrame."""
        return self._datasets.get_data(dataset_id, layer=layer)

    def list_datasets(self) -> list[DatasetMeta]:
        """List all registered datasets, sorted by registration date."""
        return self._datasets.list_datasets()

    def has_dataset(self, dataset_id: str) -> bool:
        """Check if a dataset is registered."""
        return self._datasets.has(dataset_id)

    def delete_dataset(self, dataset_id: str) -> None:
        """Delete a dataset and all its files."""
        self._datasets.delete(dataset_id)

    # ------------------------------------------------------------------
    # Experiment operations
    # ------------------------------------------------------------------

    def create_experiment(
        self, experiment_id: str, name: str, config_dict: dict[str, Any]
    ) -> None:
        """Create a new experiment record with PENDING status."""
        self._experiments.create(experiment_id, name, config_dict)

    def save_experiment_result(self, result: ExperimentResult) -> None:
        """Save (or update) the experiment result."""
        self._experiments.save_result(result)

    def save_experiment_report(
        self, experiment_id: str, report: EvaluationReport
    ) -> None:
        """Save an evaluation report for the experiment."""
        self._experiments.save_report(experiment_id, report)

    def get_experiment_result(self, experiment_id: str) -> ExperimentResult:
        """Load experiment result."""
        return self._experiments.get_result(experiment_id)

    def get_experiment_report(self, experiment_id: str) -> EvaluationReport:
        """Load evaluation report."""
        return self._experiments.get_report(experiment_id)

    def list_experiments(self) -> list[ExperimentResult]:
        """List all experiments, sorted by created_at descending."""
        return self._experiments.list_experiments()

    def has_experiment(self, experiment_id: str) -> bool:
        """Check if an experiment exists."""
        return self._experiments.has(experiment_id)

    def update_experiment_status(
        self, experiment_id: str, status: ExperimentStatus
    ) -> None:
        """Update experiment status."""
        self._experiments.update_status(experiment_id, status)

    # ------------------------------------------------------------------
    # Model operations
    # ------------------------------------------------------------------

    def register_model(
        self,
        trained_model: TrainedModel,
        experiment_id: str,
        metrics: dict[str, float] | None = None,
    ) -> ModelMeta:
        """Save a trained model and register its metadata."""
        return self._models.register(
            trained_model, experiment_id, metrics=metrics
        )

    def get_model_meta(self, model_id: str) -> ModelMeta:
        """Load metadata for a registered model."""
        return self._models.get_meta(model_id)

    def load_model(self, model_id: str) -> TrainedModel:
        """Load the actual trained model artifact."""
        return self._models.load_model(model_id)

    def list_models(self) -> list[ModelMeta]:
        """List all registered models, sorted by creation time (newest first)."""
        return self._models.list_models()

    def has_model(self, model_id: str) -> bool:
        """Check if a model is registered."""
        return self._models.has(model_id)

    def delete_model(self, model_id: str) -> None:
        """Delete a model and all its files."""
        self._models.delete(model_id)
