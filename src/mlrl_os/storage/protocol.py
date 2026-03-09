"""StorageBackend protocol — interface all storage backends must implement.

Defines a unified protocol covering dataset, experiment, and model operations.
Concrete backends (file-system, PostgreSQL) implement this protocol so that
the rest of the codebase is storage-agnostic.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import polars as pl

from mlrl_os.core.dataset import DatasetMeta, RawDataset
from mlrl_os.core.experiment import ExperimentResult
from mlrl_os.core.types import ExperimentStatus
from mlrl_os.evaluation.reports import EvaluationReport
from mlrl_os.models.algorithms.protocol import TrainedModel
from mlrl_os.models.registry import ModelMeta


@runtime_checkable
class StorageBackend(Protocol):
    """Unified storage interface for datasets, experiments, and models.

    All methods raise ``KeyError`` when a requested entity does not exist,
    and ``ValueError`` for invalid arguments (e.g. bad layer name).
    """

    # ------------------------------------------------------------------
    # Dataset operations
    # ------------------------------------------------------------------

    def register_dataset(self, raw: RawDataset, name: str = "") -> DatasetMeta:
        """Register a RawDataset and persist its data.

        Returns:
            DatasetMeta with generated ID, content hash, etc.
        """
        ...

    def get_dataset_meta(self, dataset_id: str) -> DatasetMeta:
        """Load metadata for a registered dataset.

        Raises:
            KeyError: If the dataset is not registered.
        """
        ...

    def get_dataset_data(
        self, dataset_id: str, layer: str = "snapshots"
    ) -> pl.DataFrame:
        """Load the actual data (snapshots or trajectories) as a DataFrame.

        Args:
            dataset_id: The dataset identifier.
            layer: ``"snapshots"`` or ``"trajectories"``.

        Raises:
            KeyError: If the dataset is not registered.
            ValueError: If *layer* is invalid or missing for this dataset.
        """
        ...

    def list_datasets(self) -> list[DatasetMeta]:
        """List all registered datasets, sorted by registration date."""
        ...

    def has_dataset(self, dataset_id: str) -> bool:
        """Check if a dataset is registered."""
        ...

    def delete_dataset(self, dataset_id: str) -> None:
        """Delete a dataset and all its files.

        Raises:
            KeyError: If the dataset is not registered.
        """
        ...

    # ------------------------------------------------------------------
    # Experiment operations
    # ------------------------------------------------------------------

    def create_experiment(
        self, experiment_id: str, name: str, config_dict: dict[str, Any]
    ) -> None:
        """Create a new experiment record with PENDING status.

        Raises:
            FileExistsError: If the experiment already exists.
        """
        ...

    def save_experiment_result(self, result: ExperimentResult) -> None:
        """Save (or update) the experiment result.

        Raises:
            KeyError: If the experiment does not exist.
        """
        ...

    def save_experiment_report(
        self, experiment_id: str, report: EvaluationReport
    ) -> None:
        """Save an evaluation report for the experiment.

        Raises:
            KeyError: If the experiment does not exist.
        """
        ...

    def get_experiment_result(self, experiment_id: str) -> ExperimentResult:
        """Load experiment result.

        Raises:
            KeyError: If the experiment does not exist.
        """
        ...

    def get_experiment_report(self, experiment_id: str) -> EvaluationReport:
        """Load evaluation report.

        Raises:
            KeyError: If the experiment or its report does not exist.
        """
        ...

    def list_experiments(self) -> list[ExperimentResult]:
        """List all experiments, sorted by created_at descending."""
        ...

    def has_experiment(self, experiment_id: str) -> bool:
        """Check if an experiment exists."""
        ...

    def update_experiment_status(
        self, experiment_id: str, status: ExperimentStatus
    ) -> None:
        """Update experiment status.

        Raises:
            KeyError: If the experiment does not exist.
        """
        ...

    # ------------------------------------------------------------------
    # Model operations
    # ------------------------------------------------------------------

    def register_model(
        self,
        trained_model: TrainedModel,
        experiment_id: str,
        metrics: dict[str, float] | None = None,
    ) -> ModelMeta:
        """Save a trained model and register its metadata.

        Returns:
            ModelMeta with the generated model ID.
        """
        ...

    def get_model_meta(self, model_id: str) -> ModelMeta:
        """Load metadata for a registered model.

        Raises:
            KeyError: If no model with *model_id* is registered.
        """
        ...

    def load_model(self, model_id: str) -> TrainedModel:
        """Load the actual trained model artifact.

        Raises:
            KeyError: If no model with *model_id* is registered.
        """
        ...

    def list_models(self) -> list[ModelMeta]:
        """List all registered models, sorted by creation time (newest first)."""
        ...

    def has_model(self, model_id: str) -> bool:
        """Check if a model is registered."""
        ...

    def delete_model(self, model_id: str) -> None:
        """Delete a model and all its files.

        Raises:
            KeyError: If no model with *model_id* is registered.
        """
        ...
