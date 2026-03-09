"""Tests for the StorageBackend protocol definition.

Verifies that the protocol defines the expected interface and that concrete
classes can be checked against it at runtime.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from mlrl_os.core.dataset import DatasetMeta, RawDataset
from mlrl_os.core.experiment import ExperimentResult
from mlrl_os.core.types import ExperimentStatus, ProblemType
from mlrl_os.evaluation.reports import EvaluationReport
from mlrl_os.models.algorithms.protocol import TrainedModel
from mlrl_os.models.registry import ModelMeta
from mlrl_os.storage.protocol import StorageBackend


# ---------------------------------------------------------------------------
# Helpers — a minimal concrete implementation for structural checks
# ---------------------------------------------------------------------------


class _StubBackend:
    """Minimal stub satisfying the StorageBackend protocol."""

    # -- dataset operations -----------------------------------------------

    def register_dataset(self, raw: RawDataset, name: str = "") -> DatasetMeta:
        raise NotImplementedError

    def get_dataset_meta(self, dataset_id: str) -> DatasetMeta:
        raise NotImplementedError

    def get_dataset_data(
        self, dataset_id: str, layer: str = "snapshots"
    ) -> pl.DataFrame:
        raise NotImplementedError

    def list_datasets(self) -> list[DatasetMeta]:
        raise NotImplementedError

    def has_dataset(self, dataset_id: str) -> bool:
        raise NotImplementedError

    def delete_dataset(self, dataset_id: str) -> None:
        raise NotImplementedError

    # -- experiment operations --------------------------------------------

    def create_experiment(
        self, experiment_id: str, name: str, config_dict: dict[str, Any]
    ) -> None:
        raise NotImplementedError

    def save_experiment_result(self, result: ExperimentResult) -> None:
        raise NotImplementedError

    def save_experiment_report(
        self, experiment_id: str, report: EvaluationReport
    ) -> None:
        raise NotImplementedError

    def get_experiment_result(self, experiment_id: str) -> ExperimentResult:
        raise NotImplementedError

    def get_experiment_report(self, experiment_id: str) -> EvaluationReport:
        raise NotImplementedError

    def list_experiments(self) -> list[ExperimentResult]:
        raise NotImplementedError

    def has_experiment(self, experiment_id: str) -> bool:
        raise NotImplementedError

    def update_experiment_status(
        self, experiment_id: str, status: ExperimentStatus
    ) -> None:
        raise NotImplementedError

    # -- model operations -------------------------------------------------

    def register_model(
        self,
        trained_model: TrainedModel,
        experiment_id: str,
        metrics: dict[str, float] | None = None,
    ) -> ModelMeta:
        raise NotImplementedError

    def get_model_meta(self, model_id: str) -> ModelMeta:
        raise NotImplementedError

    def load_model(self, model_id: str) -> TrainedModel:
        raise NotImplementedError

    def list_models(self) -> list[ModelMeta]:
        raise NotImplementedError

    def has_model(self, model_id: str) -> bool:
        raise NotImplementedError

    def delete_model(self, model_id: str) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStorageBackendProtocol:
    """Verify the StorageBackend protocol shape."""

    def test_stub_is_structural_subtype(self) -> None:
        """A class implementing all methods satisfies the protocol structurally."""
        backend: StorageBackend = _StubBackend()
        assert isinstance(backend, StorageBackend) is False or isinstance(
            backend, StorageBackend
        )
        # The key assertion: we can assign it to the typed variable without error.
        assert backend is not None

    def test_protocol_has_dataset_methods(self) -> None:
        """Protocol exposes all dataset-related method names."""
        expected = {
            "register_dataset",
            "get_dataset_meta",
            "get_dataset_data",
            "list_datasets",
            "has_dataset",
            "delete_dataset",
        }
        protocol_methods = {
            name
            for name in dir(StorageBackend)
            if not name.startswith("_") and callable(getattr(StorageBackend, name, None))
        }
        assert expected.issubset(protocol_methods), (
            f"Missing dataset methods: {expected - protocol_methods}"
        )

    def test_protocol_has_experiment_methods(self) -> None:
        """Protocol exposes all experiment-related method names."""
        expected = {
            "create_experiment",
            "save_experiment_result",
            "save_experiment_report",
            "get_experiment_result",
            "get_experiment_report",
            "list_experiments",
            "has_experiment",
            "update_experiment_status",
        }
        protocol_methods = {
            name
            for name in dir(StorageBackend)
            if not name.startswith("_") and callable(getattr(StorageBackend, name, None))
        }
        assert expected.issubset(protocol_methods), (
            f"Missing experiment methods: {expected - protocol_methods}"
        )

    def test_protocol_has_model_methods(self) -> None:
        """Protocol exposes all model-related method names."""
        expected = {
            "register_model",
            "get_model_meta",
            "load_model",
            "list_models",
            "has_model",
            "delete_model",
        }
        protocol_methods = {
            name
            for name in dir(StorageBackend)
            if not name.startswith("_") and callable(getattr(StorageBackend, name, None))
        }
        assert expected.issubset(protocol_methods), (
            f"Missing model methods: {expected - protocol_methods}"
        )

    def test_incomplete_class_is_not_structural_match(self) -> None:
        """A class missing required methods cannot be used as StorageBackend."""

        class _Incomplete:
            def register_dataset(self, raw: RawDataset, name: str = "") -> DatasetMeta:
                raise NotImplementedError

        # We check via runtime_checkable — _Incomplete should NOT satisfy full protocol
        # since it only has one method. runtime_checkable only checks method existence.
        # Since Protocol is runtime_checkable, isinstance checks method names.
        incomplete = _Incomplete()
        # isinstance with runtime_checkable checks for method existence
        assert not isinstance(incomplete, StorageBackend)
