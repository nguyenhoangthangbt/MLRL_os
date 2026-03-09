"""Tests for mlrl_os.features.detection module."""

from __future__ import annotations

import pytest

from mlrl_os.core.dataset import DatasetMeta
from mlrl_os.core.types import ProblemType
from mlrl_os.features.detection import ProblemTypeDetector


@pytest.fixture()
def detector() -> ProblemTypeDetector:
    return ProblemTypeDetector()


def _make_meta(
    *, has_snapshots: bool = False, has_trajectories: bool = False
) -> DatasetMeta:
    """Helper to build a minimal DatasetMeta with snapshot/trajectory flags."""
    return DatasetMeta(
        id="ds_test",
        name="Test",
        content_hash="hash",
        source_type="simos",
        source_path="/fake",
        has_snapshots=has_snapshots,
        has_trajectories=has_trajectories,
        registered_at="2025-01-01T00:00:00Z",
    )


class TestProblemTypeDetector:
    """Tests for ProblemTypeDetector.detect."""

    def test_explicit_experiment_type_wins(
        self, detector: ProblemTypeDetector
    ) -> None:
        meta = _make_meta(has_snapshots=True)
        result = detector.detect(
            meta, user_config={"experiment_type": "entity_classification"}
        )
        assert result == ProblemType.ENTITY_CLASSIFICATION

    def test_explicit_nested_experiment_type(
        self, detector: ProblemTypeDetector
    ) -> None:
        meta = _make_meta(has_snapshots=True)
        result = detector.detect(
            meta, user_config={"experiment": {"type": "entity_classification"}}
        )
        assert result == ProblemType.ENTITY_CLASSIFICATION

    def test_dataset_layer_snapshots_hint(
        self, detector: ProblemTypeDetector
    ) -> None:
        meta = _make_meta(has_snapshots=True, has_trajectories=True)
        result = detector.detect(meta, user_config={"dataset_layer": "snapshots"})
        assert result == ProblemType.TIME_SERIES

    def test_dataset_layer_trajectories_hint(
        self, detector: ProblemTypeDetector
    ) -> None:
        meta = _make_meta(has_snapshots=True, has_trajectories=True)
        result = detector.detect(meta, user_config={"dataset_layer": "trajectories"})
        assert result == ProblemType.ENTITY_CLASSIFICATION

    def test_snapshots_only_returns_time_series(
        self, detector: ProblemTypeDetector
    ) -> None:
        meta = _make_meta(has_snapshots=True, has_trajectories=False)
        assert detector.detect(meta) == ProblemType.TIME_SERIES

    def test_trajectories_only_returns_entity_classification(
        self, detector: ProblemTypeDetector
    ) -> None:
        meta = _make_meta(has_snapshots=False, has_trajectories=True)
        assert detector.detect(meta) == ProblemType.ENTITY_CLASSIFICATION

    def test_both_layers_defaults_to_time_series(
        self, detector: ProblemTypeDetector
    ) -> None:
        meta = _make_meta(has_snapshots=True, has_trajectories=True)
        assert detector.detect(meta) == ProblemType.TIME_SERIES

    def test_neither_layer_raises(
        self, detector: ProblemTypeDetector
    ) -> None:
        meta = _make_meta(has_snapshots=False, has_trajectories=False)
        with pytest.raises(ValueError, match="Cannot determine problem type"):
            detector.detect(meta)

    def test_no_user_config_detects_from_data(
        self, detector: ProblemTypeDetector
    ) -> None:
        meta = _make_meta(has_snapshots=True)
        result = detector.detect(meta, user_config=None)
        assert result == ProblemType.TIME_SERIES
