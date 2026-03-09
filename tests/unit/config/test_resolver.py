"""Tests for mlrl_os.config.resolver — config resolution and duration parsing."""

from __future__ import annotations

import re

import pytest

from mlrl_os.config.defaults import MLRLSettings
from mlrl_os.config.resolver import ConfigResolver, parse_duration
from mlrl_os.config.schemas import (
    ResolvedEntityFeatures,
    ResolvedExperimentConfig,
    ResolvedTimeSeriesFeatures,
)
from mlrl_os.core.dataset import DatasetMeta
from mlrl_os.core.types import CVStrategy, ProblemType


# ---------------------------------------------------------------------------
# parse_duration
# ---------------------------------------------------------------------------


class TestParseDuration:
    def test_seconds(self) -> None:
        assert parse_duration("30s") == 30

    def test_minutes(self) -> None:
        assert parse_duration("5m") == 300

    def test_hours(self) -> None:
        assert parse_duration("1h") == 3600

    def test_days(self) -> None:
        assert parse_duration("1d") == 86400

    def test_fractional_hours(self) -> None:
        assert parse_duration("1.5h") == 5400

    def test_uppercase(self) -> None:
        assert parse_duration("2H") == 7200

    def test_with_whitespace(self) -> None:
        assert parse_duration("  3m  ") == 180

    def test_invalid_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid duration format"):
            parse_duration("abc")

    def test_no_unit_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid duration format"):
            parse_duration("100")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid duration format"):
            parse_duration("")


# ---------------------------------------------------------------------------
# ConfigResolver.resolve
# ---------------------------------------------------------------------------


class TestConfigResolverResolve:
    """Tests for ConfigResolver.resolve()."""

    def test_zero_config_uses_defaults(
        self,
        test_settings: MLRLSettings,
        snapshot_dataset_meta: DatasetMeta,
    ) -> None:
        """Empty user config should produce a valid resolved config from defaults."""
        resolver = ConfigResolver(test_settings)
        result = resolver.resolve({}, snapshot_dataset_meta)

        assert isinstance(result, ResolvedExperimentConfig)
        assert result.experiment_type == ProblemType.TIME_SERIES
        assert result.seed == test_settings.seed_default
        assert isinstance(result.features, ResolvedTimeSeriesFeatures)
        assert result.features.target == "avg_wait"
        assert len(result.model.algorithms) > 0
        assert len(result.evaluation.metrics) > 0

    def test_explicit_experiment_type(
        self,
        test_settings: MLRLSettings,
        snapshot_dataset_meta: DatasetMeta,
    ) -> None:
        """Explicit experiment_type in user config should override auto-detection."""
        resolver = ConfigResolver(test_settings)
        # Even though dataset is snapshot-only, user can force entity classification
        user_config = {"experiment_type": "entity_classification"}
        result = resolver.resolve(user_config, snapshot_dataset_meta)
        assert result.experiment_type == ProblemType.ENTITY_CLASSIFICATION
        assert isinstance(result.features, ResolvedEntityFeatures)

    def test_auto_detects_time_series_from_snapshot_only(
        self,
        test_settings: MLRLSettings,
        snapshot_dataset_meta: DatasetMeta,
    ) -> None:
        """Snapshot-only dataset should auto-detect as TIME_SERIES."""
        resolver = ConfigResolver(test_settings)
        result = resolver.resolve({}, snapshot_dataset_meta)
        assert result.experiment_type == ProblemType.TIME_SERIES

    def test_auto_detects_entity_from_trajectory_only(
        self,
        test_settings: MLRLSettings,
        trajectory_dataset_meta: DatasetMeta,
    ) -> None:
        """Trajectory-only dataset should auto-detect as ENTITY_CLASSIFICATION."""
        resolver = ConfigResolver(test_settings)
        result = resolver.resolve({}, trajectory_dataset_meta)
        assert result.experiment_type == ProblemType.ENTITY_CLASSIFICATION
        assert isinstance(result.features, ResolvedEntityFeatures)

    def test_user_overrides_win(
        self,
        test_settings: MLRLSettings,
        snapshot_dataset_meta: DatasetMeta,
    ) -> None:
        """User overrides should take precedence over defaults."""
        resolver = ConfigResolver(test_settings)
        user_config = {
            "target": "throughput",
            "algorithms": ["random_forest"],
            "seed": 123,
        }
        result = resolver.resolve(user_config, snapshot_dataset_meta)
        assert result.features.target == "throughput"
        assert result.model.algorithms == ["random_forest"]
        assert result.seed == 123

    def test_resolved_config_has_no_none_fields(
        self,
        test_settings: MLRLSettings,
        snapshot_dataset_meta: DatasetMeta,
    ) -> None:
        """After resolution, no field should be None."""
        resolver = ConfigResolver(test_settings)
        result = resolver.resolve({}, snapshot_dataset_meta)
        data = result.model_dump()
        self._assert_no_none(data, path="")

    def test_generated_experiment_name_pattern(
        self,
        test_settings: MLRLSettings,
        snapshot_dataset_meta: DatasetMeta,
    ) -> None:
        """Auto-generated name should match experiment_YYYYMMDD_HHMMSS pattern."""
        resolver = ConfigResolver(test_settings)
        result = resolver.resolve({}, snapshot_dataset_meta)
        pattern = r"^experiment_\d{8}_\d{6}$"
        assert re.match(pattern, result.name), f"Name '{result.name}' does not match pattern"

    def test_custom_name_preserved(
        self,
        test_settings: MLRLSettings,
        snapshot_dataset_meta: DatasetMeta,
    ) -> None:
        """User-supplied name should be used as-is."""
        resolver = ConfigResolver(test_settings)
        result = resolver.resolve({"name": "my_experiment"}, snapshot_dataset_meta)
        assert result.name == "my_experiment"

    def test_dataset_id_from_meta(
        self,
        test_settings: MLRLSettings,
        snapshot_dataset_meta: DatasetMeta,
    ) -> None:
        """dataset_id should come from the DatasetMeta."""
        resolver = ConfigResolver(test_settings)
        result = resolver.resolve({}, snapshot_dataset_meta)
        assert result.dataset_id == snapshot_dataset_meta.id

    def test_dataset_layer_defaults_snapshots_for_ts(
        self,
        test_settings: MLRLSettings,
        snapshot_dataset_meta: DatasetMeta,
    ) -> None:
        resolver = ConfigResolver(test_settings)
        result = resolver.resolve({}, snapshot_dataset_meta)
        assert result.dataset_layer == "snapshots"

    def test_dataset_layer_defaults_trajectories_for_entity(
        self,
        test_settings: MLRLSettings,
        trajectory_dataset_meta: DatasetMeta,
    ) -> None:
        resolver = ConfigResolver(test_settings)
        result = resolver.resolve({}, trajectory_dataset_meta)
        assert result.dataset_layer == "trajectories"

    def test_feature_columns_auto_selected(
        self,
        test_settings: MLRLSettings,
        snapshot_dataset_meta: DatasetMeta,
    ) -> None:
        """When no feature_columns given, auto-select numeric columns excluding target and ts."""
        resolver = ConfigResolver(test_settings)
        result = resolver.resolve({}, snapshot_dataset_meta)
        assert isinstance(result.features, ResolvedTimeSeriesFeatures)
        # Should not include 'ts' or the target column
        assert "ts" not in result.features.feature_columns
        assert result.features.target not in result.features.feature_columns
        # Should include other numeric columns
        assert len(result.features.feature_columns) > 0

    # -- helpers ---

    def _assert_no_none(self, data: object, path: str) -> None:
        """Recursively assert that no value in the dict tree is None."""
        if isinstance(data, dict):
            for key, value in data.items():
                self._assert_no_none(value, f"{path}.{key}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                self._assert_no_none(item, f"{path}[{i}]")
        else:
            assert data is not None, f"Found None at {path}"
