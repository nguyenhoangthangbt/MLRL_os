"""Tests for mlrl_os.features.time_series module."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from mlrl_os.core.types import ProblemType, TaskType
from mlrl_os.features.time_series import TimeSeriesFeatureEngine, parse_duration_to_seconds


# ---------------------------------------------------------------------------
# parse_duration_to_seconds
# ---------------------------------------------------------------------------


class TestParseDurationToSeconds:
    """Tests for the parse_duration_to_seconds helper."""

    def test_seconds(self) -> None:
        assert parse_duration_to_seconds("30s") == 30

    def test_minutes(self) -> None:
        assert parse_duration_to_seconds("5m") == 300

    def test_hours(self) -> None:
        assert parse_duration_to_seconds("1h") == 3600

    def test_fractional_hours(self) -> None:
        assert parse_duration_to_seconds("1.5h") == 5400

    def test_days(self) -> None:
        assert parse_duration_to_seconds("1d") == 86400

    def test_invalid_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid duration format"):
            parse_duration_to_seconds("10x")


# ---------------------------------------------------------------------------
# TimeSeriesFeatureEngine.build_features
# ---------------------------------------------------------------------------


class TestTimeSeriesFeatureEngine:
    """Tests for TimeSeriesFeatureEngine.build_features."""

    @pytest.fixture()
    def engine(self) -> TimeSeriesFeatureEngine:
        return TimeSeriesFeatureEngine()

    def test_returns_feature_matrix_with_correct_types(
        self, engine: TimeSeriesFeatureEngine, snapshot_df: pl.DataFrame
    ) -> None:
        fm = engine.build_features(
            snapshot_df,
            target="wip",
            lookback="5m",
            horizon="5m",
            lag_intervals=["5m", "10m"],
            rolling_windows=["5m"],
        )
        assert fm.problem_type == ProblemType.TIME_SERIES
        assert fm.task_type == TaskType.REGRESSION

    def test_feature_names_include_lag_rolling_trend(
        self, engine: TimeSeriesFeatureEngine, snapshot_df: pl.DataFrame
    ) -> None:
        fm = engine.build_features(
            snapshot_df,
            target="wip",
            lookback="5m",
            horizon="5m",
            lag_intervals=["5m", "10m"],
            rolling_windows=["5m"],
            include_trend=True,
        )
        names = fm.feature_names
        assert any("lag" in n for n in names), "Expected lag features"
        assert any("rmean" in n or "rstd" in n for n in names), "Expected rolling features"
        assert any("trend" in n for n in names), "Expected trend features"

    def test_samples_fewer_than_input_rows(
        self, engine: TimeSeriesFeatureEngine, snapshot_df: pl.DataFrame
    ) -> None:
        fm = engine.build_features(
            snapshot_df,
            target="wip",
            lookback="5m",
            horizon="5m",
            lag_intervals=["5m", "10m"],
            rolling_windows=["5m"],
        )
        assert fm.X.shape[0] < len(snapshot_df)

    def test_raises_with_fewer_than_two_rows(
        self, engine: TimeSeriesFeatureEngine
    ) -> None:
        tiny = pl.DataFrame({"ts": [300.0], "wip": [5]})
        with pytest.raises(ValueError, match="at least 2"):
            engine.build_features(
                tiny,
                target="wip",
                lookback="5m",
                horizon="5m",
                lag_intervals=["5m"],
                rolling_windows=["5m"],
            )

    def test_raises_when_window_too_large(
        self, engine: TimeSeriesFeatureEngine
    ) -> None:
        # 5 rows at 300s intervals = 1500s total data.
        # lookback 1h (3600s) + horizon 1h (3600s) >> data span → no valid samples.
        small = pl.DataFrame({
            "ts": [300.0, 600.0, 900.0, 1200.0, 1500.0],
            "wip": [1, 2, 3, 4, 5],
        })
        with pytest.raises(ValueError, match="No valid samples"):
            engine.build_features(
                small,
                target="wip",
                lookback="1h",
                horizon="1h",
                lag_intervals=["1h"],
                rolling_windows=["1h"],
            )

    def test_feature_matrix_shape(
        self, engine: TimeSeriesFeatureEngine, snapshot_df: pl.DataFrame
    ) -> None:
        fm = engine.build_features(
            snapshot_df,
            target="wip",
            lookback="5m",
            horizon="5m",
            lag_intervals=["5m", "10m"],
            rolling_windows=["5m"],
        )
        n_samples, n_features = fm.X.shape
        assert n_samples > 0
        assert n_features == len(fm.feature_names)
        assert fm.y.shape == (n_samples,)

    def test_temporal_index_populated(
        self, engine: TimeSeriesFeatureEngine, snapshot_df: pl.DataFrame
    ) -> None:
        fm = engine.build_features(
            snapshot_df,
            target="wip",
            lookback="5m",
            horizon="5m",
            lag_intervals=["5m", "10m"],
            rolling_windows=["5m"],
        )
        assert fm.temporal_index is not None
        assert len(fm.temporal_index) == fm.X.shape[0]

    def test_minimal_parameters(
        self, engine: TimeSeriesFeatureEngine, snapshot_df: pl.DataFrame
    ) -> None:
        """Build features with trend, ratios, and cross-node disabled."""
        fm = engine.build_features(
            snapshot_df,
            target="wip",
            lookback="5m",
            horizon="5m",
            lag_intervals=["5m"],
            rolling_windows=["5m"],
            include_trend=False,
            include_ratios=False,
            include_cross_node=False,
        )
        assert fm.X.shape[0] > 0
        assert not any("trend" in n for n in fm.feature_names)
