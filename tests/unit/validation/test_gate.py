"""Tests for mlrl_os.validation.gate — validation rules V-02 through VE-03."""

from __future__ import annotations

import polars as pl
import pytest

from mlrl_os.config.schemas import (
    ResolvedCVConfig,
    ResolvedEntityFeatures,
    ResolvedEvaluationConfig,
    ResolvedExperimentConfig,
    ResolvedModelConfig,
    ResolvedTimeSeriesFeatures,
)
from mlrl_os.core.types import CVStrategy, ObservationPoint, ProblemType
from mlrl_os.validation.gate import ValidationGate


# ---------------------------------------------------------------------------
# Helpers: build configs and DataFrames for tests
# ---------------------------------------------------------------------------


def _ts_features(**overrides: object) -> ResolvedTimeSeriesFeatures:
    defaults = dict(
        target="avg_wait",
        lookback="5m",
        horizon="5m",
        lag_intervals=["5m"],
        rolling_windows=["5m"],
        include_trend=True,
        include_ratios=True,
        include_cross_node=False,
        feature_columns=["wip", "throughput"],
    )
    defaults.update(overrides)
    return ResolvedTimeSeriesFeatures(**defaults)


def _entity_features(**overrides: object) -> ResolvedEntityFeatures:
    defaults = dict(
        target="status",
        observation_point=ObservationPoint.ALL_STEPS,
        include_entity_state=True,
        include_node_state=True,
        include_system_state=True,
        add_progress_ratio=True,
        add_wait_trend=True,
        feature_columns=["s_elapsed", "s_cum_wait"],
    )
    defaults.update(overrides)
    return ResolvedEntityFeatures(**defaults)


def _model_config(**overrides: object) -> ResolvedModelConfig:
    defaults = dict(
        algorithms=["lightgbm"],
        selection="best_cv",
        cross_validation=ResolvedCVConfig(strategy=CVStrategy.TEMPORAL, folds=5),
        handle_imbalance=False,
        hyperparameter_tuning=False,
    )
    defaults.update(overrides)
    return ResolvedModelConfig(**defaults)


def _eval_config(**overrides: object) -> ResolvedEvaluationConfig:
    defaults = dict(
        metrics=["rmse", "mae"],
        generate_report=True,
        plot_predictions=True,
        plot_feature_importance=True,
        plot_confusion_matrix=False,
        plot_roc_curve=False,
    )
    defaults.update(overrides)
    return ResolvedEvaluationConfig(**defaults)


def _ts_config(**overrides: object) -> ResolvedExperimentConfig:
    defaults = dict(
        name="test_ts",
        experiment_type=ProblemType.TIME_SERIES,
        seed=42,
        dataset_id="ds_001",
        dataset_layer="snapshots",
        features=_ts_features(),
        model=_model_config(),
        evaluation=_eval_config(),
    )
    defaults.update(overrides)
    return ResolvedExperimentConfig(**defaults)


def _entity_config(**overrides: object) -> ResolvedExperimentConfig:
    defaults = dict(
        name="test_entity",
        experiment_type=ProblemType.ENTITY_CLASSIFICATION,
        seed=42,
        dataset_id="ds_001",
        dataset_layer="trajectories",
        features=_entity_features(),
        model=_model_config(
            cross_validation=ResolvedCVConfig(
                strategy=CVStrategy.STRATIFIED_KFOLD, folds=5,
            ),
        ),
        evaluation=_eval_config(metrics=["f1_weighted", "auc_roc"]),
    )
    defaults.update(overrides)
    return ResolvedExperimentConfig(**defaults)


def _ts_df(n: int = 100) -> pl.DataFrame:
    """Create a valid time-series DataFrame with n rows."""
    import numpy as np

    rng = np.random.RandomState(0)
    return pl.DataFrame({
        "ts": np.arange(0, n * 300, 300, dtype=float).tolist(),
        "wip": rng.randint(1, 20, size=n).tolist(),
        "throughput": rng.uniform(2, 12, size=n).round(2).tolist(),
        "avg_wait": rng.uniform(10, 120, size=n).round(2).tolist(),
    })


def _entity_df(n: int = 200) -> pl.DataFrame:
    """Create a valid entity classification DataFrame with n rows."""
    import numpy as np

    rng = np.random.RandomState(0)
    statuses = ["completed", "delayed", "failed"]
    return pl.DataFrame({
        "status": [statuses[i % len(statuses)] for i in range(n)],
        "s_elapsed": rng.uniform(50, 500, size=n).round(2).tolist(),
        "s_cum_wait": rng.uniform(10, 150, size=n).round(2).tolist(),
    })


# ---------------------------------------------------------------------------
# Tests: valid configs pass
# ---------------------------------------------------------------------------


class TestValidConfigs:
    def test_valid_ts_config_passes(self) -> None:
        gate = ValidationGate()
        result = gate.validate(_ts_config(), _ts_df(100))
        assert result.valid is True
        assert result.errors == []

    def test_valid_entity_config_passes(self) -> None:
        gate = ValidationGate()
        result = gate.validate(_entity_config(), _entity_df(200))
        assert result.valid is True
        assert result.errors == []


# ---------------------------------------------------------------------------
# Tests: universal validation rules
# ---------------------------------------------------------------------------


class TestV02MinRows:
    def test_ts_too_few_rows(self) -> None:
        gate = ValidationGate()
        result = gate.validate(_ts_config(), _ts_df(10))
        codes = [e.code for e in result.errors]
        assert "V-02" in codes
        assert result.valid is False

    def test_entity_too_few_rows(self) -> None:
        gate = ValidationGate()
        result = gate.validate(_entity_config(), _entity_df(50))
        codes = [e.code for e in result.errors]
        assert "V-02" in codes


class TestV03MissingTarget:
    def test_missing_target_column(self) -> None:
        gate = ValidationGate()
        config = _ts_config(features=_ts_features(target="nonexistent_col"))
        result = gate.validate(config, _ts_df())
        codes = [e.code for e in result.errors]
        assert "V-03" in codes


class TestV04TargetInFeatures:
    def test_target_in_feature_columns(self) -> None:
        gate = ValidationGate()
        config = _ts_config(
            features=_ts_features(feature_columns=["wip", "avg_wait"]),
        )
        result = gate.validate(config, _ts_df())
        codes = [e.code for e in result.errors]
        assert "V-04" in codes


class TestV05MissingFeatureColumn:
    def test_feature_column_not_in_data(self) -> None:
        gate = ValidationGate()
        config = _ts_config(
            features=_ts_features(feature_columns=["wip", "ghost_column"]),
        )
        result = gate.validate(config, _ts_df())
        codes = [e.code for e in result.errors]
        assert "V-05" in codes


class TestV06NonNumericFeature:
    def test_non_numeric_feature_column(self) -> None:
        gate = ValidationGate()
        df = _ts_df()
        # Add a string column and reference it as a feature
        df = df.with_columns(pl.lit("category_a").alias("label"))
        config = _ts_config(
            features=_ts_features(feature_columns=["wip", "label"]),
        )
        result = gate.validate(config, df)
        codes = [e.code for e in result.errors]
        assert "V-06" in codes


class TestV07NegativeSeed:
    def test_negative_seed(self) -> None:
        gate = ValidationGate()
        config = _ts_config(seed=-1)
        result = gate.validate(config, _ts_df())
        codes = [e.code for e in result.errors]
        assert "V-07" in codes


class TestV08UnknownAlgorithm:
    def test_unknown_algorithm(self) -> None:
        gate = ValidationGate()
        config = _ts_config(
            model=_model_config(algorithms=["lightgbm", "deep_neural_net"]),
        )
        result = gate.validate(config, _ts_df())
        codes = [e.code for e in result.errors]
        assert "V-08" in codes


class TestV09CVFolds:
    def test_cv_folds_less_than_2(self) -> None:
        gate = ValidationGate()
        config = _ts_config(
            model=_model_config(
                cross_validation=ResolvedCVConfig(strategy=CVStrategy.TEMPORAL, folds=1),
            ),
        )
        result = gate.validate(config, _ts_df())
        codes = [e.code for e in result.errors]
        assert "V-09" in codes


class TestV10UnknownMetric:
    def test_unknown_metric(self) -> None:
        gate = ValidationGate()
        config = _ts_config(
            evaluation=_eval_config(metrics=["rmse", "cosmic_loss"]),
        )
        result = gate.validate(config, _ts_df())
        codes = [e.code for e in result.errors]
        assert "V-10" in codes


class TestV11HighNullRate:
    def test_high_null_rate_in_target(self) -> None:
        gate = ValidationGate()
        import numpy as np

        rng = np.random.RandomState(0)
        n = 100
        avg_wait = rng.uniform(10, 120, size=n).round(2).tolist()
        # Set 20% of values to None (above 10% threshold)
        for i in range(20):
            avg_wait[i] = None
        df = pl.DataFrame({
            "ts": list(range(0, n * 300, 300)),
            "wip": rng.randint(1, 20, size=n).tolist(),
            "throughput": rng.uniform(2, 12, size=n).round(2).tolist(),
            "avg_wait": avg_wait,
        })
        result = gate.validate(_ts_config(), df)
        codes = [e.code for e in result.errors]
        assert "V-11" in codes


# ---------------------------------------------------------------------------
# Tests: time-series-specific rules
# ---------------------------------------------------------------------------


class TestVT01MissingTsColumn:
    def test_no_ts_column(self) -> None:
        gate = ValidationGate()
        df = _ts_df().drop("ts")
        result = gate.validate(_ts_config(), df)
        codes = [e.code for e in result.errors]
        assert "VT-01" in codes


class TestVT04NonNumericTsTarget:
    def test_string_target_for_ts(self) -> None:
        gate = ValidationGate()
        df = _ts_df().with_columns(pl.lit("high").alias("avg_wait"))
        result = gate.validate(_ts_config(), df)
        codes = [e.code for e in result.errors]
        assert "VT-04" in codes


class TestVT06NonTemporalCVForTS:
    def test_stratified_cv_rejected_for_ts(self) -> None:
        gate = ValidationGate()
        config = _ts_config(
            model=_model_config(
                cross_validation=ResolvedCVConfig(
                    strategy=CVStrategy.STRATIFIED_KFOLD, folds=5,
                ),
            ),
        )
        result = gate.validate(config, _ts_df())
        codes = [e.code for e in result.errors]
        assert "VT-06" in codes

    def test_kfold_cv_rejected_for_ts(self) -> None:
        gate = ValidationGate()
        config = _ts_config(
            model=_model_config(
                cross_validation=ResolvedCVConfig(
                    strategy=CVStrategy.KFOLD, folds=5,
                ),
            ),
        )
        result = gate.validate(config, _ts_df())
        codes = [e.code for e in result.errors]
        assert "VT-06" in codes


# ---------------------------------------------------------------------------
# Tests: entity-specific rules
# ---------------------------------------------------------------------------


class TestVE01TooManyUniqueValues:
    def test_numeric_target_too_many_unique(self) -> None:
        """Numeric target with >20 unique values should trigger VE-01."""
        import numpy as np

        gate = ValidationGate()
        rng = np.random.RandomState(0)
        n = 200
        df = pl.DataFrame({
            "status": rng.uniform(0, 100, size=n).tolist(),  # 200 unique floats
            "s_elapsed": rng.uniform(50, 500, size=n).round(2).tolist(),
            "s_cum_wait": rng.uniform(10, 150, size=n).round(2).tolist(),
        })
        result = gate.validate(_entity_config(), df)
        codes = [e.code for e in result.errors]
        assert "VE-01" in codes


class TestVE03TooManyClasses:
    def test_more_than_20_classes(self) -> None:
        import numpy as np

        gate = ValidationGate()
        rng = np.random.RandomState(0)
        n = 250
        # 25 distinct string classes
        classes = [f"class_{i}" for i in range(25)]
        df = pl.DataFrame({
            "status": [classes[i % 25] for i in range(n)],
            "s_elapsed": rng.uniform(50, 500, size=n).round(2).tolist(),
            "s_cum_wait": rng.uniform(10, 150, size=n).round(2).tolist(),
        })
        result = gate.validate(_entity_config(), df)
        codes = [e.code for e in result.errors]
        assert "VE-03" in codes


# ---------------------------------------------------------------------------
# Tests: multiple errors collected
# ---------------------------------------------------------------------------


class TestMultipleErrors:
    def test_multiple_errors_collected_at_once(self) -> None:
        """Validation should collect all errors, not stop at the first."""
        gate = ValidationGate()
        # Build a config with multiple problems:
        # - negative seed (V-07)
        # - unknown algorithm (V-08)
        # - cv_folds < 2 (V-09)
        # - unknown metric (V-10)
        config = _ts_config(
            seed=-5,
            model=_model_config(
                algorithms=["unknown_algo"],
                cross_validation=ResolvedCVConfig(strategy=CVStrategy.TEMPORAL, folds=1),
            ),
            evaluation=_eval_config(metrics=["fake_metric"]),
        )
        result = gate.validate(config, _ts_df())
        codes = {e.code for e in result.errors}
        assert "V-07" in codes
        assert "V-08" in codes
        assert "V-09" in codes
        assert "V-10" in codes
        assert len(result.errors) >= 4


# ---------------------------------------------------------------------------
# Tests: warnings
# ---------------------------------------------------------------------------


class TestWarnings:
    def test_cumulative_target_warning(self) -> None:
        gate = ValidationGate()
        df = _ts_df().with_columns(
            pl.col("avg_wait").alias("cumulative_throughput"),
        )
        config = _ts_config(
            features=_ts_features(
                target="cumulative_throughput",
                feature_columns=["wip", "throughput"],
            ),
        )
        result = gate.validate(config, df)
        warning_codes = [w.code for w in result.warnings]
        assert "W-01" in warning_codes
        # Warnings should not make the result invalid
        # (validity depends only on errors)

    def test_non_cumulative_target_no_warning(self) -> None:
        gate = ValidationGate()
        result = gate.validate(_ts_config(), _ts_df())
        warning_codes = [w.code for w in result.warnings]
        assert "W-01" not in warning_codes
