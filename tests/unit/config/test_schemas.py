"""Tests for mlrl_os.config.schemas — resolved config Pydantic models."""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ts_features(**overrides: object) -> ResolvedTimeSeriesFeatures:
    defaults = dict(
        target="avg_wait",
        lookback="8h",
        horizon="1h",
        lag_intervals=["1h", "2h", "4h"],
        rolling_windows=["2h", "4h"],
        include_trend=True,
        include_ratios=True,
        include_cross_node=False,
        feature_columns=["wip", "throughput"],
    )
    defaults.update(overrides)
    return ResolvedTimeSeriesFeatures(**defaults)


def _make_entity_features(**overrides: object) -> ResolvedEntityFeatures:
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


def _make_cv(**overrides: object) -> ResolvedCVConfig:
    defaults = dict(strategy=CVStrategy.TEMPORAL, folds=5)
    defaults.update(overrides)
    return ResolvedCVConfig(**defaults)


def _make_model(**overrides: object) -> ResolvedModelConfig:
    defaults = dict(
        algorithms=["lightgbm"],
        selection="best_cv",
        cross_validation=_make_cv(),
        handle_imbalance=False,
        hyperparameter_tuning=False,
        n_trials=20,
    )
    defaults.update(overrides)
    return ResolvedModelConfig(**defaults)


def _make_evaluation(**overrides: object) -> ResolvedEvaluationConfig:
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


def _make_experiment_config(
    problem_type: ProblemType = ProblemType.TIME_SERIES,
    **overrides: object,
) -> ResolvedExperimentConfig:
    if problem_type == ProblemType.TIME_SERIES:
        features = _make_ts_features()
    else:
        features = _make_entity_features()
    defaults = dict(
        name="test_experiment",
        experiment_type=problem_type,
        seed=42,
        dataset_id="ds_001",
        dataset_layer="snapshots",
        features=features,
        model=_make_model(),
        evaluation=_make_evaluation(),
    )
    defaults.update(overrides)
    return ResolvedExperimentConfig(**defaults)


# ---------------------------------------------------------------------------
# Tests: construction
# ---------------------------------------------------------------------------


class TestResolvedTimeSeriesFeatures:
    def test_construction(self) -> None:
        feat = _make_ts_features()
        assert feat.target == "avg_wait"
        assert feat.lookback == "8h"
        assert feat.horizon == "1h"
        assert feat.lag_intervals == ["1h", "2h", "4h"]
        assert feat.rolling_windows == ["2h", "4h"]
        assert feat.include_trend is True
        assert feat.include_ratios is True
        assert feat.include_cross_node is False
        assert feat.feature_columns == ["wip", "throughput"]
        assert feat.exclude_columns == []

    def test_exclude_columns_override(self) -> None:
        feat = _make_ts_features(exclude_columns=["bucket_idx"])
        assert feat.exclude_columns == ["bucket_idx"]


class TestResolvedEntityFeatures:
    def test_construction(self) -> None:
        feat = _make_entity_features()
        assert feat.target == "status"
        assert feat.observation_point == ObservationPoint.ALL_STEPS
        assert feat.include_entity_state is True
        assert feat.include_node_state is True
        assert feat.include_system_state is True
        assert feat.add_progress_ratio is True
        assert feat.add_wait_trend is True
        assert feat.feature_columns == ["s_elapsed", "s_cum_wait"]
        assert feat.exclude_columns == []


class TestResolvedCVConfig:
    def test_construction(self) -> None:
        cv = _make_cv()
        assert cv.strategy == CVStrategy.TEMPORAL
        assert cv.folds == 5

    def test_stratified(self) -> None:
        cv = _make_cv(strategy=CVStrategy.STRATIFIED_KFOLD, folds=10)
        assert cv.strategy == CVStrategy.STRATIFIED_KFOLD
        assert cv.folds == 10


class TestResolvedModelConfig:
    def test_construction(self) -> None:
        model = _make_model()
        assert model.algorithms == ["lightgbm"]
        assert model.selection == "best_cv"
        assert model.handle_imbalance is False
        assert model.hyperparameter_tuning is False
        assert model.n_trials == 20
        assert model.cross_validation.strategy == CVStrategy.TEMPORAL

    def test_custom_n_trials(self) -> None:
        model = _make_model(n_trials=50)
        assert model.n_trials == 50


class TestResolvedEvaluationConfig:
    def test_construction(self) -> None:
        evl = _make_evaluation()
        assert evl.metrics == ["rmse", "mae"]
        assert evl.generate_report is True
        assert evl.plot_predictions is True
        assert evl.plot_feature_importance is True
        assert evl.plot_confusion_matrix is False
        assert evl.plot_roc_curve is False


class TestResolvedExperimentConfig:
    def test_construction_ts(self) -> None:
        cfg = _make_experiment_config(ProblemType.TIME_SERIES)
        assert cfg.name == "test_experiment"
        assert cfg.experiment_type == ProblemType.TIME_SERIES
        assert cfg.seed == 42
        assert cfg.dataset_id == "ds_001"
        assert cfg.dataset_layer == "snapshots"
        assert isinstance(cfg.features, ResolvedTimeSeriesFeatures)

    def test_construction_entity(self) -> None:
        cfg = _make_experiment_config(ProblemType.ENTITY_CLASSIFICATION)
        assert cfg.experiment_type == ProblemType.ENTITY_CLASSIFICATION
        assert isinstance(cfg.features, ResolvedEntityFeatures)


# ---------------------------------------------------------------------------
# Tests: serialization round-trip
# ---------------------------------------------------------------------------


class TestSerializationRoundTrip:
    def test_ts_config_roundtrip(self) -> None:
        original = _make_experiment_config(ProblemType.TIME_SERIES)
        data = original.model_dump()
        restored = ResolvedExperimentConfig.model_validate(data)
        assert restored.name == original.name
        assert restored.experiment_type == original.experiment_type
        assert restored.seed == original.seed
        assert restored.dataset_id == original.dataset_id
        assert restored.features.target == original.features.target
        assert isinstance(restored.features, ResolvedTimeSeriesFeatures)
        assert restored.model.algorithms == original.model.algorithms
        assert restored.evaluation.metrics == original.evaluation.metrics

    def test_entity_config_roundtrip(self) -> None:
        original = _make_experiment_config(ProblemType.ENTITY_CLASSIFICATION)
        data = original.model_dump()
        restored = ResolvedExperimentConfig.model_validate(data)
        assert restored.experiment_type == original.experiment_type
        assert isinstance(restored.features, ResolvedEntityFeatures)
        assert restored.features.target == original.features.target

    def test_model_dump_returns_dict(self) -> None:
        cfg = _make_experiment_config()
        data = cfg.model_dump()
        assert isinstance(data, dict)
        assert "name" in data
        assert "features" in data
        assert "model" in data
        assert "evaluation" in data

    def test_cv_config_roundtrip(self) -> None:
        original = _make_cv(strategy=CVStrategy.STRATIFIED_KFOLD, folds=10)
        data = original.model_dump()
        restored = ResolvedCVConfig.model_validate(data)
        assert restored.strategy == original.strategy
        assert restored.folds == original.folds

    def test_evaluation_roundtrip(self) -> None:
        original = _make_evaluation(
            metrics=["f1_weighted", "auc_roc"],
            plot_confusion_matrix=True,
            plot_roc_curve=True,
        )
        data = original.model_dump()
        restored = ResolvedEvaluationConfig.model_validate(data)
        assert restored.metrics == original.metrics
        assert restored.plot_confusion_matrix is True
        assert restored.plot_roc_curve is True
