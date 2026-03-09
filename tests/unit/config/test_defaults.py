"""Tests for mlrl_os.config.defaults — settings and problem-type defaults."""

from __future__ import annotations

import pytest

from mlrl_os.config.defaults import (
    ENTITY_DEFAULTS,
    MLRLSettings,
    TS_DEFAULTS,
    get_defaults,
)
from mlrl_os.core.types import ProblemType


# ---------------------------------------------------------------------------
# MLRLSettings
# ---------------------------------------------------------------------------


class TestMLRLSettings:
    def test_default_values(self) -> None:
        settings = MLRLSettings()
        assert settings.env == "development"
        assert settings.data_dir == "./data"
        assert settings.models_dir == "./models"
        assert settings.experiments_dir == "./experiments"
        assert settings.log_level == "INFO"
        assert settings.api_port == 8001
        assert settings.cors_allow_origins == "http://localhost:5175"
        assert settings.max_training_rows == 1_000_000
        assert settings.cv_folds_default == 5
        assert settings.seed_default == 42

    def test_env_prefix(self) -> None:
        """MLRLSettings should use MLRL_ as env prefix."""
        assert MLRLSettings.model_config["env_prefix"] == "MLRL_"

    def test_custom_values(self) -> None:
        settings = MLRLSettings(
            env="production",
            data_dir="/mnt/data",
            seed_default=99,
            cv_folds_default=10,
        )
        assert settings.env == "production"
        assert settings.data_dir == "/mnt/data"
        assert settings.seed_default == 99
        assert settings.cv_folds_default == 10


# ---------------------------------------------------------------------------
# get_defaults
# ---------------------------------------------------------------------------


class TestGetDefaults:
    def test_time_series_returns_ts_defaults(self) -> None:
        defaults = get_defaults(ProblemType.TIME_SERIES)
        assert defaults["target"] == "avg_wait"
        assert defaults["lookback"] == "8h"
        assert defaults["horizon"] == "1h"

    def test_entity_returns_entity_defaults(self) -> None:
        defaults = get_defaults(ProblemType.ENTITY_CLASSIFICATION)
        assert defaults["target"] == "status"
        assert defaults["observation_point"] == "all_steps"

    def test_returns_copy_not_original(self) -> None:
        """Mutations on the returned dict must not affect the module-level originals."""
        defaults = get_defaults(ProblemType.TIME_SERIES)
        defaults["target"] = "MUTATED"
        assert TS_DEFAULTS["target"] == "avg_wait"

        defaults_entity = get_defaults(ProblemType.ENTITY_CLASSIFICATION)
        defaults_entity["target"] = "MUTATED"
        assert ENTITY_DEFAULTS["target"] == "status"

    def test_unknown_problem_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown problem type"):
            get_defaults("invalid")  # type: ignore[arg-type]

    def test_ts_defaults_include_expected_keys(self) -> None:
        defaults = get_defaults(ProblemType.TIME_SERIES)
        expected_keys = {
            "target",
            "lookback",
            "horizon",
            "lag_intervals",
            "rolling_windows",
            "include_trend",
            "include_ratios",
            "include_cross_node",
            "algorithms",
            "cv_strategy",
            "cv_folds",
            "metrics",
            "handle_imbalance",
            "hyperparameter_tuning",
            "selection",
            "generate_report",
            "plot_predictions",
            "plot_feature_importance",
            "plot_confusion_matrix",
            "plot_roc_curve",
        }
        assert expected_keys.issubset(defaults.keys())

    def test_entity_defaults_include_expected_keys(self) -> None:
        defaults = get_defaults(ProblemType.ENTITY_CLASSIFICATION)
        expected_keys = {
            "target",
            "observation_point",
            "include_entity_state",
            "include_node_state",
            "include_system_state",
            "add_progress_ratio",
            "add_wait_trend",
            "algorithms",
            "cv_strategy",
            "cv_folds",
            "metrics",
            "handle_imbalance",
            "selection",
            "generate_report",
            "plot_feature_importance",
            "plot_confusion_matrix",
            "plot_roc_curve",
        }
        assert expected_keys.issubset(defaults.keys())

    def test_ts_defaults_algorithms(self) -> None:
        defaults = get_defaults(ProblemType.TIME_SERIES)
        assert "lightgbm" in defaults["algorithms"]
        assert "xgboost" in defaults["algorithms"]

    def test_entity_defaults_algorithms(self) -> None:
        defaults = get_defaults(ProblemType.ENTITY_CLASSIFICATION)
        assert "lightgbm" in defaults["algorithms"]

    def test_ts_defaults_cv_strategy_temporal(self) -> None:
        defaults = get_defaults(ProblemType.TIME_SERIES)
        assert defaults["cv_strategy"] == "temporal"

    def test_entity_defaults_cv_strategy_stratified(self) -> None:
        defaults = get_defaults(ProblemType.ENTITY_CLASSIFICATION)
        assert defaults["cv_strategy"] == "stratified_kfold"
