"""Tests for mlrl_os.models.tuning — Optuna hyperparameter search spaces."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mlrl_os.models.tuning import (
    DEFAULT_N_TRIALS,
    has_search_space,
    suggest_params,
)


# ---------------------------------------------------------------------------
# has_search_space
# ---------------------------------------------------------------------------


class TestHasSearchSpace:
    def test_known_algorithms_have_search_space(self) -> None:
        for name in ("lightgbm", "xgboost", "random_forest", "extra_trees", "linear"):
            assert has_search_space(name), f"{name} should have a search space"

    def test_unknown_algorithm_has_no_search_space(self) -> None:
        assert not has_search_space("unknown_algo")
        assert not has_search_space("")

    def test_default_n_trials_is_positive(self) -> None:
        assert DEFAULT_N_TRIALS > 0


# ---------------------------------------------------------------------------
# suggest_params — mock-based (verify trial methods are called correctly)
# ---------------------------------------------------------------------------


def _make_trial() -> MagicMock:
    """Create a mock Optuna trial that returns reasonable values."""
    trial = MagicMock()
    trial.suggest_int.side_effect = lambda name, low, high, **kw: low
    trial.suggest_float.side_effect = lambda name, low, high, **kw: low
    trial.suggest_categorical.side_effect = lambda name, choices: choices[0]
    return trial


class TestSuggestLightGBM:
    def test_returns_expected_keys(self) -> None:
        trial = _make_trial()
        params = suggest_params(trial, "lightgbm", "classification")
        expected = {
            "n_estimators", "learning_rate", "max_depth", "num_leaves",
            "min_child_samples", "subsample", "colsample_bytree",
            "reg_alpha", "reg_lambda",
        }
        assert set(params.keys()) == expected

    def test_values_are_numeric(self) -> None:
        trial = _make_trial()
        params = suggest_params(trial, "lightgbm", "regression")
        for v in params.values():
            assert isinstance(v, (int, float))


class TestSuggestXGBoost:
    def test_returns_expected_keys(self) -> None:
        trial = _make_trial()
        params = suggest_params(trial, "xgboost", "classification")
        expected = {
            "n_estimators", "learning_rate", "max_depth",
            "subsample", "colsample_bytree", "reg_alpha", "reg_lambda",
            "min_child_weight", "gamma",
        }
        assert set(params.keys()) == expected


class TestSuggestRandomForest:
    def test_returns_expected_keys(self) -> None:
        trial = _make_trial()
        params = suggest_params(trial, "random_forest", "classification")
        expected = {
            "n_estimators", "max_depth", "min_samples_split",
            "min_samples_leaf", "max_features",
        }
        assert set(params.keys()) == expected


class TestSuggestExtraTrees:
    def test_returns_expected_keys(self) -> None:
        trial = _make_trial()
        params = suggest_params(trial, "extra_trees", "regression")
        expected = {
            "n_estimators", "max_depth", "min_samples_split",
            "min_samples_leaf", "max_features",
        }
        assert set(params.keys()) == expected


class TestSuggestLinear:
    def test_regression_returns_alpha(self) -> None:
        trial = _make_trial()
        params = suggest_params(trial, "linear", "regression")
        assert "alpha" in params
        assert "C" not in params

    def test_classification_returns_c_and_max_iter(self) -> None:
        trial = _make_trial()
        params = suggest_params(trial, "linear", "classification")
        assert "C" in params
        assert "max_iter" in params
        assert "alpha" not in params


class TestSuggestUnknown:
    def test_unknown_algorithm_returns_empty(self) -> None:
        trial = _make_trial()
        params = suggest_params(trial, "unknown_algo", "regression")
        assert params == {}
