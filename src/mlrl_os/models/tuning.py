"""Hyperparameter tuning via Optuna (TPE-based Bayesian optimization).

Each algorithm defines a search space as Optuna trial suggestions.
The tuner runs N trials, evaluating each via cross-validation on the
primary metric, and returns the best hyperparameters found.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Default number of Optuna trials per algorithm
DEFAULT_N_TRIALS = 20


def suggest_params(
    trial: Any,  # optuna.trial.Trial — Any to avoid import at module level
    algorithm_name: str,
    task: str,
) -> dict[str, Any]:
    """Suggest hyperparameters for a single Optuna trial.

    Args:
        trial: Optuna trial object.
        algorithm_name: Name of the algorithm.
        task: "regression" or "classification".

    Returns:
        Dict of suggested hyperparameters to pass as **kwargs to train().
    """
    if algorithm_name == "lightgbm":
        return _suggest_lightgbm(trial)

    if algorithm_name == "xgboost":
        return _suggest_xgboost(trial)

    if algorithm_name == "random_forest":
        return _suggest_random_forest(trial)

    if algorithm_name == "extra_trees":
        return _suggest_extra_trees(trial)

    if algorithm_name == "linear":
        return _suggest_linear(trial, task)

    return {}


def has_search_space(algorithm_name: str) -> bool:
    """Check if an algorithm has a defined search space."""
    return algorithm_name in {
        "lightgbm", "xgboost", "random_forest", "extra_trees", "linear",
    }


# ---------------------------------------------------------------------------
# Per-algorithm search spaces
# ---------------------------------------------------------------------------


def _suggest_lightgbm(trial: Any) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def _suggest_xgboost(trial: Any) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
    }


def _suggest_random_forest(trial: Any) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
        "max_depth": trial.suggest_int("max_depth", 4, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2"]
        ),
    }


def _suggest_extra_trees(trial: Any) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
        "max_depth": trial.suggest_int("max_depth", 4, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2"]
        ),
    }


def _suggest_linear(trial: Any, task: str) -> dict[str, Any]:
    if task == "regression":
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True),
        }
    # LogisticRegression
    return {
        "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
        "max_iter": trial.suggest_categorical("max_iter", [500, 1000, 2000]),
    }
