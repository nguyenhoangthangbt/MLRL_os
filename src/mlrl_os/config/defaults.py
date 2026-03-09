"""Default configurations and environment-based settings for ML/RL OS.

Provides sensible defaults for both problem types so that zero-config
experiments are valid. All defaults can be overridden via user config.
"""

from __future__ import annotations

from typing import Any

from pydantic_settings import BaseSettings

from mlrl_os.core.types import ProblemType


class MLRLSettings(BaseSettings):
    """Application settings loaded from environment variables.

    All variables are prefixed with ``MLRL_`` (e.g. ``MLRL_ENV=production``).
    """

    model_config = {"env_prefix": "MLRL_"}

    env: str = "development"
    data_dir: str = "./data"
    models_dir: str = "./models"
    experiments_dir: str = "./experiments"
    log_level: str = "INFO"
    api_port: int = 8001
    cors_allow_origins: str = "http://localhost:5175"
    max_training_rows: int = 1_000_000
    cv_folds_default: int = 5
    seed_default: int = 42
    storage_backend: str = "file"
    database_url: str = ""


# ---------------------------------------------------------------------------
# Problem-type defaults
# ---------------------------------------------------------------------------

TS_DEFAULTS: dict[str, Any] = {
    "target": "avg_wait",
    "lookback": "8h",
    "horizon": "1h",
    "lag_intervals": ["1h", "2h", "4h", "8h"],
    "rolling_windows": ["2h", "4h"],
    "include_trend": True,
    "include_ratios": True,
    "include_cross_node": True,
    "algorithms": ["lightgbm", "xgboost"],
    "cv_strategy": "temporal",
    "cv_folds": 5,
    "metrics": ["rmse", "mae", "mape"],
    "handle_imbalance": False,
    "hyperparameter_tuning": False,
    "n_trials": 20,
    "selection": "best_cv",
    "generate_report": True,
    "plot_predictions": True,
    "plot_feature_importance": True,
    "plot_confusion_matrix": False,
    "plot_roc_curve": False,
}

ENTITY_DEFAULTS: dict[str, Any] = {
    "target": "status",
    "observation_point": "all_steps",
    "include_entity_state": True,
    "include_node_state": True,
    "include_system_state": True,
    "add_progress_ratio": True,
    "add_wait_trend": True,
    "algorithms": ["lightgbm"],
    "cv_strategy": "stratified_kfold",
    "cv_folds": 5,
    "metrics": ["f1_weighted", "auc_roc", "precision", "recall"],
    "handle_imbalance": True,
    "hyperparameter_tuning": False,
    "n_trials": 20,
    "selection": "best_cv",
    "generate_report": True,
    "plot_predictions": False,
    "plot_feature_importance": True,
    "plot_confusion_matrix": True,
    "plot_roc_curve": True,
}


def get_defaults(problem_type: ProblemType) -> dict[str, Any]:
    """Return a *copy* of the default configuration for a problem type.

    Args:
        problem_type: The problem type to retrieve defaults for.

    Returns:
        A fresh dictionary containing all default values. Callers may
        mutate the returned dict without affecting the module-level defaults.

    Raises:
        ValueError: If *problem_type* is not a recognised ``ProblemType``.
    """
    if problem_type == ProblemType.TIME_SERIES:
        return dict(TS_DEFAULTS)
    if problem_type == ProblemType.ENTITY_CLASSIFICATION:
        return dict(ENTITY_DEFAULTS)
    msg = f"Unknown problem type: {problem_type}"
    raise ValueError(msg)
