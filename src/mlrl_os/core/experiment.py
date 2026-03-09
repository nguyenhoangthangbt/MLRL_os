"""Experiment result models for ML/RL OS."""

from __future__ import annotations

from pydantic import BaseModel, Field

from mlrl_os.core.types import ExperimentStatus, ProblemType


class AlgorithmScore(BaseModel):
    """Performance of one algorithm across cross-validation folds."""

    algorithm: str
    metrics: dict[str, float] = Field(default_factory=dict)
    metrics_std: dict[str, float] = Field(default_factory=dict)
    rank: int = 0


class FeatureImportanceEntry(BaseModel):
    """One feature's importance score."""

    feature: str
    importance: float
    rank: int = 0


class ExperimentResult(BaseModel):
    """Full result of an experiment run."""

    experiment_id: str
    name: str
    status: ExperimentStatus
    experiment_type: ProblemType
    created_at: str
    completed_at: str | None = None
    duration_seconds: float | None = None

    best_algorithm: str | None = None
    metrics: dict[str, float] | None = None
    all_algorithm_scores: list[AlgorithmScore] | None = None
    feature_importance: list[FeatureImportanceEntry] | None = None
    model_id: str | None = None

    sample_count: int | None = None
    feature_count: int | None = None
    resolved_config: dict | None = None  # type: ignore[type-arg]

    error_message: str | None = None
