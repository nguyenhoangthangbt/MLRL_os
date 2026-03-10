"""API request/response Pydantic models.

These are the serialization models for the REST API layer.
They are separate from core/config models to decouple the API
contract from internal representations.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    name: str
    source_type: str
    has_snapshots: bool
    has_trajectories: bool
    snapshot_row_count: int | None = None
    trajectory_row_count: int | None = None
    registered_at: str


class DatasetListItem(BaseModel):
    id: str
    name: str
    source_type: str
    has_snapshots: bool
    has_trajectories: bool
    snapshot_row_count: int | None = None
    trajectory_row_count: int | None = None
    registered_at: str


class DatasetDetailResponse(BaseModel):
    """Full dataset metadata — mirrors DatasetMeta."""

    id: str
    name: str
    version: int
    content_hash: str
    source_type: str
    source_path: str | None = None
    has_snapshots: bool
    has_trajectories: bool
    snapshot_row_count: int | None = None
    trajectory_row_count: int | None = None
    snapshot_column_count: int | None = None
    trajectory_column_count: int | None = None
    snapshot_columns: list[dict[str, Any]] | None = None
    trajectory_columns: list[dict[str, Any]] | None = None
    simos_metadata: dict[str, Any] | None = None
    simos_summary: dict[str, Any] | None = None
    source_instrument: str | None = None
    source_job_id: str | None = None
    source_template: str | None = None
    registered_at: str


class SchemaResponse(BaseModel):
    dataset_id: str
    layer: str
    columns: list[dict[str, Any]]


class PreviewResponse(BaseModel):
    dataset_id: str
    layer: str
    columns: list[str]
    rows: list[dict[str, Any]]
    total_rows: int


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


class ExperimentRequest(BaseModel):
    """Request body for submitting or validating an experiment.

    All fields except ``dataset_id`` are optional — the config
    resolver fills in defaults.
    """

    dataset_id: str
    name: str | None = None
    seed: int | None = None
    dataset_layer: str | None = None
    experiment_type: str | None = None
    target: str | None = None
    lookback: str | None = None
    horizon: str | None = None
    lag_intervals: list[str] | None = None
    rolling_windows: list[str] | None = None
    observation_point: str | None = None
    feature_columns: list[str] | None = None
    exclude_columns: list[str] | None = None
    algorithms: list[str] | None = None
    cv_folds: int | None = None
    cv_strategy: str | None = None
    metrics: list[str] | None = None
    handle_imbalance: bool | None = None
    hyperparameter_tuning: bool | None = None
    n_trials: int | None = None
    generate_report: bool | None = None

    def to_user_config(self) -> dict[str, Any]:
        """Convert to the flat dict expected by ConfigResolver.resolve()."""
        cfg: dict[str, Any] = {}
        for key, value in self.model_dump(exclude_none=True).items():
            if key == "dataset_id":
                continue
            cfg[key] = value
        return cfg


class ExperimentSubmitResponse(BaseModel):
    experiment_id: str
    status: str
    experiment_type: str
    name: str


class ExperimentListItem(BaseModel):
    experiment_id: str
    name: str
    status: str
    experiment_type: str
    created_at: str
    completed_at: str | None = None
    best_algorithm: str | None = None
    duration_seconds: float | None = None


class ExperimentDetailResponse(BaseModel):
    """Full experiment result."""

    experiment_id: str
    name: str
    status: str
    experiment_type: str
    created_at: str
    completed_at: str | None = None
    duration_seconds: float | None = None
    best_algorithm: str | None = None
    metrics: dict[str, float] | None = None
    all_algorithm_scores: list[dict[str, Any]] | None = None
    feature_importance: list[dict[str, Any]] | None = None
    model_id: str | None = None
    sample_count: int | None = None
    feature_count: int | None = None
    resolved_config: dict[str, Any] | None = None
    error_message: str | None = None


class ValidationRequest(ExperimentRequest):
    """Same shape as ExperimentRequest."""


class ValidationErrorDetail(BaseModel):
    code: str
    field: str
    message: str
    suggestion: str | None = None


class ValidationResponse(BaseModel):
    valid: bool
    resolved_config: dict[str, Any] | None = None
    errors: list[ValidationErrorDetail] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DefaultsResponse(BaseModel):
    problem_type: str
    defaults: dict[str, Any]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ModelListItem(BaseModel):
    id: str
    experiment_id: str
    algorithm_name: str
    task: str
    metrics: dict[str, float] = Field(default_factory=dict)
    created_at: str


class ModelDetailResponse(BaseModel):
    id: str
    experiment_id: str
    algorithm_name: str
    task: str
    feature_names: list[str]
    metrics: dict[str, float] = Field(default_factory=dict)
    created_at: str
    file_path: str


class PredictRequest(BaseModel):
    data: list[dict[str, Any]]


class PredictResponse(BaseModel):
    model_id: str
    predictions: list[float | int | str]
    task_type: str
    probabilities: list[dict[str, float]] | None = None


class FeatureImportanceResponse(BaseModel):
    model_id: str
    features: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class ConfigResolveRequest(ExperimentRequest):
    """Same shape as ExperimentRequest."""


class ConfigResolveResponse(BaseModel):
    resolved_config: dict[str, Any]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    detail: str
