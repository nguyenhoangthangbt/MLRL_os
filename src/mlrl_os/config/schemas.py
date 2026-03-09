"""Pydantic models for all resolved configuration types.

After config resolution, every field has a concrete value -- no optional fields.
Pipeline code never needs to check for None.
"""

from __future__ import annotations

from pydantic import BaseModel

from mlrl_os.core.types import CVStrategy, ObservationPoint, ProblemType


class ResolvedTimeSeriesFeatures(BaseModel):
    """Fully resolved feature configuration for time-series forecasting."""

    target: str
    lookback: str
    horizon: str
    lag_intervals: list[str]
    rolling_windows: list[str]
    include_trend: bool
    include_ratios: bool
    include_cross_node: bool
    feature_columns: list[str]
    exclude_columns: list[str] = []


class ResolvedEntityFeatures(BaseModel):
    """Fully resolved feature configuration for entity classification."""

    target: str
    observation_point: ObservationPoint
    include_entity_state: bool
    include_node_state: bool
    include_system_state: bool
    add_progress_ratio: bool
    add_wait_trend: bool
    feature_columns: list[str]
    exclude_columns: list[str] = []
    sla_column: str | None = None
    sla_threshold: float | None = None


class ResolvedCVConfig(BaseModel):
    """Fully resolved cross-validation configuration."""

    strategy: CVStrategy
    folds: int


class ResolvedModelConfig(BaseModel):
    """Fully resolved model training configuration."""

    algorithms: list[str]
    selection: str
    cross_validation: ResolvedCVConfig
    handle_imbalance: bool
    hyperparameter_tuning: bool


class ResolvedEvaluationConfig(BaseModel):
    """Fully resolved evaluation configuration."""

    metrics: list[str]
    generate_report: bool
    plot_predictions: bool
    plot_feature_importance: bool
    plot_confusion_matrix: bool
    plot_roc_curve: bool


class ResolvedExperimentConfig(BaseModel):
    """Fully resolved experiment configuration.

    Every field is concrete -- no optionals. Produced by ``ConfigResolver.resolve()``
    and consumed by the experiment pipeline. Passed through the validation gate
    before any training occurs.
    """

    name: str
    experiment_type: ProblemType
    seed: int
    dataset_id: str
    dataset_layer: str
    features: ResolvedTimeSeriesFeatures | ResolvedEntityFeatures
    model: ResolvedModelConfig
    evaluation: ResolvedEvaluationConfig
