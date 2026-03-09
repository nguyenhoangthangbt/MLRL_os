"""Core types, enums, and base models used across ML/RL OS."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    pass


class ProblemType(str, Enum):
    """Supported prediction problem types."""

    TIME_SERIES = "time_series"
    ENTITY_CLASSIFICATION = "entity_classification"
    REINFORCEMENT_LEARNING = "reinforcement_learning"


class TaskType(str, Enum):
    """ML task type."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class ExperimentStatus(str, Enum):
    """Experiment lifecycle status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ObservationPoint(str, Enum):
    """At which step(s) of an entity's journey to make predictions."""

    ALL_STEPS = "all_steps"
    ENTRY_ONLY = "entry_only"
    MIDPOINT = "midpoint"


class CVStrategy(str, Enum):
    """Cross-validation strategy."""

    TEMPORAL = "temporal"
    STRATIFIED_KFOLD = "stratified_kfold"
    KFOLD = "kfold"


class ColumnInfo(BaseModel):
    """Schema information for a single data column."""

    name: str
    dtype: str
    null_count: int = 0
    null_rate: float = 0.0
    unique_count: int = 0
    is_numeric: bool = False
    is_categorical: bool = False
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    categories: list[str] | None = None
    category_counts: dict[str, int] | None = None


class TargetInfo(BaseModel):
    """Information about a potential prediction target."""

    column: str
    task_type: TaskType
    is_default: bool = False
    null_rate: float = 0.0
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    unique_count: int | None = None
    classes: list[str] | None = None
    class_balance: dict[str, float] | None = None


class AvailableTargets(BaseModel):
    """Auto-discovered targets from a dataset."""

    dataset_id: str
    time_series_targets: list[TargetInfo] = Field(default_factory=list)
    entity_targets: list[TargetInfo] = Field(default_factory=list)


class FeatureMatrix:
    """Model-ready data containing feature array, target array, and metadata.

    Not a Pydantic model because it holds numpy arrays.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        problem_type: ProblemType,
        target_name: str,
        task_type: TaskType,
        temporal_index: np.ndarray | None = None,
        entity_ids: np.ndarray | None = None,
        class_names: list[str] | None = None,
    ) -> None:
        if X.ndim != 2:
            msg = f"X must be 2D, got {X.ndim}D"
            raise ValueError(msg)
        if X.shape[0] != y.shape[0]:
            msg = f"X rows ({X.shape[0]}) != y rows ({y.shape[0]})"
            raise ValueError(msg)
        if X.shape[1] != len(feature_names):
            msg = f"X columns ({X.shape[1]}) != feature_names length ({len(feature_names)})"
            raise ValueError(msg)
        if temporal_index is not None and temporal_index.shape[0] != X.shape[0]:
            msg = f"temporal_index length ({temporal_index.shape[0]}) != X rows ({X.shape[0]})"
            raise ValueError(msg)
        if entity_ids is not None and entity_ids.shape[0] != X.shape[0]:
            msg = f"entity_ids length ({entity_ids.shape[0]}) != X rows ({X.shape[0]})"
            raise ValueError(msg)

        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.problem_type = problem_type
        self.target_name = target_name
        self.task_type = task_type
        self.temporal_index = temporal_index
        self.entity_ids = entity_ids
        self.class_names = class_names

    @property
    def sample_count(self) -> int:
        """Number of samples (rows)."""
        return int(self.X.shape[0])

    @property
    def feature_count(self) -> int:
        """Number of features (columns)."""
        return int(self.X.shape[1])
