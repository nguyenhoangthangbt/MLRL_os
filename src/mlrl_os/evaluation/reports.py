"""Evaluation report generation for ML/RL OS."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from mlrl_os.core.experiment import (
    AlgorithmScore,
    ExperimentResult,
    FeatureImportanceEntry,
)


class PredictionSample(BaseModel):
    """One prediction vs actual pair for visualization."""

    index: int
    actual: float | str
    predicted: float | str
    probabilities: dict[str, float] | None = None


class EvaluationReport(BaseModel):
    """Full evaluation report — stored as JSON."""

    experiment_id: str
    experiment_name: str
    experiment_type: str
    created_at: str

    dataset_id: str
    dataset_name: str
    sample_count: int
    feature_count: int
    target: str
    task_type: str

    best_algorithm: str
    best_metrics: dict[str, float] = Field(default_factory=dict)

    algorithm_scores: list[AlgorithmScore] = Field(default_factory=list)
    feature_importance: list[FeatureImportanceEntry] = Field(default_factory=list)
    predictions_sample: list[PredictionSample] = Field(default_factory=list)

    confusion_matrix: list[list[int]] | None = None
    class_names: list[str] | None = None

    resolved_config: dict[str, Any] = Field(default_factory=dict)


class ReportGenerator:
    """Generate evaluation reports from experiment results."""

    def generate(
        self,
        result: ExperimentResult,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None = None,
        class_names: list[str] | None = None,
        dataset_name: str = "",
        max_samples: int = 100,
    ) -> EvaluationReport:
        """Generate a structured evaluation report.

        Args:
            result: Experiment result with metrics and model info.
            y_true: True target values.
            y_pred: Predicted values.
            y_proba: Class probabilities (classification only).
            class_names: Class names for classification.
            dataset_name: Name of the dataset used.
            max_samples: Maximum number of prediction samples to include.

        Returns:
            Complete EvaluationReport.
        """
        # Build prediction samples
        n_samples = min(max_samples, len(y_true))
        samples: list[PredictionSample] = []
        for i in range(n_samples):
            actual: float | str = float(y_true[i]) if isinstance(y_true[i], (int, float, np.integer, np.floating)) else str(y_true[i])
            predicted: float | str = float(y_pred[i]) if isinstance(y_pred[i], (int, float, np.integer, np.floating)) else str(y_pred[i])

            proba_dict = None
            if y_proba is not None and class_names:
                proba_dict = {
                    class_names[j]: round(float(y_proba[i][j]), 4)
                    for j in range(len(class_names))
                }

            samples.append(
                PredictionSample(
                    index=i,
                    actual=actual,
                    predicted=predicted,
                    probabilities=proba_dict,
                )
            )

        # Build confusion matrix for classification
        confusion = None
        if class_names and len(class_names) > 0:
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(y_true, y_pred)
            confusion = cm.tolist()

        return EvaluationReport(
            experiment_id=result.experiment_id,
            experiment_name=result.name,
            experiment_type=result.experiment_type.value,
            created_at=result.created_at,
            dataset_id=result.resolved_config.get("dataset_id", "") if result.resolved_config else "",
            dataset_name=dataset_name,
            sample_count=result.sample_count or len(y_true),
            feature_count=result.feature_count or 0,
            target=result.resolved_config.get("features", {}).get("target", "") if result.resolved_config else "",
            task_type="classification" if class_names else "regression",
            best_algorithm=result.best_algorithm or "",
            best_metrics=result.metrics or {},
            algorithm_scores=result.all_algorithm_scores or [],
            feature_importance=result.feature_importance or [],
            predictions_sample=samples,
            confusion_matrix=confusion,
            class_names=class_names,
            resolved_config=result.resolved_config or {},
        )
