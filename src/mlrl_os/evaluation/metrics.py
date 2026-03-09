"""Metric registry and computation for ML/RL OS."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Avoid division by zero
    mask = y_true != 0
    if not mask.any():
        return 0.0
    return float(mean_absolute_percentage_error(y_true[mask], y_pred[mask]))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))


def _f1_weighted(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def _f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(precision_score(y_true, y_pred, average="weighted", zero_division=0))


def _recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(recall_score(y_true, y_pred, average="weighted", zero_division=0))


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def _auc_roc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> float:
    """Compute AUC-ROC. Requires probability predictions."""
    if y_proba is None:
        return 0.0

    n_classes = len(np.unique(y_true))
    if n_classes <= 1:
        return 0.0

    try:
        if n_classes == 2:
            # Binary: use probability of positive class
            proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            return float(roc_auc_score(y_true, proba))
        # Multiclass: one-vs-rest
        return float(
            roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
        )
    except ValueError:
        return 0.0


# Metric registry: name -> (function, requires_proba)
MetricFn = tuple[Any, bool]  # (callable, needs_proba)

METRIC_REGISTRY: dict[str, MetricFn] = {
    # Regression metrics
    "rmse": (_rmse, False),
    "mae": (_mae, False),
    "mape": (_mape, False),
    "r2": (_r2, False),
    # Classification metrics
    "f1_weighted": (_f1_weighted, False),
    "f1_macro": (_f1_macro, False),
    "precision": (_precision, False),
    "recall": (_recall, False),
    "accuracy": (_accuracy, False),
    "auc_roc": (_auc_roc, True),
}

REGRESSION_METRICS = {"rmse", "mae", "mape", "r2"}
CLASSIFICATION_METRICS = {"f1_weighted", "f1_macro", "precision", "recall", "accuracy", "auc_roc"}
ALL_KNOWN_METRICS = set(METRIC_REGISTRY.keys())


def compute_metric(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> float:
    """Compute a single metric by name.

    Args:
        name: Metric name (must be in METRIC_REGISTRY).
        y_true: True values.
        y_pred: Predicted values.
        y_proba: Class probabilities (required for auc_roc).

    Returns:
        Computed metric value.

    Raises:
        KeyError: If metric name is unknown.
    """
    if name not in METRIC_REGISTRY:
        msg = f"Unknown metric '{name}'. Available: {sorted(METRIC_REGISTRY.keys())}"
        raise KeyError(msg)

    fn, needs_proba = METRIC_REGISTRY[name]
    if needs_proba:
        return float(fn(y_true, y_pred, y_proba))
    return float(fn(y_true, y_pred))


def compute_metrics(
    metric_names: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute multiple metrics.

    Args:
        metric_names: List of metric names.
        y_true: True values.
        y_pred: Predicted values.
        y_proba: Class probabilities.

    Returns:
        Dict of metric_name -> value.
    """
    results: dict[str, float] = {}
    for name in metric_names:
        results[name] = compute_metric(name, y_true, y_pred, y_proba)
    return results
