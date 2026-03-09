"""Tests for mlrl_os.evaluation.metrics."""

from __future__ import annotations

import numpy as np
import pytest

from mlrl_os.evaluation.metrics import (
    METRIC_REGISTRY,
    REGRESSION_METRICS,
    CLASSIFICATION_METRICS,
    compute_metric,
    compute_metrics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def regression_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Simple regression y_true and y_pred."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
    return y_true, y_pred


@pytest.fixture()
def classification_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple binary classification y_true, y_pred, y_proba."""
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 1, 1, 1, 0])
    y_proba = np.array([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.3, 0.7],
        [0.8, 0.2],
        [0.1, 0.9],
        [0.4, 0.6],
        [0.15, 0.85],
        [0.6, 0.4],
    ])
    return y_true, y_pred, y_proba


# ---------------------------------------------------------------------------
# compute_metric
# ---------------------------------------------------------------------------


class TestComputeMetric:
    """Tests for compute_metric()."""

    @pytest.mark.parametrize("metric_name", sorted(REGRESSION_METRICS))
    def test_regression_metrics(
        self, metric_name: str, regression_arrays: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Each known regression metric computes without error."""
        y_true, y_pred = regression_arrays
        value = compute_metric(metric_name, y_true, y_pred)
        assert isinstance(value, float)

    @pytest.mark.parametrize(
        "metric_name",
        sorted(CLASSIFICATION_METRICS - {"auc_roc"}),
    )
    def test_classification_metrics(
        self,
        metric_name: str,
        classification_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Each known classification metric (except auc_roc) computes without error."""
        y_true, y_pred, _ = classification_arrays
        value = compute_metric(metric_name, y_true, y_pred)
        assert isinstance(value, float)

    def test_auc_roc_with_proba(
        self,
        classification_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """auc_roc computes when y_proba is provided."""
        y_true, y_pred, y_proba = classification_arrays
        value = compute_metric("auc_roc", y_true, y_pred, y_proba=y_proba)
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0

    def test_unknown_metric_raises_key_error(self) -> None:
        """compute_metric raises KeyError for an unknown metric name."""
        with pytest.raises(KeyError, match="Unknown metric"):
            compute_metric("nonexistent", np.array([1]), np.array([1]))


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    """Tests for compute_metrics()."""

    def test_returns_dict_with_all_requested(
        self, regression_arrays: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """compute_metrics returns a dict with an entry for each requested metric."""
        y_true, y_pred = regression_arrays
        names = ["rmse", "mae", "r2"]
        result = compute_metrics(names, y_true, y_pred)

        assert isinstance(result, dict)
        assert set(result.keys()) == set(names)
        for v in result.values():
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# Specific metric properties
# ---------------------------------------------------------------------------


class TestMetricProperties:
    """Tests for specific metric value properties."""

    def test_rmse_is_non_negative(
        self, regression_arrays: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """RMSE is always >= 0."""
        y_true, y_pred = regression_arrays
        rmse = compute_metric("rmse", y_true, y_pred)
        assert rmse >= 0.0

    def test_r2_perfect_predictions(self) -> None:
        """R2 = 1.0 when predictions match truth exactly."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        r2 = compute_metric("r2", y, y)
        assert r2 == pytest.approx(1.0)

    def test_accuracy_perfect_predictions(self) -> None:
        """Accuracy = 1.0 when all predictions are correct."""
        y = np.array([0, 1, 1, 0, 1])
        acc = compute_metric("accuracy", y, y)
        assert acc == pytest.approx(1.0)

    def test_auc_roc_returns_zero_without_proba(self) -> None:
        """auc_roc returns 0.0 when y_proba is None."""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0])
        value = compute_metric("auc_roc", y_true, y_pred, y_proba=None)
        assert value == 0.0

    def test_mape_handles_zero_values(self) -> None:
        """MAPE handles y_true with zeros without raising errors."""
        y_true = np.array([0.0, 0.0, 3.0, 4.0])
        y_pred = np.array([0.1, 0.2, 3.1, 3.9])
        value = compute_metric("mape", y_true, y_pred)
        assert isinstance(value, float)
        assert value >= 0.0
