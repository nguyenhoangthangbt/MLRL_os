"""Tests for mlrl_os.evaluation.reports."""

from __future__ import annotations

import numpy as np
import pytest

from mlrl_os.core.experiment import ExperimentResult
from mlrl_os.core.types import ExperimentStatus, ProblemType
from mlrl_os.evaluation.reports import EvaluationReport, ReportGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def experiment_result() -> ExperimentResult:
    """Minimal ExperimentResult for report generation."""
    return ExperimentResult(
        experiment_id="exp_001",
        name="Test",
        status=ExperimentStatus.COMPLETED,
        experiment_type=ProblemType.TIME_SERIES,
        created_at="2025-01-15T10:00:00Z",
        best_algorithm="linear",
        metrics={"rmse": 0.5},
        sample_count=100,
        feature_count=5,
    )


@pytest.fixture()
def generator() -> ReportGenerator:
    """Fresh ReportGenerator."""
    return ReportGenerator()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReportGenerator:
    """Tests for ReportGenerator.generate()."""

    def test_generate_returns_evaluation_report(
        self,
        generator: ReportGenerator,
        experiment_result: ExperimentResult,
    ) -> None:
        """generate() returns an EvaluationReport instance."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])

        report = generator.generate(experiment_result, y_true, y_pred)

        assert isinstance(report, EvaluationReport)
        assert report.experiment_id == "exp_001"
        assert report.experiment_name == "Test"
        assert report.best_algorithm == "linear"

    def test_report_contains_prediction_samples(
        self,
        generator: ReportGenerator,
        experiment_result: ExperimentResult,
    ) -> None:
        """Report includes prediction samples with actual and predicted values."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.2, 2.8])

        report = generator.generate(experiment_result, y_true, y_pred)

        assert len(report.predictions_sample) == 3
        sample = report.predictions_sample[0]
        assert sample.index == 0
        assert sample.actual == pytest.approx(1.0)
        assert sample.predicted == pytest.approx(1.1)

    def test_report_contains_confusion_matrix_for_classification(
        self,
        generator: ReportGenerator,
        experiment_result: ExperimentResult,
    ) -> None:
        """Report includes a confusion matrix when class_names is provided."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])

        report = generator.generate(
            experiment_result,
            y_true,
            y_pred,
            class_names=["neg", "pos"],
        )

        assert report.confusion_matrix is not None
        assert report.class_names == ["neg", "pos"]
        # Confusion matrix should be a list of lists (2x2 for binary).
        assert len(report.confusion_matrix) == 2
        assert len(report.confusion_matrix[0]) == 2

    def test_no_confusion_matrix_for_regression(
        self,
        generator: ReportGenerator,
        experiment_result: ExperimentResult,
    ) -> None:
        """Report has no confusion matrix when class_names is None."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.2, 2.8])

        report = generator.generate(experiment_result, y_true, y_pred, class_names=None)

        assert report.confusion_matrix is None
        assert report.class_names is None

    def test_prediction_sample_count_respects_max_samples(
        self,
        generator: ReportGenerator,
        experiment_result: ExperimentResult,
    ) -> None:
        """predictions_sample count is <= max_samples."""
        rng = np.random.RandomState(0)
        y_true = rng.randn(500)
        y_pred = rng.randn(500)

        report = generator.generate(
            experiment_result, y_true, y_pred, max_samples=50
        )

        assert len(report.predictions_sample) <= 50
        assert len(report.predictions_sample) == 50
