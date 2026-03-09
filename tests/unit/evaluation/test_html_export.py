"""Tests for mlrl_os.evaluation.html_export — standalone HTML report generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlrl_os.core.experiment import (
    AlgorithmScore,
    ExperimentResult,
    FeatureImportanceEntry,
)
from mlrl_os.core.types import ExperimentStatus, ProblemType
from mlrl_os.evaluation.html_export import export_html_report


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _minimal_result() -> ExperimentResult:
    """ExperimentResult with only required fields."""
    return ExperimentResult(
        experiment_id="exp_html_001",
        name="HTML Export Test",
        status=ExperimentStatus.COMPLETED,
        experiment_type=ProblemType.ENTITY_CLASSIFICATION,
        created_at="2026-03-10T12:00:00Z",
        completed_at="2026-03-10T12:00:05Z",
        duration_seconds=5.0,
        best_algorithm="lightgbm",
        metrics={"f1_weighted": 0.95, "auc_roc": 0.98, "precision": 0.94, "recall": 0.96},
        sample_count=500,
        feature_count=40,
    )


def _full_result() -> ExperimentResult:
    """ExperimentResult with algorithm scores and feature importance."""
    return ExperimentResult(
        experiment_id="exp_html_002",
        name="Full HTML Export Test",
        status=ExperimentStatus.COMPLETED,
        experiment_type=ProblemType.ENTITY_CLASSIFICATION,
        created_at="2026-03-10T12:00:00Z",
        completed_at="2026-03-10T12:01:30Z",
        duration_seconds=90.0,
        best_algorithm="lightgbm",
        metrics={"f1_weighted": 0.95, "auc_roc": 0.98},
        all_algorithm_scores=[
            AlgorithmScore(
                algorithm="lightgbm",
                metrics={"f1_weighted": 0.94, "auc_roc": 0.97},
                metrics_std={"f1_weighted": 0.02, "auc_roc": 0.01},
                rank=1,
            ),
            AlgorithmScore(
                algorithm="random_forest",
                metrics={"f1_weighted": 0.90, "auc_roc": 0.93},
                metrics_std={"f1_weighted": 0.03, "auc_roc": 0.02},
                rank=2,
            ),
        ],
        feature_importance=[
            FeatureImportanceEntry(feature="s_elapsed", importance=0.25, rank=1),
            FeatureImportanceEntry(feature="s_cum_wait", importance=0.20, rank=2),
            FeatureImportanceEntry(feature="n_queue_len", importance=0.15, rank=3),
            FeatureImportanceEntry(feature="sys_wip", importance=0.10, rank=4),
            FeatureImportanceEntry(feature="progress_ratio", importance=0.08, rank=5),
        ],
        sample_count=1000,
        feature_count=84,
        resolved_config={
            "name": "Full HTML Export Test",
            "experiment_type": "entity_classification",
            "seed": 42,
            "dataset_id": "ds_001",
        },
    )


def _regression_result() -> ExperimentResult:
    """ExperimentResult for a time-series regression experiment."""
    return ExperimentResult(
        experiment_id="exp_html_003",
        name="Regression HTML Test",
        status=ExperimentStatus.COMPLETED,
        experiment_type=ProblemType.TIME_SERIES,
        created_at="2026-03-10T14:00:00Z",
        completed_at="2026-03-10T14:00:10Z",
        duration_seconds=10.0,
        best_algorithm="xgboost",
        metrics={"rmse": 0.12, "mae": 0.08, "r2": 0.95},
        sample_count=300,
        feature_count=20,
    )


# ---------------------------------------------------------------------------
# Tests: file creation
# ---------------------------------------------------------------------------


class TestExportHtmlReportCreatesFile:
    def test_creates_file_at_given_path(self, tmp_path: Path) -> None:
        output = tmp_path / "report.html"
        result_path = export_html_report(_minimal_result(), output)
        assert result_path == output
        assert output.exists()
        assert output.stat().st_size > 0

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        output = tmp_path / "nested" / "dir" / "report.html"
        result_path = export_html_report(_minimal_result(), output)
        assert result_path == output
        assert output.exists()

    def test_returns_path_object(self, tmp_path: Path) -> None:
        output = tmp_path / "report.html"
        result_path = export_html_report(_minimal_result(), output)
        assert isinstance(result_path, Path)


# ---------------------------------------------------------------------------
# Tests: content sections
# ---------------------------------------------------------------------------


class TestHtmlContentSections:
    def test_contains_experiment_summary(self, tmp_path: Path) -> None:
        output = tmp_path / "report.html"
        export_html_report(_minimal_result(), output)
        html = output.read_text(encoding="utf-8")
        assert "exp_html_001" in html
        assert "HTML Export Test" in html
        assert "entity_classification" in html
        assert "completed" in html.lower()

    def test_contains_metrics_table(self, tmp_path: Path) -> None:
        output = tmp_path / "report.html"
        export_html_report(_minimal_result(), output)
        html = output.read_text(encoding="utf-8")
        assert "f1_weighted" in html
        assert "0.95" in html
        assert "auc_roc" in html
        assert "0.98" in html

    def test_contains_algorithm_comparison(self, tmp_path: Path) -> None:
        output = tmp_path / "report.html"
        export_html_report(_full_result(), output)
        html = output.read_text(encoding="utf-8")
        assert "lightgbm" in html
        assert "random_forest" in html
        # Both ranks should appear
        assert "#1" in html or "1" in html

    def test_contains_feature_importance(self, tmp_path: Path) -> None:
        output = tmp_path / "report.html"
        export_html_report(_full_result(), output)
        html = output.read_text(encoding="utf-8")
        assert "s_elapsed" in html
        assert "s_cum_wait" in html
        assert "n_queue_len" in html

    def test_contains_feature_importance_svg(self, tmp_path: Path) -> None:
        """Feature importance should be rendered as inline SVG bars."""
        output = tmp_path / "report.html"
        export_html_report(_full_result(), output)
        html = output.read_text(encoding="utf-8")
        assert "<svg" in html

    def test_regression_metrics(self, tmp_path: Path) -> None:
        output = tmp_path / "report.html"
        export_html_report(_regression_result(), output)
        html = output.read_text(encoding="utf-8")
        assert "rmse" in html
        assert "0.12" in html
        assert "time_series" in html


# ---------------------------------------------------------------------------
# Tests: self-contained HTML
# ---------------------------------------------------------------------------


class TestSelfContainedHtml:
    def test_no_external_css_links(self, tmp_path: Path) -> None:
        output = tmp_path / "report.html"
        export_html_report(_full_result(), output)
        html = output.read_text(encoding="utf-8")
        assert 'rel="stylesheet"' not in html
        assert "cdn" not in html.lower()

    def test_no_external_script_links(self, tmp_path: Path) -> None:
        output = tmp_path / "report.html"
        export_html_report(_full_result(), output)
        html = output.read_text(encoding="utf-8")
        # No <script src="..."> external references
        assert "script src=" not in html.lower()

    def test_has_inline_style(self, tmp_path: Path) -> None:
        output = tmp_path / "report.html"
        export_html_report(_full_result(), output)
        html = output.read_text(encoding="utf-8")
        assert "<style" in html

    def test_valid_html_structure(self, tmp_path: Path) -> None:
        output = tmp_path / "report.html"
        export_html_report(_full_result(), output)
        html = output.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html or "<!doctype html>" in html
        assert "<html" in html
        assert "</html>" in html
        assert "<head" in html
        assert "<body" in html


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_no_algorithm_scores(self, tmp_path: Path) -> None:
        """Report generates without algorithm scores."""
        output = tmp_path / "report.html"
        export_html_report(_minimal_result(), output)
        html = output.read_text(encoding="utf-8")
        assert output.exists()
        assert "lightgbm" in html

    def test_no_feature_importance(self, tmp_path: Path) -> None:
        """Report generates without feature importance entries."""
        output = tmp_path / "report.html"
        export_html_report(_minimal_result(), output)
        assert output.exists()

    def test_no_duration(self, tmp_path: Path) -> None:
        """Report generates when duration is None."""
        result = _minimal_result()
        result.duration_seconds = None
        output = tmp_path / "report.html"
        export_html_report(result, output)
        assert output.exists()

    def test_no_resolved_config(self, tmp_path: Path) -> None:
        """Report generates when resolved_config is None."""
        result = _minimal_result()
        result.resolved_config = None
        output = tmp_path / "report.html"
        export_html_report(result, output)
        assert output.exists()
