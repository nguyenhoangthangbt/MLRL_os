"""Tests for mlrl_os.data.validation."""

from __future__ import annotations

import polars as pl
import pytest

from mlrl_os.data.validation import DataQualityReport, validate_data_quality


# =====================================================================
# Quality classification tests
# =====================================================================


class TestQualityClassification:
    """Tests for row-count-based quality classification."""

    def test_excellent_quality(self) -> None:
        df = pl.DataFrame({"a": list(range(1000))})
        report = validate_data_quality(df)
        assert report.quality == "excellent"

    def test_good_quality(self) -> None:
        df = pl.DataFrame({"a": list(range(200))})
        report = validate_data_quality(df)
        assert report.quality == "good"

    def test_minimal_quality(self) -> None:
        df = pl.DataFrame({"a": list(range(50))})
        report = validate_data_quality(df)
        assert report.quality == "minimal"

    def test_insufficient_quality(self) -> None:
        df = pl.DataFrame({"a": list(range(10))})
        report = validate_data_quality(df)
        assert report.quality == "insufficient"

    def test_insufficient_generates_issue(self) -> None:
        df = pl.DataFrame({"a": list(range(10))})
        report = validate_data_quality(df)
        assert any("Insufficient" in issue for issue in report.issues)


# =====================================================================
# Basic report fields
# =====================================================================


class TestReportFields:
    """Tests for DataQualityReport basic fields."""

    def test_row_and_column_counts(self) -> None:
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        report = validate_data_quality(df)
        assert report.row_count == 3
        assert report.column_count == 2

    def test_report_is_pydantic_model(self) -> None:
        df = pl.DataFrame({"a": [1]})
        report = validate_data_quality(df)
        assert isinstance(report, DataQualityReport)
        # Verify serializable
        d = report.model_dump()
        assert "quality" in d


# =====================================================================
# Missing rate checks
# =====================================================================


class TestMissingRates:
    """Tests for missing value detection."""

    def test_no_missing_values(self) -> None:
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        report = validate_data_quality(df)
        assert report.missing_rates["a"] == 0.0
        assert report.missing_rates["b"] == 0.0

    def test_missing_rate_computed_correctly(self) -> None:
        df = pl.DataFrame({"a": [1, None, None, None, 5]})
        report = validate_data_quality(df)
        assert report.missing_rates["a"] == pytest.approx(0.6, abs=1e-5)

    def test_high_missing_rate_generates_issue(self) -> None:
        # >30% missing should flag an issue
        df = pl.DataFrame({"a": [1, None, None, None, 5]})
        report = validate_data_quality(df)
        issues_about_a = [i for i in report.issues if "'a'" in i]
        assert len(issues_about_a) > 0

    def test_low_missing_rate_no_issue(self) -> None:
        # 10% missing should NOT flag an issue
        values: list[int | None] = list(range(9)) + [None]
        df = pl.DataFrame({"a": values})
        report = validate_data_quality(df)
        missing_issues = [
            i for i in report.issues if "'a'" in i and "missing" in i
        ]
        assert len(missing_issues) == 0


# =====================================================================
# Constant column checks
# =====================================================================


class TestConstantColumns:
    """Tests for constant column detection."""

    def test_constant_column_detected(self) -> None:
        df = pl.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]})
        report = validate_data_quality(df)
        assert "a" in report.constant_columns
        assert "b" not in report.constant_columns

    def test_constant_column_generates_warning(self) -> None:
        df = pl.DataFrame({"const": [42, 42, 42]})
        report = validate_data_quality(df)
        assert any("const" in w for w in report.warnings)

    def test_all_null_column_is_constant(self) -> None:
        df = pl.DataFrame({"a": [None, None, None]})
        report = validate_data_quality(df)
        assert "a" in report.constant_columns

    def test_varied_column_not_constant(self) -> None:
        df = pl.DataFrame({"a": [1, 2, 3]})
        report = validate_data_quality(df)
        assert "a" not in report.constant_columns


# =====================================================================
# Duplicate row checks
# =====================================================================


class TestDuplicateRows:
    """Tests for duplicate row detection."""

    def test_no_duplicates(self) -> None:
        df = pl.DataFrame({"a": [1, 2, 3]})
        report = validate_data_quality(df)
        assert report.duplicate_count == 0

    def test_duplicates_counted(self) -> None:
        df = pl.DataFrame({"a": [1, 1, 2]})
        report = validate_data_quality(df)
        assert report.duplicate_count == 1

    def test_high_duplicate_rate_generates_issue(self) -> None:
        # 50% duplicates, well above 10% threshold
        df = pl.DataFrame({"a": [1, 1, 1, 1, 2, 3]})
        report = validate_data_quality(df)
        dup_issues = [i for i in report.issues if "duplicate" in i]
        assert len(dup_issues) > 0

    def test_low_duplicate_rate_no_issue(self) -> None:
        # 1 duplicate in 100 rows = 1%, below threshold
        values = list(range(99)) + [0]
        df = pl.DataFrame({"a": values})
        report = validate_data_quality(df)
        dup_issues = [i for i in report.issues if "duplicate" in i]
        assert len(dup_issues) == 0


# =====================================================================
# Edge cases
# =====================================================================


class TestEdgeCases:
    """Tests for edge cases in validation."""

    def test_single_row(self) -> None:
        df = pl.DataFrame({"a": [1]})
        report = validate_data_quality(df)
        assert report.row_count == 1
        assert report.quality == "insufficient"

    def test_empty_dataframe(self) -> None:
        df = pl.DataFrame({"a": pl.Series([], dtype=pl.Int64)})
        report = validate_data_quality(df)
        assert report.row_count == 0
        assert report.duplicate_count == 0

    def test_multiple_issues_returned(self) -> None:
        # Insufficient rows AND high missing rate
        df = pl.DataFrame({"a": [1, None, None, None, None]})
        report = validate_data_quality(df)
        assert len(report.issues) >= 2  # insufficient + missing
