"""Data quality validation for ML/RL OS datasets.

Provides automated quality checks to surface issues before
experiments are run: missing values, constant columns, duplicates,
and minimum sample-size requirements.
"""

from __future__ import annotations

import logging

import polars as pl
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Quality thresholds ──────────────────────────────────────────────
_QUALITY_THRESHOLDS: list[tuple[int, str]] = [
    (1000, "excellent"),
    (200, "good"),
    (50, "minimal"),
]
_QUALITY_DEFAULT = "insufficient"

_MISSING_RATE_THRESHOLD = 0.30
_DUPLICATE_RATE_THRESHOLD = 0.10


class DataQualityReport(BaseModel):
    """Results of automated data quality checks on a DataFrame."""

    row_count: int
    column_count: int
    missing_rates: dict[str, float] = Field(default_factory=dict)
    constant_columns: list[str] = Field(default_factory=list)
    duplicate_count: int = 0
    quality: str = "insufficient"
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _classify_quality(row_count: int) -> str:
    """Classify data quality level based on row count."""
    for threshold, label in _QUALITY_THRESHOLDS:
        if row_count >= threshold:
            return label
    return _QUALITY_DEFAULT


def validate_data_quality(df: pl.DataFrame) -> DataQualityReport:
    """Run data quality checks on a Polars DataFrame.

    Checks performed:
    - **Row count quality**: insufficient (<50), minimal (50-200),
      good (200-1000), excellent (>1000).
    - **Missing rate per column**: flagged as issue if >30%.
    - **Constant columns**: columns with exactly one unique non-null value.
    - **Duplicate rows**: flagged as issue if >10% of rows are duplicates.

    Args:
        df: The DataFrame to validate.

    Returns:
        A DataQualityReport summarising findings.
    """
    row_count = len(df)
    column_count = len(df.columns)
    issues: list[str] = []
    warnings: list[str] = []

    # ── Quality level ───────────────────────────────────────────
    quality = _classify_quality(row_count)
    if quality == "insufficient":
        issues.append(
            f"Insufficient data: only {row_count} rows (minimum 50 recommended)"
        )

    # ── Missing rates ───────────────────────────────────────────
    missing_rates: dict[str, float] = {}
    for col_name in df.columns:
        null_count = df[col_name].null_count()
        rate = null_count / row_count if row_count > 0 else 0.0
        rate = round(rate, 6)
        missing_rates[col_name] = rate
        if rate > _MISSING_RATE_THRESHOLD:
            issues.append(
                f"Column '{col_name}' has {rate:.1%} missing values "
                f"(threshold: {_MISSING_RATE_THRESHOLD:.0%})"
            )

    # ── Constant columns ────────────────────────────────────────
    constant_columns: list[str] = []
    for col_name in df.columns:
        n_unique = df[col_name].drop_nulls().n_unique()
        if n_unique <= 1 and row_count > 0:
            constant_columns.append(col_name)
            warnings.append(
                f"Column '{col_name}' is constant (single unique value)"
            )

    # ── Duplicate rows ──────────────────────────────────────────
    duplicate_count = 0
    if row_count > 0:
        unique_count = df.unique().height
        duplicate_count = row_count - unique_count
        duplicate_rate = duplicate_count / row_count
        if duplicate_rate > _DUPLICATE_RATE_THRESHOLD:
            issues.append(
                f"{duplicate_count} duplicate rows ({duplicate_rate:.1%} of data, "
                f"threshold: {_DUPLICATE_RATE_THRESHOLD:.0%})"
            )

    report = DataQualityReport(
        row_count=row_count,
        column_count=column_count,
        missing_rates=missing_rates,
        constant_columns=constant_columns,
        duplicate_count=duplicate_count,
        quality=quality,
        issues=issues,
        warnings=warnings,
    )

    logger.info(
        "Data quality: %s (%d rows, %d issues, %d warnings)",
        quality,
        row_count,
        len(issues),
        len(warnings),
    )
    return report
