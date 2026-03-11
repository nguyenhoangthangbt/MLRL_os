"""Validation gate: all validation rules from the CONSTITUTION.

Experiments MUST pass validation before training. All errors are collected
and returned at once so users can fix everything in a single pass. There is
no bypass flag.
"""

from __future__ import annotations

import polars as pl
from pydantic import BaseModel

from mlrl_os.config.resolver import parse_duration
from mlrl_os.config.schemas import (
    ResolvedEntityFeatures,
    ResolvedExperimentConfig,
    ResolvedTimeSeriesFeatures,
)
from mlrl_os.core.types import CVStrategy, ObservationPoint, ProblemType
from mlrl_os.features.target_derivation import is_derived_target

# ---------------------------------------------------------------------------
# Known registries
# ---------------------------------------------------------------------------

KNOWN_ALGORITHMS: frozenset[str] = frozenset({
    "lightgbm",
    "xgboost",
    "random_forest",
    "extra_trees",
    "linear",
    "lstm",
})

KNOWN_METRICS: frozenset[str] = frozenset({
    "rmse",
    "mae",
    "mape",
    "r2",
    "f1_weighted",
    "f1_macro",
    "auc_roc",
    "precision",
    "recall",
    "accuracy",
})

# Columns whose cumulative nature makes them poor prediction targets
_CUMULATIVE_COLUMNS: frozenset[str] = frozenset({
    "cumulative_throughput",
    "cumulative_arrivals",
    "total_processed",
})

# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class ValidationError(BaseModel):
    """A single validation error."""

    code: str
    field: str
    message: str
    suggestion: str | None = None


class ValidationResult(BaseModel):
    """Outcome of running the validation gate."""

    valid: bool
    errors: list[ValidationError] = []
    warnings: list[ValidationError] = []


# ---------------------------------------------------------------------------
# Validation gate
# ---------------------------------------------------------------------------


class ValidationGate:
    """Validates a resolved experiment config against the data.

    Implements rules V-01 through V-11 (universal), VT-01 through VT-06
    (time-series specific), and VE-01 through VE-04 (entity specific).
    All errors are collected -- not just the first.
    """

    def validate(
        self,
        config: ResolvedExperimentConfig,
        df: pl.DataFrame,
    ) -> ValidationResult:
        """Run all applicable validation rules.

        Args:
            config: Fully resolved experiment configuration.
            df: The actual data (snapshot or trajectory DataFrame).

        Returns:
            A ``ValidationResult`` indicating pass/fail with all errors
            and warnings collected.
        """
        errors: list[ValidationError] = []
        warnings: list[ValidationError] = []

        # Universal rules
        self._v02_min_rows(config, df, errors)
        self._v03_target_exists(config, df, errors)
        self._v04_target_not_in_features(config, errors)
        self._v05_feature_columns_exist(config, df, errors)
        self._v06_feature_columns_numeric(config, df, errors)
        self._v07_seed_non_negative(config, errors)
        self._v08_algorithms_registered(config, errors)
        self._v09_cv_folds(config, errors)
        self._v10_metrics_registered(config, errors)
        self._v11_target_null_rate(config, df, errors)

        # Problem-type-specific rules
        if config.experiment_type == ProblemType.TIME_SERIES:
            self._vt01_ts_column(df, errors)
            self._vt02_lookback_horizon_duration(config, df, errors)
            self._vt03_lookback_windows(config, df, errors)
            self._vt04_target_numeric(config, df, errors)
            self._vt05_lag_intervals(config, errors)
            self._vt06_cv_temporal(config, errors)
        else:
            self._ve01_target_categorical(config, df, errors)
            self._ve02_class_min_samples(config, df, errors)
            self._ve03_max_classes(config, df, errors)
            self._ve04_observation_point(config, errors)

        # Warnings
        self._warn_cumulative_target(config, warnings)

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    # -----------------------------------------------------------------------
    # Universal rules (V-01 to V-11)
    # -----------------------------------------------------------------------

    @staticmethod
    def _v02_min_rows(
        config: ResolvedExperimentConfig,
        df: pl.DataFrame,
        errors: list[ValidationError],
    ) -> None:
        """V-02: Dataset must have a minimum number of rows."""
        min_rows = 50 if config.experiment_type == ProblemType.TIME_SERIES else 100
        if len(df) < min_rows:
            errors.append(ValidationError(
                code="V-02",
                field="dataset",
                message=(
                    f"Dataset has {len(df)} rows but requires at least {min_rows} "
                    f"for {config.experiment_type.value} experiments."
                ),
                suggestion="Provide a larger dataset or use a different problem type.",
            ))

    @staticmethod
    def _v03_target_exists(
        config: ResolvedExperimentConfig,
        df: pl.DataFrame,
        errors: list[ValidationError],
    ) -> None:
        """V-03: Target column must exist in the data (skip for derived targets)."""
        target = config.features.target
        if is_derived_target(target):
            return  # Derived targets are computed during feature engineering
        if target not in df.columns:
            errors.append(ValidationError(
                code="V-03",
                field="features.target",
                message=f"Target column '{target}' not found in dataset.",
                suggestion=f"Available columns: {', '.join(df.columns[:20])}",
            ))

    @staticmethod
    def _v04_target_not_in_features(
        config: ResolvedExperimentConfig,
        errors: list[ValidationError],
    ) -> None:
        """V-04: Target must not appear in feature columns (data leakage)."""
        target = config.features.target
        if target in config.features.feature_columns:
            errors.append(ValidationError(
                code="V-04",
                field="features.feature_columns",
                message=(
                    f"Target column '{target}' is included in feature_columns. "
                    "This causes data leakage."
                ),
                suggestion="Remove the target from feature_columns.",
            ))

    @staticmethod
    def _v05_feature_columns_exist(
        config: ResolvedExperimentConfig,
        df: pl.DataFrame,
        errors: list[ValidationError],
    ) -> None:
        """V-05: All feature columns must exist in the data."""
        missing = [c for c in config.features.feature_columns if c not in df.columns]
        if missing:
            errors.append(ValidationError(
                code="V-05",
                field="features.feature_columns",
                message=f"Feature columns not found in dataset: {missing}",
                suggestion="Check column names or remove missing columns from the config.",
            ))

    @staticmethod
    def _v06_feature_columns_numeric(
        config: ResolvedExperimentConfig,
        df: pl.DataFrame,
        errors: list[ValidationError],
    ) -> None:
        """V-06: All feature columns must be numeric."""
        non_numeric = [
            c
            for c in config.features.feature_columns
            if c in df.columns and not df[c].dtype.is_numeric()
        ]
        if non_numeric:
            errors.append(ValidationError(
                code="V-06",
                field="features.feature_columns",
                message=f"Non-numeric feature columns: {non_numeric}",
                suggestion=(
                    "Remove non-numeric columns from feature_columns "
                    "or encode them before training."
                ),
            ))

    @staticmethod
    def _v07_seed_non_negative(
        config: ResolvedExperimentConfig,
        errors: list[ValidationError],
    ) -> None:
        """V-07: Seed must be non-negative."""
        if config.seed < 0:
            errors.append(ValidationError(
                code="V-07",
                field="seed",
                message=f"Seed must be non-negative, got {config.seed}.",
                suggestion="Use a non-negative integer for reproducibility.",
            ))

    @staticmethod
    def _v08_algorithms_registered(
        config: ResolvedExperimentConfig,
        errors: list[ValidationError],
    ) -> None:
        """V-08: All algorithm names must be registered."""
        unknown = [a for a in config.model.algorithms if a not in KNOWN_ALGORITHMS]
        if unknown:
            errors.append(ValidationError(
                code="V-08",
                field="model.algorithms",
                message=f"Unknown algorithms: {unknown}",
                suggestion=f"Known algorithms: {sorted(KNOWN_ALGORITHMS)}",
            ))

    @staticmethod
    def _v09_cv_folds(
        config: ResolvedExperimentConfig,
        errors: list[ValidationError],
    ) -> None:
        """V-09: CV folds must be >= 2."""
        folds = config.model.cross_validation.folds
        if folds < 2:
            errors.append(ValidationError(
                code="V-09",
                field="model.cross_validation.folds",
                message=f"CV folds must be >= 2, got {folds}.",
            ))

    @staticmethod
    def _v10_metrics_registered(
        config: ResolvedExperimentConfig,
        errors: list[ValidationError],
    ) -> None:
        """V-10: All metric names must be registered."""
        unknown = [m for m in config.evaluation.metrics if m not in KNOWN_METRICS]
        if unknown:
            errors.append(ValidationError(
                code="V-10",
                field="evaluation.metrics",
                message=f"Unknown metrics: {unknown}",
                suggestion=f"Known metrics: {sorted(KNOWN_METRICS)}",
            ))

    @staticmethod
    def _v11_target_null_rate(
        config: ResolvedExperimentConfig,
        df: pl.DataFrame,
        errors: list[ValidationError],
    ) -> None:
        """V-11: Target null rate must be <= 10%."""
        target = config.features.target
        if is_derived_target(target):
            return  # Derived targets are computed during feature engineering
        if target not in df.columns:
            return  # Already caught by V-03
        null_rate = df[target].null_count() / len(df) if len(df) > 0 else 0.0
        if null_rate > 0.10:
            errors.append(ValidationError(
                code="V-11",
                field="features.target",
                message=(
                    f"Target column '{target}' has {null_rate:.1%} null values "
                    "(maximum allowed is 10%)."
                ),
                suggestion="Impute or filter null target values before training.",
            ))

    # -----------------------------------------------------------------------
    # Time-series rules (VT-01 to VT-06)
    # -----------------------------------------------------------------------

    @staticmethod
    def _vt01_ts_column(
        df: pl.DataFrame,
        errors: list[ValidationError],
    ) -> None:
        """VT-01: Data must have a 'ts' column."""
        if "ts" not in df.columns:
            errors.append(ValidationError(
                code="VT-01",
                field="dataset",
                message="Time-series data must contain a 'ts' timestamp column.",
                suggestion="Ensure the dataset has a 'ts' column with timestamps.",
            ))

    @staticmethod
    def _vt02_lookback_horizon_duration(
        config: ResolvedExperimentConfig,
        df: pl.DataFrame,
        errors: list[ValidationError],
    ) -> None:
        """VT-02: lookback + horizon must be <= data duration."""
        features = config.features
        if not isinstance(features, ResolvedTimeSeriesFeatures):
            return
        if "ts" not in df.columns:
            return  # Already caught by VT-01

        try:
            ts_col = df["ts"]
            # Handle both datetime and numeric timestamp columns
            if ts_col.dtype.is_numeric():
                ts_min = float(ts_col.min())  # type: ignore[arg-type]
                ts_max = float(ts_col.max())  # type: ignore[arg-type]
                data_duration = ts_max - ts_min
            else:
                ts_min = ts_col.min()
                ts_max = ts_col.max()
                if ts_min is None or ts_max is None:
                    return
                data_duration = (ts_max - ts_min).total_seconds()

            required = parse_duration(features.lookback) + parse_duration(features.horizon)
            if required > data_duration:
                errors.append(ValidationError(
                    code="VT-02",
                    field="features.lookback",
                    message=(
                        f"lookback ({features.lookback}) + horizon ({features.horizon}) "
                        f"= {required}s exceeds data duration of {data_duration:.0f}s."
                    ),
                    suggestion="Reduce lookback/horizon or provide a longer dataset.",
                ))
        except (ValueError, TypeError):
            return  # Duration parsing issues will surface elsewhere

    @staticmethod
    def _vt03_lookback_windows(
        config: ResolvedExperimentConfig,
        df: pl.DataFrame,
        errors: list[ValidationError],
    ) -> None:
        """VT-03: lookback must produce >= 50 sliding windows from the data.

        Sliding windows: each data point after the lookback period is one
        usable sample.  The count is ``(data_duration - lookback - horizon)
        / bucket_interval``, **not** ``data_duration / lookback``.
        """
        features = config.features
        if not isinstance(features, ResolvedTimeSeriesFeatures):
            return
        if "ts" not in df.columns:
            return

        try:
            ts_col = df["ts"]
            if ts_col.dtype.is_numeric():
                ts_min = float(ts_col.min())  # type: ignore[arg-type]
                ts_max = float(ts_col.max())  # type: ignore[arg-type]
                data_duration = ts_max - ts_min
            else:
                ts_min = ts_col.min()
                ts_max = ts_col.max()
                if ts_min is None or ts_max is None:
                    return
                data_duration = (ts_max - ts_min).total_seconds()

            lookback_secs = parse_duration(features.lookback)
            horizon_secs = parse_duration(features.horizon)
            if lookback_secs <= 0:
                return

            # Estimate bucket interval from median gap between consecutive timestamps
            n_rows = len(df)
            if n_rows < 2:
                return
            bucket_interval = data_duration / (n_rows - 1)
            if bucket_interval <= 0:
                return

            # Sliding windows: each point after lookback+horizon is a sample
            usable_duration = data_duration - lookback_secs - horizon_secs
            if usable_duration <= 0:
                # VT-02 already catches lookback+horizon > data_duration
                return
            windows = usable_duration / bucket_interval
            if windows < 50:
                errors.append(ValidationError(
                    code="VT-03",
                    field="features.lookback",
                    message=(
                        f"Lookback of {features.lookback} produces ~{windows:.0f} "
                        "usable sliding windows from the data (minimum 50 required)."
                    ),
                    suggestion="Reduce lookback duration or provide a longer dataset.",
                ))
        except (ValueError, TypeError):
            return

    @staticmethod
    def _vt04_target_numeric(
        config: ResolvedExperimentConfig,
        df: pl.DataFrame,
        errors: list[ValidationError],
    ) -> None:
        """VT-04: Time-series target must be numeric."""
        target = config.features.target
        if target not in df.columns:
            return  # Already caught by V-03
        if not df[target].dtype.is_numeric():
            errors.append(ValidationError(
                code="VT-04",
                field="features.target",
                message=f"Time-series target '{target}' must be numeric, got {df[target].dtype}.",
                suggestion="Choose a numeric column as the target for time-series forecasting.",
            ))

    @staticmethod
    def _vt05_lag_intervals(
        config: ResolvedExperimentConfig,
        errors: list[ValidationError],
    ) -> None:
        """VT-05: All lag intervals must be <= lookback."""
        features = config.features
        if not isinstance(features, ResolvedTimeSeriesFeatures):
            return

        try:
            lookback_secs = parse_duration(features.lookback)
            oversized = [
                lag
                for lag in features.lag_intervals
                if parse_duration(lag) > lookback_secs
            ]
            if oversized:
                errors.append(ValidationError(
                    code="VT-05",
                    field="features.lag_intervals",
                    message=(
                        f"Lag intervals {oversized} exceed lookback "
                        f"of {features.lookback}."
                    ),
                    suggestion="Remove lag intervals larger than the lookback window.",
                ))
        except ValueError:
            return  # Parsing errors handled separately

    @staticmethod
    def _vt06_cv_temporal(
        config: ResolvedExperimentConfig,
        errors: list[ValidationError],
    ) -> None:
        """VT-06: Time-series must use temporal CV strategy."""
        if config.model.cross_validation.strategy != CVStrategy.TEMPORAL:
            errors.append(ValidationError(
                code="VT-06",
                field="model.cross_validation.strategy",
                message=(
                    f"Time-series experiments require temporal CV, "
                    f"got '{config.model.cross_validation.strategy.value}'."
                ),
                suggestion="Set cross_validation.strategy to 'temporal'.",
            ))

    # -----------------------------------------------------------------------
    # Entity classification rules (VE-01 to VE-04)
    # -----------------------------------------------------------------------

    @staticmethod
    def _ve01_target_categorical(
        config: ResolvedExperimentConfig,
        df: pl.DataFrame,
        errors: list[ValidationError],
    ) -> None:
        """VE-01: Target must be categorical or have <= 20 unique values."""
        target = config.features.target
        if is_derived_target(target):
            return  # Derived targets are always categorical
        if target not in df.columns:
            return  # Already caught by V-03
        col = df[target]
        is_categorical = col.dtype == pl.Utf8 or col.dtype == pl.Categorical
        if not is_categorical and col.n_unique() > 20:
            errors.append(ValidationError(
                code="VE-01",
                field="features.target",
                message=(
                    f"Entity target '{target}' has {col.n_unique()} unique values. "
                    "Must be categorical or have <= 20 unique values."
                ),
                suggestion=(
                    "Choose a categorical target or bin the values into "
                    "<= 20 classes."
                ),
            ))

    @staticmethod
    def _ve02_class_min_samples(
        config: ResolvedExperimentConfig,
        df: pl.DataFrame,
        errors: list[ValidationError],
    ) -> None:
        """VE-02: All target classes must have >= 10 samples."""
        target = config.features.target
        if is_derived_target(target):
            return  # Derived targets are validated post-derivation
        if target not in df.columns:
            return
        value_counts = df[target].drop_nulls().value_counts()
        # value_counts returns a DataFrame with the target column and "count"
        small_classes = value_counts.filter(pl.col("count") < 10)
        if len(small_classes) > 0:
            class_names = small_classes[target].to_list()
            errors.append(ValidationError(
                code="VE-02",
                field="features.target",
                message=(
                    f"Target classes with fewer than 10 samples: {class_names}. "
                    "Each class must have at least 10 samples."
                ),
                suggestion="Merge rare classes or collect more data.",
            ))

    @staticmethod
    def _ve03_max_classes(
        config: ResolvedExperimentConfig,
        df: pl.DataFrame,
        errors: list[ValidationError],
    ) -> None:
        """VE-03: Number of classes must be <= 20."""
        target = config.features.target
        if is_derived_target(target):
            return  # Derived targets have at most 3 classes
        if target not in df.columns:
            return
        n_classes = df[target].drop_nulls().n_unique()
        if n_classes > 20:
            errors.append(ValidationError(
                code="VE-03",
                field="features.target",
                message=(
                    f"Target has {n_classes} classes (maximum 20 allowed)."
                ),
                suggestion="Reduce the number of classes by merging or filtering.",
            ))

    @staticmethod
    def _ve04_observation_point(
        config: ResolvedExperimentConfig,
        errors: list[ValidationError],
    ) -> None:
        """VE-04: observation_point must be a valid enum value."""
        features = config.features
        if not isinstance(features, ResolvedEntityFeatures):
            return
        # Since we use the enum in the model, this is already enforced by
        # Pydantic. But we add an explicit check for defensive programming.
        try:
            ObservationPoint(features.observation_point)
        except ValueError:
            errors.append(ValidationError(
                code="VE-04",
                field="features.observation_point",
                message=(
                    f"Invalid observation_point: '{features.observation_point}'. "
                    f"Valid values: {[e.value for e in ObservationPoint]}"
                ),
            ))

    # -----------------------------------------------------------------------
    # Warnings
    # -----------------------------------------------------------------------

    @staticmethod
    def _warn_cumulative_target(
        config: ResolvedExperimentConfig,
        warnings: list[ValidationError],
    ) -> None:
        """Warn if the target column is a cumulative metric."""
        target = config.features.target
        if target in _CUMULATIVE_COLUMNS:
            warnings.append(ValidationError(
                code="W-01",
                field="features.target",
                message=(
                    f"Target '{target}' appears to be a cumulative metric. "
                    "Cumulative columns tend to be trivially predictable "
                    "and may not produce meaningful models."
                ),
                suggestion="Consider using a rate-based or instantaneous metric instead.",
            ))
