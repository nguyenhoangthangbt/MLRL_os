"""Target auto-discovery from dataset schema.

Inspects loaded data to find all valid prediction targets and marks
sensible defaults. The experiment pipeline uses discovered targets
to present options in the Builder UI and to auto-configure when the
user provides zero configuration.
"""

from __future__ import annotations

import logging

import polars as pl

from mlrl_os.core.dataset import DatasetMeta, compute_column_info
from mlrl_os.core.types import AvailableTargets, ColumnInfo, TargetInfo, TaskType

logger = logging.getLogger(__name__)

# Snapshot columns that are structural, not valid prediction targets
_SNAPSHOT_EXCLUDE = frozenset({"ts", "bucket_idx"})

# Default time-series target (overridable)
_DEFAULT_TS_TARGET = "avg_wait"

# Canonical entity target column
_ENTITY_STATUS_COLUMN = "status"


class TargetDiscovery:
    """Discover available prediction targets from a dataset.

    For **time-series** data (snapshots), every numeric column except
    structural indices (``ts``, ``bucket_idx``) is a candidate target.
    The default is ``avg_wait``.

    For **entity** data (trajectories), the primary target is
    ``episode_status`` (mapped to canonical ``status``). Additional
    derived targets ``sla_breach`` and ``delay_severity`` are offered
    when the underlying columns are present.
    """

    def discover(
        self,
        dataset_meta: DatasetMeta,
        snapshots_df: pl.DataFrame | None = None,
        trajectories_df: pl.DataFrame | None = None,
    ) -> AvailableTargets:
        """Discover all available prediction targets.

        Args:
            dataset_meta: Registered dataset metadata (used for ``id``
                and pre-computed column info when DataFrames are not
                provided).
            snapshots_df: Optional snapshots DataFrame for live
                inspection.
            trajectories_df: Optional trajectories DataFrame for live
                inspection.

        Returns:
            AvailableTargets with time-series and entity target lists.
        """
        ts_targets: list[TargetInfo] = []
        entity_targets: list[TargetInfo] = []

        # ── Time-series targets from snapshots ──────────────────
        if snapshots_df is not None:
            ts_targets = self._discover_ts_targets(snapshots_df)
        elif dataset_meta.snapshot_columns:
            ts_targets = self._discover_ts_targets_from_info(
                dataset_meta.snapshot_columns
            )

        # ── Entity targets from trajectories ────────────────────
        if trajectories_df is not None:
            entity_targets = self._discover_entity_targets(trajectories_df)
        elif dataset_meta.trajectory_columns:
            entity_targets = self._discover_entity_targets_from_info(
                dataset_meta.trajectory_columns
            )

        logger.info(
            "Discovered %d time-series targets, %d entity targets for dataset %s",
            len(ts_targets),
            len(entity_targets),
            dataset_meta.id,
        )

        return AvailableTargets(
            dataset_id=dataset_meta.id,
            time_series_targets=ts_targets,
            entity_targets=entity_targets,
        )

    # ── Private: time-series target discovery ───────────────────

    def _discover_ts_targets(
        self, df: pl.DataFrame
    ) -> list[TargetInfo]:
        """Discover time-series targets from a live snapshots DataFrame."""
        col_infos = compute_column_info(df)
        return self._discover_ts_targets_from_info(col_infos)

    def _discover_ts_targets_from_info(
        self, columns: list[ColumnInfo]
    ) -> list[TargetInfo]:
        """Build target list from pre-computed ColumnInfo."""
        targets: list[TargetInfo] = []

        for col in columns:
            if col.name in _SNAPSHOT_EXCLUDE:
                continue
            if not col.is_numeric:
                continue

            is_default = col.name == _DEFAULT_TS_TARGET
            target = TargetInfo(
                column=col.name,
                task_type=TaskType.REGRESSION,
                is_default=is_default,
                null_rate=col.null_rate,
                mean=col.mean,
                std=col.std,
                min=col.min,
                max=col.max,
                unique_count=col.unique_count,
            )
            targets.append(target)

        # If default target not found, mark the first target as default
        if targets and not any(t.is_default for t in targets):
            targets[0].is_default = True

        return targets

    # ── Private: entity target discovery ────────────────────────

    def _discover_entity_targets(
        self, df: pl.DataFrame
    ) -> list[TargetInfo]:
        """Discover entity targets from a live trajectories DataFrame."""
        targets: list[TargetInfo] = []

        # Primary target: episode_status (canonical: "status")
        if _ENTITY_STATUS_COLUMN in df.columns:
            targets.append(self._build_classification_target(
                df, _ENTITY_STATUS_COLUMN, is_default=True
            ))

        # Derived target: sla_breach (binary classification)
        # Derivable if cum_sla_breaches or sla columns exist
        if self._can_derive_sla_breach(df):
            targets.append(self._build_derived_sla_breach(df))

        # Derived target: delay_severity (multi-class)
        # Derivable if total_time column exists
        if "total_time" in df.columns:
            targets.append(self._build_derived_delay_severity(df))

        # If no targets found, fall back to any categorical columns
        if not targets:
            col_infos = compute_column_info(df)
            for col in col_infos:
                if col.is_categorical:
                    targets.append(TargetInfo(
                        column=col.name,
                        task_type=TaskType.CLASSIFICATION,
                        is_default=len(targets) == 0,
                        null_rate=col.null_rate,
                        unique_count=col.unique_count,
                        classes=col.categories,
                    ))

        return targets

    def _discover_entity_targets_from_info(
        self, columns: list[ColumnInfo]
    ) -> list[TargetInfo]:
        """Build entity target list from pre-computed ColumnInfo."""
        targets: list[TargetInfo] = []

        for col in columns:
            if col.name == _ENTITY_STATUS_COLUMN:
                classes = col.categories
                class_balance: dict[str, float] | None = None
                if col.category_counts:
                    total = sum(col.category_counts.values())
                    if total > 0:
                        class_balance = {
                            k: round(v / total, 6)
                            for k, v in col.category_counts.items()
                        }
                targets.append(TargetInfo(
                    column=col.name,
                    task_type=TaskType.CLASSIFICATION,
                    is_default=True,
                    null_rate=col.null_rate,
                    unique_count=col.unique_count,
                    classes=classes,
                    class_balance=class_balance,
                ))

        # Derived targets are only discoverable from live DataFrames,
        # not from pre-computed info, since they require computation.
        return targets

    # ── Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _build_classification_target(
        df: pl.DataFrame,
        column: str,
        is_default: bool = False,
    ) -> TargetInfo:
        """Build a TargetInfo for a classification column."""
        series = df[column].drop_nulls()
        if series.dtype != pl.Utf8:
            series = series.cast(pl.Utf8)

        classes = sorted(series.unique().to_list())
        total = len(series)
        counts = series.value_counts()
        class_balance: dict[str, float] = {}
        for row in counts.iter_rows(named=True):
            label = str(row[column])
            class_balance[label] = round(row["count"] / total, 6) if total > 0 else 0.0

        null_rate = df[column].null_count() / len(df) if len(df) > 0 else 0.0

        return TargetInfo(
            column=column,
            task_type=TaskType.CLASSIFICATION,
            is_default=is_default,
            null_rate=round(null_rate, 6),
            unique_count=len(classes),
            classes=classes,
            class_balance=class_balance,
        )

    @staticmethod
    def _can_derive_sla_breach(df: pl.DataFrame) -> bool:
        """Check if SLA breach can be derived from available columns."""
        return "s_cum_wait" in df.columns or "total_time" in df.columns

    @staticmethod
    def _build_derived_sla_breach(df: pl.DataFrame) -> TargetInfo:
        """Build a derived binary SLA breach target.

        Uses ``done`` column combined with ``status`` to infer breaches,
        or falls back to a placeholder when exact derivation requires
        domain thresholds (set at experiment time).
        """
        return TargetInfo(
            column="sla_breach",
            task_type=TaskType.CLASSIFICATION,
            is_default=False,
            classes=["no_breach", "breach"],
        )

    @staticmethod
    def _build_derived_delay_severity(df: pl.DataFrame) -> TargetInfo:
        """Build a derived multi-class delay severity target.

        Severity bins are determined at feature-engineering time based
        on the distribution of ``total_time``. This target info records
        the availability; actual binning is deferred.
        """
        return TargetInfo(
            column="delay_severity",
            task_type=TaskType.CLASSIFICATION,
            is_default=False,
            classes=["low", "medium", "high"],
        )
