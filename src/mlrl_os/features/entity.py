"""Entity feature engineering.

Transforms entity trajectory records into classification features
by extracting state vectors, computing derived features, and
filtering by observation point.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from mlrl_os.core.types import (
    FeatureMatrix,
    ObservationPoint,
    ProblemType,
    TaskType,
)

# Canonical state column groups (from CONTRACTS.md)
ENTITY_STATE_COLS = [
    "s_priority",
    "s_elapsed",
    "s_cum_wait",
    "s_cum_processing",
    "s_wait_ratio",
    "s_steps_done",
    "s_cum_cost",
    "s_cum_transit_cost",
    "s_cum_setup",
    "s_source_idx",
]

NODE_STATE_COLS = [
    "s_node_util",
    "s_node_avg_queue",
    "s_node_max_queue",
    "s_node_avg_wait",
    "s_node_avg_processing",
    "s_node_throughput",
    "s_node_concurrency",
    "s_node_setup_count",
    "s_node_mutation_count",
]

SYSTEM_STATE_COLS = [
    "s_sys_util",
    "s_sys_throughput",
    "s_sys_bottleneck_util",
    "s_sys_node_count",
    "s_sys_total_capacity",
]


class EntityFeatureEngine:
    """Transform entity trajectory records into classification features."""

    def build_features(
        self,
        df: pl.DataFrame,
        target: str,
        observation_point: ObservationPoint = ObservationPoint.ALL_STEPS,
        include_entity_state: bool = True,
        include_node_state: bool = True,
        include_system_state: bool = True,
        add_progress_ratio: bool = True,
        add_wait_trend: bool = True,
        feature_columns: list[str] | None = None,
        exclude_columns: list[str] | None = None,
    ) -> FeatureMatrix:
        """Extract features from entity trajectory state vectors.

        Args:
            df: Polars DataFrame with trajectory data.
            target: Target column name (e.g., "status", "sla_breach").
            observation_point: At which step(s) to observe entities.
            include_entity_state: Include 10 entity state features.
            include_node_state: Include 9 node state features.
            include_system_state: Include 5 system state features.
            add_progress_ratio: Add steps_done / total_steps derived feature.
            add_wait_trend: Add wait_ratio trend derived feature.
            feature_columns: Explicit feature columns (overrides auto-selection).
            exclude_columns: Columns to exclude from auto-selection.

        Returns:
            FeatureMatrix ready for classification model training.
        """
        # Filter by observation point
        filtered = self._filter_by_observation_point(df, observation_point)

        if len(filtered) == 0:
            msg = f"No samples after filtering by observation_point={observation_point.value}"
            raise ValueError(msg)

        # Select feature columns
        if feature_columns:
            selected_cols = [
                c for c in feature_columns if c in filtered.columns and c != target
            ]
        else:
            selected_cols = self._auto_select_features(
                filtered,
                target,
                include_entity_state,
                include_node_state,
                include_system_state,
                exclude_columns,
            )

        # Add derived features
        derived_cols: list[str] = []
        if add_progress_ratio and "s_steps_done" in filtered.columns:
            total_steps_col = "s_sys_node_count"
            if total_steps_col in filtered.columns:
                filtered = filtered.with_columns(
                    (
                        pl.when(pl.col(total_steps_col) > 0)
                        .then(pl.col("s_steps_done") / pl.col(total_steps_col))
                        .otherwise(0.0)
                    ).alias("d_progress_ratio")
                )
                derived_cols.append("d_progress_ratio")

        if add_wait_trend and "s_cum_wait" in filtered.columns:
            if "s_elapsed" in filtered.columns:
                filtered = filtered.with_columns(
                    (
                        pl.when(pl.col("s_elapsed") > 0)
                        .then(pl.col("s_cum_wait") / pl.col("s_elapsed"))
                        .otherwise(0.0)
                    ).alias("d_wait_trend")
                )
                derived_cols.append("d_wait_trend")

        # Add relative features
        if "s_node_avg_queue" in filtered.columns and "s_node_util" in filtered.columns:
            if "s_sys_util" in filtered.columns:
                filtered = filtered.with_columns(
                    (
                        pl.when(pl.col("s_sys_util") > 0)
                        .then(pl.col("s_node_util") / pl.col("s_sys_util"))
                        .otherwise(1.0)
                    ).alias("d_util_relative")
                )
                derived_cols.append("d_util_relative")

        if "s_cum_cost" in filtered.columns and "s_elapsed" in filtered.columns:
            filtered = filtered.with_columns(
                (
                    pl.when(pl.col("s_elapsed") > 0)
                    .then(pl.col("s_cum_cost") / pl.col("s_elapsed"))
                    .otherwise(0.0)
                ).alias("d_cost_rate")
            )
            derived_cols.append("d_cost_rate")

        all_feature_cols = selected_cols + derived_cols

        # Verify target exists
        if target not in filtered.columns:
            msg = f"Target column '{target}' not found in data"
            raise ValueError(msg)

        # Determine task type from target values
        target_series = filtered[target]
        task_type, class_names, y_encoded = self._encode_target(target_series)

        # Build feature matrix
        # Filter to only numeric feature columns that exist
        valid_feature_cols = [
            c for c in all_feature_cols
            if c in filtered.columns and filtered[c].dtype.is_numeric()
        ]

        if len(valid_feature_cols) == 0:
            msg = "No valid numeric feature columns found"
            raise ValueError(msg)

        # Drop rows with nulls in features or target
        subset = valid_feature_cols + [target]
        clean_df = filtered.drop_nulls(subset=subset)

        if len(clean_df) == 0:
            msg = "No valid samples after dropping nulls"
            raise ValueError(msg)

        X = clean_df.select(valid_feature_cols).to_numpy().astype(np.float64)

        # Re-encode target from clean data
        _, class_names, y_encoded = self._encode_target(clean_df[target])

        # Entity IDs for grouping
        entity_ids = None
        if "eid" in clean_df.columns:
            entity_ids = clean_df["eid"].to_numpy()

        return FeatureMatrix(
            X=X,
            y=y_encoded,
            feature_names=valid_feature_cols,
            problem_type=ProblemType.ENTITY_CLASSIFICATION,
            target_name=target,
            task_type=task_type,
            entity_ids=entity_ids,
            class_names=class_names,
        )

    def _filter_by_observation_point(
        self, df: pl.DataFrame, point: ObservationPoint
    ) -> pl.DataFrame:
        """Filter trajectory rows by observation point."""
        if point == ObservationPoint.ALL_STEPS:
            return df

        if point == ObservationPoint.ENTRY_ONLY:
            return df.filter(pl.col("step") == 0)

        if point == ObservationPoint.MIDPOINT:
            # For each entity, find the step closest to the midpoint
            if "eid" not in df.columns or "step" not in df.columns:
                return df

            # Get max step per entity, then filter to midpoint
            max_steps = df.group_by("eid").agg(pl.col("step").max().alias("max_step"))
            df_with_max = df.join(max_steps, on="eid")
            midpoint_df = df_with_max.filter(
                pl.col("step") == (pl.col("max_step") / 2).round(0).cast(pl.Int64)
            ).drop("max_step")
            return midpoint_df

        return df

    def _auto_select_features(
        self,
        df: pl.DataFrame,
        target: str,
        include_entity_state: bool,
        include_node_state: bool,
        include_system_state: bool,
        exclude_columns: list[str] | None,
    ) -> list[str]:
        """Auto-select feature columns based on config."""
        exclude = set(exclude_columns or [])
        exclude.update(
            {
                target,
                "eid",
                "etype",
                "node",
                "done",
                "status",
                "total_time",
                "t_enter",
                "t_complete",
            }
        )

        candidates: list[str] = []

        if include_entity_state:
            candidates.extend(ENTITY_STATE_COLS)

        if include_node_state:
            candidates.extend(NODE_STATE_COLS)

        if include_system_state:
            candidates.extend(SYSTEM_STATE_COLS)

        # Add dynamic resource and attr columns
        for col in df.columns:
            if col.startswith("s_r_") or col.startswith("s_attr_"):
                candidates.append(col)

        # Filter to columns that exist, are numeric, and not excluded
        result = []
        for col in candidates:
            if col in df.columns and col not in exclude and df[col].dtype.is_numeric():
                result.append(col)

        return result

    def _encode_target(
        self, series: pl.Series
    ) -> tuple[TaskType, list[str] | None, np.ndarray]:
        """Encode target column for classification.

        Returns:
            Tuple of (task_type, class_names, encoded_array).
        """
        if series.dtype == pl.Utf8 or series.dtype == pl.Categorical:
            # Categorical target → classification
            unique_vals = sorted(series.drop_nulls().unique().to_list())
            class_names = [str(v) for v in unique_vals]
            mapping = {v: i for i, v in enumerate(class_names)}
            encoded = series.map_elements(
                lambda x: mapping.get(str(x), -1), return_dtype=pl.Int64
            ).to_numpy()
            return TaskType.CLASSIFICATION, class_names, encoded.astype(np.int64)

        if series.dtype.is_numeric():
            unique_count = series.n_unique()
            if unique_count <= 20:
                # Few unique numeric values → treat as classification
                unique_vals = sorted(series.drop_nulls().unique().to_list())
                class_names = [str(int(v)) if float(v) == int(v) else str(v) for v in unique_vals]
                mapping = {v: i for i, v in enumerate(unique_vals)}
                encoded = series.map_elements(
                    lambda x: mapping.get(x, -1), return_dtype=pl.Int64
                ).to_numpy()
                return TaskType.CLASSIFICATION, class_names, encoded.astype(np.int64)

            # Many unique numeric values → regression
            return TaskType.REGRESSION, None, series.to_numpy().astype(np.float64)

        msg = f"Unsupported target dtype: {series.dtype}"
        raise ValueError(msg)
