"""Time-series feature engineering.

Transforms snapshot time-series into supervised learning features
using lag values, rolling statistics, trend extraction, and ratio features.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from mlrl_os.core.types import FeatureMatrix, ProblemType, TaskType


def parse_duration_to_seconds(s: str) -> int:
    """Parse duration string to seconds.

    Supported formats: "30s", "5m", "1h", "1.5h", "8h", "1d".
    """
    s = s.strip().lower()
    if s.endswith("d"):
        return int(float(s[:-1]) * 86400)
    if s.endswith("h"):
        return int(float(s[:-1]) * 3600)
    if s.endswith("m"):
        return int(float(s[:-1]) * 60)
    if s.endswith("s"):
        return int(float(s[:-1]))
    msg = f"Invalid duration format: '{s}'. Use s/m/h/d suffix."
    raise ValueError(msg)


class TimeSeriesFeatureEngine:
    """Transform snapshot time-series into supervised learning features."""

    def build_features(
        self,
        df: pl.DataFrame,
        target: str,
        lookback: str,
        horizon: str,
        lag_intervals: list[str],
        rolling_windows: list[str],
        include_trend: bool = True,
        include_ratios: bool = True,
        include_cross_node: bool = True,
        feature_columns: list[str] | None = None,
        exclude_columns: list[str] | None = None,
    ) -> FeatureMatrix:
        """Apply windowing and feature engineering to snapshot data.

        Args:
            df: Polars DataFrame with snapshot data, must have 'ts' column.
            target: Target column name for prediction.
            lookback: Lookback window (e.g., "8h").
            horizon: Forecast horizon (e.g., "1h").
            lag_intervals: Intervals for lag features (e.g., ["1h", "2h"]).
            rolling_windows: Windows for rolling stats (e.g., ["2h", "4h"]).
            include_trend: Whether to include linear trend features.
            include_ratios: Whether to include ratio features.
            include_cross_node: Whether to include cross-node imbalance features.
            feature_columns: Explicit feature columns. If None, auto-select.
            exclude_columns: Columns to exclude from auto-selection.

        Returns:
            FeatureMatrix ready for model training.
        """
        # Sort by timestamp
        df = df.sort("ts")

        # Determine bucket interval (seconds between snapshots)
        ts_vals = df["ts"].to_numpy()
        if len(ts_vals) < 2:
            msg = "Need at least 2 snapshots for time-series features"
            raise ValueError(msg)

        bucket_interval = float(ts_vals[1] - ts_vals[0])
        if bucket_interval <= 0:
            msg = "Snapshot timestamps must be strictly increasing"
            raise ValueError(msg)

        # Parse durations to steps
        lookback_secs = parse_duration_to_seconds(lookback)
        horizon_secs = parse_duration_to_seconds(horizon)
        lookback_steps = max(1, int(lookback_secs / bucket_interval))
        horizon_steps = max(1, int(horizon_secs / bucket_interval))

        # Select source columns for features
        source_cols = self._select_source_columns(
            df, target, feature_columns, exclude_columns
        )

        # Build feature DataFrame
        feature_df = df.select(["ts"] + source_cols).clone()

        # Generate lag features
        lag_steps = [
            max(1, int(parse_duration_to_seconds(interval) / bucket_interval))
            for interval in lag_intervals
        ]
        lag_cols: list[str] = []
        for col in source_cols:
            for steps, label in zip(lag_steps, lag_intervals, strict=False):
                col_name = f"{col}_lag_{label}"
                feature_df = feature_df.with_columns(
                    pl.col(col).shift(steps).alias(col_name)
                )
                lag_cols.append(col_name)

        # Generate rolling features
        rolling_cols: list[str] = []
        for col in source_cols:
            for window_str in rolling_windows:
                window_steps = max(
                    2, int(parse_duration_to_seconds(window_str) / bucket_interval)
                )
                mean_name = f"{col}_rmean_{window_str}"
                std_name = f"{col}_rstd_{window_str}"
                feature_df = feature_df.with_columns(
                    pl.col(col)
                    .rolling_mean(window_size=window_steps)
                    .alias(mean_name),
                    pl.col(col)
                    .rolling_std(window_size=window_steps)
                    .alias(std_name),
                )
                rolling_cols.extend([mean_name, std_name])

        # Generate trend features (linear slope over lookback window)
        trend_cols: list[str] = []
        if include_trend:
            for col in source_cols:
                trend_name = f"{col}_trend"
                feature_df = feature_df.with_columns(
                    self._compute_trend(feature_df[col], lookback_steps).alias(
                        trend_name
                    )
                )
                trend_cols.append(trend_name)

        # Generate ratio features
        ratio_cols: list[str] = []
        if include_ratios:
            ratio_cols = self._add_ratio_features(feature_df, source_cols)
            # Re-select after adding ratios (handled in-place via with_columns)

        # Generate cross-node imbalance features
        cross_cols: list[str] = []
        if include_cross_node:
            cross_cols = self._add_cross_node_features(feature_df, source_cols)

        # Create target column (value at t + horizon)
        target_col_name = f"{target}_target"
        feature_df = feature_df.with_columns(
            pl.col(target).shift(-horizon_steps).alias(target_col_name)
        )

        # Collect all feature column names
        all_feature_cols = lag_cols + rolling_cols + trend_cols + ratio_cols + cross_cols

        # Filter to only columns that exist in the DataFrame
        existing_cols = set(feature_df.columns)
        all_feature_cols = [c for c in all_feature_cols if c in existing_cols]

        # Drop rows with NaN (from lookback window and horizon shift)
        subset_cols = all_feature_cols + [target_col_name]
        feature_df = feature_df.drop_nulls(subset=subset_cols)

        if len(feature_df) == 0:
            msg = (
                f"No valid samples after windowing. "
                f"Lookback={lookback}, horizon={horizon}, data rows={len(df)}. "
                f"Reduce lookback or horizon."
            )
            raise ValueError(msg)

        # Extract numpy arrays
        X = feature_df.select(all_feature_cols).to_numpy().astype(np.float64)
        y = feature_df[target_col_name].to_numpy().astype(np.float64)
        temporal_index = feature_df["ts"].to_numpy().astype(np.float64)

        return FeatureMatrix(
            X=X,
            y=y,
            feature_names=all_feature_cols,
            problem_type=ProblemType.TIME_SERIES,
            target_name=target,
            task_type=TaskType.REGRESSION,
            temporal_index=temporal_index,
        )

    def _select_source_columns(
        self,
        df: pl.DataFrame,
        target: str,
        feature_columns: list[str] | None,
        exclude_columns: list[str] | None,
    ) -> list[str]:
        """Select numeric columns to use as feature sources."""
        if feature_columns:
            return [c for c in feature_columns if c != target and c in df.columns]

        exclude = set(exclude_columns or [])
        exclude.add("ts")
        exclude.add("bucket_idx")
        exclude.add(target)

        cols = []
        for col_name in df.columns:
            if col_name in exclude:
                continue
            if df[col_name].dtype.is_numeric():
                cols.append(col_name)

        return cols

    def _compute_trend(self, series: pl.Series, window: int) -> pl.Series:
        """Compute rolling linear trend (slope) over a window.

        Uses simple linear regression slope: cov(x,y) / var(x)
        where x = [0, 1, ..., window-1].
        """
        values = series.to_numpy().astype(np.float64)
        n = len(values)
        result = np.full(n, np.nan)

        if window < 2:
            return pl.Series(name="trend", values=result)

        x = np.arange(window, dtype=np.float64)
        x_mean = x.mean()
        x_var = np.sum((x - x_mean) ** 2)

        if x_var == 0:
            return pl.Series(name="trend", values=result)

        for i in range(window - 1, n):
            y_window = values[i - window + 1 : i + 1]
            if np.any(np.isnan(y_window)):
                continue
            y_mean = y_window.mean()
            cov_xy = np.sum((x - x_mean) * (y_window - y_mean))
            result[i] = cov_xy / x_var

        return pl.Series(name="trend", values=result)

    def _add_ratio_features(
        self, df: pl.DataFrame, source_cols: list[str]
    ) -> list[str]:
        """Add ratio features between related columns."""
        added: list[str] = []

        # WIP / capacity ratio (if both exist)
        if "wip" in source_cols and "wip_ratio" not in source_cols:
            # wip_ratio might already be a column from SimOS
            pass

        # Queue-to-busy ratio
        if "in_queue" in source_cols and "busy" in source_cols:
            name = "queue_to_busy_ratio"
            df_temp = pl.when(pl.col("busy") > 0).then(
                pl.col("in_queue") / pl.col("busy")
            ).otherwise(0.0)
            try:
                df.insert_column(len(df.columns), df_temp.alias(name))
                added.append(name)
            except Exception:
                pass

        return added

    def _add_cross_node_features(
        self, df: pl.DataFrame, source_cols: list[str]
    ) -> list[str]:
        """Add cross-node imbalance features."""
        added: list[str] = []

        # Find all node queue columns
        queue_cols = [c for c in source_cols if c.startswith("n_") and c.endswith("_queue")]
        if len(queue_cols) >= 2:
            # Queue imbalance: max / mean
            name = "queue_imbalance"
            try:
                max_q = pl.max_horizontal(*[pl.col(c) for c in queue_cols])
                mean_q = pl.mean_horizontal(*[pl.col(c) for c in queue_cols])
                expr = pl.when(mean_q > 0).then(max_q / mean_q).otherwise(1.0)
                df.insert_column(len(df.columns), expr.alias(name))
                added.append(name)
            except Exception:
                pass

        # Find all node utilization columns
        util_cols = [c for c in source_cols if c.startswith("n_") and c.endswith("_util")]
        if len(util_cols) >= 2:
            name = "util_imbalance"
            try:
                max_u = pl.max_horizontal(*[pl.col(c) for c in util_cols])
                mean_u = pl.mean_horizontal(*[pl.col(c) for c in util_cols])
                expr = pl.when(mean_u > 0).then(max_u / mean_u).otherwise(1.0)
                df.insert_column(len(df.columns), expr.alias(name))
                added.append(name)
            except Exception:
                pass

        return added
