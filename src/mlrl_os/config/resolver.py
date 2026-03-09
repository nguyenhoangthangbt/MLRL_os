"""Config resolution: merge user config + defaults into a fully resolved config.

The resolver guarantees that the output ``ResolvedExperimentConfig`` has no
optional fields -- every setting is concrete and ready for the pipeline.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from mlrl_os.config.defaults import MLRLSettings, get_defaults
from mlrl_os.config.schemas import (
    ResolvedCVConfig,
    ResolvedEntityFeatures,
    ResolvedEvaluationConfig,
    ResolvedExperimentConfig,
    ResolvedModelConfig,
    ResolvedTimeSeriesFeatures,
)
from mlrl_os.core.dataset import DatasetMeta
from mlrl_os.core.types import CVStrategy, ObservationPoint, ProblemType

# ---------------------------------------------------------------------------
# Duration parsing
# ---------------------------------------------------------------------------

_DURATION_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*(s|m|h|d)$", re.IGNORECASE)

_SUFFIX_MULTIPLIERS: dict[str, int] = {
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
}


def parse_duration(s: str) -> int:
    """Parse a human-readable duration string into seconds.

    Supported formats: ``"30s"``, ``"5m"``, ``"1h"``, ``"1.5h"``, ``"1d"``.

    Args:
        s: Duration string with a numeric value followed by a unit suffix
           (``s`` for seconds, ``m`` for minutes, ``h`` for hours, ``d`` for days).

    Returns:
        Duration in whole seconds (rounded to the nearest integer).

    Raises:
        ValueError: If *s* does not match the expected format.
    """
    match = _DURATION_RE.match(s.strip())
    if not match:
        msg = (
            f"Invalid duration format: {s!r}. "
            "Expected a number followed by s, m, h, or d (e.g. '30s', '1.5h')."
        )
        raise ValueError(msg)
    value = float(match.group(1))
    unit = match.group(2).lower()
    return int(value * _SUFFIX_MULTIPLIERS[unit])


# ---------------------------------------------------------------------------
# Config resolver
# ---------------------------------------------------------------------------


class ConfigResolver:
    """Merge user-supplied (partial) config onto problem-type defaults.

    The resolver performs the following steps:

    1. Detect the problem type from the user config or dataset metadata.
    2. Load defaults for the detected problem type.
    3. Deep-merge user overrides onto defaults (user wins).
    4. Auto-resolve feature columns from dataset metadata when not provided.
    5. Generate an experiment name and seed if missing.
    6. Build and return a ``ResolvedExperimentConfig`` with no optional fields.
    """

    def __init__(self, settings: MLRLSettings | None = None) -> None:
        self._settings = settings or MLRLSettings()

    # -- public API ---------------------------------------------------------

    def resolve(
        self,
        user_config: dict[str, Any],
        dataset_meta: DatasetMeta,
    ) -> ResolvedExperimentConfig:
        """Resolve a partial user config into a fully specified experiment config.

        Args:
            user_config: Partial configuration dict supplied by the user.
                         May be empty for zero-config experiments.
            dataset_meta: Metadata for the dataset being used.

        Returns:
            A ``ResolvedExperimentConfig`` ready for validation and training.
        """
        problem_type = self._detect_problem_type(user_config, dataset_meta)
        defaults = get_defaults(problem_type)
        merged = self._deep_merge(defaults, user_config)

        # Resolve top-level fields
        name = merged.get("name") or self._generate_name()
        seed = merged.get("seed", self._settings.seed_default)
        dataset_layer = self._resolve_layer(problem_type, merged)

        # Resolve sub-configs
        features = self._resolve_features(problem_type, merged, dataset_meta)
        model = self._resolve_model(merged)
        evaluation = self._resolve_evaluation(merged)

        return ResolvedExperimentConfig(
            name=name,
            experiment_type=problem_type,
            seed=seed,
            dataset_id=dataset_meta.id,
            dataset_layer=dataset_layer,
            features=features,
            model=model,
            evaluation=evaluation,
        )

    # -- problem type detection ---------------------------------------------

    @staticmethod
    def _detect_problem_type(
        user_config: dict[str, Any],
        dataset_meta: DatasetMeta,
    ) -> ProblemType:
        """Detect the problem type from user config or dataset metadata.

        Priority order:
        1. Explicit ``experiment_type`` in user config.
        2. Presence of snapshot vs trajectory data in dataset metadata.
        3. Default to time-series if both are available.
        """
        explicit = user_config.get("experiment_type")
        if explicit is not None:
            return ProblemType(explicit)

        if dataset_meta.has_snapshots and not dataset_meta.has_trajectories:
            return ProblemType.TIME_SERIES
        if dataset_meta.has_trajectories and not dataset_meta.has_snapshots:
            return ProblemType.ENTITY_CLASSIFICATION
        # Both or neither -- default to time-series
        return ProblemType.TIME_SERIES

    # -- deep merge ---------------------------------------------------------

    @staticmethod
    def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
        """Recursively merge *overrides* onto *base*. Overrides win."""
        result = dict(base)
        for key, value in overrides.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = ConfigResolver._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    # -- layer resolution ---------------------------------------------------

    @staticmethod
    def _resolve_layer(
        problem_type: ProblemType,
        merged: dict[str, Any],
    ) -> str:
        """Determine which dataset layer to use."""
        explicit = merged.get("dataset_layer")
        if explicit:
            return str(explicit)
        if problem_type == ProblemType.TIME_SERIES:
            return "snapshots"
        return "trajectories"

    # -- feature resolution -------------------------------------------------

    def _resolve_features(
        self,
        problem_type: ProblemType,
        merged: dict[str, Any],
        dataset_meta: DatasetMeta,
    ) -> ResolvedTimeSeriesFeatures | ResolvedEntityFeatures:
        """Build the resolved feature config for the problem type."""
        features_section = merged.get("features", {})
        # Merge feature-level overrides with top-level defaults that were already merged
        combined = dict(merged)
        combined.update(features_section)

        target = combined.get("target", merged.get("target", ""))
        exclude_columns = combined.get("exclude_columns", [])

        # Auto-resolve feature columns from dataset metadata
        feature_columns = combined.get("feature_columns")
        if not feature_columns:
            feature_columns = self._auto_select_features(
                problem_type, dataset_meta, target, exclude_columns,
            )

        if problem_type == ProblemType.TIME_SERIES:
            return ResolvedTimeSeriesFeatures(
                target=target,
                lookback=combined.get("lookback", "8h"),
                horizon=combined.get("horizon", "1h"),
                lag_intervals=combined.get("lag_intervals", ["1h", "2h", "4h", "8h"]),
                rolling_windows=combined.get("rolling_windows", ["2h", "4h"]),
                include_trend=combined.get("include_trend", True),
                include_ratios=combined.get("include_ratios", True),
                include_cross_node=combined.get("include_cross_node", True),
                feature_columns=feature_columns,
                exclude_columns=exclude_columns,
            )

        return ResolvedEntityFeatures(
            target=target,
            observation_point=ObservationPoint(
                combined.get("observation_point", "all_steps"),
            ),
            include_entity_state=combined.get("include_entity_state", True),
            include_node_state=combined.get("include_node_state", True),
            include_system_state=combined.get("include_system_state", True),
            add_progress_ratio=combined.get("add_progress_ratio", True),
            add_wait_trend=combined.get("add_wait_trend", True),
            feature_columns=feature_columns,
            exclude_columns=exclude_columns,
        )

    @staticmethod
    def _auto_select_features(
        problem_type: ProblemType,
        dataset_meta: DatasetMeta,
        target: str,
        exclude_columns: list[str],
    ) -> list[str]:
        """Auto-select all numeric columns as features, excluding the target.

        For time-series, uses snapshot columns. For entity classification,
        uses trajectory columns. Non-numeric columns and explicitly excluded
        columns are omitted.
        """
        if problem_type == ProblemType.TIME_SERIES:
            columns = dataset_meta.snapshot_columns or []
        else:
            columns = dataset_meta.trajectory_columns or []

        excluded = {target, *exclude_columns}
        # Also exclude the timestamp column which should not be a feature
        excluded.add("ts")

        return [
            col.name
            for col in columns
            if col.is_numeric and col.name not in excluded
        ]

    # -- model resolution ---------------------------------------------------

    @staticmethod
    def _resolve_model(merged: dict[str, Any]) -> ResolvedModelConfig:
        """Build the resolved model config."""
        model_section = merged.get("model", {})
        combined = dict(merged)
        combined.update(model_section)

        cv_section = combined.get("cross_validation", {})
        cv_config = ResolvedCVConfig(
            strategy=CVStrategy(
                cv_section.get("strategy", combined.get("cv_strategy", "temporal")),
            ),
            folds=cv_section.get("folds", combined.get("cv_folds", 5)),
        )

        return ResolvedModelConfig(
            algorithms=combined.get("algorithms", ["lightgbm"]),
            selection=combined.get("selection", "best_cv"),
            cross_validation=cv_config,
            handle_imbalance=combined.get("handle_imbalance", False),
            hyperparameter_tuning=combined.get("hyperparameter_tuning", False),
        )

    # -- evaluation resolution ----------------------------------------------

    @staticmethod
    def _resolve_evaluation(merged: dict[str, Any]) -> ResolvedEvaluationConfig:
        """Build the resolved evaluation config."""
        eval_section = merged.get("evaluation", {})
        combined = dict(merged)
        combined.update(eval_section)

        return ResolvedEvaluationConfig(
            metrics=combined.get("metrics", ["rmse", "mae"]),
            generate_report=combined.get("generate_report", True),
            plot_predictions=combined.get("plot_predictions", True),
            plot_feature_importance=combined.get("plot_feature_importance", True),
            plot_confusion_matrix=combined.get("plot_confusion_matrix", False),
            plot_roc_curve=combined.get("plot_roc_curve", False),
        )

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _generate_name() -> str:
        """Generate a timestamped experiment name."""
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"experiment_{ts}"
