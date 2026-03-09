"""Problem type auto-detection from dataset and user config."""

from __future__ import annotations

from mlrl_os.core.dataset import DatasetMeta
from mlrl_os.core.types import ProblemType


class ProblemTypeDetector:
    """Auto-detect problem type from dataset metadata and user config.

    Detection logic:
    1. If user explicitly specifies type → use it
    2. If dataset has snapshots layer → TIME_SERIES
    3. If dataset has trajectories layer → ENTITY_CLASSIFICATION
    4. If both → default to TIME_SERIES (user should specify)
    5. If neither → raise error
    """

    def detect(
        self,
        dataset_meta: DatasetMeta,
        user_config: dict | None = None,  # type: ignore[type-arg]
    ) -> ProblemType:
        """Detect problem type.

        Args:
            dataset_meta: Registered dataset metadata.
            user_config: Optional user configuration dict.

        Returns:
            Detected ProblemType.

        Raises:
            ValueError: If problem type cannot be determined.
        """
        # 1. User explicitly specified
        if user_config:
            explicit_type = user_config.get("experiment_type") or user_config.get(
                "experiment", {}
            ).get("type")
            if explicit_type:
                return ProblemType(explicit_type)

            # Check dataset_layer hint
            layer = user_config.get("dataset_layer") or user_config.get(
                "dataset", {}
            ).get("layer")
            if layer == "snapshots":
                return ProblemType.TIME_SERIES
            if layer == "trajectories":
                return ProblemType.ENTITY_CLASSIFICATION

        # 2. Detect from dataset
        if dataset_meta.has_snapshots and not dataset_meta.has_trajectories:
            return ProblemType.TIME_SERIES

        if dataset_meta.has_trajectories and not dataset_meta.has_snapshots:
            return ProblemType.ENTITY_CLASSIFICATION

        if dataset_meta.has_snapshots and dataset_meta.has_trajectories:
            # Both available — default to time-series, user should specify
            return ProblemType.TIME_SERIES

        msg = (
            "Cannot determine problem type: dataset has neither snapshots nor trajectories. "
            "Please specify experiment_type explicitly."
        )
        raise ValueError(msg)
