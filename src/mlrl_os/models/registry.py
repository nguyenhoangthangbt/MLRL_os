"""Model versioning and storage registry.

Manages trained model artifacts on the file system. Models are stored
under ``{models_dir}/models/{model_id}/`` and are immutable once
registered.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from pydantic import BaseModel, Field

from mlrl_os.config.defaults import MLRLSettings
from mlrl_os.models.algorithms.protocol import TrainedModel

logger = logging.getLogger(__name__)


class ModelMeta(BaseModel):
    """Metadata for a registered model."""

    id: str
    experiment_id: str
    algorithm_name: str
    task: str  # "regression" | "classification"
    feature_names: list[str]
    metrics: dict[str, float] = Field(default_factory=dict)
    created_at: str
    file_path: str


class ModelRegistry:
    """File-system based model registry.

    Models are stored as:

    .. code-block:: text

        {models_dir}/models/{id}/
            meta.json
            model.joblib
    """

    def __init__(self, settings: MLRLSettings | None = None) -> None:
        self._settings = settings or MLRLSettings()
        self._base_dir = Path(self._settings.models_dir) / "models"
        self._base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        trained_model: TrainedModel,
        experiment_id: str,
        metrics: dict[str, float] | None = None,
    ) -> ModelMeta:
        """Save a trained model and register its metadata.

        Returns:
            :class:`ModelMeta` with the generated model ID.
        """
        now = datetime.now(timezone.utc)
        timestamp = now.isoformat()
        model_id = self._generate_id(
            experiment_id, trained_model.algorithm_name, timestamp
        )

        model_dir = self._base_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        # Persist the model artifact
        artifact_path = model_dir / "model.joblib"
        joblib.dump(trained_model, artifact_path)
        logger.info("Saved model artifact to %s", artifact_path)

        # Build and persist metadata
        meta = ModelMeta(
            id=model_id,
            experiment_id=experiment_id,
            algorithm_name=trained_model.algorithm_name,
            task=trained_model.task,
            feature_names=trained_model.feature_names,
            metrics=metrics or {},
            created_at=timestamp,
            file_path=str(artifact_path),
        )

        meta_path = model_dir / "meta.json"
        meta_path.write_text(meta.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Registered model %s for experiment %s", model_id, experiment_id)

        return meta

    def get_meta(self, model_id: str) -> ModelMeta:
        """Load metadata for a registered model.

        Raises:
            KeyError: If no model with *model_id* is registered.
        """
        meta_path = self._meta_path(model_id)
        if not meta_path.exists():
            msg = f"Model not found: {model_id}"
            raise KeyError(msg)

        return ModelMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))

    def load_model(self, model_id: str) -> TrainedModel:
        """Load the actual trained model artifact.

        Raises:
            KeyError: If no model with *model_id* is registered.
        """
        meta = self.get_meta(model_id)
        artifact_path = Path(meta.file_path)

        if not artifact_path.exists():
            msg = f"Model artifact missing for {model_id}: {artifact_path}"
            raise KeyError(msg)

        model: TrainedModel = joblib.load(artifact_path)
        logger.info("Loaded model %s from %s", model_id, artifact_path)
        return model

    def list_models(self) -> list[ModelMeta]:
        """List all registered models, sorted by creation time (newest first)."""
        models: list[ModelMeta] = []
        if not self._base_dir.exists():
            return models

        for meta_path in self._base_dir.glob("*/meta.json"):
            try:
                meta = ModelMeta.model_validate_json(
                    meta_path.read_text(encoding="utf-8")
                )
                models.append(meta)
            except Exception:
                logger.warning("Skipping corrupt metadata: %s", meta_path)

        models.sort(key=lambda m: m.created_at, reverse=True)
        return models

    def has(self, model_id: str) -> bool:
        """Check if a model is registered."""
        return self._meta_path(model_id).exists()

    def delete(self, model_id: str) -> None:
        """Delete a model and all its files.

        Raises:
            KeyError: If no model with *model_id* is registered.
        """
        model_dir = self._base_dir / model_id
        if not model_dir.exists():
            msg = f"Model not found: {model_id}"
            raise KeyError(msg)

        shutil.rmtree(model_dir)
        logger.info("Deleted model %s", model_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _meta_path(self, model_id: str) -> Path:
        """Return the path to a model's metadata file."""
        return self._base_dir / model_id / "meta.json"

    @staticmethod
    def _generate_id(
        experiment_id: str, algorithm_name: str, timestamp: str
    ) -> str:
        """Generate a deterministic 12-character model ID.

        Uses the first 12 hex characters of the SHA-256 hash of
        ``experiment_id + algorithm_name + timestamp``.
        """
        raw = f"{experiment_id}{algorithm_name}{timestamp}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]
