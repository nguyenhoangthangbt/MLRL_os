"""Model management endpoints."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Request

from mlrl_os.api.schemas import (
    FeatureImportanceResponse,
    ModelDetailResponse,
    ModelListItem,
    PredictRequest,
    PredictResponse,
)
from mlrl_os.models.algorithms.registry import default_registry

router = APIRouter()


@router.get("/", response_model=list[ModelListItem])
async def list_models(request: Request) -> list[ModelListItem]:
    """List all registered models."""
    registry = request.app.state.model_registry
    models = registry.list_models()
    return [
        ModelListItem(
            id=m.id,
            experiment_id=m.experiment_id,
            algorithm_name=m.algorithm_name,
            task=m.task,
            metrics=m.metrics,
            created_at=m.created_at,
        )
        for m in models
    ]


@router.get("/{model_id}", response_model=ModelDetailResponse)
async def get_model(request: Request, model_id: str) -> ModelDetailResponse:
    """Get model metadata."""
    registry = request.app.state.model_registry
    meta = registry.get_meta(model_id)
    return ModelDetailResponse(**meta.model_dump())


@router.post("/{model_id}/predict", response_model=PredictResponse)
async def predict(request: Request, model_id: str, body: PredictRequest) -> PredictResponse:
    """Run prediction with a trained model."""
    registry = request.app.state.model_registry
    trained_model = registry.load_model(model_id)

    algo_registry = default_registry()
    algorithm = algo_registry.get(trained_model.algorithm_name)

    # Convert list of dicts to numpy array, ordered by feature_names
    feature_names = trained_model.feature_names
    rows = []
    for row_dict in body.data:
        row = [float(row_dict.get(f, 0.0)) for f in feature_names]
        rows.append(row)
    X = np.array(rows)

    y_pred = algorithm.predict(trained_model, X)
    predictions: list[float | int | str] = y_pred.tolist()

    probabilities = None
    if trained_model.task == "classification":
        y_proba = algorithm.predict_proba(trained_model, X)
        if y_proba is not None:
            probabilities = []
            # Get class names from the model if available
            classes = getattr(trained_model.model, "classes_", None)
            for row_probs in y_proba:
                if classes is not None:
                    prob_dict = {str(c): float(p) for c, p in zip(classes, row_probs)}
                else:
                    prob_dict = {str(i): float(p) for i, p in enumerate(row_probs)}
                probabilities.append(prob_dict)

    return PredictResponse(
        model_id=model_id,
        predictions=predictions,
        task_type=trained_model.task,
        probabilities=probabilities,
    )


@router.get("/{model_id}/feature-importance", response_model=FeatureImportanceResponse)
async def get_feature_importance(
    request: Request,
    model_id: str,
) -> FeatureImportanceResponse:
    """Get feature importance from a trained model."""
    registry = request.app.state.model_registry
    trained_model = registry.load_model(model_id)

    algo_registry = default_registry()
    algorithm = algo_registry.get(trained_model.algorithm_name)

    raw_importance = algorithm.feature_importance(trained_model) or {}

    features = [
        {"feature": name, "importance": imp}
        for name, imp in sorted(raw_importance.items(), key=lambda kv: kv[1], reverse=True)
    ]

    return FeatureImportanceResponse(model_id=model_id, features=features)
