"""WebSocket endpoints for real-time ML inference.

Provides two endpoints:
- ``/predict/{model_id}`` — accept observation batches, return ML predictions
- ``/policy/{policy_id}`` — accept states, return RL policy actions (v0.2 stub)
"""

from __future__ import annotations

import logging
import time

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/predict/{model_id}")
async def ws_predict(websocket: WebSocket, model_id: str) -> None:
    """Accept observation batches and return ML predictions in real-time.

    Protocol:
        1. Client connects to ``/ws/v1/predict/{model_id}``
        2. Server loads the model; closes with 4004 if not found
        3. Client sends JSON: ``{"observations": [[...], [...]]}``
        4. Server responds with JSON: ``{"predictions": [...], ...}``
        5. Loop continues until client disconnects

    The response always includes ``predictions``, ``model_id``, and
    ``timestamp``. For classification models, ``probabilities`` is also
    included.
    """
    from mlrl_os.models.algorithms.registry import default_registry
    from mlrl_os.models.registry import ModelRegistry

    registry = ModelRegistry()

    try:
        meta = registry.get_meta(model_id)
        model = registry.load_model(model_id)
        algo = default_registry().get(model.algorithm_name)
    except KeyError:
        await websocket.close(code=4004, reason=f"Model {model_id} not found")
        return

    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            observations = np.array(data["observations"], dtype=np.float32)
            predictions = algo.predict(model, observations)

            response: dict[str, object] = {
                "predictions": predictions.tolist(),
                "model_id": model_id,
                "timestamp": time.time(),
            }

            proba = algo.predict_proba(model, observations)
            if proba is not None:
                response["probabilities"] = proba.tolist()

            await websocket.send_json(response)
    except WebSocketDisconnect:
        logger.info("Client disconnected from model %s", model_id)


@router.websocket("/policy/{policy_id}")
async def ws_policy(websocket: WebSocket, policy_id: str) -> None:
    """Accept states and return RL policy actions.

    This is a v0.2 stub. For now it always closes with 4004 since no
    RL policies are trained yet.
    """
    await websocket.close(code=4004, reason=f"Policy {policy_id} not found")
