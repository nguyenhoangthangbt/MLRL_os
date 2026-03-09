"""FastAPI routes for RL experiment management.

Provides endpoints for submitting RL experiments, listing RL experiments,
and listing trained RL policies.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


# -- Request / Response models -----------------------------------------------


class RLExperimentRequest(BaseModel):
    """Request body for submitting an RL experiment."""

    name: str
    template: str = "healthcare_er"
    algorithm: str = "dqn"
    max_episodes: int = 100
    seed: int = 42
    simos_url: str = "ws://localhost:8000"
    reward_function: str = "throughput"


class RLExperimentResponse(BaseModel):
    """Response after submitting an RL experiment."""

    experiment_id: str
    status: str
    name: str


class RLPolicyResponse(BaseModel):
    """Metadata for a trained RL policy."""

    policy_id: str
    algorithm: str
    template: str


# -- Endpoints ---------------------------------------------------------------


@router.get("/experiments")
async def list_rl_experiments(request: Request) -> list[dict[str, Any]]:
    """List RL experiments filtered from the experiment tracker."""
    tracker = request.app.state.experiment_tracker
    experiments = tracker.list_experiments()
    return [
        e.model_dump(mode="json")
        for e in experiments
        if e.experiment_type is not None
        and e.experiment_type.value == "reinforcement_learning"
    ]


@router.get("/policies")
async def list_rl_policies(request: Request) -> list[dict[str, Any]]:
    """List trained RL policies (v0.2 placeholder)."""
    return []


@router.post("/experiments", status_code=202)
async def submit_rl_experiment(
    request: Request, body: RLExperimentRequest
) -> RLExperimentResponse:
    """Submit an RL experiment for asynchronous execution.

    Runs the full RL pipeline: environment setup -> training -> evaluation.
    """
    from mlrl_os.rl.runner import RLExperimentRunner

    config: dict[str, Any] = body.model_dump()
    config["experiment_type"] = "reinforcement_learning"
    config["reward"] = {"function": body.reward_function}

    runner = RLExperimentRunner()
    result = await runner.run(config)

    return RLExperimentResponse(
        experiment_id=result.experiment_id,
        status=result.status.value,
        name=result.name,
    )
