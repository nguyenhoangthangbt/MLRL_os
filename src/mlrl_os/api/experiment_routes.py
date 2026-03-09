"""Experiment management endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request, status

from mlrl_os.api.schemas import (
    DefaultsResponse,
    ExperimentDetailResponse,
    ExperimentListItem,
    ExperimentRequest,
    ExperimentSubmitResponse,
    ValidationErrorDetail,
    ValidationRequest,
    ValidationResponse,
)
from mlrl_os.config.defaults import get_defaults
from mlrl_os.core.types import ProblemType

logger = logging.getLogger(__name__)

router = APIRouter()


# NOTE: /validate and /defaults/* must be defined BEFORE /{id}
# to avoid FastAPI matching them as experiment IDs.


@router.post("/validate", response_model=ValidationResponse)
async def validate_experiment(
    request: Request,
    body: ValidationRequest,
) -> ValidationResponse:
    """Validate an experiment config without running it."""
    runner = request.app.state.experiment_runner
    registry = request.app.state.dataset_registry
    user_config = body.to_user_config()

    try:
        meta = registry.get_meta(body.dataset_id)
        validation = runner.validate_only(user_config, meta)
    except (KeyError, Exception) as exc:
        return ValidationResponse(
            valid=False,
            errors=[
                ValidationErrorDetail(
                    code="RESOLVE_ERROR",
                    field="config",
                    message=str(exc),
                )
            ],
        )

    errors = [
        ValidationErrorDetail(
            code=e.code,
            field=e.field,
            message=e.message,
            suggestion=getattr(e, "suggestion", None),
        )
        for e in validation.errors
    ]
    warnings = [w.message for w in validation.warnings]

    resolved_config = None
    if validation.valid:
        resolver = request.app.state.experiment_runner._config_resolver
        config = resolver.resolve(user_config, meta)
        resolved_config = config.model_dump(mode="json")

    return ValidationResponse(
        valid=validation.valid,
        resolved_config=resolved_config,
        errors=errors,
        warnings=warnings,
    )


@router.get("/defaults/{problem_type}", response_model=DefaultsResponse)
async def get_experiment_defaults(problem_type: str) -> DefaultsResponse:
    """Get default configuration for a problem type."""
    pt = ProblemType(problem_type)
    defaults = get_defaults(pt)
    return DefaultsResponse(problem_type=problem_type, defaults=defaults)


@router.post("/", status_code=status.HTTP_202_ACCEPTED, response_model=ExperimentSubmitResponse)
async def submit_experiment(
    request: Request,
    body: ExperimentRequest,
) -> ExperimentSubmitResponse:
    """Submit and run an experiment.

    For v0.1, the experiment runs synchronously and the result is
    available immediately. The 202 status indicates the experiment
    was accepted and processed.
    """
    runner = request.app.state.experiment_runner
    registry = request.app.state.dataset_registry
    meta = registry.get_meta(body.dataset_id)
    user_config = body.to_user_config()

    result = runner.run(user_config, meta)

    return ExperimentSubmitResponse(
        experiment_id=result.experiment_id,
        status=result.status.value if hasattr(result.status, "value") else str(result.status),
        experiment_type=(
            result.experiment_type.value
            if hasattr(result.experiment_type, "value")
            else str(result.experiment_type)
        ),
        name=result.name,
    )


@router.get("/", response_model=list[ExperimentListItem])
async def list_experiments(request: Request) -> list[ExperimentListItem]:
    """List all experiments."""
    tracker = request.app.state.experiment_tracker
    results = tracker.list_experiments()
    return [
        ExperimentListItem(
            experiment_id=r.experiment_id,
            name=r.name,
            status=r.status.value if hasattr(r.status, "value") else str(r.status),
            experiment_type=(
                r.experiment_type.value
                if hasattr(r.experiment_type, "value")
                else str(r.experiment_type)
            ),
            created_at=r.created_at,
            completed_at=r.completed_at,
            best_algorithm=r.best_algorithm,
            duration_seconds=r.duration_seconds,
        )
        for r in results
    ]


@router.get("/{experiment_id}", response_model=ExperimentDetailResponse)
async def get_experiment(request: Request, experiment_id: str) -> ExperimentDetailResponse:
    """Get full experiment result."""
    tracker = request.app.state.experiment_tracker
    result = tracker.get_result(experiment_id)
    dump = result.model_dump(mode="json")
    return ExperimentDetailResponse(**dump)


@router.get("/{experiment_id}/report")
async def get_experiment_report(request: Request, experiment_id: str) -> dict:
    """Get the evaluation report for an experiment."""
    tracker = request.app.state.experiment_tracker
    report = tracker.get_report(experiment_id)
    return report.model_dump(mode="json")
