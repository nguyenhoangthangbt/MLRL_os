"""Config resolution endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request

from mlrl_os.api.schemas import ConfigResolveRequest, ConfigResolveResponse

router = APIRouter()


@router.post("/resolve", response_model=ConfigResolveResponse)
async def resolve_config(
    request: Request,
    body: ConfigResolveRequest,
) -> ConfigResolveResponse:
    """Preview the fully resolved config without running an experiment."""
    runner = request.app.state.experiment_runner
    registry = request.app.state.dataset_registry
    meta = registry.get_meta(body.dataset_id)
    user_config = body.to_user_config()

    config = runner._config_resolver.resolve(user_config, meta)
    return ConfigResolveResponse(resolved_config=config.model_dump(mode="json"))
