"""FastAPI application factory for ML/RL OS."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from mlrl_os.api.schemas import HealthResponse
from mlrl_os.config.defaults import MLRLSettings
from mlrl_os.data.registry import DatasetRegistry
from mlrl_os.experiment.runner import ExperimentRunner
from mlrl_os.experiment.tracker import ExperimentTracker
from mlrl_os.models.registry import ModelRegistry


def create_app(settings: MLRLSettings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    All shared dependencies are stored on ``app.state`` so that route
    handlers can access them via ``request.app.state``.
    """
    settings = settings or MLRLSettings()

    app = FastAPI(
        title="ML/RL OS",
        version="0.1.0",
        description="Predictive Intelligence Instrument for Operational Systems",
    )

    # ── CORS ──────────────────────────────────────────────
    origins = [o.strip() for o in settings.cors_allow_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Shared state ──────────────────────────────────────
    app.state.settings = settings
    app.state.dataset_registry = DatasetRegistry(settings)
    app.state.model_registry = ModelRegistry(settings)
    app.state.experiment_tracker = ExperimentTracker(settings)
    app.state.experiment_runner = ExperimentRunner(
        settings=settings,
        dataset_registry=app.state.dataset_registry,
        model_registry=app.state.model_registry,
        experiment_tracker=app.state.experiment_tracker,
    )

    # ── Exception handlers ────────────────────────────────
    @app.exception_handler(KeyError)
    async def key_error_handler(request: Request, exc: KeyError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    # ── Health endpoint ───────────────────────────────────
    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse()

    # ── Include routers ───────────────────────────────────
    from mlrl_os.api.config_routes import router as config_router
    from mlrl_os.api.data_routes import router as data_router
    from mlrl_os.api.experiment_routes import router as experiment_router
    from mlrl_os.api.model_routes import router as model_router
    from mlrl_os.streaming.ws_inference import router as ws_router

    app.include_router(data_router, prefix="/api/v1/datasets", tags=["datasets"])
    app.include_router(experiment_router, prefix="/api/v1/experiments", tags=["experiments"])
    app.include_router(model_router, prefix="/api/v1/models", tags=["models"])
    app.include_router(config_router, prefix="/api/v1/config", tags=["config"])
    app.include_router(ws_router, prefix="/ws/v1", tags=["streaming"])

    return app
