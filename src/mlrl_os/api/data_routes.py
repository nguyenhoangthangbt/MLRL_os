"""Dataset management endpoints."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Form, Query, Request, UploadFile, status

from mlrl_os.api.schemas import (
    DatasetDetailResponse,
    DatasetListItem,
    DatasetUploadResponse,
    PreviewResponse,
    SchemaResponse,
)
from mlrl_os.core.types import AvailableTargets
from mlrl_os.data.discovery import TargetDiscovery
from mlrl_os.data.external_loader import ExternalLoader
from mlrl_os.data.simos_loader import SimosLoader

router = APIRouter()


@router.post("/", status_code=status.HTTP_201_CREATED, response_model=DatasetUploadResponse)
async def upload_dataset(
    request: Request,
    file: UploadFile,
    name: str | None = Form(default=None),
    source_instrument: str | None = Form(default=None),
    source_job_id: str | None = Form(default=None),
    source_template: str | None = Form(default=None),
) -> DatasetUploadResponse:
    """Upload and register a dataset (JSON SimOS export, CSV, or Parquet)."""
    registry = request.app.state.dataset_registry

    # Save to temp file, preserving the original extension
    suffix = Path(file.filename or "data").suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        if suffix == ".json":
            raw = SimosLoader().load(tmp_path)
        elif suffix == ".parquet":
            raw = ExternalLoader().load_parquet(tmp_path)
        else:
            # Default to CSV
            raw = ExternalLoader().load_csv(tmp_path)

        dataset_name = name or (file.filename or "unnamed_dataset")
        meta = registry.register(
            raw,
            name=dataset_name,
            source_instrument=source_instrument,
            source_job_id=source_job_id,
            source_template=source_template,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    return DatasetUploadResponse(
        dataset_id=meta.id,
        name=meta.name,
        source_type=meta.source_type,
        has_snapshots=meta.has_snapshots,
        has_trajectories=meta.has_trajectories,
        snapshot_row_count=meta.snapshot_row_count,
        trajectory_row_count=meta.trajectory_row_count,
        registered_at=meta.registered_at,
    )


@router.get("/", response_model=list[DatasetListItem])
async def list_datasets(request: Request) -> list[DatasetListItem]:
    """List all registered datasets."""
    registry = request.app.state.dataset_registry
    metas = registry.list_datasets()
    return [
        DatasetListItem(
            id=m.id,  # noqa: E501
            name=m.name,
            source_type=m.source_type,
            has_snapshots=m.has_snapshots,
            has_trajectories=m.has_trajectories,
            snapshot_row_count=m.snapshot_row_count,
            trajectory_row_count=m.trajectory_row_count,
            registered_at=m.registered_at,
        )
        for m in metas
    ]


@router.get("/{dataset_id}", response_model=DatasetDetailResponse)
async def get_dataset(request: Request, dataset_id: str) -> DatasetDetailResponse:
    """Get full metadata for a dataset."""
    registry = request.app.state.dataset_registry
    meta = registry.get_meta(dataset_id)
    dump = meta.model_dump(mode="json")
    return DatasetDetailResponse(**dump)


@router.get("/{dataset_id}/schema", response_model=SchemaResponse)
async def get_schema(
    request: Request,
    dataset_id: str,
    layer: str = Query(default="snapshots", pattern="^(snapshots|trajectories)$"),
) -> SchemaResponse:
    """Get column schema with types and stats for a dataset layer."""
    registry = request.app.state.dataset_registry
    meta = registry.get_meta(dataset_id)

    if layer == "snapshots" and meta.snapshot_columns:
        columns = [c.model_dump(mode="json") for c in meta.snapshot_columns]
    elif layer == "trajectories" and meta.trajectory_columns:
        columns = [c.model_dump(mode="json") for c in meta.trajectory_columns]
    else:
        columns = []

    return SchemaResponse(dataset_id=dataset_id, layer=layer, columns=columns)


@router.get("/{dataset_id}/available-targets", response_model=AvailableTargets)
async def get_available_targets(request: Request, dataset_id: str) -> AvailableTargets:
    """Discover available prediction targets for a dataset."""
    registry = request.app.state.dataset_registry
    meta = registry.get_meta(dataset_id)

    snapshots_df = None
    trajectories_df = None
    try:
        snapshots_df = registry.get_data(dataset_id, "snapshots")
    except (KeyError, ValueError):
        pass
    try:
        trajectories_df = registry.get_data(dataset_id, "trajectories")
    except (KeyError, ValueError):
        pass

    discovery = TargetDiscovery()
    return discovery.discover(meta, snapshots_df, trajectories_df)


@router.get("/{dataset_id}/preview", response_model=PreviewResponse)
async def preview_dataset(
    request: Request,
    dataset_id: str,
    layer: str = Query(default="snapshots", pattern="^(snapshots|trajectories)$"),
    rows: int = Query(default=10, ge=1, le=100),
) -> PreviewResponse:
    """Preview first N rows of a dataset layer."""
    registry = request.app.state.dataset_registry
    df = registry.get_data(dataset_id, layer)

    preview_df = df.head(rows)
    return PreviewResponse(
        dataset_id=dataset_id,
        layer=layer,
        columns=df.columns,
        rows=preview_df.to_dicts(),
        total_rows=len(df),
    )
