"""Pipeline API routes.

Endpoints for managing the model build pipeline (generate, train, validate,
merge, convert). Superuser JWT required for all endpoints.
"""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from config.service_config import get_auth_url
from jarvis_settings_client import create_superuser_auth
from models.pipeline_models import (
    ArtifactsResponse,
    BuildRequest,
    PipelineStatus,
)
from services.pipeline_service import get_pipeline_service

logger = logging.getLogger("uvicorn")

require_superuser = create_superuser_auth(get_auth_url)

router = APIRouter(prefix="/v1/pipeline", tags=["pipeline"])


@router.get("/status", response_model=PipelineStatus)
async def get_pipeline_status(
    _auth: None = Depends(require_superuser),
) -> PipelineStatus:
    """Get current pipeline state (idle/running/step info)."""
    service = get_pipeline_service()
    return service.get_status()


@router.post("/build", response_model=PipelineStatus)
async def start_build(
    request: BuildRequest,
    _auth: None = Depends(require_superuser),
) -> PipelineStatus:
    """Start a pipeline build (full or individual steps).

    Only one pipeline run at a time. Returns 409 if already running.
    """
    service = get_pipeline_service()
    try:
        status = await service.start_build(request)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return status


@router.post("/cancel", response_model=PipelineStatus)
async def cancel_build(
    _auth: None = Depends(require_superuser),
) -> PipelineStatus:
    """Cancel a running pipeline."""
    service = get_pipeline_service()
    return await service.cancel()


@router.get("/logs")
async def stream_logs(
    _auth: None = Depends(require_superuser),
) -> StreamingResponse:
    """SSE stream of pipeline log lines.

    Each event is a JSON object: {"line": "...", "timestamp": "..."}
    Sends a sentinel event when the pipeline finishes.
    """
    service = get_pipeline_service()

    async def event_generator():
        async for line in service.stream_logs():
            data = json.dumps({"line": line})
            yield f"data: {data}\n\n"
        # Send completion sentinel
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/artifacts", response_model=ArtifactsResponse)
async def get_artifacts(
    _auth: None = Depends(require_superuser),
) -> ArtifactsResponse:
    """List models, adapters, GGUF/MLX files on disk."""
    service = get_pipeline_service()
    return service.get_artifacts()
