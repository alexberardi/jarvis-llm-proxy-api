"""Health check API routes.

Endpoints for checking service health and proxying to model service health.
"""

import os

from fastapi import APIRouter
import httpx
from services.settings_helpers import get_float_setting, get_setting

router = APIRouter(tags=["health"])


async def _get_health_status():
    """Internal health check logic."""
    model_service_url = get_setting("model_service.url", "MODEL_SERVICE_URL", "")
    internal_token = os.getenv("MODEL_SERVICE_TOKEN") or os.getenv(
        "LLM_PROXY_INTERNAL_TOKEN"
    )
    if not model_service_url:
        return {"status": "degraded", "reason": "MODEL_SERVICE_URL not set"}
    url = model_service_url.rstrip("/") + "/health"
    headers = {}
    if internal_token:
        headers["X-Internal-Token"] = internal_token
    timeout = get_float_setting(
        "model_service.timeout_seconds", "MODEL_SERVICE_TIMEOUT", 60.0
    )
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                return {
                    "status": "degraded",
                    "reason": f"Model service error {resp.status_code}",
                    "body": resp.text[:200],
                }
            data = resp.json()
            return {"status": "healthy", "model_service": data}
    except httpx.ConnectError:
        return {"status": "degraded", "reason": f"Cannot reach model service at {url}"}
    except httpx.TimeoutException:
        return {"status": "degraded", "reason": "Model service health check timed out"}


@router.get("/health")
async def root_health():
    """Root-level health endpoint (standardized across all services)."""
    return await _get_health_status()


@router.get("/v1/health")
async def health():
    """Health proxy to model service (legacy endpoint)."""
    return await _get_health_status()
