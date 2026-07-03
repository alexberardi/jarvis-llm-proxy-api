"""Health check API routes.

Endpoints for checking service health and proxying to model service health.

The HTTP status code here is load-bearing: the docker healthcheck does
urllib.request.urlopen('http://localhost:7704/health') (or curl -f) and fails
on any non-2xx — that is the mechanism that finally surfaces a dead or
degraded model service as an unhealthy container. Returning 200 with a
"degraded" body (the pre-2026-07 behavior) kept a container "healthy" while
100% of completions failed for hours.
"""

import os

from fastapi import APIRouter
from fastapi.responses import JSONResponse
import httpx
from services.settings_helpers import get_float_setting, get_setting

router = APIRouter(tags=["health"])

# Health probes must be fast: the configured model_service.timeout_seconds is
# an INFERENCE timeout (default 60s) — reusing it here would let a wedged
# model service stall every healthcheck. Cap the probe at 5s.
_PROBE_TIMEOUT_CAP_S = 5.0

# Grace window for slow model loads (e.g. a 32B GGUF): while the model service
# reports the live slot still loading AND its uptime is under this window, we
# stay 200 ("initializing") so container health doesn't flap during startup.
_LOADING_GRACE_WINDOW_S = 900.0


async def _get_health_status() -> JSONResponse:
    """Internal health check logic — returns a real HTTP status code."""
    model_service_url = get_setting("model_service.url", "MODEL_SERVICE_URL", "")
    internal_token = os.getenv("MODEL_SERVICE_TOKEN") or os.getenv(
        "LLM_PROXY_INTERNAL_TOKEN"
    )
    if not model_service_url:
        # Deliberate passthrough posture (macOS dev runs the model service on
        # the host): no model service configured is NOT a failure — stay 200.
        return JSONResponse(
            status_code=200,
            content={"status": "degraded", "reason": "MODEL_SERVICE_URL not set"},
        )
    url = model_service_url.rstrip("/") + "/health"
    headers = {}
    if internal_token:
        headers["X-Internal-Token"] = internal_token
    timeout = min(
        get_float_setting(
            "model_service.timeout_seconds", "MODEL_SERVICE_TIMEOUT", 60.0
        ),
        _PROBE_TIMEOUT_CAP_S,
    )
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, headers=headers)
    except httpx.TimeoutException:
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "reason": "Model service health check timed out",
            },
        )
    except httpx.HTTPError:
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "reason": f"Cannot reach model service at {url}",
            },
        )

    if resp.status_code != 200:
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "reason": f"Model service error {resp.status_code}",
                "body": resp.text[:200],
            },
        )

    data = resp.json()
    live_status = (data.get("slots") or {}).get("live", {}).get("status")

    # "ok" is the model service's literal for "live slot ready" (older model
    # services without slots always report "ok" — treat them as healthy too).
    if data.get("status") == "ok" or live_status == "ready":
        return JSONResponse(
            status_code=200, content={"status": "healthy", "model_service": data}
        )

    if live_status == "failed":
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "reason": "Model service live model failed to load",
                "model_service": data,
            },
        )

    # Live slot still loading: grace window so slow 32B loads don't flap
    # container health. A body without uptime_s (older model service) is
    # treated as within grace.
    uptime_s = data.get("uptime_s")
    if uptime_s is None or uptime_s < _LOADING_GRACE_WINDOW_S:
        return JSONResponse(
            status_code=200,
            content={"status": "initializing", "model_service": data},
        )
    return JSONResponse(
        status_code=503,
        content={
            "status": "degraded",
            "reason": "Model service live model still loading past grace window",
            "model_service": data,
        },
    )


@router.get("/health")
async def root_health() -> JSONResponse:
    """Root-level health endpoint (standardized across all services)."""
    return await _get_health_status()


@router.get("/v1/health")
async def health() -> JSONResponse:
    """Health proxy to model service (legacy endpoint)."""
    return await _get_health_status()
