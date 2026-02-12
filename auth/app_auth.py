from __future__ import annotations

import json
from typing import Optional

import httpx
from fastapi import Header, HTTPException, Request

from config.service_config import get_auth_url


async def require_app_auth(
    request: Request,
    x_jarvis_app_id: Optional[str] = Header(None),
    x_jarvis_app_key: Optional[str] = Header(None),
):
    """
    Enforce app-to-app authentication by forwarding headers to jarvis-auth /internal/app-ping.
    Health endpoints remain unauthenticated.
    """
    # Skip auth for health endpoints
    if request.url.path in ("/health", "/health/live", "/health/ready"):
        return

    if not x_jarvis_app_id or not x_jarvis_app_key:
        raise HTTPException(status_code=401, detail="Missing app credentials")

    jarvis_auth_base = get_auth_url()

    app_ping = jarvis_auth_base.rstrip("/") + "/internal/app-ping"
    async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
        try:
            resp = await client.request(
                "GET",
                app_ping,
                headers={
                    "X-Jarvis-App-Id": x_jarvis_app_id,
                    "X-Jarvis-App-Key": x_jarvis_app_key,
                },
            )
        except httpx.RequestError as exc:  # pragma: no cover - network error path
            raise HTTPException(
                status_code=502,
                detail=f"Auth service unavailable when calling {app_ping}: {exc}",
            ) from exc

    if resp.status_code != 200:
        if resp.status_code == 401:
            raise HTTPException(status_code=401, detail="Invalid app credentials")
        raise HTTPException(status_code=resp.status_code, detail="App auth failed")

    # Optionally stash calling app in state
    try:
        body = resp.json()
        request.state.calling_app_id = body.get("app_id")
    except (json.JSONDecodeError, KeyError, AttributeError):
        request.state.calling_app_id = None

