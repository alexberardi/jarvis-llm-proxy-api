"""Model listing and engine info API routes.

Endpoints for listing available models and getting inference engine information.
"""

import logging
import os

from fastapi import APIRouter
import httpx

from models.api_models import ModelListResponse
from services.response_helpers import openai_error

logger = logging.getLogger("uvicorn")

router = APIRouter(prefix="/v1", tags=["models"])


@router.get("/models", response_model=ModelListResponse)
async def list_models():
    """Proxy model list from model service."""
    model_service_url = os.getenv("MODEL_SERVICE_URL")
    internal_token = os.getenv("MODEL_SERVICE_TOKEN") or os.getenv(
        "LLM_PROXY_INTERNAL_TOKEN"
    )
    if not model_service_url:
        openai_error(
            "internal_server_error",
            "MODEL_SERVICE_URL is not set; cannot list models.",
            500,
        )
    url = model_service_url.rstrip("/") + "/internal/model/models"
    headers = {}
    if internal_token:
        headers["X-Internal-Token"] = internal_token
    async with httpx.AsyncClient(
        timeout=float(os.getenv("MODEL_SERVICE_TIMEOUT", "60"))
    ) as client:
        resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            openai_error(
                "internal_server_error",
                f"Model service error {resp.status_code}: {resp.text}",
                500,
            )
        data = resp.json()
        models = []
        for m in data.get("models", []):
            models.append(
                {
                    "id": m.get("id"),
                    "object": "model",
                    "created": 0,
                    "owned_by": "jarvis",
                }
            )
        return ModelListResponse(object="list", data=models)


@router.get("/engine")
async def get_engine_info():
    """Get inference engine information.

    Returns the current inference engine type and its caching capabilities.
    Use `allows_caching` to determine if warmup messages provide benefit.

    Response:
        inference_engine: "llama_cpp" | "vllm" | "transformers"
        allows_caching: true if engine benefits from warmup messages (prefix caching)
        backend_type: the model backend type
        description: human-readable description of caching behavior
    """
    model_service_url = os.getenv("MODEL_SERVICE_URL")
    internal_token = os.getenv("MODEL_SERVICE_TOKEN") or os.getenv(
        "LLM_PROXY_INTERNAL_TOKEN"
    )
    if not model_service_url:
        openai_error(
            "internal_server_error",
            "MODEL_SERVICE_URL is not set; cannot get engine info.",
            500,
        )
    url = model_service_url.rstrip("/") + "/internal/model/engine"
    headers = {}
    if internal_token:
        headers["X-Internal-Token"] = internal_token
    async with httpx.AsyncClient(
        timeout=float(os.getenv("MODEL_SERVICE_TIMEOUT", "60"))
    ) as client:
        resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            openai_error(
                "internal_server_error",
                f"Model service error {resp.status_code}: {resp.text}",
                500,
            )
        return resp.json()
