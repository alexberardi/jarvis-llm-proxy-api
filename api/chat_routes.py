"""Chat completions API routes.

OpenAI-compatible chat completions endpoint that proxies to the model service.
"""

import logging
import os

from fastapi import APIRouter, Depends, HTTPException
import httpx

from auth.app_auth import require_app_auth
from managers.model_manager import ModelManager
from models.api_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from services.message_service import request_has_images
from services.response_helpers import create_openai_response, openai_error

logger = logging.getLogger("uvicorn")

router = APIRouter(tags=["chat"])


@router.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    dependencies=[Depends(require_app_auth)],
)
async def chat_completions(req: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint.

    Supports:
    - Text-only messages (string content)
    - Multimodal messages (structured content with text + images)
    - Model selection via model field (supports aliases: full, lightweight, vision, cloud)
    """
    # Validate: reject images for non-vision models early
    if request_has_images(req.messages):
        model_manager = ModelManager()
        model_config = model_manager.get_model_config(req.model)
        if not model_config.supports_images:
            openai_error(
                "invalid_request_error",
                f"Model '{req.model}' does not support images. Use a vision-capable model instead.",
            )

    try:
        model_service_url = os.getenv("MODEL_SERVICE_URL")
        internal_token = os.getenv("MODEL_SERVICE_TOKEN") or os.getenv(
            "LLM_PROXY_INTERNAL_TOKEN"
        )
        if not model_service_url:
            openai_error(
                "internal_server_error",
                "MODEL_SERVICE_URL is not set; API is passthrough-only.",
                500,
            )
        url = model_service_url.rstrip("/") + "/internal/model/chat"
        headers = {}
        if internal_token:
            headers["X-Internal-Token"] = internal_token
        # Use longer timeout for requests with date context (model loading can take time)
        base_timeout = float(os.getenv("MODEL_SERVICE_TIMEOUT", "60"))
        timeout = base_timeout + 60 if req.include_date_context else base_timeout

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=req.dict(), headers=headers)
            if resp.status_code != 200:
                openai_error(
                    "internal_server_error",
                    f"Model service error {resp.status_code}: {resp.text}",
                    500,
                )
            data = resp.json()
            content = data.get("content")
            usage = data.get("usage")
            date_keys = data.get("date_keys")  # Jarvis extension
            model_name = req.model
            return create_openai_response(
                content=content, model_name=model_name, usage=usage, date_keys=date_keys
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during chat completion: {e}")
        import traceback

        traceback.print_exc()
        openai_error("internal_server_error", f"Internal error: {str(e)}", 500)
