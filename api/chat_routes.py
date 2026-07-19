"""Chat completions API routes.

OpenAI-compatible chat completions endpoint that proxies to the model service.
"""

import json
import logging
import os
import uuid

import anyio
from fastapi import APIRouter, Depends, Header, HTTPException
import httpx

from auth.app_auth import require_app_auth
from models.api_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from services.response_helpers import create_openai_response, openai_error
from services.settings_helpers import get_float_setting, get_setting
from services.streaming import ClosingStreamingResponse

logger = logging.getLogger("uvicorn")

router = APIRouter(tags=["chat"])


@router.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    dependencies=[Depends(require_app_auth)],
)
async def chat_completions(
    req: ChatCompletionRequest,
    x_request_id: str | None = Header(default=None),
):
    """OpenAI-compatible chat completions endpoint.

    Supports:
    - Text-only messages (string content)
    - Multimodal messages (structured content with text + images)

    Defaults to the **live** model. Consumers may explicitly request
    "background" when needed (e.g. heavy summarisation that shouldn't
    block the live model).

    NOTE — `stream=true` is NOT OpenAI chunk format. Frames are
    `{"delta": "tok"}` per token, then `{"done": true, content, usage,
    tool_calls, finish_reason}`; errors are `{"error": "..."}` and
    cancellations `{"cancelled": true}`. There is no `data: [DONE]`
    sentinel. The response echoes X-Request-Id (supplied or generated);
    POST /v1/chat/completions/cancel/{request_id} aborts an in-flight
    streamed generation.
    """
    # Default to live; allow explicit "background" pass-through
    if not req.model or req.model.lower() not in ("live", "background"):
        req.model = "live"

    try:
        model_service_url = get_setting("model_service.url", "MODEL_SERVICE_URL", "")
        internal_token = os.getenv("MODEL_SERVICE_TOKEN") or os.getenv(
            "LLM_PROXY_INTERNAL_TOKEN"
        )
        if not model_service_url:
            openai_error(
                "internal_server_error",
                "MODEL_SERVICE_URL is not set; API is passthrough-only.",
                500,
            )
        headers = {}
        if internal_token:
            headers["X-Internal-Token"] = internal_token
        timeout = get_float_setting(
            "model_service.timeout_seconds", "MODEL_SERVICE_TIMEOUT", 60.0
        )

        # Streaming mode: proxy SSE from model service
        if req.stream:
            url = model_service_url.rstrip("/") + "/internal/model/chat/stream"
            # One id across both hops: caller-supplied or generated here,
            # forwarded to the model service and echoed back — it's the handle
            # for the cancel endpoint.
            request_id = x_request_id or uuid.uuid4().hex
            headers["X-Request-Id"] = request_id

            async def stream_proxy():
                client = httpx.AsyncClient(timeout=timeout)
                resp = None
                try:
                    stream_cm = client.stream(
                        "POST", url, json=req.model_dump(), headers=headers
                    )
                    resp = await stream_cm.__aenter__()
                    if resp.status_code != 200:
                        frame = {"error": f"Model service error {resp.status_code}"}
                        yield f"data: {json.dumps(frame)}\n\n"
                        return
                    async for line in resp.aiter_lines():
                        if line:
                            yield f"{line}\n\n"
                except httpx.HTTPError as e:
                    # Transport failures (connect refused, read timeout, mid-
                    # stream drop) must honor the frame contract — a stream
                    # that just truncates with no error/done frame hangs the
                    # consumer's parser.
                    frame = {"error": f"Model service connection error: {e}"}
                    yield f"data: {json.dumps(frame)}\n\n"
                finally:
                    # Deterministic upstream close: on consumer disconnect the
                    # model service must see the socket drop NOW (its abort
                    # machinery keys off it), not at GC. Shielded because this
                    # finally runs under a pending cancellation, where any
                    # unshielded await re-raises and skips the close. Nested
                    # so a raising resp.aclose() can't leak the client pool.
                    with anyio.CancelScope(shield=True):
                        try:
                            if resp is not None:
                                await resp.aclose()
                        finally:
                            await client.aclose()

            # ClosingStreamingResponse acloses stream_proxy explicitly on
            # teardown — the finally above must run NOW on consumer
            # disconnect, not at GC, or the model service never sees the drop.
            return ClosingStreamingResponse(
                stream_proxy(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "X-Request-Id": request_id,
                },
            )

        # Non-streaming mode: existing behavior
        url = model_service_url.rstrip("/") + "/internal/model/chat"
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=req.model_dump(), headers=headers)
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
            tool_calls = data.get("tool_calls")  # Native tool calling
            finish_reason = data.get("finish_reason")  # Native tool calling
            model_name = req.model
            return create_openai_response(
                content=content,
                model_name=model_name,
                usage=usage,
                date_keys=date_keys,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during chat completion: {e}")
        import traceback

        traceback.print_exc()
        openai_error("internal_server_error", f"Internal error: {str(e)}", 500)


@router.post(
    "/v1/chat/completions/cancel/{request_id}",
    dependencies=[Depends(require_app_auth)],
)
async def cancel_chat_completion(request_id: str):
    """Cancel an in-flight streamed generation by its X-Request-Id.

    Proxies to the model service's cancel endpoint. Cancellation is
    cooperative (next token boundary); the stream ends with a
    {"cancelled": true} frame. 404 when no stream with that id is active.
    """
    model_service_url = get_setting("model_service.url", "MODEL_SERVICE_URL", "")
    if not model_service_url:
        openai_error(
            "internal_server_error",
            "MODEL_SERVICE_URL is not set; API is passthrough-only.",
            500,
        )
    internal_token = os.getenv("MODEL_SERVICE_TOKEN") or os.getenv(
        "LLM_PROXY_INTERNAL_TOKEN"
    )
    headers = {}
    if internal_token:
        headers["X-Internal-Token"] = internal_token
    timeout = get_float_setting(
        "model_service.timeout_seconds", "MODEL_SERVICE_TIMEOUT", 60.0
    )
    url = model_service_url.rstrip("/") + f"/internal/model/cancel/{request_id}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, headers=headers)
    if resp.status_code == 404:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "type": "not_found",
                    "message": f"No active stream with request id '{request_id}'",
                    "code": "request_not_found",
                }
            },
        )
    if resp.status_code != 200:
        openai_error(
            "internal_server_error",
            f"Model service error {resp.status_code}",
            500,
        )
    return resp.json()
