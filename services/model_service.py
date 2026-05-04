import atexit
import json
import logging
import os
from typing import Optional, List

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from managers.chat_types import GenerationParams
from managers.model_manager import ModelManager
from models.api_models import ChatCompletionRequest, Message
from services.chat_runner import run_chat_completion, normalize_messages
from services.date_keys import extract_date_keys_fast, build_date_hint_message
from services.date_key_matcher import extract_date_keys as extract_date_keys_regex
from api.settings_routes import router as settings_router
from services.settings_helpers import get_setting

load_dotenv()

logger = logging.getLogger("uvicorn")

app = FastAPI()

# Include settings routes
app.include_router(settings_router)


# ============================================================================
# Remote logging setup (jarvis-logs)
# ============================================================================

_jarvis_logger: Optional[logging.Logger] = None


def _setup_remote_logging() -> None:
    """Set up remote logging to jarvis-logs server for forwarded worker logs."""
    global _jarvis_logger
    try:
        from jarvis_log_client import init as init_log_client, JarvisLogHandler

        app_id = os.getenv("JARVIS_APP_ID", "llm-proxy")
        app_key = os.getenv("JARVIS_APP_KEY")
        if not app_key:
            logger.info("JARVIS_APP_KEY not set, remote logging disabled")
            return

        init_log_client(app_id=app_id, app_key=app_key)

        # Create a dedicated logger for forwarded worker logs
        _jarvis_logger = logging.getLogger("llm-proxy-worker")
        _jarvis_logger.setLevel(logging.DEBUG)

        handler = JarvisLogHandler(
            service="llm-proxy-worker",
            level=logging.DEBUG,
        )
        _jarvis_logger.addHandler(handler)
        logger.info("Remote logging enabled for worker log forwarding")
    except ImportError:
        logger.warning("jarvis-log-client not installed, remote logging disabled")
    except Exception as e:
        logger.warning(f"Failed to initialize remote logging: {e}")


# Initialize remote logging on module load
_setup_remote_logging()


# ============================================================================
# Auth helper (defined early for use by all endpoints)
# ============================================================================

def require_internal_token(x_internal_token: str | None):
    expected = os.getenv("MODEL_SERVICE_TOKEN") or os.getenv("LLM_PROXY_INTERNAL_TOKEN")
    if expected and x_internal_token != expected:
        raise HTTPException(
            status_code=401,
            detail={"error": {"type": "unauthorized", "message": "Invalid internal token", "code": "unauthorized"}},
        )


# ============================================================================
# Log forwarding endpoint for workers
# ============================================================================

class LogEntry(BaseModel):
    level: str  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    message: str
    logger_name: Optional[str] = None
    timestamp: Optional[str] = None
    extra: Optional[dict] = None


class LogBatch(BaseModel):
    entries: List[LogEntry]


@app.post("/internal/log")
def forward_log(
    entry: LogEntry,
    x_internal_token: str | None = Header(default=None),
):
    """Forward a single log entry from worker to jarvis-logs."""
    require_internal_token(x_internal_token)
    _forward_log_entry(entry)
    return {"status": "ok"}


@app.post("/internal/logs")
def forward_logs_batch(
    batch: LogBatch,
    x_internal_token: str | None = Header(default=None),
):
    """Forward multiple log entries from worker to jarvis-logs."""
    require_internal_token(x_internal_token)
    for entry in batch.entries:
        _forward_log_entry(entry)
    return {"status": "ok", "count": len(batch.entries)}


def _forward_log_entry(entry: LogEntry) -> None:
    """Forward a log entry to jarvis-logs via the remote logger."""
    level = getattr(logging, entry.level.upper(), logging.INFO)

    if _jarvis_logger:
        _jarvis_logger.log(level, entry.message, extra=entry.extra or {})
    else:
        # Fallback to local logging if remote not available
        logger.log(level, f"[worker] {entry.message}")

model_manager = ModelManager()

# Safety net: atexit handler runs even if FastAPI shutdown event doesn't fire
# (e.g., when process is killed before uvicorn can run shutdown hooks)
_cleanup_done = False

def _atexit_cleanup():
    global _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True
    logger.info("🧹 [atexit] Cleaning up model service...")
    try:
        model_manager.unload_all()
    except Exception as e:
        logger.warning(f"⚠️  [atexit] Cleanup error: {e}")

atexit.register(_atexit_cleanup)


@app.on_event("shutdown")
def shutdown_event():
    global _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True
    logger.info("🧹 Shutting down model service, unloading models...")
    model_manager.unload_all()


@app.post("/internal/model/unload")
def unload_models(x_internal_token: str | None = Header(default=None)):
    require_internal_token(x_internal_token)
    logger.info("🧹 Unloading models (debug pause)...")
    model_manager.unload_all()
    logger.info("✅ Unload complete")
    return {"status": "ok"}


@app.post("/internal/model/reload")
def reload_models(x_internal_token: str | None = Header(default=None)):
    require_internal_token(x_internal_token)
    global model_manager
    logger.info("🔄 Reloading models (debug resume)")

    # Invalidate settings cache before reloading
    try:
        from services.settings_service import get_settings_service
        settings = get_settings_service()
        settings.invalidate_cache()
        logger.info("🔄 Settings cache invalidated")
    except Exception as e:
        logger.warning(f"⚠️  Failed to invalidate settings cache: {e}")

    try:
        model_manager.unload_all()
    except (RuntimeError, AttributeError) as e:
        logger.debug(f"unload_all during reload failed: {e}")
    # Reset singleton so __init__ re-reads settings
    ModelManager._instance = None
    ModelManager._initialized = False
    model_manager = ModelManager()
    return {"status": "ok"}


def _get_user_text(req: ChatCompletionRequest) -> str:
    """Extract the last user message text for date key extraction."""
    for msg in reversed(req.messages):
        if msg.role == "user":
            content = msg.content
            if isinstance(content, str):
                return content
            # Handle structured content (text + images)
            if isinstance(content, list):
                for part in content:
                    if hasattr(part, "type") and part.type == "text":
                        return part.text
                    elif isinstance(part, dict) and part.get("type") == "text":
                        return part.get("text", "")
    return ""


@app.post("/internal/model/chat")
async def model_chat(req: ChatCompletionRequest, x_internal_token: str | None = Header(default=None)):
    require_internal_token(x_internal_token)
    logger.info(f"▶️  /internal/model/chat model={req.model} messages={len(req.messages)}")

    # Extract date keys using deterministic regex matcher (~0ms).
    # Returns keys directly — no confidence thresholds or model hints needed.
    # Falls back to FastText if regex returns empty (belt-and-suspenders).
    date_keys: Optional[List[str]] = None
    if req.include_date_context:
        user_text = _get_user_text(req)
        if user_text:
            logger.debug(f"🗓️  Extracting date keys from: {user_text!r}")
            date_keys = extract_date_keys_regex(user_text)
            logger.debug(f"🗓️  Regex matcher result: keys={date_keys}")

            # Fallback: if regex found nothing, try FastText + hint
            if not date_keys:
                ft_result = extract_date_keys_fast(user_text)
                if ft_result.keys:
                    date_keys = ft_result.keys
                    logger.debug(f"🗓️  FastText fallback: keys={ft_result.keys}")
                    hint_msg = build_date_hint_message(ft_result)
                    if hint_msg:
                        req.messages.append(Message(**hint_msg))

    result = await run_chat_completion(model_manager, req, allow_images=True)
    preview = (result.content or "")[:1000]
    logger.debug(f"🧾 /internal/model/chat response_preview={preview!r}")
    logger.info(f"✅ /internal/model/chat model={req.model} done")

    response = {"content": result.content, "usage": result.usage}
    if date_keys is not None:
        response["date_keys"] = date_keys
    if result.tool_calls is not None:
        response["tool_calls"] = result.tool_calls
    if result.finish_reason is not None:
        response["finish_reason"] = result.finish_reason
    return response


@app.post("/internal/model/chat/stream")
async def model_chat_stream(
    req: ChatCompletionRequest,
    x_internal_token: str | None = Header(default=None),
):
    """SSE streaming chat completion endpoint.

    Returns Server-Sent Events with token deltas, followed by a final
    event containing the full content and usage stats.
    """
    require_internal_token(x_internal_token)
    logger.info(f"▶️  /internal/model/chat/stream model={req.model} messages={len(req.messages)}")

    # Date key extraction (same as non-streaming)
    date_keys: Optional[List[str]] = None
    if req.include_date_context:
        user_text = _get_user_text(req)
        if user_text:
            date_keys = extract_date_keys_regex(user_text)
            if not date_keys:
                ft_result = extract_date_keys_fast(user_text)
                if ft_result.keys:
                    date_keys = ft_result.keys
                    hint_msg = build_date_hint_message(ft_result)
                    if hint_msg:
                        req.messages.append(Message(**hint_msg))

    model_config = model_manager.get_model_config(req.model)
    if not model_config:
        raise HTTPException(status_code=404, detail=f"Model '{req.model}' not found")

    backend = model_config.backend_instance
    if not hasattr(backend, "generate_text_chat_stream"):
        raise HTTPException(status_code=501, detail="Backend does not support streaming")

    normalized = normalize_messages(req.messages)
    params = GenerationParams(
        temperature=req.temperature or 0.7,
        max_tokens=req.max_tokens,
        stream=True,
        adapter_settings=(
            req.adapter_settings.model_dump()
            if req.adapter_settings and req.adapter_settings.enabled
            else None
        ),
    )

    def event_generator():
        try:
            for event in backend.generate_text_chat_stream(model_config, normalized, params):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models": list(model_manager.registry.keys()),
        "aliases": model_manager.aliases,
    }


@app.get("/internal/model/models")
async def list_models(x_internal_token: str | None = Header(default=None)):
    require_internal_token(x_internal_token)
    data = []
    for model_id, cfg in model_manager.registry.items():
        data.append(
            {
                "id": model_id,
                "backend": cfg.backend_type,
                "supports_images": cfg.supports_images,
                "context_length": cfg.context_length,
            }
        )
    return {"models": data, "aliases": model_manager.aliases}


@app.get("/internal/model/engine")
async def get_engine_info(x_internal_token: str | None = Header(default=None)):
    """Get information about the current inference engine.

    Returns engine type and capabilities like prefix caching support.
    - llama_cpp: Benefits from warmup messages (prefix caching with matched content)
    - vllm: Automatic prefix caching, no benefit from explicit warmup messages
    - transformers: No prefix caching optimization
    """
    require_internal_token(x_internal_token)

    # Get the main model's inference engine
    inference_engine = get_setting(
        "inference.general.engine", "JARVIS_INFERENCE_ENGINE", "llama_cpp"
    ).lower()
    backend_type = get_setting(
        "model.main.backend", "JARVIS_MODEL_BACKEND", "GGUF"
    ).upper()

    # Check if live_model backend instance has inference_engine attribute (more accurate)
    if model_manager.live_model and hasattr(model_manager.live_model, "inference_engine"):
        inference_engine = model_manager.live_model.inference_engine

    # llama.cpp and MLX benefit from warmup messages (prefix caching reuses KV cache for matching prefixes)
    # vLLM has automatic prefix caching but doesn't benefit from explicit warmup messages
    # transformers has no prefix caching optimization
    allows_caching = inference_engine in ("llama_cpp", "mlx")

    return {
        "inference_engine": inference_engine,
        "backend_type": backend_type,
        "allows_caching": allows_caching,
        "description": {
            "llama_cpp": "llama.cpp with prefix caching - benefits from warmup messages",
            "mlx": "MLX with KV cache prefix caching - benefits from warmup messages",
            "vllm": "vLLM with automatic prefix caching - no warmup needed",
            "transformers": "HuggingFace Transformers - no prefix caching",
        }.get(inference_engine, "Unknown engine"),
    }

