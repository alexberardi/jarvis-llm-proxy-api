import atexit
import logging
import os
from typing import Optional, List

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from managers.model_manager import ModelManager
from models.api_models import ChatCompletionRequest
from services.chat_runner import run_chat_completion
from services.date_keys import extract_date_keys
from api.settings_routes import router as settings_router

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

    # Extract date keys if requested
    date_keys: Optional[List[str]] = None
    if req.include_date_context:
        user_text = _get_user_text(req)
        if user_text:
            logger.debug(f"🗓️  Extracting date keys from: {user_text!r}")
            model_chat_fn = None
            if model_manager.main_model and hasattr(model_manager.main_model, "chat_with_temperature"):
                model_chat_fn = model_manager.main_model.chat_with_temperature
            date_keys = extract_date_keys(user_text, model_chat_fn=model_chat_fn)
            logger.debug(f"🗓️  Extracted date keys: {date_keys}")

    result = await run_chat_completion(model_manager, req, allow_images=True)
    preview = (result.content or "")[:1000]
    logger.debug(f"🧾 /internal/model/chat response_preview={preview!r}")
    logger.info(f"✅ /internal/model/chat model={req.model} done")
    
    response = {"content": result.content, "usage": result.usage}
    if date_keys is not None:
        response["date_keys"] = date_keys
    return response


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
    inference_engine = os.getenv("JARVIS_INFERENCE_ENGINE", "llama_cpp").lower()
    backend_type = os.getenv("JARVIS_MODEL_BACKEND", "GGUF").upper()

    # Check if main_model backend instance has inference_engine attribute (more accurate)
    if model_manager.main_model and hasattr(model_manager.main_model, "inference_engine"):
        inference_engine = model_manager.main_model.inference_engine

    # llama.cpp benefits from warmup messages (prefix caching reuses KV cache for matching prefixes)
    # vLLM has automatic prefix caching but doesn't benefit from explicit warmup messages
    # transformers has no prefix caching optimization
    allows_caching = inference_engine == "llama_cpp"

    return {
        "inference_engine": inference_engine,
        "backend_type": backend_type,
        "allows_caching": allows_caching,
        "description": {
            "llama_cpp": "llama.cpp with prefix caching - benefits from warmup messages",
            "vllm": "vLLM with automatic prefix caching - no warmup needed",
            "transformers": "HuggingFace Transformers - no prefix caching",
        }.get(inference_engine, "Unknown engine"),
    }

