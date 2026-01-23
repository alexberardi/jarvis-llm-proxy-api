import os
import asyncio
from typing import Optional, List

from fastapi import FastAPI, Header, HTTPException
from dotenv import load_dotenv

from managers.model_manager import ModelManager
from models.api_models import ChatCompletionRequest
from services.chat_runner import run_chat_completion
from services.date_keys import extract_date_keys, HybridDateKeyExtractor

load_dotenv()

app = FastAPI()

model_manager = ModelManager()

# Pre-load date key extractor in background if adapter exists
def _preload_date_key_extractor():
    """Pre-load the date key extractor to avoid first-request latency."""
    from services.date_keys import is_adapter_trained
    if is_adapter_trained():
        print("[model] 🗓️  Pre-loading date key extractor...", flush=True)
        try:
            extractor = HybridDateKeyExtractor()
            # Pre-warm by doing a dummy extraction
            extractor.extract("tomorrow")
        except Exception as e:
            print(f"[model] ⚠️  Failed to pre-load date key extractor: {e}", flush=True)

# Run in a separate thread to not block startup
import threading
_preload_thread = threading.Thread(target=_preload_date_key_extractor, daemon=True)
_preload_thread.start()


@app.on_event("shutdown")
def shutdown_event():
    print("[model] 🧹 Shutting down model service, unloading models...")
    model_manager.unload_all()


@app.post("/internal/model/unload")
def unload_models(x_internal_token: str | None = Header(default=None)):
    require_internal_token(x_internal_token)
    print("[model] 🧹 Unloading models (debug pause)...", flush=True)
    model_manager.unload_all()
    print("[model] ✅ Unload complete", flush=True)
    return {"status": "ok"}


@app.post("/internal/model/reload")
def reload_models(x_internal_token: str | None = Header(default=None)):
    require_internal_token(x_internal_token)
    global model_manager
    print("[model] 🔄 Reloading models (debug resume)")
    try:
        model_manager.unload_all()
    except Exception:
        pass
    model_manager = ModelManager()
    return {"status": "ok"}


def require_internal_token(x_internal_token: str | None):
    expected = os.getenv("MODEL_SERVICE_TOKEN") or os.getenv("LLM_PROXY_INTERNAL_TOKEN")
    if expected and x_internal_token != expected:
        raise HTTPException(
            status_code=401,
            detail={"error": {"type": "unauthorized", "message": "Invalid internal token", "code": "unauthorized"}},
        )


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
    print(f"[model] ▶️  /internal/model/chat model={req.model} messages={len(req.messages)}")
    
    # Extract date keys if requested
    date_keys: Optional[List[str]] = None
    if req.include_date_context:
        user_text = _get_user_text(req)
        if user_text:
            print(f"[model] 🗓️  Extracting date keys from: {user_text!r}", flush=True)
            date_keys = extract_date_keys(user_text)
            print(f"[model] 🗓️  Extracted date keys: {date_keys}", flush=True)
    
    result = await run_chat_completion(model_manager, req, allow_images=True)
    preview = (result.content or "")[:1000]
    print(f"[model] 🧾 /internal/model/chat response_preview={preview!r}")
    print(f"[model] ✅ /internal/model/chat model={req.model} done")
    
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

