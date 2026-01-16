import os
import asyncio

from fastapi import FastAPI, Header, HTTPException
from dotenv import load_dotenv

from managers.model_manager import ModelManager
from models.api_models import ChatCompletionRequest
from services.chat_runner import run_chat_completion

load_dotenv()

app = FastAPI()

model_manager = ModelManager()


def require_internal_token(x_internal_token: str | None):
    expected = os.getenv("MODEL_SERVICE_TOKEN") or os.getenv("LLM_PROXY_INTERNAL_TOKEN")
    if expected and x_internal_token != expected:
        raise HTTPException(
            status_code=401,
            detail={"error": {"type": "unauthorized", "message": "Invalid internal token", "code": "unauthorized"}},
        )


@app.post("/internal/model/chat")
async def model_chat(req: ChatCompletionRequest, x_internal_token: str | None = Header(default=None)):
    require_internal_token(x_internal_token)
    print(f"[model] ▶️  /internal/model/chat model={req.model} messages={len(req.messages)}")
    result = await run_chat_completion(model_manager, req, allow_images=True)
    preview = (result.content or "")[:1000]
    print(f"[model] 🧾 /internal/model/chat response_preview={preview!r}")
    print(f"[model] ✅ /internal/model/chat model={req.model} done")
    return {"content": result.content, "usage": result.usage}


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

