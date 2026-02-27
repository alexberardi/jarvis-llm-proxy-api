"""Embedding API routes — OpenAI-compatible /v1/embeddings endpoint.

General-purpose embeddings endpoint. Any service can call it for
semantic search, clustering, similarity scoring, etc.
"""

import logging
import time

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from auth.app_auth import require_app_auth
from managers.embedding_manager import EmbeddingManager

logger = logging.getLogger("uvicorn")

router = APIRouter(tags=["embeddings"])


# ---------------------------------------------------------------------------
# Request / response models (OpenAI-compatible)
# ---------------------------------------------------------------------------


class EmbeddingRequest(BaseModel):
    input: str | list[str] = Field(..., description="Text or list of texts to embed")
    model: str = Field(default="all-MiniLM-L6-v2", description="Embedding model name")


class EmbeddingObject(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingObject]
    model: str
    usage: EmbeddingUsage


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post(
    "/v1/embeddings",
    response_model=EmbeddingResponse,
    dependencies=[Depends(require_app_auth)],
)
async def create_embeddings(req: EmbeddingRequest) -> EmbeddingResponse:
    """Generate embeddings for the given input text(s).

    OpenAI-compatible: accepts a string or list of strings, returns
    embedding vectors with usage info.
    """
    t0 = time.time()

    # Normalize input to list
    texts: list[str] = [req.input] if isinstance(req.input, str) else req.input

    if not texts:
        return EmbeddingResponse(
            data=[],
            model=req.model,
            usage=EmbeddingUsage(prompt_tokens=0, total_tokens=0),
        )

    manager = EmbeddingManager.get_instance()
    vectors = manager.encode(texts)

    # Approximate token count (rough: ~4 chars per token)
    total_chars = sum(len(t) for t in texts)
    approx_tokens = max(1, total_chars // 4)

    elapsed_ms = (time.time() - t0) * 1000
    logger.info(f"Embedded {len(texts)} text(s) in {elapsed_ms:.1f}ms")

    return EmbeddingResponse(
        data=[
            EmbeddingObject(embedding=vec, index=i)
            for i, vec in enumerate(vectors)
        ],
        model=manager.model_name,
        usage=EmbeddingUsage(prompt_tokens=approx_tokens, total_tokens=approx_tokens),
    )
