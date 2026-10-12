"""GET /v1/models must advertise which models accept image content.

The model service already reports `supports_images` and `context_length` per
model (`/internal/model/models`), but `api/model_routes.py` used to rebuild each
entry with only id/object/created/owned_by — so a client could not tell that the
live slot was vision-capable, and `ModelInfo` didn't even declare the field.

Consumers (command-center) need this to decide whether to send an image or fall
back to text, and a silent drop here is invisible until a request 400s.

Run with:
    pytest tests/test_models_route.py -v
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI

import api.model_routes as model_routes
from models.api_models import ModelInfo, ModelListResponse

MODEL_SERVICE_PAYLOAD = {
    "models": [
        {
            "id": "Qwen3.8-27B-UD-Q4_K_M.gguf",
            "backend": "REST",
            "supports_images": True,
            "context_length": 8192,
        },
        {
            "id": ".models/Qwen3-Coder-30B-A3B-Instruct-IQ4_XS.gguf",
            "backend": "REST",
            "supports_images": False,
            "context_length": 131072,
        },
    ],
    "aliases": {"live": "Qwen3.8-27B-UD-Q4_K_M.gguf"},
}


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(model_routes.router)
    return app


def _mock_response(payload):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = payload
    return resp


async def _call_list_models(payload=None):
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=_mock_response(payload or MODEL_SERVICE_PAYLOAD))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    with patch.object(model_routes.httpx, "AsyncClient", return_value=mock_client), \
         patch.object(model_routes, "get_setting", lambda *a, **k: "http://model-service:7705"):
        return await model_routes.list_models()


def test_model_info_declares_image_capability():
    """The response schema must carry the field, else pydantic strips it."""
    m = ModelInfo(id="x")
    assert m.supports_images is False
    assert m.context_length is None


def test_list_models_passes_through_supports_images():
    result = asyncio_run(_call_list_models())
    by_id = {m.id: m for m in result.data}
    assert by_id["Qwen3.8-27B-UD-Q4_K_M.gguf"].supports_images is True
    assert by_id[".models/Qwen3-Coder-30B-A3B-Instruct-IQ4_XS.gguf"].supports_images is False


def test_list_models_passes_through_context_length():
    result = asyncio_run(_call_list_models())
    by_id = {m.id: m for m in result.data}
    assert by_id["Qwen3.8-27B-UD-Q4_K_M.gguf"].context_length == 8192
    assert by_id[".models/Qwen3-Coder-30B-A3B-Instruct-IQ4_XS.gguf"].context_length == 131072


def test_list_models_keeps_openai_required_fields():
    """Adding fields must not break OpenAI compatibility of the envelope."""
    result = asyncio_run(_call_list_models())
    assert result.object == "list"
    for m in result.data:
        assert m.object == "model"
        assert m.created == 0
        assert m.owned_by == "jarvis"


def test_missing_capability_fields_default_false():
    """An older model service that omits the keys must not 500."""
    result = asyncio_run(
        _call_list_models({"models": [{"id": "legacy", "backend": "GGUF"}]})
    )
    assert result.data[0].supports_images is False
    assert result.data[0].context_length is None


def test_response_model_accepts_populated_capabilities():
    resp = ModelListResponse(
        object="list",
        data=[ModelInfo(id="a", supports_images=True, context_length=4096)],
    )
    dumped = resp.model_dump()["data"][0]
    assert dumped["supports_images"] is True
    assert dumped["context_length"] == 4096


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)
