"""A model-server 4xx must reach the client as a 4xx, not become a 500.

Live symptom: a voice request failed with

    HTTP 500
    Internal error: HTTP 400: {"error":{"code":400,
      "message":"request (10502 tokens) exceeds the available context size
                 (8192 tokens)","type":"exceed_context_size_error", ...}}

The 400 was correct — llama-server refusing an over-long prompt. Two layers
threw that status away:

1. ``RestClient.chat_with_temperature`` caught ``httpx.HTTPStatusError`` and
   re-raised ``Exception(f"HTTP {code}: {text}")``, stringifying the status
   into the message so nothing downstream could read it.
2. ``chat_runner``'s broad ``except Exception`` wrapped it as
   ``internal_server_error`` / 500.

So an ordinary, client-fixable condition — the conversation outgrew the
context window — looked like a broken server. Anything keying on 5xx retries
forever against a request that can never succeed.

The same wrapper shape exists in ``api/chat_routes.py`` (fixed in #86) but
one layer further down, at the model service itself.

Run with:
    pytest tests/test_upstream_error_propagation.py -v
"""

import asyncio

import pytest
from fastapi import HTTPException

import services.chat_runner as chat_runner
from backends.rest_backend import BackendHTTPError, RestClient
from models.api_models import ChatCompletionRequest
from services.response_helpers import error_type_for_status

CTX_ERROR = (
    '{"error":{"code":400,"message":"request (10502 tokens) exceeds the available '
    'context size (8192 tokens), try increasing it","type":"exceed_context_size_error",'
    '"n_prompt_tokens":10502,"n_ctx":8192}}'
)


# --- layer 1: the backend must not discard the status ------------------------


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        "backends.rest_backend.get_setting",
        lambda key, env_fallback, default, **k: ("openai" if key == "rest.provider" else default),
    )
    c = RestClient(base_url="http://llama:8080/", model_name="qwen3")
    return c


def _error_response(status: int, text: str):
    import httpx

    return httpx.Response(status, text=text, request=httpx.Request("POST", "http://llama:8080/"))


async def _post_returning(resp):
    """Stand in for httpx's AsyncClient.post."""
    return resp


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422, 429, 500, 503])
def test_backend_preserves_upstream_status(client, status, monkeypatch):
    monkeypatch.setattr(client.client, "post", lambda *a, **k: _post_returning(_error_response(status, CTX_ERROR)))
    with pytest.raises(BackendHTTPError) as exc:
        asyncio.run(client.chat_with_temperature([{"role": "user", "content": "hi"}], 0.7))
    assert exc.value.status_code == status


def test_backend_error_keeps_upstream_message_unwrapped(client, monkeypatch):
    monkeypatch.setattr(client.client, "post", lambda *a, **k: _post_returning(_error_response(400, CTX_ERROR)))
    with pytest.raises(BackendHTTPError) as e:
        asyncio.run(client.chat_with_temperature([{"role": "user", "content": "hi"}], 0.7))
    assert e.value.upstream_message == CTX_ERROR
    assert e.value.status_code == 400
    # the status must be readable as data, not scraped out of the message
    assert "exceed_context_size_error" in e.value.upstream_message


def test_backend_error_is_not_a_bare_exception():
    """Callers must be able to branch on status without string parsing."""
    err = BackendHTTPError(400, CTX_ERROR)
    assert isinstance(err, Exception)
    assert err.status_code == 400
    assert err.upstream_message == CTX_ERROR
    assert "400" in str(err)


# --- layer 2: chat_runner must map it to the right status --------------------


def test_status_to_error_type_mapping():
    assert error_type_for_status(400) == "invalid_request_error"
    assert error_type_for_status(404) == "not_found_error"
    assert error_type_for_status(429) == "rate_limit_error"
    assert error_type_for_status(500) == "internal_server_error"
    assert error_type_for_status(503) == "internal_server_error"
    # unknown 4xx is still a client error
    assert error_type_for_status(418) == "invalid_request_error"
    assert error_type_for_status(599) == "internal_server_error"


def _run_with_failing_backend(status: int, message: str) -> HTTPException:
    """Drive the real run_chat_completion with a backend that fails."""
    backend = type(
        "B",
        (),
        {
            "generate_text_chat": staticmethod(
                lambda *a, **k: (_ for _ in ()).throw(BackendHTTPError(status, message))
            ),
            "backend_type": "rest",
        },
    )()
    model_config = type(
        "C",
        (),
        {
            "model": "live",
            "supports_images": False,
            "backend_type": "rest",
            "backend_instance": backend,
        },
    )()
    manager = type(
        "M",
        (),
        {
            "get_model_config": staticmethod(lambda name: model_config),
            "registry": {"live": model_config},
            "aliases": {"live": "live"},
        },
    )()
    req = ChatCompletionRequest(
        model="live", messages=[{"role": "user", "content": "tell a joke"}]
    )
    with pytest.raises(HTTPException) as exc:
        asyncio.run(chat_runner.run_chat_completion(manager, req))
    return exc.value


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 413, 422, 429, 500, 502, 503, 504])
def test_chat_runner_preserves_backend_status(status):
    """The real regression: a 400 must not come back as a 500."""
    exc = _run_with_failing_backend(status, CTX_ERROR)
    assert exc.status_code == status, f"{status} surfaced as {exc.status_code}"


def test_context_overflow_message_reaches_client_intact():
    """The operator needs '10502 tokens exceeds 8192', not a generic 500."""
    exc = _run_with_failing_backend(400, CTX_ERROR)
    body = str(exc.detail)
    assert exc.status_code == 400
    assert "10502" in body and "8192" in body
    assert "exceed_context_size_error" in body
    assert "Internal error" not in body


def test_overflow_is_not_labelled_internal_server_error():
    exc = _run_with_failing_backend(400, CTX_ERROR)
    assert exc.detail["error"]["type"] == "invalid_request_error"
