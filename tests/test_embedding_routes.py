"""Tests for the /v1/embeddings API endpoint."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.conftest import apply_auth_mock


@pytest.fixture
def client():
    """Create a test client with mocked auth and embedding manager."""
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384

    def mock_encode(texts, normalize_embeddings=True, show_progress_bar=False):
        return np.array([[0.1] * 384 for _ in texts])

    mock_model.encode = mock_encode
    mock_st = MagicMock(return_value=mock_model)
    fake_module = MagicMock()
    fake_module.SentenceTransformer = mock_st

    from managers.embedding_manager import EmbeddingManager
    EmbeddingManager.reset()

    with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
        from main import app
        apply_auth_mock(app)
        with TestClient(app) as c:
            yield c

    EmbeddingManager.reset()


class TestEmbeddingEndpoint:
    def test_single_string_input(self, client):
        resp = client.post("/v1/embeddings", json={"input": "hello world"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        assert len(body["data"]) == 1
        assert body["data"][0]["index"] == 0
        assert len(body["data"][0]["embedding"]) == 384
        assert "model" in body
        assert body["usage"]["prompt_tokens"] > 0

    def test_list_input(self, client):
        resp = client.post("/v1/embeddings", json={"input": ["one", "two", "three"]})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["data"]) == 3
        for i, obj in enumerate(body["data"]):
            assert obj["index"] == i

    def test_empty_list_input(self, client):
        resp = client.post("/v1/embeddings", json={"input": []})
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"] == []
        assert body["usage"]["prompt_tokens"] == 0

    def test_missing_input_returns_422(self, client):
        resp = client.post("/v1/embeddings", json={})
        assert resp.status_code == 422

    def test_response_has_openai_format(self, client):
        resp = client.post("/v1/embeddings", json={"input": "test"})
        body = resp.json()
        assert "object" in body
        assert "data" in body
        assert "model" in body
        assert "usage" in body
        assert body["data"][0]["object"] == "embedding"
