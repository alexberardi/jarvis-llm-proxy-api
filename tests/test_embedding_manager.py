"""Tests for the EmbeddingManager singleton."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from managers.embedding_manager import EmbeddingManager


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton between tests."""
    EmbeddingManager.reset()
    yield
    EmbeddingManager.reset()


# Create a mock module for sentence_transformers so the lazy import works
_mock_st_class = MagicMock()
_mock_st_module = MagicMock()
_mock_st_module.SentenceTransformer = _mock_st_class


class TestSingleton:
    def test_get_instance_returns_same_object(self):
        a = EmbeddingManager.get_instance()
        b = EmbeddingManager.get_instance()
        assert a is b

    def test_reset_clears_singleton(self):
        a = EmbeddingManager.get_instance()
        EmbeddingManager.reset()
        b = EmbeddingManager.get_instance()
        assert a is not b


class TestEncode:
    def _make_mock_model(self):
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ])
        return mock_model

    def test_encode_returns_list_of_lists(self):
        mock_model = self._make_mock_model()
        mock_st = MagicMock(return_value=mock_model)
        fake_module = MagicMock()
        fake_module.SentenceTransformer = mock_st

        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            mgr = EmbeddingManager.get_instance()
            result = mgr.encode(["hello", "world"])

        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert result[0] == pytest.approx([0.1, 0.2, 0.3])

    def test_encode_calls_with_normalize(self):
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.array([[1.0, 0.0, 0.0]])
        mock_st = MagicMock(return_value=mock_model)
        fake_module = MagicMock()
        fake_module.SentenceTransformer = mock_st

        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            mgr = EmbeddingManager.get_instance()
            mgr.encode(["test"], normalize=True)

        mock_model.encode.assert_called_once_with(
            ["test"],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def test_encode_without_normalize(self):
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.array([[1.0, 0.0, 0.0]])
        mock_st = MagicMock(return_value=mock_model)
        fake_module = MagicMock()
        fake_module.SentenceTransformer = mock_st

        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            mgr = EmbeddingManager.get_instance()
            mgr.encode(["test"], normalize=False)

        mock_model.encode.assert_called_once_with(
            ["test"],
            normalize_embeddings=False,
            show_progress_bar=False,
        )

    def test_dimension_property(self):
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st = MagicMock(return_value=mock_model)
        fake_module = MagicMock()
        fake_module.SentenceTransformer = mock_st

        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            mgr = EmbeddingManager.get_instance()
            assert mgr.dimension == 384

    def test_model_loaded_once(self):
        """Model should only be loaded once even with multiple encode calls."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.array([[0.0] * 384])
        mock_st = MagicMock(return_value=mock_model)
        fake_module = MagicMock()
        fake_module.SentenceTransformer = mock_st

        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            mgr = EmbeddingManager.get_instance()
            mgr.encode(["one"])
            mgr.encode(["two"])

        mock_st.assert_called_once()
