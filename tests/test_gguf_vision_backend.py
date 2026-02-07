"""
Tests for GGUF Vision Backend using llama-cpp-python's multimodal API.

This backend supports LLaVA-style GGUF models with separate CLIP vision encoders.
It allows running quantized vision models on CPU while the main model uses GPU.

Run with:
    pytest tests/test_gguf_vision_backend.py -v
"""

import base64
import sys
from unittest.mock import MagicMock, patch

import pytest

from managers.chat_types import (
    ChatResult,
    GenerationParams,
    ImagePart,
    NormalizedMessage,
    TextPart,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llama_cpp_vision():
    """Mock llama-cpp-python's Llama and Llava15ChatHandler for testing.

    Since llama_cpp is a C extension that may not be installed in test env,
    we mock the entire module before importing the backend.
    """
    # Create mock classes
    mock_llama_class = MagicMock()
    mock_handler_class = MagicMock()

    mock_model = MagicMock()
    mock_model.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "A cat in the image."}}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}
    }
    mock_llama_class.return_value = mock_model

    mock_handler_instance = MagicMock()
    mock_handler_class.return_value = mock_handler_instance

    # Create mock module structure
    mock_llama_cpp = MagicMock()
    mock_llama_cpp.Llama = mock_llama_class

    mock_chat_format = MagicMock()
    mock_chat_format.Llava15ChatHandler = mock_handler_class
    mock_llama_cpp.llama_chat_format = mock_chat_format

    # Patch sys.modules to inject mock
    with patch.dict(sys.modules, {
        "llama_cpp": mock_llama_cpp,
        "llama_cpp.llama_chat_format": mock_chat_format,
    }):
        yield mock_llama_class, mock_handler_class, mock_model


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Generate a minimal valid PNG image for testing."""
    # Minimal 1x1 red PNG
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    )


@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    """Generate a minimal valid JPEG image for testing."""
    # Minimal 1x1 white JPEG
    return base64.b64decode(
        "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0a"
        "HBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIy"
        "MjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIA"
        "AhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEB"
        "AQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCwAB//2Q=="
    )


# ---------------------------------------------------------------------------
# Constructor Tests
# ---------------------------------------------------------------------------


class TestGGUFVisionClientConstructor:
    """Tests for GGUFVisionClient initialization."""

    def test_constructor_requires_model_path(self, mock_llama_cpp_vision) -> None:
        """GGUFVisionClient should raise ValueError if model_path is empty."""
        from backends.gguf_vision_backend import GGUFVisionClient

        with pytest.raises(ValueError, match="[Mm]odel.*path.*required"):
            GGUFVisionClient(
                model_path="",
                clip_model_path="/path/to/clip.gguf",
            )

    def test_constructor_requires_clip_model_path(self, mock_llama_cpp_vision) -> None:
        """GGUFVisionClient should raise ValueError if clip_model_path is empty."""
        from backends.gguf_vision_backend import GGUFVisionClient

        with pytest.raises(ValueError, match="CLIP.*path.*required"):
            GGUFVisionClient(
                model_path="/path/to/model.gguf",
                clip_model_path="",
            )

    def test_constructor_sets_inference_engine(self, mock_llama_cpp_vision) -> None:
        """GGUFVisionClient should set inference_engine to 'gguf_vision'."""
        from backends.gguf_vision_backend import GGUFVisionClient

        client = GGUFVisionClient(
            model_path="/path/to/model.gguf",
            clip_model_path="/path/to/clip.gguf",
        )

        assert client.inference_engine == "gguf_vision"

    def test_constructor_stores_model_name(self, mock_llama_cpp_vision) -> None:
        """GGUFVisionClient should store model_path as model_name."""
        from backends.gguf_vision_backend import GGUFVisionClient

        client = GGUFVisionClient(
            model_path="/path/to/my-vision-model.gguf",
            clip_model_path="/path/to/clip.gguf",
        )

        assert client.model_name == "/path/to/my-vision-model.gguf"

    def test_constructor_initializes_llama_with_chat_handler(
        self, mock_llama_cpp_vision
    ) -> None:
        """GGUFVisionClient should initialize Llama with Llava15ChatHandler."""
        mock_llama, mock_handler, _ = mock_llama_cpp_vision
        from backends.gguf_vision_backend import GGUFVisionClient

        GGUFVisionClient(
            model_path="/path/to/model.gguf",
            clip_model_path="/path/to/clip.gguf",
        )

        # Verify chat handler was created with clip model path
        mock_handler.assert_called_once()
        handler_call_kwargs = mock_handler.call_args.kwargs
        assert handler_call_kwargs.get("clip_model_path") == "/path/to/clip.gguf"

        # Verify Llama was initialized with the chat handler
        mock_llama.assert_called_once()
        llama_call_kwargs = mock_llama.call_args.kwargs
        assert llama_call_kwargs.get("model_path") == "/path/to/model.gguf"
        assert "chat_handler" in llama_call_kwargs

    def test_constructor_respects_context_window(self, mock_llama_cpp_vision) -> None:
        """GGUFVisionClient should pass context_window to Llama as n_ctx."""
        mock_llama, _, _ = mock_llama_cpp_vision
        from backends.gguf_vision_backend import GGUFVisionClient

        GGUFVisionClient(
            model_path="/path/to/model.gguf",
            clip_model_path="/path/to/clip.gguf",
            context_window=4096,
        )

        llama_call_kwargs = mock_llama.call_args.kwargs
        assert llama_call_kwargs.get("n_ctx") == 4096

    def test_constructor_respects_n_gpu_layers(self, mock_llama_cpp_vision) -> None:
        """GGUFVisionClient should pass n_gpu_layers to Llama."""
        mock_llama, _, _ = mock_llama_cpp_vision
        from backends.gguf_vision_backend import GGUFVisionClient

        GGUFVisionClient(
            model_path="/path/to/model.gguf",
            clip_model_path="/path/to/clip.gguf",
            n_gpu_layers=10,
        )

        llama_call_kwargs = mock_llama.call_args.kwargs
        assert llama_call_kwargs.get("n_gpu_layers") == 10

    def test_constructor_n_gpu_layers_defaults_to_zero(self, mock_llama_cpp_vision) -> None:
        """GGUFVisionClient should default n_gpu_layers to 0 (CPU only)."""
        mock_llama, _, _ = mock_llama_cpp_vision
        from backends.gguf_vision_backend import GGUFVisionClient

        GGUFVisionClient(
            model_path="/path/to/model.gguf",
            clip_model_path="/path/to/clip.gguf",
        )

        llama_call_kwargs = mock_llama.call_args.kwargs
        assert llama_call_kwargs.get("n_gpu_layers") == 0


# ---------------------------------------------------------------------------
# Vision Chat Tests
# ---------------------------------------------------------------------------


class TestGGUFVisionChatGeneration:
    """Tests for generate_vision_chat method."""

    def test_generate_vision_chat_with_single_image(
        self, mock_llama_cpp_vision, sample_image_bytes
    ) -> None:
        """generate_vision_chat should handle a message with a single image."""
        _, _, mock_model = mock_llama_cpp_vision
        from backends.gguf_vision_backend import GGUFVisionClient

        client = GGUFVisionClient(
            model_path="/path/to/model.gguf",
            clip_model_path="/path/to/clip.gguf",
        )

        messages = [
            NormalizedMessage(
                role="user",
                content=[
                    TextPart(text="What is in this image?"),
                    ImagePart(data=sample_image_bytes, mime_type="image/png"),
                ],
            )
        ]
        params = GenerationParams(temperature=0.7, max_tokens=256)

        result = client.generate_vision_chat(None, messages, params)

        # Verify model was called
        mock_model.create_chat_completion.assert_called_once()

        # Verify result
        assert isinstance(result, ChatResult)
        assert result.content == "A cat in the image."

    def test_generate_vision_chat_returns_chat_result(
        self, mock_llama_cpp_vision, sample_image_bytes
    ) -> None:
        """generate_vision_chat should return ChatResult with content and usage."""
        _, _, mock_model = mock_llama_cpp_vision
        from backends.gguf_vision_backend import GGUFVisionClient

        client = GGUFVisionClient(
            model_path="/path/to/model.gguf",
            clip_model_path="/path/to/clip.gguf",
        )

        messages = [
            NormalizedMessage(
                role="user",
                content=[
                    TextPart(text="Describe this."),
                    ImagePart(data=sample_image_bytes, mime_type="image/png"),
                ],
            )
        ]
        params = GenerationParams()

        result = client.generate_vision_chat(None, messages, params)

        assert isinstance(result, ChatResult)
        assert result.usage is not None
        assert result.usage["prompt_tokens"] == 50
        assert result.usage["completion_tokens"] == 10
        assert result.usage["total_tokens"] == 60

    def test_generate_vision_chat_respects_temperature(
        self, mock_llama_cpp_vision, sample_image_bytes
    ) -> None:
        """generate_vision_chat should pass temperature to create_chat_completion."""
        _, _, mock_model = mock_llama_cpp_vision
        from backends.gguf_vision_backend import GGUFVisionClient

        client = GGUFVisionClient(
            model_path="/path/to/model.gguf",
            clip_model_path="/path/to/clip.gguf",
        )

        messages = [
            NormalizedMessage(
                role="user",
                content=[
                    TextPart(text="What's here?"),
                    ImagePart(data=sample_image_bytes, mime_type="image/png"),
                ],
            )
        ]
        params = GenerationParams(temperature=0.3, max_tokens=128)

        client.generate_vision_chat(None, messages, params)

        call_kwargs = mock_model.create_chat_completion.call_args.kwargs
        assert call_kwargs.get("temperature") == 0.3
        assert call_kwargs.get("max_tokens") == 128

    def test_generate_vision_chat_with_multiple_images(
        self, mock_llama_cpp_vision, sample_image_bytes
    ) -> None:
        """generate_vision_chat should handle multiple images in a message."""
        _, _, mock_model = mock_llama_cpp_vision
        from backends.gguf_vision_backend import GGUFVisionClient

        client = GGUFVisionClient(
            model_path="/path/to/model.gguf",
            clip_model_path="/path/to/clip.gguf",
        )

        messages = [
            NormalizedMessage(
                role="user",
                content=[
                    TextPart(text="Compare these images:"),
                    ImagePart(data=sample_image_bytes, mime_type="image/png"),
                    ImagePart(data=sample_image_bytes, mime_type="image/png"),
                ],
            )
        ]
        params = GenerationParams()

        result = client.generate_vision_chat(None, messages, params)

        # Should still work with multiple images
        assert isinstance(result, ChatResult)
        mock_model.create_chat_completion.assert_called_once()

    def test_generate_vision_chat_with_conversation_history(
        self, mock_llama_cpp_vision, sample_image_bytes
    ) -> None:
        """generate_vision_chat should handle multi-turn conversations."""
        _, _, mock_model = mock_llama_cpp_vision
        from backends.gguf_vision_backend import GGUFVisionClient

        client = GGUFVisionClient(
            model_path="/path/to/model.gguf",
            clip_model_path="/path/to/clip.gguf",
        )

        messages = [
            NormalizedMessage(
                role="user",
                content=[
                    TextPart(text="What is this?"),
                    ImagePart(data=sample_image_bytes, mime_type="image/png"),
                ],
            ),
            NormalizedMessage(
                role="assistant",
                content=[TextPart(text="This is a red dot.")],
            ),
            NormalizedMessage(
                role="user",
                content=[TextPart(text="What color is it?")],
            ),
        ]
        params = GenerationParams()

        result = client.generate_vision_chat(None, messages, params)

        assert isinstance(result, ChatResult)
        # The messages should be passed to the model
        call_args = mock_model.create_chat_completion.call_args
        assert len(call_args.kwargs.get("messages", [])) == 3


# ---------------------------------------------------------------------------
# Image Conversion Tests
# ---------------------------------------------------------------------------


class TestImageConversion:
    """Tests for image data URL conversion."""

    def test_converts_image_part_to_data_url(
        self, mock_llama_cpp_vision, sample_image_bytes
    ) -> None:
        """Image parts should be converted to data URLs for llama.cpp."""
        _, _, mock_model = mock_llama_cpp_vision
        from backends.gguf_vision_backend import GGUFVisionClient

        client = GGUFVisionClient(
            model_path="/path/to/model.gguf",
            clip_model_path="/path/to/clip.gguf",
        )

        messages = [
            NormalizedMessage(
                role="user",
                content=[
                    TextPart(text="Look at this:"),
                    ImagePart(data=sample_image_bytes, mime_type="image/png"),
                ],
            )
        ]
        params = GenerationParams()

        client.generate_vision_chat(None, messages, params)

        # Check the messages passed to create_chat_completion
        call_kwargs = mock_model.create_chat_completion.call_args.kwargs
        sent_messages = call_kwargs.get("messages", [])
        assert len(sent_messages) == 1

        user_content = sent_messages[0]["content"]
        # Should have text and image_url parts
        assert isinstance(user_content, list)
        assert len(user_content) == 2

        # Find image_url part
        image_part = next(p for p in user_content if p.get("type") == "image_url")
        assert image_part is not None
        assert "image_url" in image_part
        assert image_part["image_url"]["url"].startswith("data:image/png;base64,")

    def test_handles_png_and_jpeg(
        self, mock_llama_cpp_vision, sample_image_bytes, sample_jpeg_bytes
    ) -> None:
        """Should correctly set mime type for both PNG and JPEG images."""
        _, _, mock_model = mock_llama_cpp_vision
        from backends.gguf_vision_backend import GGUFVisionClient

        client = GGUFVisionClient(
            model_path="/path/to/model.gguf",
            clip_model_path="/path/to/clip.gguf",
        )

        # Test with PNG
        png_messages = [
            NormalizedMessage(
                role="user",
                content=[
                    TextPart(text="PNG image:"),
                    ImagePart(data=sample_image_bytes, mime_type="image/png"),
                ],
            )
        ]
        client.generate_vision_chat(None, png_messages, GenerationParams())

        png_call = mock_model.create_chat_completion.call_args.kwargs
        png_content = png_call["messages"][0]["content"]
        png_image_part = next(p for p in png_content if p.get("type") == "image_url")
        assert "data:image/png;base64," in png_image_part["image_url"]["url"]

        mock_model.reset_mock()

        # Test with JPEG
        jpeg_messages = [
            NormalizedMessage(
                role="user",
                content=[
                    TextPart(text="JPEG image:"),
                    ImagePart(data=sample_jpeg_bytes, mime_type="image/jpeg"),
                ],
            )
        ]
        client.generate_vision_chat(None, jpeg_messages, GenerationParams())

        jpeg_call = mock_model.create_chat_completion.call_args.kwargs
        jpeg_content = jpeg_call["messages"][0]["content"]
        jpeg_image_part = next(p for p in jpeg_content if p.get("type") == "image_url")
        assert "data:image/jpeg;base64," in jpeg_image_part["image_url"]["url"]


# ---------------------------------------------------------------------------
# Lifecycle Tests
# ---------------------------------------------------------------------------


class TestGGUFVisionClientLifecycle:
    """Tests for model lifecycle management."""

    def test_unload_clears_model(self, mock_llama_cpp_vision) -> None:
        """unload() should set model and chat_handler to None."""
        from backends.gguf_vision_backend import GGUFVisionClient

        client = GGUFVisionClient(
            model_path="/path/to/model.gguf",
            clip_model_path="/path/to/clip.gguf",
        )

        # Verify model is loaded
        assert client.model is not None

        client.unload()

        assert client.model is None
        assert client.chat_handler is None

    def test_generate_text_chat_raises_not_implemented(
        self, mock_llama_cpp_vision
    ) -> None:
        """generate_text_chat should raise NotImplementedError (vision-only backend)."""
        from backends.gguf_vision_backend import GGUFVisionClient

        client = GGUFVisionClient(
            model_path="/path/to/model.gguf",
            clip_model_path="/path/to/clip.gguf",
        )

        messages = [
            NormalizedMessage(role="user", content=[TextPart(text="Hello")])
        ]
        params = GenerationParams()

        with pytest.raises(NotImplementedError, match="vision-only"):
            client.generate_text_chat(None, messages, params)

    def test_last_usage_updated_after_generation(
        self, mock_llama_cpp_vision, sample_image_bytes
    ) -> None:
        """last_usage should be updated after generate_vision_chat."""
        from backends.gguf_vision_backend import GGUFVisionClient

        client = GGUFVisionClient(
            model_path="/path/to/model.gguf",
            clip_model_path="/path/to/clip.gguf",
        )

        # Initially None
        assert client.last_usage is None

        messages = [
            NormalizedMessage(
                role="user",
                content=[
                    TextPart(text="Describe:"),
                    ImagePart(data=sample_image_bytes, mime_type="image/png"),
                ],
            )
        ]
        client.generate_vision_chat(None, messages, GenerationParams())

        assert client.last_usage is not None
        assert client.last_usage["total_tokens"] == 60


# ---------------------------------------------------------------------------
# Thread Safety Tests
# ---------------------------------------------------------------------------


class TestGGUFVisionClientThreadSafety:
    """Tests for thread-safe operation."""

    def test_has_lock_attribute(self, mock_llama_cpp_vision) -> None:
        """GGUFVisionClient should have a _lock attribute for thread safety."""
        from backends.gguf_vision_backend import GGUFVisionClient

        client = GGUFVisionClient(
            model_path="/path/to/model.gguf",
            clip_model_path="/path/to/clip.gguf",
        )

        assert hasattr(client, "_lock")


# ---------------------------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------------------------


class TestGGUFVisionClientErrorHandling:
    """Tests for error handling."""

    def test_handles_empty_response(self, mock_llama_cpp_vision, sample_image_bytes) -> None:
        """Should handle empty/None content in model response gracefully."""
        _, _, mock_model = mock_llama_cpp_vision
        mock_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": None}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10}
        }

        from backends.gguf_vision_backend import GGUFVisionClient

        client = GGUFVisionClient(
            model_path="/path/to/model.gguf",
            clip_model_path="/path/to/clip.gguf",
        )

        messages = [
            NormalizedMessage(
                role="user",
                content=[
                    TextPart(text="Test"),
                    ImagePart(data=sample_image_bytes, mime_type="image/png"),
                ],
            )
        ]

        result = client.generate_vision_chat(None, messages, GenerationParams())

        # Should return empty string, not None
        assert result.content == ""
