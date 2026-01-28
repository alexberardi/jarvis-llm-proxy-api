"""Abstract base class for LLM inference backends."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from managers.chat_types import ChatResult, GenerationParams, NormalizedMessage


class LLMBackendBase(ABC):
    """Abstract base class for LLM inference backends.

    All backends must implement the modern generation interface
    (generate_text_chat). Legacy compatibility methods (chat,
    chat_with_temperature) have default implementations that delegate
    to the modern interface.

    Required attributes (set in __init__):
        model_name: Human-readable model identifier
        inference_engine: Backend engine name ("vllm", "llama_cpp", "transformers", etc.)

    Optional attributes:
        last_usage: Token usage from last generation
    """

    # =========================================================================
    # REQUIRED ATTRIBUTES (must be set in subclass __init__)
    # =========================================================================

    model_name: str
    inference_engine: str
    last_usage: Optional[Dict[str, int]] = None

    # =========================================================================
    # MODERN INTERFACE (REQUIRED)
    # =========================================================================

    @abstractmethod
    def generate_text_chat(
        self,
        model_cfg: Any,
        messages: List[NormalizedMessage],
        params: GenerationParams,
    ) -> ChatResult:
        """Generate text response using modern interface.

        Args:
            model_cfg: Model configuration object
            messages: List of normalized messages with structured content
            params: Generation parameters (temperature, max_tokens, etc.)

        Returns:
            ChatResult with content and usage information
        """
        pass

    def generate_vision_chat(
        self,
        model_cfg: Any,
        messages: List[NormalizedMessage],
        params: GenerationParams,
    ) -> ChatResult:
        """Generate response with vision/multimodal support.

        Default implementation raises NotImplementedError.
        Override in backends that support vision.

        Args:
            model_cfg: Model configuration object
            messages: List of normalized messages (may contain ImagePart)
            params: Generation parameters

        Returns:
            ChatResult with content and usage information

        Raises:
            NotImplementedError: If backend doesn't support vision
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support vision/multimodal"
        )

    # =========================================================================
    # LEGACY COMPATIBILITY METHODS
    # =========================================================================

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """Legacy compatibility method.

        Default implementation converts to modern interface.
        Override if backend needs custom handling.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        from managers.chat_types import NormalizedMessage, TextPart

        normalized = [
            NormalizedMessage(role=m["role"], content=[TextPart(text=m["content"])])
            for m in messages
        ]
        params = GenerationParams(temperature=temperature)
        result = self.generate_text_chat(None, normalized, params)
        return result.content

    def chat_with_temperature(
        self, messages: List[Dict[str, str]], temperature: float = 0.7
    ) -> str:
        """Legacy compatibility method with explicit temperature.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        return self.chat(messages, temperature)

    # =========================================================================
    # ADAPTER SUPPORT (OPTIONAL)
    # =========================================================================

    def load_adapter(self, adapter_path: str, scale: float = 1.0) -> None:
        """Load a LoRA adapter for dynamic fine-tuning.

        Default implementation raises NotImplementedError.
        Override in backends that support adapters.

        Args:
            adapter_path: Path to adapter directory or file
            scale: Adapter scaling factor (default 1.0)

        Raises:
            NotImplementedError: If backend doesn't support adapters
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support LoRA adapters"
        )

    def remove_adapter(self) -> None:
        """Remove the currently loaded adapter, reverting to base model.

        Default implementation raises NotImplementedError.
        Override in backends that support adapters.

        Raises:
            NotImplementedError: If backend doesn't support adapters
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support LoRA adapters"
        )

    def get_current_adapter(self) -> Optional[str]:
        """Get the path of the currently loaded adapter, if any.

        Returns:
            Adapter path if loaded, None otherwise
        """
        return None

    # =========================================================================
    # LIFECYCLE MANAGEMENT
    # =========================================================================

    @abstractmethod
    def unload(self) -> None:
        """Unload model and free resources.

        Must handle:
        - Deleting model from memory
        - Stopping background processes
        - Clearing GPU/accelerator caches
        - Closing network connections (for remote backends)
        """
        pass

    # =========================================================================
    # INTROSPECTION
    # =========================================================================

    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information for debugging/monitoring.

        Returns:
            Dict with model metadata
        """
        return {
            "model_name": self.model_name,
            "inference_engine": self.inference_engine,
            "last_usage": self.last_usage,
        }
