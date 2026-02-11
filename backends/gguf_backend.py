import logging
import os
import threading
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from .power_metrics import PowerMetrics
from managers.chat_types import NormalizedMessage, TextPart, GenerationParams, ChatResult
from backends.base import LLMBackendBase

logger = logging.getLogger("uvicorn")


class GGUFClient(LLMBackendBase):
    def __init__(self, model_path: str, chat_format: str, stop_tokens: List[str] = None, context_window: int = None):
        if not model_path:
            raise ValueError("Model path is required")

        # Store model name for unload functionality
        self.model_name = model_path
        self.model_path = model_path
        self.chat_format = chat_format
        self.model = None
        self.last_usage = None
        self._lock = threading.Lock()  # Add thread safety

        # Adapter state tracking (constructor-based loading)
        self._current_adapter_hash: Optional[str] = None
        self._current_adapter_path: Optional[str] = None
        self._current_adapter_scale: float = 1.0

        # Context cache for prefix matching optimization
        self.context_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Initialize power monitoring (optional)
        self.power_metrics = PowerMetrics()
        self.power_metrics.start_monitoring()

        # Get context window from parameter or environment variable, default to 4096 for better performance
        if context_window is None:
            context_window = int(os.getenv("JARVIS_MODEL_CONTEXT_WINDOW", "4096"))

        # Get optimal thread count based on CPU cores
        n_threads = int(os.getenv("JARVIS_N_THREADS", min(10, os.cpu_count() or 4)))

        # Get GPU layers - be more conservative to avoid memory issues
        n_gpu_layers = int(os.getenv("JARVIS_N_GPU_LAYERS", "-1"))


        # Memory management settings
        self.enable_cache = os.getenv("JARVIS_ENABLE_CONTEXT_CACHE", "true").lower() == "true"
        self.max_cache_size = int(os.getenv("JARVIS_MAX_CACHE_SIZE", "100"))

        # LLaMA.cpp optimization parameters
        n_batch = int(os.getenv("JARVIS_N_BATCH", "512"))
        n_ubatch = int(os.getenv("JARVIS_N_UBATCH", "512"))
        self.flash_attn = os.getenv("JARVIS_FLASH_ATTN", "true").lower() == "true"
        rope_scaling_type = int(os.getenv("JARVIS_ROPE_SCALING_TYPE", "0"))
        mul_mat_q = os.getenv("JARVIS_MUL_MAT_Q", "true").lower() == "true"
        f16_kv = os.getenv("JARVIS_F16_KV", "true").lower() == "true"
        seed = int(os.getenv("JARVIS_SEED", "42"))
        verbose = os.getenv("JARVIS_VERBOSE", "false").lower() == "true"

        # Inference parameters
        self.max_tokens = int(os.getenv("JARVIS_MAX_TOKENS", "7000"))
        self.top_p = float(os.getenv("JARVIS_TOP_P", "0.95"))
        self.top_k = int(os.getenv("JARVIS_TOP_K", "40"))
        self.repeat_penalty = float(os.getenv("JARVIS_REPEAT_PENALTY", "1.1"))
        self.mirostat_mode = int(os.getenv("JARVIS_MIROSTAT_MODE", "0"))
        self.mirostat_tau = float(os.getenv("JARVIS_MIROSTAT_TAU", "5.0"))
        self.mirostat_eta = float(os.getenv("JARVIS_MIROSTAT_ETA", "0.1"))

        # Check inference engine preference
        inference_engine = os.getenv("JARVIS_INFERENCE_ENGINE", "llama_cpp").lower()

        logger.debug(f"🔍 Debug: Inference engine: {inference_engine}")
        logger.debug(f"🔍 Debug: Model path: {model_path}")
        logger.debug(f"🔍 Debug: Chat format: {chat_format}")
        logger.debug(f"🔍 Debug: Context window: {context_window}")
        logger.debug(f"🔍 Debug: Threads: {n_threads}")
        logger.debug(f"🔍 Debug: GPU layers: {n_gpu_layers}")
        logger.debug(f"🔍 Debug: Context cache: {'enabled' if self.enable_cache else 'disabled'}")
        logger.debug(f"🔍 Debug: Batch size: {n_batch}")
        logger.debug(f"🔍 Debug: Micro batch size: {n_ubatch}")
        logger.debug(f"🔍 Debug: F16 KV cache: {'enabled' if f16_kv else 'disabled'}")
        logger.debug(f"🔍 Debug: Matrix multiplication: {'enabled' if mul_mat_q else 'disabled'}")

        if inference_engine == "vllm":
            logger.info(f"🚀 Using vLLM inference engine")
            self._init_vllm(model_path, chat_format, stop_tokens, context_window, n_threads, n_gpu_layers, verbose, seed, n_batch, n_ubatch, rope_scaling_type, mul_mat_q, f16_kv)
        else:
            logger.info(f"🦙 Using llama.cpp inference engine")
            self._init_llama_cpp(model_path, chat_format, stop_tokens, context_window, n_threads, n_gpu_layers, verbose, seed, context_window, n_batch, n_ubatch, rope_scaling_type, mul_mat_q, f16_kv)

    def _init_vllm(self, model_path: str, chat_format: str, stop_tokens: List[str], context_window: int, n_threads: int, n_gpu_layers: int, verbose: bool, seed: int, n_batch: int, n_ubatch: int, rope_scaling_type: int, mul_mat_q: bool, f16_kv: bool):
        """Initialize vLLM backend"""
        try:
            from .vllm_backend import VLLMClient
            self.backend = VLLMClient(model_path, chat_format, stop_tokens, context_window)
            self.inference_engine = "vllm"
        except ValueError as e:
            if "vLLM requires HuggingFace model names" in str(e):
                logger.warning("🔧 CONFIGURATION SUGGESTION:")
                logger.warning("   For GGUF files, use: JARVIS_INFERENCE_ENGINE=llama_cpp")
                logger.warning("   For vLLM, switch to TRANSFORMERS backend with HF models:")
                logger.warning("   JARVIS_MODEL_BACKEND=TRANSFORMERS")
                logger.warning("   JARVIS_MODEL_NAME=microsoft/Phi-3-mini-4k-instruct")
                logger.warning("   JARVIS_INFERENCE_ENGINE=vllm")
                logger.info(f"🔄 Falling back to llama.cpp for GGUF file...")
                self._init_llama_cpp(model_path, chat_format, stop_tokens, context_window, n_threads, n_gpu_layers, verbose, seed, context_window, n_batch, n_ubatch, rope_scaling_type, mul_mat_q, f16_kv)
            else:
                raise

    def _init_llama_cpp(
        self,
        model_path: str,
        chat_format: str,
        stop_tokens: List[str],
        context_window: int,
        n_threads: int,
        n_gpu_layers: int,
        verbose: bool,
        seed: int,
        ctx_window: int,
        n_batch: int,
        n_ubatch: int,
        rope_scaling_type: int,
        mul_mat_q: bool,
        f16_kv: bool,
        lora_path: Optional[str] = None,
        lora_scale: float = 1.0,
    ):
        """Initialize llama.cpp backend, optionally with a LoRA adapter."""
        from llama_cpp import Llama

        # Store init kwargs so we can reload with a different adapter later
        self._llama_init_kwargs = {
            "model_path": model_path,
            "n_threads": n_threads,
            "n_gpu_layers": n_gpu_layers,
            "verbose": verbose,
            "seed": seed,
            "n_ctx": context_window,
            "n_batch": n_batch,
            "n_ubatch": n_ubatch,
            "rope_scaling_type": rope_scaling_type,
            "mul_mat_q": mul_mat_q,
            "f16_kv": f16_kv,
            "flash_attn": self.flash_attn,
        }

        logger.debug(f"🔍 Debug: LLAMA_METAL env var: {os.getenv('LLAMA_METAL', 'not set')}")
        logger.debug(f"🔍 Debug: Metal will be enabled: {os.getenv('LLAMA_METAL', 'false').lower() == 'true'}")
        logger.debug(f"🔍 Debug: Loading model with n_gpu_layers={n_gpu_layers}")

        # Build constructor kwargs, adding LoRA if provided
        ctor_kwargs = dict(self._llama_init_kwargs)
        if lora_path:
            ctor_kwargs["lora_path"] = lora_path
            ctor_kwargs["lora_scale"] = lora_scale
            logger.info(f"🧩 Loading model with LoRA adapter: {lora_path} (scale={lora_scale})")

        self.model = Llama(**ctor_kwargs)
        self.backend = self
        self.inference_engine = "llama_cpp"

        # Debug: Check model loading results
        logger.info(f"✅ Model loaded successfully!")
        logger.debug(f"🔍 Debug: Model context size: {self.model.n_ctx()}")
        logger.debug(f"🔍 Debug: Model vocab size: {self.model.n_vocab()}")

        # Warm up the model with a small inference
        self._warmup()

        logger.debug(f"🔍 Debug: Model initialization complete")

    def _warmup(self) -> None:
        """Warm up the model with a small inference."""
        try:
            logger.debug(f"🔍 Debug: Warming up model...")
            self.model.create_chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1,
                temperature=0.0,
                stream=False,
            )
            logger.debug(f"🔍 Debug: Model warmup completed successfully")
        except Exception as e:
            logger.warning(f"⚠️  Debug: Model warmup failed: {e}")

    # =========================================================================
    # ADAPTER SUPPORT (constructor-based reload)
    # =========================================================================

    def _resolve_gguf_adapter(self, adapter_hash: str) -> Optional[str]:
        """Resolve adapter hash to a GGUF adapter file path.

        Looks up the adapter via adapter_cache, then checks for:
        1. gguf/adapter.gguf (preferred, from dual-format training)
        2. Any *.gguf file in the adapter directory root (fallback)

        Returns:
            Path to the .gguf adapter file, or None if not found.
        """
        from services import adapter_cache

        adapter_dir = adapter_cache.get_adapter_path(adapter_hash)
        if adapter_dir is None:
            logger.warning(f"Adapter {adapter_hash} not found in cache/storage")
            return None

        # Preferred: gguf/adapter.gguf from dual-format training output
        gguf_subdir = adapter_dir / "gguf" / "adapter.gguf"
        if gguf_subdir.is_file():
            logger.debug(f"Resolved GGUF adapter: {gguf_subdir}")
            return str(gguf_subdir)

        # Fallback: any *.gguf file in the adapter root
        for candidate in adapter_dir.glob("*.gguf"):
            if candidate.is_file():
                logger.debug(f"Resolved GGUF adapter (fallback): {candidate}")
                return str(candidate)

        logger.warning(f"No .gguf adapter file found in {adapter_dir}")
        return None

    def _reload_with_adapter(self, lora_path: Optional[str], lora_scale: float = 1.0) -> None:
        """Destroy the current model and recreate it with (or without) a LoRA adapter.

        Must be called with self._lock held.
        """
        logger.info(f"🔄 Reloading model {'with adapter ' + lora_path if lora_path else 'without adapter'}...")
        start = time.time()

        # Destroy current model
        if self.model is not None:
            del self.model
            self.model = None

        from llama_cpp import Llama

        ctor_kwargs = dict(self._llama_init_kwargs)
        if lora_path:
            ctor_kwargs["lora_path"] = lora_path
            ctor_kwargs["lora_scale"] = lora_scale

        self.model = Llama(**ctor_kwargs)
        self._warmup()

        elapsed = time.time() - start
        logger.info(f"✅ Model reloaded in {elapsed:.2f}s")

    def load_adapter(self, adapter_path: str, scale: float = 1.0) -> None:
        """Load a LoRA adapter by reloading the model with the adapter baked in."""
        with self._lock:
            self._reload_with_adapter(adapter_path, scale)
            self._current_adapter_path = adapter_path
            self._current_adapter_scale = scale

    def remove_adapter(self) -> None:
        """Remove the current adapter by reloading the base model."""
        with self._lock:
            if self._current_adapter_path is None:
                return
            self._reload_with_adapter(None)
            self._current_adapter_hash = None
            self._current_adapter_path = None
            self._current_adapter_scale = 1.0

    def get_current_adapter(self) -> Optional[str]:
        """Return the currently loaded adapter path, or None if base model."""
        return self._current_adapter_path

    # =========================================================================
    # CHAT METHODS
    # =========================================================================

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """Chat method with temperature support"""
        return self.chat_with_temperature(messages, temperature)

    def chat_with_temperature(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """Delegate to appropriate backend with thread safety"""
        # Use thread lock to prevent concurrent access issues
        with self._lock:
            if self.inference_engine == "vllm":
                return self._chat_vllm(messages, temperature)
            else:
                return self._chat_llama_cpp(messages, temperature)

    def _chat_vllm(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """Chat using vLLM backend"""
        logger.debug(f"🚀 vLLM chat with {len(messages)} messages, temperature: {temperature}")

        try:
            response_text, usage = self.backend.generate(
                messages=messages,
                temperature=temperature,
                max_tokens=int(os.getenv("JARVIS_MAX_TOKENS", "7000")),
                top_p=0.95
            )

            # Update last usage
            self.last_usage = time.time()

            return response_text

        except Exception as e:
            logger.error(f"❌ vLLM chat error: {e}")
            raise

    def _chat_llama_cpp(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        # Start timing
        start_time = time.time()

        # Enhanced logging for prefix matching diagnosis
        logger.debug(f"🔍 PREFIX DEBUG: Starting chat with {len(messages)} messages")
        logger.debug(f"🔍 PREFIX DEBUG: Temperature: {temperature}")

        # Log each message in detail for prefix matching analysis
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            content_preview = content[:100] + "..." if len(content) > 100 else content
            logger.debug(f"🔍 PREFIX DEBUG: Message {i+1} [{role}]: {content_preview}")
            logger.debug(f"🔍 PREFIX DEBUG: Message {i+1} length: {len(content)} chars, {len(content.split())} words")

        # Capture initial power metrics (if available)
        initial_gpu_power = self.power_metrics.gpu_power
        initial_cpu_power = self.power_metrics.cpu_power

        # llama_cpp supports structured chat messages directly
        logger.debug(f"🔍 PREFIX DEBUG: Calling LLaMA.cpp create_chat_completion...")

        # Log the exact messages being sent to help diagnose prefix matching
        logger.debug(f"🔍 PREFIX DEBUG: Raw messages being sent to LLaMA.cpp:")
        for i, msg in enumerate(messages):
            logger.debug(f"🔍 PREFIX DEBUG: Raw[{i}]: {msg}")

        try:
            # llama.cpp's create_chat_completion automatically detects chat format from the model
            response = self.model.create_chat_completion(
                messages=messages,  # type: ignore
                temperature=temperature,
                max_tokens=self.max_tokens,  # Use configurable max tokens
                top_p=self.top_p,  # Use configurable top_p
                top_k=self.top_k,  # Use configurable top_k
                repeat_penalty=self.repeat_penalty,  # Use configurable repeat penalty
                stream=False,
                mirostat_mode=self.mirostat_mode,  # Use configurable mirostat mode
                mirostat_tau=self.mirostat_tau,  # Use configurable mirostat tau
                mirostat_eta=self.mirostat_eta,  # Use configurable mirostat eta
            )
            logger.debug(f"🔍 PREFIX DEBUG: LLaMA.cpp response received")

        except Exception as e:
            logger.warning(f"⚠️  Error during inference: {e}")
            # Try to recover by reinitializing the model context
            try:
                logger.info(f"🔄 Attempting to recover from inference error...")
                # Force a small warmup inference to reset context
                self.model.create_chat_completion(
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1,
                    temperature=0.0,
                    stream=False,
                )
                logger.info(f"✅ Recovery successful, retrying original request...")
                # Retry the original request
                response = self.model.create_chat_completion(
                    messages=messages,  # type: ignore
                    temperature=temperature,
                    max_tokens=self.max_tokens,  # Use configurable max tokens
                    top_p=self.top_p,  # Use configurable top_p
                    top_k=self.top_k,  # Use configurable top_k
                    repeat_penalty=self.repeat_penalty,  # Use configurable repeat penalty
                    stream=False,
                )
            except Exception as retry_e:
                logger.error(f"❌ Recovery failed: {retry_e}")
                return ""

        # Calculate timing
        end_time = time.time()
        total_time = end_time - start_time

        # Capture final power metrics (if available)
        final_gpu_power = self.power_metrics.gpu_power
        final_cpu_power = self.power_metrics.cpu_power
        avg_gpu_power = (initial_gpu_power + final_gpu_power) / 2
        avg_cpu_power = (initial_cpu_power + final_cpu_power) / 2

        # Extract token stats and print performance metrics
        try:
            content = response["choices"][0]["message"]["content"] or ""  # type: ignore
            usage = response.get("usage", {})  # type: ignore

            # Store usage information for OpenAI-style response
            self.last_usage = usage

            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            # Calculate tokens per second
            tokens_per_second = completion_tokens / total_time if total_time > 0 else 0

            # Calculate first token latency (approximate)
            first_token_time = total_time / completion_tokens if completion_tokens > 0 else 0

            # Enhanced performance stats
            logger.debug(f"🚀 Generated {completion_tokens} tokens in {total_time:.2f}s ({tokens_per_second:.1f} tok/s)")
            logger.debug(f"📊 Prompt: {prompt_tokens} tokens | Completion: {completion_tokens} tokens | Total: {total_tokens} tokens")
            logger.debug(f"⚡ First token latency: ~{first_token_time*1000:.0f}ms")
            logger.debug(f"🌡️  Temperature: {temperature}")

            # Power metrics (if available)
            if self.power_metrics.sudo_available:
                logger.debug(f"🔋 GPU Power: {avg_gpu_power:.0f}mW ({avg_gpu_power/1000:.1f}W) | CPU Power: {avg_cpu_power:.0f}mW ({avg_cpu_power/1000:.1f}W)")
                logger.debug(f"⚙️  GPU: {self.power_metrics.gpu_frequency}MHz @ {self.power_metrics.gpu_utilization:.1f}% utilization")

                # Energy efficiency
                if tokens_per_second > 0:
                    energy_per_token = (avg_gpu_power + avg_cpu_power) / tokens_per_second
                    logger.debug(f"🌱 Energy efficiency: {energy_per_token:.1f}mW per token/s")
            else:
                logger.debug("💡 For power monitoring, run: sudo powermetrics --samplers gpu_power,cpu_power --sample-count 1 -n 1")

            return content

        except (KeyError, IndexError, TypeError):
            return ""

    def process_context(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Process context without generating a response - for warm-up purposes"""
        try:
            # Use a minimal inference to process the context
            # This creates internal representations that can be reused
            response = self.model.create_chat_completion(
                messages=messages,
                temperature=0.0,  # Deterministic
                max_tokens=1,     # Minimal tokens
                top_p=1.0,
                top_k=1,
                repeat_penalty=1.0,
                stream=False,
            )

            # Extract the internal context representation
            # This is a simplified approach - in practice, you might want to
            # extract embeddings or other internal representations
            processed_context = {
                "messages": messages,
                "context_processed": True,
                "timestamp": time.time()
            }

            logger.debug(f"🔥 Context processed for {len(messages)} messages")
            return processed_context

        except Exception as e:
            logger.warning(f"⚠️  Error processing context: {e}")
            # Fallback to storing raw messages
            return {
                "messages": messages,
                "context_processed": False,
                "timestamp": time.time()
            }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.context_cache),
            "max_cache_size": self.max_cache_size
        }

    def clear_cache(self):
        """Clear the context cache to free memory"""
        self.context_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.debug("🧹 Context cache cleared")

    def _get_context_key(self, messages: List[Dict[str, str]]) -> str:
        """Generate a cache key for the message context"""
        # Create a hash of the message content for caching
        import hashlib
        content = "".join([msg.get("content", "") for msg in messages])
        return hashlib.md5(content.encode()).hexdigest()

    def _manage_cache_size(self):
        """Manage cache size to prevent memory issues"""
        if len(self.context_cache) > self.max_cache_size:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self.context_cache.keys())[:len(self.context_cache) - self.max_cache_size]
            for key in keys_to_remove:
                del self.context_cache[key]
            logger.debug(f"🧹 Removed {len(keys_to_remove)} old cache entries")

    def unload(self):
        """Unload the model and clean up resources"""
        if hasattr(self, 'model'):
            del self.model
            self.model = None
        if hasattr(self, 'power_metrics'):
            self.power_metrics.stop_monitoring()
        self._current_adapter_hash = None
        self._current_adapter_path = None
        self._current_adapter_scale = 1.0
        logger.info(f"🔄 Unloaded model: {self.model_path}")

    def __del__(self):
        """Clean up power monitoring on destruction"""
        if hasattr(self, 'power_metrics'):
            self.power_metrics.stop_monitoring()

    def generate_text_chat(
        self,
        model_cfg: Any,
        messages: List[NormalizedMessage],
        params: GenerationParams,
    ) -> ChatResult:
        """
        Generate text chat response using GGUF backend.
        Supports both llama.cpp and vLLM inference engines.
        Handles per-request adapter switching via model reload.
        """
        # Convert NormalizedMessage to legacy dict format
        legacy_messages = []
        for msg in messages:
            # Concatenate all text parts
            text_content = " ".join(
                part.text for part in msg.content if isinstance(part, TextPart)
            )
            legacy_messages.append({"role": msg.role, "content": text_content})

        # Use thread lock for thread safety
        with self._lock:
            # Handle adapter switching for llama.cpp engine
            if self.inference_engine == "llama_cpp" and params.adapter_settings:
                self._handle_adapter_switch(params.adapter_settings)

            if self.inference_engine == "vllm":
                # Use vLLM backend
                response_text, usage = self.backend.generate(
                    messages=legacy_messages,
                    temperature=params.temperature,
                    max_tokens=params.max_tokens or self.max_tokens,
                    top_p=self.top_p,
                    # Pass response_format for JSON support
                    response_format=params.response_format,
                )
                # vLLM returns usage dict, but we need to store it
                if isinstance(usage, dict):
                    self.last_usage = usage
                return ChatResult(content=response_text, usage=usage)
            else:
                # Use llama.cpp backend
                # Note: llama.cpp's create_chat_completion automatically detects chat format from the model
                completion_kwargs = {
                    "messages": legacy_messages,  # type: ignore
                    "temperature": params.temperature,
                    "max_tokens": params.max_tokens or self.max_tokens,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "repeat_penalty": self.repeat_penalty,
                    "stream": False,
                    "mirostat_mode": self.mirostat_mode,
                    "mirostat_tau": self.mirostat_tau,
                    "mirostat_eta": self.mirostat_eta,
                }
                if params.grammar:
                    try:
                        from llama_cpp import LlamaGrammar
                        completion_kwargs["grammar"] = LlamaGrammar.from_string(params.grammar)
                        logger.info("🧩 GGUF grammar applied to llama.cpp request")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to apply GGUF grammar: {e}")
                response = self.model.create_chat_completion(**completion_kwargs)

                content = response["choices"][0]["message"]["content"] or ""  # type: ignore
                usage = response.get("usage", {})  # type: ignore
                self.last_usage = usage

                return ChatResult(content=content, usage=usage)

    def _handle_adapter_switch(self, adapter_settings: dict) -> None:
        """Check adapter_settings and reload model if adapter changed.

        Must be called with self._lock held.

        Sticky behavior: if adapter_settings is provided but has no hash,
        the current adapter stays loaded (avoids unnecessary reloads).
        """
        adapter_hash = adapter_settings.get("hash")
        adapter_scale = adapter_settings.get("scale", 1.0)
        adapter_enabled = adapter_settings.get("enabled", True)

        if not adapter_hash or not adapter_enabled:
            # No adapter requested or explicitly disabled — keep current state
            if adapter_hash and not adapter_enabled:
                logger.debug(f"🧩 [GGUF] Adapter disabled by request: hash={adapter_hash}")
            return

        # Same adapter already loaded — skip reload
        if adapter_hash == self._current_adapter_hash:
            logger.debug(f"🧩 [GGUF] Adapter already loaded: {adapter_hash[:8]}")
            return

        # Resolve adapter hash to a .gguf file path
        gguf_path = self._resolve_gguf_adapter(adapter_hash)
        if gguf_path is None:
            logger.warning(f"⚠️  [GGUF] Adapter not found: hash={adapter_hash}")
            return

        # Reload model with the new adapter
        logger.info(f"🧩 [GGUF] Switching adapter: {adapter_hash[:8]} (scale={adapter_scale})")
        self._reload_with_adapter(gguf_path, adapter_scale)
        self._current_adapter_hash = adapter_hash
        self._current_adapter_path = gguf_path
        self._current_adapter_scale = adapter_scale
