import logging
import os
import threading
import time
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

from .power_metrics import PowerMetrics
from managers.chat_types import NormalizedMessage, TextPart, GenerationParams, ChatResult
from backends.base import LLMBackendBase

logger = logging.getLogger("uvicorn")


class GGUFAdapterNotSupportedError(RuntimeError):
    """Raised when GGUF LoRA adapter is configured but not currently supported."""
    pass


# =============================================================================
# ORPHANED CODE: llama.cpp Per-Request Adapter Loading
# =============================================================================
#
# THIS CODE WILL NOT WORK UNTIL LLAMA.CPP FIXES THEIR SCHEDULER BUG
#
# Bug: GGML_ASSERT(hash_set.size == hash_set.keys.size) in ggml-backend.c
# Status: Open as of January 2025
# Tracking: https://github.com/ggerganov/llama.cpp/issues/7742
# Related: https://github.com/ggerganov/llama.cpp/issues/4485
#
# The llama.cpp scheduler has a bug that causes assertions to fail when
# using LoRA adapters with certain model/adapter combinations. This happens
# during the graph scheduling phase when the hash_set size doesn't match
# the number of keys.
#
# When this bug is fixed upstream, uncomment and integrate this code into
# GGUFClient to enable per-request adapter swapping similar to vLLM.
#
# Expected API (llama-cpp-python):
#   from llama_cpp import Llama
#
#   # Initialize with LoRA support
#   llm = Llama(
#       model_path="model.gguf",
#       lora_path="adapter.gguf",  # Optional initial adapter
#       lora_scale=1.0,
#   )
#
#   # Per-request adapter loading (DOES NOT WORK YET)
#   # The llama_cpp.Llama class would need to expose methods like:
#   # llm.load_lora_adapter(path, scale=1.0)
#   # llm.unload_lora_adapter()
#
# =============================================================================

class _GGUFAdapterManager:
    """
    ORPHANED: Per-request adapter manager for llama.cpp / GGUF models.

    THIS CLASS IS NOT FUNCTIONAL due to upstream llama.cpp bug.
    See: https://github.com/ggerganov/llama.cpp/issues/7742

    When the bug is fixed, this class can be used to manage adapter loading:

    Usage (future):
        manager = _GGUFAdapterManager(llama_model)
        manager.load_adapter("/path/to/adapter.gguf")
        response = llama_model.create_chat_completion(...)
        manager.unload_adapter()
    """

    def __init__(self, model: Any):
        """
        Initialize adapter manager.

        Args:
            model: A llama_cpp.Llama instance
        """
        self._model = model
        self._current_adapter_path: Optional[str] = None
        self._adapter_loaded: bool = False

    def load_adapter(self, adapter_path: str, scale: float = 1.0) -> bool:
        """
        Load a LoRA adapter for subsequent inference calls.

        THIS METHOD DOES NOT WORK due to llama.cpp scheduler bug.
        See: https://github.com/ggerganov/llama.cpp/issues/7742

        Args:
            adapter_path: Path to .gguf adapter file
            scale: LoRA scaling factor (default 1.0)

        Returns:
            True if adapter loaded successfully, False otherwise

        Raises:
            GGUFAdapterNotSupportedError: Always, until bug is fixed
        """
        raise GGUFAdapterNotSupportedError(
            "llama.cpp LoRA adapters are disabled due to upstream scheduler bug. "
            "Track fix at: https://github.com/ggerganov/llama.cpp/issues/7742"
        )

        # --- ORPHANED IMPLEMENTATION (uncomment when bug is fixed) ---
        #
        # if self._current_adapter_path == adapter_path:
        #     print(f"🧩 GGUF adapter already loaded: {adapter_path}")
        #     return True
        #
        # adapter_file = Path(adapter_path)
        # if not adapter_file.exists():
        #     print(f"⚠️  GGUF adapter not found: {adapter_path}")
        #     return False
        #
        # if adapter_file.suffix.lower() != ".gguf":
        #     print(f"⚠️  GGUF adapter must be .gguf file: {adapter_path}")
        #     return False
        #
        # try:
        #     # Unload current adapter if any
        #     if self._adapter_loaded:
        #         self.unload_adapter()
        #
        #     print(f"🧩 GGUF loading adapter: {adapter_path} (scale={scale})")
        #     start_time = time.time()
        #
        #     # llama-cpp-python API for loading adapters
        #     # Note: This requires llama-cpp-python built with LoRA support
        #     # and the underlying llama.cpp scheduler bug to be fixed
        #     self._model.load_lora_adapter(adapter_path, scale=scale)
        #
        #     self._current_adapter_path = adapter_path
        #     self._adapter_loaded = True
        #
        #     elapsed = time.time() - start_time
        #     print(f"✅ GGUF adapter loaded in {elapsed:.2f}s")
        #     return True
        #
        # except AttributeError as e:
        #     print(f"⚠️  llama-cpp-python doesn't support load_lora_adapter: {e}")
        #     print("    This feature requires llama-cpp-python >= X.X.X")
        #     return False
        # except Exception as e:
        #     print(f"⚠️  GGUF adapter load failed: {e}")
        #     return False

    def unload_adapter(self) -> bool:
        """
        Unload current LoRA adapter, reverting to base model.

        THIS METHOD DOES NOT WORK due to llama.cpp scheduler bug.
        See: https://github.com/ggerganov/llama.cpp/issues/7742

        Returns:
            True if adapter unloaded successfully, False otherwise
        """
        if not self._adapter_loaded:
            logger.debug("🧩 GGUF no adapter to unload (base model)")
            return True

        raise GGUFAdapterNotSupportedError(
            "llama.cpp LoRA adapters are disabled due to upstream scheduler bug. "
            "Track fix at: https://github.com/ggerganov/llama.cpp/issues/7742"
        )

        # --- ORPHANED IMPLEMENTATION (uncomment when bug is fixed) ---
        #
        # try:
        #     print("🧩 GGUF unloading adapter, reverting to base model...")
        #     start_time = time.time()
        #
        #     self._model.unload_lora_adapter()
        #
        #     self._current_adapter_path = None
        #     self._adapter_loaded = False
        #
        #     elapsed = time.time() - start_time
        #     print(f"✅ GGUF adapter unloaded in {elapsed:.2f}s")
        #     return True
        #
        # except AttributeError as e:
        #     print(f"⚠️  llama-cpp-python doesn't support unload_lora_adapter: {e}")
        #     return False
        # except Exception as e:
        #     print(f"⚠️  GGUF adapter unload failed: {e}")
        #     return False

    def get_current_adapter(self) -> Optional[str]:
        """Return the currently loaded adapter path, or None if base model."""
        return self._current_adapter_path

    @property
    def is_adapter_loaded(self) -> bool:
        """Check if an adapter is currently loaded."""
        return self._adapter_loaded


# =============================================================================
# END ORPHANED CODE
# =============================================================================


class GGUFClient(LLMBackendBase):
    def __init__(self, model_path: str, chat_format: str, stop_tokens: List[str] = None, context_window: int = None):
        if not model_path:
            raise ValueError("Model path is required")
        
        # ----------------------------------------------------------------
        # GGUF LoRA adapters are currently disabled due to a llama.cpp
        # scheduler bug (GGML_ASSERT hash_set.size). If an adapter path
        # is configured, raise an error directing users to vLLM/Transformers.
        # ----------------------------------------------------------------
        adapter_path_env = os.getenv("JARVIS_ADAPTER_PATH", "").strip()
        if adapter_path_env:
            raise GGUFAdapterNotSupportedError(
                "GGUF LoRA adapters are currently not supported due to a llama.cpp "
                "runtime bug (scheduler hash_set assert). Please use the vLLM or "
                "Transformers backend for adapter-enabled inference. "
                "To use the GGUF backend without adapters, unset JARVIS_ADAPTER_PATH."
            )
        
        # Store model name for unload functionality
        self.model_name = model_path
        self.model_path = model_path
        self.chat_format = chat_format
        self.model = None
        self.last_usage = None
        self._lock = threading.Lock()  # Add thread safety
        
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
        
        # GGUF LoRA is disabled - no adapter loading
        self.lora_path = None

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

    def _init_llama_cpp(self, model_path: str, chat_format: str, stop_tokens: List[str], context_window: int, n_threads: int, n_gpu_layers: int, verbose: bool, seed: int, ctx_window: int, n_batch: int, n_ubatch: int, rope_scaling_type: int, mul_mat_q: bool, f16_kv: bool):
        """Initialize llama.cpp backend"""
        from llama_cpp import Llama

        logger.debug(f"🔍 Debug: LLAMA_METAL env var: {os.getenv('LLAMA_METAL', 'not set')}")
        logger.debug(f"🔍 Debug: Metal will be enabled: {os.getenv('LLAMA_METAL', 'false').lower() == 'true'}")
        logger.debug(f"🔍 Debug: Loading model with n_gpu_layers={n_gpu_layers}")
        
        # Stable initialization with llama-cpp-python (no LoRA - disabled)
        self.model = Llama(
            model_path=model_path,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,  # Use configurable verbose setting
            seed=seed,  # Use configurable seed
            n_ctx=context_window,
            n_batch=n_batch,  # Use configurable batch size
            n_ubatch=n_ubatch,  # Use configurable micro batch size
            rope_scaling_type=rope_scaling_type,  # Use configurable RoPE scaling
            mul_mat_q=mul_mat_q,  # Use configurable matrix multiplication
            f16_kv=f16_kv,  # Use configurable F16 KV cache
            flash_attn=self.flash_attn,
        )
        self.backend = self
        self.inference_engine = "llama_cpp"
        
        # Debug: Check model loading results
        logger.info(f"✅ Model loaded successfully!")
        logger.debug(f"🔍 Debug: Model context size: {self.model.n_ctx()}")
        logger.debug(f"🔍 Debug: Model vocab size: {self.model.n_vocab()}")

        # Warm up the model with a small inference
        try:
            logger.debug(f"🔍 Debug: Warming up model...")
            warmup_response = self.model.create_chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1,
                temperature=0.0,
                stream=False,
            )
            logger.debug(f"🔍 Debug: Model warmup completed successfully")
        except Exception as e:
            logger.warning(f"⚠️  Debug: Model warmup failed: {e}")

        logger.debug(f"🔍 Debug: Model initialization complete")

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
    