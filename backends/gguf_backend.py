import os
import time
from typing import List, Dict, Any, Union, Optional
from .power_metrics import PowerMetrics
import threading
from managers.chat_types import NormalizedMessage, TextPart, GenerationParams, ChatResult

class GGUFClient:
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
        
        print(f"🔍 Debug: Inference engine: {inference_engine}")
        print(f"🔍 Debug: Model path: {model_path}")
        print(f"🔍 Debug: Chat format: {chat_format}")
        print(f"🔍 Debug: Context window: {context_window}")
        print(f"🔍 Debug: Threads: {n_threads}")
        print(f"🔍 Debug: GPU layers: {n_gpu_layers}")
        print(f"🔍 Debug: Context cache: {'enabled' if self.enable_cache else 'disabled'}")
        print(f"🔍 Debug: Batch size: {n_batch}")
        print(f"🔍 Debug: Micro batch size: {n_ubatch}")
        print(f"🔍 Debug: F16 KV cache: {'enabled' if f16_kv else 'disabled'}")
        print(f"🔍 Debug: Matrix multiplication: {'enabled' if mul_mat_q else 'disabled'}")
        
        if inference_engine == "vllm":
            print(f"🚀 Using vLLM inference engine")
            self._init_vllm(model_path, chat_format, stop_tokens, context_window, n_threads, n_gpu_layers, verbose, seed, n_batch, n_ubatch, rope_scaling_type, mul_mat_q, f16_kv)
        else:
            print(f"🦙 Using llama.cpp inference engine")
            self._init_llama_cpp(model_path, chat_format, stop_tokens, context_window, n_threads, n_gpu_layers, verbose, seed, context_window, n_batch, n_ubatch, rope_scaling_type, mul_mat_q, f16_kv)

    def _init_vllm(self, model_path: str, chat_format: str, stop_tokens: List[str], context_window: int, n_threads: int, n_gpu_layers: int, verbose: bool, seed: int, n_batch: int, n_ubatch: int, rope_scaling_type: int, mul_mat_q: bool, f16_kv: bool):
        """Initialize vLLM backend"""
        try:
            from .vllm_backend import VLLMClient
            self.backend = VLLMClient(model_path, chat_format, stop_tokens, context_window)
            self.inference_engine = "vllm"
        except ValueError as e:
            if "vLLM requires HuggingFace model names" in str(e):
                print(f"")
                print(f"🔧 CONFIGURATION SUGGESTION:")
                print(f"   For GGUF files, use: JARVIS_INFERENCE_ENGINE=llama_cpp")
                print(f"   For vLLM, switch to TRANSFORMERS backend with HF models:")
                print(f"   ")
                print(f"   JARVIS_MODEL_BACKEND=TRANSFORMERS")
                print(f"   JARVIS_MODEL_NAME=microsoft/Phi-3-mini-4k-instruct")
                print(f"   JARVIS_INFERENCE_ENGINE=vllm")
                print(f"")
                print(f"🔄 Falling back to llama.cpp for GGUF file...")
                self._init_llama_cpp(model_path, chat_format, stop_tokens, context_window, n_threads, n_gpu_layers, verbose, seed, context_window, n_batch, n_ubatch, rope_scaling_type, mul_mat_q, f16_kv)
            else:
                raise

    def _init_llama_cpp(self, model_path: str, chat_format: str, stop_tokens: List[str], context_window: int, n_threads: int, n_gpu_layers: int, verbose: bool, seed: int, ctx_window: int, n_batch: int, n_ubatch: int, rope_scaling_type: int, mul_mat_q: bool, f16_kv: bool):
        """Initialize llama.cpp backend"""
        from llama_cpp import Llama
        
        print(f"🔍 Debug: LLAMA_METAL env var: {os.getenv('LLAMA_METAL', 'not set')}")
        print(f"🔍 Debug: Metal will be enabled: {os.getenv('LLAMA_METAL', 'false').lower() == 'true'}")
        print(f"🔍 Debug: Loading model with n_gpu_layers={n_gpu_layers}")
        
        # Stable initialization with llama-cpp-python
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
        )
        self.backend = self
        self.inference_engine = "llama_cpp"
        
        # Debug: Check model loading results
        print(f"✅ Model loaded successfully!")
        print(f"🔍 Debug: Model context size: {self.model.n_ctx()}")
        print(f"🔍 Debug: Model vocab size: {self.model.n_vocab()}")
        
        # Warm up the model with a small inference
        try:
            print(f"🔍 Debug: Warming up model...")
            warmup_response = self.model.create_chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1,
                temperature=0.0,
                stream=False,
            )
            print(f"🔍 Debug: Model warmup completed successfully")
        except Exception as e:
            print(f"⚠️  Debug: Model warmup failed: {e}")
            
        print(f"🔍 Debug: Model initialization complete")

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
        print(f"🚀 vLLM chat with {len(messages)} messages, temperature: {temperature}")
        
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
            print(f"❌ vLLM chat error: {e}")
            raise
    
    def _chat_llama_cpp(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        # Start timing
        start_time = time.time()

        # Enhanced logging for prefix matching diagnosis
        print(f"🔍 PREFIX DEBUG: Starting chat with {len(messages)} messages")
        print(f"🔍 PREFIX DEBUG: Temperature: {temperature}")
        
        # Log each message in detail for prefix matching analysis
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            content_preview = content[:100] + "..." if len(content) > 100 else content
            print(f"🔍 PREFIX DEBUG: Message {i+1} [{role}]: {content_preview}")
            print(f"🔍 PREFIX DEBUG: Message {i+1} length: {len(content)} chars, {len(content.split())} words")
        
        # Capture initial power metrics (if available)
        initial_gpu_power = self.power_metrics.gpu_power
        initial_cpu_power = self.power_metrics.cpu_power
        
        # llama_cpp supports structured chat messages directly
        print(f"🔍 PREFIX DEBUG: Calling LLaMA.cpp create_chat_completion...")
        
        # Log the exact messages being sent to help diagnose prefix matching
        print(f"🔍 PREFIX DEBUG: Raw messages being sent to LLaMA.cpp:")
        for i, msg in enumerate(messages):
            print(f"🔍 PREFIX DEBUG: Raw[{i}]: {msg}")
        
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
            print(f"🔍 PREFIX DEBUG: LLaMA.cpp response received")
            
        except Exception as e:
            print(f"⚠️  Error during inference: {e}")
            # Try to recover by reinitializing the model context
            try:
                print(f"🔄 Attempting to recover from inference error...")
                # Force a small warmup inference to reset context
                self.model.create_chat_completion(
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1,
                    temperature=0.0,
                    stream=False,
                )
                print(f"✅ Recovery successful, retrying original request...")
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
                print(f"❌ Recovery failed: {retry_e}")
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
            print(f"🚀 Generated {completion_tokens} tokens in {total_time:.2f}s ({tokens_per_second:.1f} tok/s)")
            print(f"📊 Prompt: {prompt_tokens} tokens | Completion: {completion_tokens} tokens | Total: {total_tokens} tokens")
            print(f"⚡ First token latency: ~{first_token_time*1000:.0f}ms")
            print(f"🌡️  Temperature: {temperature}")
            
            # Power metrics (if available)
            if self.power_metrics.sudo_available:
                print(f"🔋 GPU Power: {avg_gpu_power:.0f}mW ({avg_gpu_power/1000:.1f}W) | CPU Power: {avg_cpu_power:.0f}mW ({avg_cpu_power/1000:.1f}W)")
                print(f"⚙️  GPU: {self.power_metrics.gpu_frequency}MHz @ {self.power_metrics.gpu_utilization:.1f}% utilization")
                
                # Energy efficiency
                if tokens_per_second > 0:
                    energy_per_token = (avg_gpu_power + avg_cpu_power) / tokens_per_second
                    print(f"🌱 Energy efficiency: {energy_per_token:.1f}mW per token/s")
            else:
                print("💡 For power monitoring, run: sudo powermetrics --samplers gpu_power,cpu_power --sample-count 1 -n 1")
            
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
            
            print(f"🔥 Context processed for {len(messages)} messages")
            return processed_context
            
        except Exception as e:
            print(f"⚠️  Error processing context: {e}")
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
        print("🧹 Context cache cleared")
    
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
            print(f"🧹 Removed {len(keys_to_remove)} old cache entries")
    
    def unload(self):
        """Unload the model and clean up resources"""
        if hasattr(self, 'model'):
            del self.model
            self.model = None
        if hasattr(self, 'power_metrics'):
            self.power_metrics.stop_monitoring()
        print(f"🔄 Unloaded model: {self.model_path}")
    
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
                # Enable JSON grammar for structured output enforcement
                # Grammar ensures valid JSON structure at token generation level,
                # preventing issues like duplicate keys, malformed syntax, etc.
                # The system message provides schema guidance, while grammar ensures syntax correctness.
                # Can be disabled via JARVIS_DISABLE_JSON_GRAMMAR=true if causing issues
                # Note: Qwen3 models have known grammar compatibility issues, so we disable it for them
                grammar = None
                grammar_disabled = os.getenv("JARVIS_DISABLE_JSON_GRAMMAR", "false").lower() == "true"
                
                # Check if this is a Qwen model (known to have grammar issues with llama.cpp)
                # Qwen models have reported llama_decode errors and output control problems
                model_path_lower = self.model_path.lower() if hasattr(self, 'model_path') and self.model_path else ""
                model_id_lower = ""
                try:
                    model_id_lower = model_cfg.model_id.lower() if hasattr(model_cfg, "model_id") and model_cfg.model_id else ""
                except Exception:
                    model_id_lower = ""
                is_qwen = "qwen" in model_path_lower or "qwen" in model_id_lower
                if is_qwen and not grammar_disabled:
                    print("⚠️  Qwen model detected - disabling grammar due to known compatibility issues")
                    print("   (Qwen models have reported llama_decode errors with grammar)")
                    grammar_disabled = True

                # Global switch: disable grammar by default for JSON enforcement to avoid
                # runaway generations / truncation; rely on downstream JSON validation +
                # retry instead. Set JARVIS_ENABLE_JSON_GRAMMAR=true to re-enable.
                grammar_enabled = os.getenv("JARVIS_ENABLE_JSON_GRAMMAR", "false").lower() == "true"
                if not grammar_enabled:
                    grammar_disabled = True
                    print("⚠️  JSON grammar disabled by default (JARVIS_ENABLE_JSON_GRAMMAR!=true); using validation + retry pipeline")
                
                if params.response_format and params.response_format.get("type") == "json_object" and not grammar_disabled:
                    grammar = self._get_json_grammar()
                    if grammar is not None:
                        print("✅ Using JSON grammar for structured output enforcement")
                    else:
                        print("⚠️  JSON grammar not available, falling back to system message only")
                elif grammar_disabled:
                    print("⚠️  JSON grammar disabled via JARVIS_DISABLE_JSON_GRAMMAR, using system message only")
                
                # Call llama.cpp with optional grammar
                # Note: grammar parameter may not be supported in all versions of llama-cpp-python
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
                
                # Only add grammar if it's available
                if grammar is not None:
                    completion_kwargs["grammar"] = grammar
                
                # Try calling with grammar, fall back without it if not supported or if it causes errors
                grammar_failed = False
                try:
                    response = self.model.create_chat_completion(**completion_kwargs)
                except (TypeError, RuntimeError, ValueError) as e:
                    # Grammar can cause various errors:
                    # - TypeError: parameter not supported
                    # - RuntimeError: llama_decode errors (e.g., -3, -1, etc.)
                    # - ValueError: invalid grammar
                    error_str = str(e).lower()
                    error_repr = repr(e).lower()
                    
                    # Check if this looks like a grammar-related error
                    is_grammar_error = (
                        grammar is not None and (
                            "grammar" in error_str or 
                            "llama_decode" in error_str or
                            "llama_decode" in error_repr or
                            "decode" in error_str or
                            "-3" in error_str or
                            "-3" in error_repr or
                            "returned -" in error_str or
                            "returned -" in error_repr
                        )
                    )
                    
                    if is_grammar_error:
                        print(f"⚠️  Grammar caused error: {e}")
                        print("⚠️  Falling back to system message only for JSON enforcement")
                        grammar_failed = True
                        # Retry without grammar
                        completion_kwargs.pop("grammar", None)
                        try:
                            response = self.model.create_chat_completion(**completion_kwargs)
                        except Exception as retry_e:
                            print(f"❌ Error even without grammar: {retry_e}")
                            raise
                    else:
                        # Not a grammar-related error, re-raise
                        raise
                except Exception as e:
                    # Catch-all for other exceptions - only fall back if grammar was used
                    if grammar is not None:
                        error_str = str(e).lower()
                        # If it's an unexpected error and we're using grammar, try without it
                        print(f"⚠️  Unexpected error with grammar: {e}")
                        print("⚠️  Attempting fallback without grammar")
                        grammar_failed = True
                        completion_kwargs.pop("grammar", None)
                        try:
                            response = self.model.create_chat_completion(**completion_kwargs)
                        except Exception as retry_e:
                            # If it still fails, the error is not grammar-related
                            print(f"❌ Error persists without grammar, re-raising original: {retry_e}")
                            raise e  # Re-raise original error
                    else:
                        # No grammar was used, re-raise
                        raise
                
                content = response["choices"][0]["message"]["content"] or ""  # type: ignore
                usage = response.get("usage", {})  # type: ignore
                self.last_usage = usage
                
                return ChatResult(content=content, usage=usage)
    
    def _get_json_grammar(self) -> Optional[Any]:
        """
        Get JSON grammar for llama.cpp if supported.
        Returns None if grammar is not available.
        """
        try:
            from llama_cpp import LlamaGrammar
            
            # Create a simple JSON object grammar
            # This is a basic JSON grammar - can be expanded for more complex schemas
            json_grammar = r'''
root ::= object
object ::= "{" ws ( string ":" ws value ( "," ws string ":" ws value )* )? "}"
value ::= string | number | object | array | "true" | "false" | "null"
string ::= "\"" ([^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4}))* "\""
number ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
array ::= "[" ws ( value ( "," ws value )* )? "]"
ws ::= [ \t\n\r]*
'''
            
            return LlamaGrammar.from_string(json_grammar, verbose=False)
        except (ImportError, AttributeError, Exception) as e:
            # Grammar not supported or not available
            print(f"⚠️  JSON grammar not available: {e}")
            return None