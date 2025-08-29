from llama_cpp import Llama
import os
import time
from typing import List, Dict, Any, Union, Optional
from .power_metrics import PowerMetrics

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

        # Initialize power monitoring (optional)
        self.power_metrics = PowerMetrics()
        self.power_metrics.start_monitoring()
        
        # Get context window from parameter or environment variable, default to 512
        if context_window is None:
            context_window = int(os.getenv("JARVIS_MODEL_CONTEXT_WINDOW", "512"))
        
        print(f"🔍 Debug: LLAMA_METAL env var: {os.getenv('LLAMA_METAL', 'not set')}")
        print(f"🔍 Debug: Metal will be enabled: {os.getenv('LLAMA_METAL', 'false').lower() == 'true'}")
        print(f"🔍 Debug: Model path: {model_path}")
        print(f"🔍 Debug: Chat format: {chat_format}")
        print(f"🔍 Debug: Context window: {context_window}")
        print(f"🔍 Debug: Loading model with n_gpu_layers=-1 (all layers on GPU)")
        
        # Stable initialization with llama-cpp-python 0.2.64
        self.model = Llama(
            model_path=model_path,
            n_threads=10,
            n_gpu_layers=-1,
            verbose=True,
            seed=-0,
            n_ctx=context_window,  # Use configurable context window
        )
        
        # Debug: Check model loading results
        print(f"✅ Model loaded successfully!")
        print(f"🔍 Debug: Model context size: {self.model.n_ctx()}")
        print(f"🔍 Debug: Model vocab size: {self.model.n_vocab()}")
        
        # Try to get GPU memory info if available
        try:
            print(f"🔍 Debug: Checking GPU memory usage...")
            # Force a small inference to see if GPU is used
            test_response = self.model.create_completion("Hello", max_tokens=1)
            print(f"🔍 Debug: Test inference completed successfully")
        except Exception as e:
            print(f"⚠️  Debug: Test inference failed: {e}")
            
        print(f"🔍 Debug: Model initialization complete")

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """Chat method with temperature support"""
        return self.chat_with_temperature(messages, temperature)
    
    def chat_with_temperature(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
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
        
        response = self.model.create_chat_completion(
            messages=messages,  # type: ignore
            temperature=temperature,
            max_tokens=7000,
            top_p=0.95,
            top_k=40,
            repeat_penalty=1.1,
            stream=False,
        )
        print(f"🔍 PREFIX DEBUG: LLaMA.cpp response received")
        
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