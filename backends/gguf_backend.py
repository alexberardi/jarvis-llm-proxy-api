from llama_cpp import Llama
import os
import time
from typing import List, Dict, Any, Union, Optional
from .power_metrics import PowerMetrics

class GGUFClient:
    def __init__(self, model_path: str, chat_format: str, context_window: int = None):
        if not model_path:
            raise ValueError("Model path is required")
        
        # Store model name for unload functionality
        self.model_name = model_path
        self.model_path = model_path
        self.chat_format = chat_format
        self.model = None
        
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
            n_threads_batch=10,
            n_gpu_layers=-1,
            chat_format=chat_format,
            verbose=True,
            metal=True,
            use_mlock=False,
            use_mmap=True,
            seed=-1,
            rope_freq_scale=1.0,
            rope_freq_base=1000000.0,
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

    def chat(self, messages: List[Dict[str, str]]) -> str:
        # Start timing
        start_time = time.time()
        
        # Capture initial power metrics (if available)
        initial_gpu_power = self.power_metrics.gpu_power
        initial_cpu_power = self.power_metrics.cpu_power
        
        # llama_cpp supports structured chat messages directly
        response = self.model.create_chat_completion(
            messages=messages,  # type: ignore
            temperature=0.7,
            max_tokens=1024,
            top_p=0.95,
            top_k=40,
            repeat_penalty=1.1,
            stream=False,
        )
        
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