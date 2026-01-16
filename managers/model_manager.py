import os
from typing import Optional, Dict, Any


def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default

class ModelConfig:
    """Configuration for a single model"""
    def __init__(
        self,
        model_id: str,
        backend_type: str,
        backend_instance: Any,
        supports_images: bool = False,
        context_length: int = 4096,
    ):
        self.model_id = model_id
        self.backend_type = backend_type
        self.backend_instance = backend_instance
        self.supports_images = supports_images
        self.context_length = context_length

class ModelManager:
    def __init__(self):
        self.main_model = None
        self.lightweight_model = None
        self.cloud_model = None
        self.vision_model = None
        self.cloud_model_name = None
        self.vision_model_name = None
        
        # Model registry: maps model_id -> ModelConfig
        self.registry: Dict[str, ModelConfig] = {}
        
        # Aliases for stable client references
        self.aliases: Dict[str, str] = {}
        
        self._initialize_models()
    
    def _initialize_models(self):
        # Get environment variables
        model_backend = os.getenv("JARVIS_MODEL_BACKEND", "OLLAMA").upper()
        lightweight_model_backend = os.getenv("JARVIS_LIGHTWEIGHT_MODEL_BACKEND")
        if lightweight_model_backend:
            lightweight_model_backend = lightweight_model_backend.upper()
        cloud_model_backend = os.getenv("JARVIS_CLOUD_MODEL_BACKEND", "REST").upper()
        vision_model_backend = os.getenv("JARVIS_VISION_MODEL_BACKEND")
        if vision_model_backend:
            vision_model_backend = vision_model_backend.upper()
        
        main_model_path = os.getenv("JARVIS_MODEL_NAME")
        lightweight_model_path = os.getenv("JARVIS_LIGHTWEIGHT_MODEL_NAME")
        cloud_model_path = os.getenv("JARVIS_CLOUD_MODEL_NAME")
        vision_model_path = os.getenv("JARVIS_VISION_MODEL_NAME")
        vision_rest_url = os.getenv("JARVIS_VISION_REST_MODEL_URL", os.getenv("JARVIS_REST_MODEL_URL"))
        main_chat_format = os.getenv("JARVIS_MODEL_CHAT_FORMAT")
        main_stop_tokens = os.getenv("JARVIS_MODEL_STOP_TOKENS")
        lightweight_chat_format = os.getenv("JARVIS_LIGHTWEIGHT_MODEL_CHAT_FORMAT")
        lightweight_stop_tokens = os.getenv("JARVIS_LIGHTWEIGHT_MODEL_STOP_TOKENS")
        cloud_chat_format = os.getenv("JARVIS_CLOUD_MODEL_CHAT_FORMAT")
        main_context_window = _get_int_env("JARVIS_MODEL_CONTEXT_WINDOW", 512)
        lightweight_context_window = _get_int_env("JARVIS_LIGHTWEIGHT_MODEL_CONTEXT_WINDOW", 512)
        cloud_context_window = _get_int_env("JARVIS_CLOUD_MODEL_CONTEXT_WINDOW", 512)
        vision_context_window = _get_int_env("JARVIS_VISION_MODEL_CONTEXT_WINDOW", 131072)
        
        # Check if we should share vLLM models (same name or empty lightweight name/backend)
        should_share_vllm = (
            model_backend == "VLLM" and 
            (not lightweight_model_backend or lightweight_model_backend == "VLLM") and
            (not lightweight_model_path or lightweight_model_path == main_model_path)
        )
        
        if should_share_vllm:
            print(f"🔄 vLLM Memory Optimization: Using shared model instance")
            if not lightweight_model_backend:
                print(f"   → Lightweight backend empty, defaulting to shared vLLM: {main_model_path}")
            elif not lightweight_model_path:
                print(f"   → Lightweight model name empty, using main: {main_model_path}")
            else:
                print(f"   → Identical model names detected: {main_model_path}")
            print(f"   → Memory savings: ~50% (single model instance)")
        
        # Initialize main model
        if model_backend == "MOCK":
            from backends.mock_backend import MockBackend
            self.main_model = MockBackend(main_model_path or "mock-main")
        elif model_backend == "MLX":
            from backends.mlx_backend import MlxClient
            self.main_model = MlxClient(main_model_path)
        elif model_backend == "GGUF":
            from backends.gguf_backend import GGUFClient 
            self.main_model = GGUFClient(main_model_path, main_chat_format, main_stop_tokens, main_context_window)
        elif model_backend == "TRANSFORMERS":
            from backends.transformers_backend import TransformersClient
            self.main_model = TransformersClient(main_model_path, main_chat_format, main_stop_tokens, main_context_window)
        elif model_backend == "VLLM":
            from backends.vllm_backend import VLLMClient
            self.main_model = VLLMClient(main_model_path, main_chat_format, main_stop_tokens, main_context_window)
        elif model_backend == "REST":
            from backends.rest_backend import RestClient
            rest_url = os.getenv("JARVIS_REST_MODEL_URL")
            if not rest_url:
                raise ValueError("JARVIS_REST_MODEL_URL must be set when using REST backend")
            self.main_model = RestClient(rest_url, main_model_path or "jarvis-llm", "main")
        # elif model_backend == "OLLAMA":
            # from backends.ollama_backend import OllamaClient
            # self.main_model = OllamaClient()
        else:
            raise ValueError("Unsupported MODEL_BACKEND. Use 'MOCK', 'MLX', 'GGUF', 'TRANSFORMERS', 'VLLM', 'REST', or 'OLLAMA'.")

        # Initialize lightweight model
        if should_share_vllm:
            # Share the main vLLM model instance to save memory
            self.lightweight_model = self.main_model
            print(f"✅ Lightweight model sharing main vLLM instance")
        elif lightweight_model_backend == "MOCK":
            from backends.mock_backend import MockBackend
            self.lightweight_model = MockBackend(lightweight_model_path or "mock-lightweight")
        elif lightweight_model_backend == "MLX":
            from backends.mlx_backend import MlxClient
            self.lightweight_model = MlxClient(lightweight_model_path)
        elif lightweight_model_backend == "GGUF":
            from backends.gguf_backend import GGUFClient 
            self.lightweight_model = GGUFClient(lightweight_model_path, lightweight_chat_format, lightweight_stop_tokens, lightweight_context_window)
        elif lightweight_model_backend == "TRANSFORMERS":
            from backends.transformers_backend import TransformersClient
            self.lightweight_model = TransformersClient(lightweight_model_path, lightweight_chat_format, lightweight_stop_tokens, lightweight_context_window)
        elif lightweight_model_backend == "VLLM":
            from backends.vllm_backend import VLLMClient
            self.lightweight_model = VLLMClient(lightweight_model_path, lightweight_chat_format, lightweight_stop_tokens, lightweight_context_window)
        elif lightweight_model_backend == "REST":
            from backends.rest_backend import RestClient
            rest_url = os.getenv("JARVIS_REST_LIGHTWEIGHT_MODEL_URL")
            if not rest_url:
                raise ValueError("JARVIS_REST_LIGHTWEIGHT_MODEL_URL must be set when using REST backend")
            self.lightweight_model = RestClient(rest_url, lightweight_model_path or "jarvis-llm", "lightweight")
        # elif lightweight_model_backend == "OLLAMA":
            # from backends.ollama_backend import OllamaClient
            # self.lightweight_model = OllamaClient()
        else:
            if not lightweight_model_backend:
                raise ValueError("JARVIS_LIGHTWEIGHT_MODEL_BACKEND must be set when not using vLLM model sharing. Use 'MOCK', 'MLX', 'GGUF', 'TRANSFORMERS', 'VLLM', 'REST', or 'OLLAMA'.")
            else:
                raise ValueError(f"Unsupported LIGHTWEIGHT_MODEL_BACKEND '{lightweight_model_backend}'. Use 'MOCK', 'MLX', 'GGUF', 'TRANSFORMERS', 'VLLM', 'REST', or 'OLLAMA'.")

        # Initialize cloud model (REST-backed)
        cloud_rest_url = os.getenv("JARVIS_CLOUD_REST_MODEL_URL", os.getenv("JARVIS_REST_MODEL_URL"))
        if cloud_model_path or cloud_rest_url:
            if cloud_model_backend != "REST":
                raise ValueError("Unsupported CLOUD_MODEL_BACKEND. Only 'REST' is supported for cloud.")
            if not cloud_rest_url:
                raise ValueError("JARVIS_CLOUD_REST_MODEL_URL (or JARVIS_REST_MODEL_URL) must be set when enabling cloud model")
            from backends.rest_backend import RestClient
            resolved_cloud_name = cloud_model_path or "jarvis-cloud"
            self.cloud_model_name = resolved_cloud_name
            self.cloud_model = RestClient(cloud_rest_url, resolved_cloud_name, "cloud")
        
        # Initialize vision model (optional)
        if vision_model_path or vision_rest_url:
            if not vision_model_backend:
                if vision_rest_url:
                    vision_model_backend = "REST"
                else:
                    raise ValueError("JARVIS_VISION_MODEL_BACKEND must be set when JARVIS_VISION_MODEL_NAME is set")

            resolved_vision_name = vision_model_path or "jarvis-vision"
            print(f"🔁 Initializing vision model: {resolved_vision_name}")

            if vision_model_backend == "MOCK":
                from backends.mock_backend import MockBackend
                self.vision_model = MockBackend(resolved_vision_name)
                self.vision_model_name = resolved_vision_name
            elif vision_model_backend in ("MLX", "MLX_VISION"):
                # Both MLX and MLX_VISION use mlx-vlm for vision models
                from backends.mlx_vision_backend import MlxVisionClient
                self.vision_model = MlxVisionClient(resolved_vision_name)
                self.vision_model_name = resolved_vision_name
            elif vision_model_backend == "TRANSFORMERS":
                from backends.transformers_vision_backend import TransformersVisionClient
                self.vision_model = TransformersVisionClient(resolved_vision_name)
                self.vision_model_name = resolved_vision_name
            elif vision_model_backend == "REST":
                from backends.rest_backend import RestClient
                if not vision_rest_url:
                    raise ValueError("JARVIS_VISION_REST_MODEL_URL (or JARVIS_REST_MODEL_URL) must be set when using REST backend for vision")
                self.vision_model = RestClient(vision_rest_url, resolved_vision_name, "vision")
                self.vision_model_name = resolved_vision_name
            else:
                raise ValueError(f"Unsupported VISION_MODEL_BACKEND '{vision_model_backend}'. Use 'MOCK', 'MLX', 'MLX_VISION', 'TRANSFORMERS', or 'REST'.")
            print(f"✅ Vision model loaded: {self.vision_model_name}")
        
        # Populate model registry
        self._populate_registry()
    
    def swap_main_model(self, new_model: str, new_model_backend: str, new_model_chat_format: str, new_model_context_window: int = None):
        try:
            # Unload current model
            if hasattr(self.main_model, 'unload'):
                self.main_model.unload()
            
            # Load new model based on backend
            if new_model_backend.upper() == "MLX":
                from backends.mlx_backend import MlxClient
                self.main_model = MlxClient(new_model)
            elif new_model_backend.upper() == "GGUF":
                from backends.gguf_backend import GGUFClient 
                self.main_model = GGUFClient(new_model, new_model_chat_format, None, new_model_context_window)
            elif new_model_backend.upper() == "TRANSFORMERS":
                from backends.transformers_backend import TransformersClient
                self.main_model = TransformersClient(new_model, new_model_chat_format, None, new_model_context_window)
            elif new_model_backend.upper() == "VLLM":
                from backends.vllm_backend import VLLMClient
                self.main_model = VLLMClient(new_model, new_model_chat_format, None, new_model_context_window)
            elif new_model_backend.upper() == "REST":
                from backends.rest_backend import RestClient
                rest_url = os.getenv("JARVIS_REST_MODEL_URL")
                if not rest_url:
                    raise ValueError("JARVIS_REST_MODEL_URL must be set when using REST backend")
                self.main_model = RestClient(rest_url, new_model or "jarvis-llm", "main")
            else:
                raise ValueError("Unsupported MODEL_BACKEND. Use 'MLX', 'GGUF', 'TRANSFORMERS', or 'REST'.")
            
            return {"status": "success", "message": f"Model swapped to {new_model}"}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def swap_lightweight_model(self, new_model: str, new_model_backend: str, new_model_chat_format: str, new_model_context_window: int = None):
        try:
            # Unload current lightweight model
            if hasattr(self.lightweight_model, 'unload'):
                self.lightweight_model.unload()
            
            # Load new lightweight model based on backend
            if new_model_backend.upper() == "MLX":
                from backends.mlx_backend import MlxClient
                self.lightweight_model = MlxClient(new_model)
            elif new_model_backend.upper() == "GGUF":
                from backends.gguf_backend import GGUFClient 
                self.lightweight_model = GGUFClient(new_model, new_model_chat_format, None, new_model_context_window)
            elif new_model_backend.upper() == "TRANSFORMERS":
                from backends.transformers_backend import TransformersClient
                self.lightweight_model = TransformersClient(new_model, new_model_chat_format, None, new_model_context_window)
            elif new_model_backend.upper() == "VLLM":
                from backends.vllm_backend import VLLMClient
                self.lightweight_model = VLLMClient(new_model, new_model_chat_format, None, new_model_context_window)
            elif new_model_backend.upper() == "REST":
                from backends.rest_backend import RestClient
                rest_url = os.getenv("JARVIS_REST_LIGHTWEIGHT_MODEL_URL")
                if not rest_url:
                    raise ValueError("JARVIS_REST_LIGHTWEIGHT_MODEL_URL must be set when using REST backend")
                self.lightweight_model = RestClient(rest_url, new_model or "jarvis-llm", "lightweight")
            else:
                raise ValueError("Unsupported MODEL_BACKEND. Use 'MLX', 'GGUF', 'TRANSFORMERS', or 'REST'.")
            
            return {"status": "success", "message": f"Lightweight model swapped to {new_model}"}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _populate_registry(self):
        """Populate the model registry and aliases after models are initialized"""
        # Add main model to registry
        if self.main_model:
            main_model_id = os.getenv("JARVIS_MODEL_NAME", "jarvis-text-8b")
            main_backend = os.getenv("JARVIS_MODEL_BACKEND", "OLLAMA").upper()
            main_context = int(os.getenv("JARVIS_MODEL_CONTEXT_WINDOW", "4096"))
            self.registry[main_model_id] = ModelConfig(
                model_id=main_model_id,
                backend_type=main_backend,
                backend_instance=self.main_model,
                supports_images=False,
                context_length=main_context,
            )
            # Alias: "full" -> main model
            self.aliases["full"] = main_model_id
            print(f"📋 Registered main model: {main_model_id} (alias: 'full')")
        
        # Add lightweight model to registry
        if self.lightweight_model and self.lightweight_model != self.main_model:
            lightweight_model_id = os.getenv("JARVIS_LIGHTWEIGHT_MODEL_NAME", "jarvis-text-1b")
            lightweight_backend = os.getenv("JARVIS_LIGHTWEIGHT_MODEL_BACKEND", "").upper()
            lightweight_context = int(os.getenv("JARVIS_LIGHTWEIGHT_MODEL_CONTEXT_WINDOW", "4096"))
            self.registry[lightweight_model_id] = ModelConfig(
                model_id=lightweight_model_id,
                backend_type=lightweight_backend,
                backend_instance=self.lightweight_model,
                supports_images=False,
                context_length=lightweight_context,
            )
            # Alias: "lightweight" -> lightweight model
            self.aliases["lightweight"] = lightweight_model_id
            print(f"📋 Registered lightweight model: {lightweight_model_id} (alias: 'lightweight')")
        
        # Add cloud model to registry
        if self.cloud_model:
            cloud_model_id = self.cloud_model_name or "jarvis-cloud"
            cloud_backend = os.getenv("JARVIS_CLOUD_MODEL_BACKEND", "REST").upper()
            cloud_context = int(os.getenv("JARVIS_CLOUD_MODEL_CONTEXT_WINDOW", "4096"))
            self.registry[cloud_model_id] = ModelConfig(
                model_id=cloud_model_id,
                backend_type=cloud_backend,
                backend_instance=self.cloud_model,
                supports_images=False,
                context_length=cloud_context,
            )
            # Alias: "cloud" -> cloud model
            self.aliases["cloud"] = cloud_model_id
            print(f"📋 Registered cloud model: {cloud_model_id} (alias: 'cloud')")
        
        # Add vision model to registry
        if self.vision_model:
            vision_model_id = self.vision_model_name or "jarvis-vision-11b"
            vision_backend = os.getenv("JARVIS_VISION_MODEL_BACKEND", "").upper()
            vision_context = _get_int_env("JARVIS_VISION_MODEL_CONTEXT_WINDOW", 131072)
            self.registry[vision_model_id] = ModelConfig(
                model_id=vision_model_id,
                backend_type=vision_backend,
                backend_instance=self.vision_model,
                supports_images=True,
                context_length=vision_context,
            )
            # Alias: "vision" -> vision model
            self.aliases["vision"] = vision_model_id
            print(f"📋 Registered vision model: {vision_model_id} (alias: 'vision', supports_images=True)")
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """
        Get model configuration by name (supports both direct model IDs and aliases).
        
        Args:
            model_name: Either a direct model ID or an alias (full, lightweight, vision, cloud)
        
        Returns:
            ModelConfig if found, None otherwise
        """
        # First check if it's an alias
        if model_name.lower() in self.aliases:
            resolved_id = self.aliases[model_name.lower()]
            return self.registry.get(resolved_id)
        
        # Otherwise treat it as a direct model ID
        return self.registry.get(model_name)
    
    def list_models(self) -> list:
        """List all available models in the registry"""
        models = []
        for model_id, config in self.registry.items():
            models.append({
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "jarvis",
                "supports_images": config.supports_images,
                "context_length": config.context_length,
            })
        return models 