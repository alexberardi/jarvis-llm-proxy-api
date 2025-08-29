import os

class ModelManager:
    def __init__(self):
        self.main_model = None
        self.lightweight_model = None
        self._initialize_models()
    
    def _initialize_models(self):
        # Get environment variables
        model_backend = os.getenv("JARVIS_MODEL_BACKEND", "OLLAMA").upper()
        lightweight_model_backend = os.getenv("JARVIS_LIGHTWEIGHT_MODEL_BACKEND", "OLLAMA").upper()
        
        main_model_path = os.getenv("JARVIS_MODEL_NAME")
        lightweight_model_path = os.getenv("JARVIS_LIGHTWEIGHT_MODEL_NAME")
        main_chat_format = os.getenv("JARVIS_MODEL_CHAT_FORMAT")
        main_stop_tokens = os.getenv("JARVIS_MODEL_STOP_TOKENS")
        lightweight_chat_format = os.getenv("JARVIS_LIGHTWEIGHT_MODEL_CHAT_FORMAT")
        lightweight_stop_tokens = os.getenv("JARVIS_LIGHTWEIGHT_MODEL_STOP_TOKENS")
        main_context_window = int(os.getenv("JARVIS_MODEL_CONTEXT_WINDOW", "512"))
        lightweight_context_window = int(os.getenv("JARVIS_LIGHTWEIGHT_MODEL_CONTEXT_WINDOW", "512"))
        
        # Initialize main model
        if model_backend == "MLX":
            from backends.mlx_backend import MlxClient
            self.main_model = MlxClient(main_model_path, main_stop_tokens)
        elif model_backend == "GGUF":
            from backends.gguf_backend import GGUFClient 
            self.main_model = GGUFClient(main_model_path, main_chat_format, main_stop_tokens, main_context_window)
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
            raise ValueError("Unsupported MODEL_BACKEND. Use 'MLX', 'GGUF', 'REST', or 'OLLAMA'.")

        # Initialize lightweight model
        if lightweight_model_backend == "MLX":
            from backends.mlx_backend import MlxClient
            self.lightweight_model = MlxClient(lightweight_model_path, lightweight_stop_tokens)
        elif lightweight_model_backend == "GGUF":
            from backends.gguf_backend import GGUFClient 
            self.lightweight_model = GGUFClient(lightweight_model_path, lightweight_chat_format, lightweight_stop_tokens, lightweight_context_window)
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
            raise ValueError("Unsupported LIGHTWEIGHT_MODEL_BACKEND. Use 'MLX', 'GGUF', 'REST', or 'OLLAMA'.")
    
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
                self.main_model = GGUFClient(new_model, new_model_chat_format, new_model_context_window)
            elif new_model_backend.upper() == "REST":
                from backends.rest_backend import RestClient
                rest_url = os.getenv("JARVIS_REST_MODEL_URL")
                if not rest_url:
                    raise ValueError("JARVIS_REST_MODEL_URL must be set when using REST backend")
                self.main_model = RestClient(rest_url, new_model or "jarvis-llm", "main")
            else:
                raise ValueError("Unsupported MODEL_BACKEND. Use 'MLX', 'GGUF', or 'REST'.")
            
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
                self.lightweight_model = GGUFClient(new_model, new_model_chat_format, new_model_context_window)
            elif new_model_backend.upper() == "REST":
                from backends.rest_backend import RestClient
                rest_url = os.getenv("JARVIS_REST_LIGHTWEIGHT_MODEL_URL")
                if not rest_url:
                    raise ValueError("JARVIS_REST_LIGHTWEIGHT_MODEL_URL must be set when using REST backend")
                self.lightweight_model = RestClient(rest_url, new_model or "jarvis-llm", "lightweight")
            else:
                raise ValueError("Unsupported MODEL_BACKEND. Use 'MLX', 'GGUF', or 'REST'.")
            
            return {"status": "success", "message": f"Lightweight model swapped to {new_model}"}
            
        except Exception as e:
            return {"status": "error", "message": str(e)} 