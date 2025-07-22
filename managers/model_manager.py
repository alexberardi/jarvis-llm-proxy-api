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
        lightweight_chat_format = os.getenv("JARVIS_LIGHTWEIGHT_MODEL_CHAT_FORMAT")
        main_context_window = int(os.getenv("JARVIS_MODEL_CONTEXT_WINDOW", "512"))
        lightweight_context_window = int(os.getenv("JARVIS_LIGHTWEIGHT_MODEL_CONTEXT_WINDOW", "512"))
        
        # Initialize main model
        if model_backend == "MLX":
            from backends.mlx_backend import MlxClient
            self.main_model = MlxClient(main_model_path)
        elif model_backend == "GGUF":
            from backends.gguf_backend import GGUFClient 
            self.main_model = GGUFClient(main_model_path, main_chat_format, main_context_window)
        # elif model_backend == "OLLAMA":
            # from backends.ollama_backend import OllamaClient
            # self.main_model = OllamaClient()
        else:
            raise ValueError("Unsupported MODEL_BACKEND. Use 'MLX', 'GGUF', or 'OLLAMA'.")

        # Initialize lightweight model
        if lightweight_model_backend == "MLX":
            from backends.mlx_backend import MlxClient
            self.lightweight_model = MlxClient(lightweight_model_path)
        elif lightweight_model_backend == "GGUF":
            from backends.gguf_backend import GGUFClient 
            self.lightweight_model = GGUFClient(lightweight_model_path, lightweight_chat_format, lightweight_context_window)
        # elif lightweight_model_backend == "OLLAMA":
            # from backends.ollama_backend import OllamaClient
            # self.lightweight_model = OllamaClient()
        else:
            raise ValueError("Unsupported LIGHTWEIGHT_MODEL_BACKEND. Use 'MLX', 'GGUF', or 'OLLAMA'.")
    
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
            else:
                raise ValueError("Unsupported MODEL_BACKEND. Use 'MLX' or 'GGUF'.")
            
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
            else:
                raise ValueError("Unsupported MODEL_BACKEND. Use 'MLX' or 'GGUF'.")
            
            return {"status": "success", "message": f"Lightweight model swapped to {new_model}"}
            
        except Exception as e:
            return {"status": "error", "message": str(e)} 