from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Dict
import os
from dotenv import load_dotenv
from managers.model_manager import ModelManager

load_dotenv()

app = FastAPI()

model_backend = os.getenv("JARVIS_MODEL_BACKEND", "OLLAMA").upper()
lightweight_model_backend = os.getenv("JARVIS_LIGHTWEIGHT_MODEL_BACKEND", "OLLAMA").upper()

# Debug setup - only enable when DEBUG=true
debug_enabled = os.getenv("DEBUG", "false").lower() == "true"
debug_port = int(os.getenv("DEBUG_PORT", "5678"))
if debug_enabled:
    try:
        import debugpy
        debugpy.listen(("0.0.0.0", debug_port))
        print(f"🐛 Debugger listening on port {debug_port}")
    except ImportError:
        print("❌ debugpy is not installed, but DEBUG is set to true")

# Get model paths and chat formats
main_model_path = os.getenv("JARVIS_MODEL_NAME")
lightweight_model_path = os.getenv("JARVIS_LIGHTWEIGHT_MODEL_NAME")
main_chat_format = os.getenv("JARVIS_MODEL_CHAT_FORMAT")
lightweight_chat_format = os.getenv("JARVIS_LIGHTWEIGHT_MODEL_CHAT_FORMAT")
main_context_window = int(os.getenv("JARVIS_MODEL_CONTEXT_WINDOW", "512"))
lightweight_context_window = int(os.getenv("JARVIS_LIGHTWEIGHT_MODEL_CONTEXT_WINDOW", "512"))

print(main_chat_format)

# Initialize model manager
model_manager = ModelManager()

# Accepts messages instead of just prompt
class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]

class ModelSwapRequest(BaseModel):
    new_model: str
    new_model_backend: str
    new_model_chat_format: str
    new_model_context_window: int = None  # Optional, will use env var if not provided

@app.post("/api/v{version:int}/chat")
async def chat(version: int, req: ChatRequest):
    # Debug logging
    print(f"🔍 Debug: Received chat request with {len(req.messages)} messages")
    for i, msg in enumerate(req.messages):
        print(f"🔍 Debug: Message {i}: role='{msg['role']}', content_length={len(msg['content'])}")
        if len(msg['content']) < 200:
            print(f"🔍 Debug: Message {i} content: {msg['content']}")
        else:
            print(f"🔍 Debug: Message {i} content preview: {msg['content'][:200]}...")
    
    output = model_manager.main_model.chat(req.messages)
    
    # Debug logging for response
    print(f"🔍 Debug: Model response length: {len(output)}")
    print(f"🔍 Debug: Model response preview: {output[:200]}...")
    
    return {"response": output}

@app.post("/api/v{version:int}/lightweight/chat")
async def lightweight_chat(version: int, req: ChatRequest):
    output = model_manager.lightweight_model.chat(req.messages)
    return {"response": output}

@app.post("/api/v{version:int}/model-swap")
async def model_swap(version: int, req: ModelSwapRequest):
    return model_manager.swap_main_model(
        req.new_model, 
        req.new_model_backend, 
        req.new_model_chat_format, 
        req.new_model_context_window
    )

@app.post("/api/v{version:int}/lightweight/model-swap")
async def lightweight_model_swap(version: int, req: ModelSwapRequest):
    return model_manager.swap_lightweight_model(
        req.new_model, 
        req.new_model_backend, 
        req.new_model_chat_format, 
        req.new_model_context_window
    )

@app.get("/api/v{version:int}/health")
async def health(version: int):
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_backend": model_backend,
        "lightweight_model_backend": lightweight_model_backend,
        "main_model": main_model_path,
        "lightweight_model": lightweight_model_path
    }

@app.post("/api/v{version:int}/debug-request")
async def debug_request(version: int, request: Request):
    """Debug endpoint to inspect incoming requests"""
    try:
        # Get raw body
        body = await request.body()
        body_str = body.decode('utf-8')
        
        # Get headers
        headers = dict(request.headers)
        
        # Try to parse as JSON
        try:
            import json
            parsed_json = json.loads(body_str)
            json_valid = True
        except json.JSONDecodeError as e:
            json_valid = False
            json_error = str(e)
        
        return {
            "body_length": len(body),
            "body_preview": body_str[:500] + "..." if len(body_str) > 500 else body_str,
            "headers": {k: v for k, v in headers.items() if k.lower() in ['content-type', 'content-length', 'user-agent', 'accept']},
            "json_valid": json_valid,
            "json_error": json_error if not json_valid else None,
            "content_type": headers.get('content-type', 'not-set')
        }
    except Exception as e:
        return {"error": str(e)}