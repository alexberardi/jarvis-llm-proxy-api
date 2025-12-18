import multiprocessing
# Fix for vLLM CUDA multiprocessing issue - set spawn method early
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Any
import os
import time
import uuid
import asyncio
import base64
import re
from dotenv import load_dotenv
from managers.model_manager import ModelManager
from managers.chat_types import (
    NormalizedMessage,
    TextPart,
    ImagePart,
    GenerationParams,
    ChatResult,
)

load_dotenv()

app = FastAPI()

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

# Initialize model manager
model_manager = ModelManager()


# ============================================================================
# OpenAI-compatible request/response models
# ============================================================================

class ImageUrl(BaseModel):
    url: str
    detail: Optional[str] = None


class ContentPart(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None


class Message(BaseModel):
    role: str
    content: Union[str, List[ContentPart]]


class ChatCompletionRequest(BaseModel):
    model: str = "full"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class ErrorDetail(BaseModel):
    type: str
    message: str
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "jarvis"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# ============================================================================
# Helper functions for message normalization
# ============================================================================

def parse_data_url(data_url: str) -> tuple[bytes, str]:
    """
    Parse a data URL and return (image_bytes, mime_type).
    
    Expected format: data:image/png;base64,iVBORw0KG...
    """
    match = re.match(r'^data:([^;]+);base64,(.+)$', data_url)
    if not match:
        raise ValueError("Invalid data URL format. Expected: data:<mime_type>;base64,<data>")
    
    mime_type = match.group(1)
    b64_data = match.group(2)
    
    try:
        image_bytes = base64.b64decode(b64_data)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image data: {e}")
    
    return image_bytes, mime_type


def normalize_message(message: Message) -> NormalizedMessage:
    """
    Convert an OpenAI-style message to a NormalizedMessage.
    
    Handles both:
    - content as string: {"role": "user", "content": "Hello"}
    - content as array: {"role": "user", "content": [{"type": "text", "text": "..."}, {"type": "image_url", ...}]}
    """
    content_parts: List[Union[TextPart, ImagePart]] = []
    
    if isinstance(message.content, str):
        # Simple string content
        content_parts.append(TextPart(text=message.content))
    elif isinstance(message.content, list):
        # Structured content array
        for part in message.content:
            if part.type == "text" and part.text is not None:
                content_parts.append(TextPart(text=part.text))
            elif part.type == "image_url" and part.image_url is not None:
                url = part.image_url.url
                # For now, only support data URLs
                if not url.startswith("data:"):
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": {
                                "type": "invalid_request_error",
                                "message": "Only data URLs are supported for images. HTTP(S) URLs are not yet supported.",
                                "code": None,
                            }
                        }
                    )
                image_bytes, mime_type = parse_data_url(url)
                content_parts.append(
                    ImagePart(
                        data=image_bytes,
                        mime_type=mime_type,
                        detail=part.image_url.detail,
                    )
                )
            else:
                # Unknown part type or missing required field
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "type": "invalid_request_error",
                            "message": f"Unsupported content part type or missing field: {part.type}",
                            "code": None,
                        }
                    }
                )
    else:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "type": "invalid_request_error",
                    "message": "Message content must be either a string or an array of content parts",
                    "code": None,
                }
            }
        )
    
    return NormalizedMessage(role=message.role, content=content_parts)


def normalize_messages(messages: List[Message]) -> List[NormalizedMessage]:
    """Convert a list of OpenAI-style messages to NormalizedMessage format"""
    return [normalize_message(msg) for msg in messages]


def has_images(messages: List[NormalizedMessage]) -> bool:
    """Check if any message contains an ImagePart"""
    for msg in messages:
        for part in msg.content:
            if isinstance(part, ImagePart):
                return True
    return False


def create_openai_response(
    content: str,
    model_name: str,
    usage: Optional[Dict] = None,
) -> ChatCompletionResponse:
    """Create an OpenAI-style chat completion response"""
    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    
    if usage is None:
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    return ChatCompletionResponse(
        id=response_id,
        created=created,
        model=model_name,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=Message(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=Usage(**usage),
    )


def openai_error(error_type: str, message: str, status_code: int = 400):
    """Create an OpenAI-style error response"""
    raise HTTPException(
        status_code=status_code,
        detail={
            "error": {
                "type": error_type,
                "message": message,
                "code": None,
            }
        }
    )


# ============================================================================
# Routes
# ============================================================================

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    
    Supports:
    - Text-only messages (string content)
    - Multimodal messages (structured content with text + images)
    - Model selection via model field (supports aliases: full, lightweight, vision, cloud)
    """
    print(f"🔍 Received chat completion request for model: {req.model}")
    print(f"🔍 Messages: {len(req.messages)}, Temperature: {req.temperature}")
    
    # 1. Resolve model
    model_config = model_manager.get_model_config(req.model)
    if not model_config:
        openai_error(
            "model_not_found",
            f"Model '{req.model}' does not exist. Available models: {list(model_manager.registry.keys())} and aliases: {list(model_manager.aliases.keys())}",
            404,
        )
    
    # 2. Normalize messages
    try:
        normalized_messages = normalize_messages(req.messages)
    except HTTPException:
        raise
    except Exception as e:
        openai_error("invalid_request_error", f"Failed to parse messages: {str(e)}")
    
    # 3. Detect images
    has_images_flag = has_images(normalized_messages)
    if has_images_flag:
        print(f"🖼️  Detected images in request")
    
    # 4. Check model capabilities
    if has_images_flag and not model_config.supports_images:
        openai_error(
            "invalid_request_error",
            f"Model '{req.model}' does not support images. Use a vision-capable model instead (try 'vision' alias or a model with supports_images=true).",
        )
    
    # 5. Get backend instance
    backend = model_config.backend_instance
    
    # 6. Prepare generation parameters
    params = GenerationParams(
        temperature=req.temperature or 0.7,
        max_tokens=req.max_tokens,
        stream=req.stream or False,
    )
    
    # 7. Dispatch to appropriate backend method
    try:
        start_time = time.time()
        
        if has_images_flag:
            # Vision path
            print(f"🎨 Routing to vision backend: {model_config.backend_type}")
            if not hasattr(backend, "generate_vision_chat"):
                openai_error(
                    "internal_server_error",
                    f"Backend '{model_config.backend_type}' does not support vision (missing generate_vision_chat method).",
                    500,
                )
            
            if asyncio.iscoroutinefunction(backend.generate_vision_chat):
                result: ChatResult = await backend.generate_vision_chat(
                    model_config, normalized_messages, params
                )
            else:
                result: ChatResult = backend.generate_vision_chat(
                    model_config, normalized_messages, params
                )
        else:
            # Text-only path
            print(f"📝 Routing to text backend: {model_config.backend_type}")
            
            # Check if backend has modern generate_text_chat method
            if hasattr(backend, "generate_text_chat"):
                # Use modern interface with NormalizedMessage
                if asyncio.iscoroutinefunction(backend.generate_text_chat):
                    result: ChatResult = await backend.generate_text_chat(
                        model_config, normalized_messages, params
                    )
                else:
                    result: ChatResult = backend.generate_text_chat(
                        model_config, normalized_messages, params
                    )
            else:
                # Convert NormalizedMessage back to simple dict format for legacy backends
                legacy_messages = []
                for msg in normalized_messages:
                    # Concatenate all text parts
                    text_content = " ".join(
                        part.text for part in msg.content if isinstance(part, TextPart)
                    )
                    legacy_messages.append({"role": msg.role, "content": text_content})
                
                # Use the existing chat methods on backends
                if hasattr(backend, 'chat_with_temperature'):
                    if asyncio.iscoroutinefunction(backend.chat_with_temperature):
                        output = await backend.chat_with_temperature(legacy_messages, params.temperature)
                    else:
                        output = backend.chat_with_temperature(legacy_messages, params.temperature)
                else:
                    if asyncio.iscoroutinefunction(backend.chat):
                        output = await backend.chat(legacy_messages)
                    else:
                        output = backend.chat(legacy_messages)
                
                # Create ChatResult from string output
                usage = None
                if hasattr(backend, 'last_usage'):
                    usage = backend.last_usage
                result = ChatResult(content=output, usage=usage)
        
        end_time = time.time()
        print(f"⏱️  Request completed in {end_time - start_time:.2f}s")
        
        # 8. Convert to OpenAI response format
        response = create_openai_response(
            content=result.content,
            model_name=model_config.model_id,
            usage=result.usage,
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error during chat completion: {e}")
        import traceback
        traceback.print_exc()
        openai_error("internal_server_error", f"Internal error: {str(e)}", 500)


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    """List all available models in OpenAI-compatible format"""
    models = model_manager.list_models()
    return ModelListResponse(object="list", data=models)


@app.get("/v1/health")
async def health():
    """Health check endpoint"""
    # Get backend info
    model_backend = os.getenv("JARVIS_MODEL_BACKEND", "OLLAMA").upper()
    lightweight_model_backend = os.getenv("JARVIS_LIGHTWEIGHT_MODEL_BACKEND", "").upper()
    vision_model_backend = os.getenv("JARVIS_VISION_MODEL_BACKEND", "").upper()
    cloud_model_backend = os.getenv("JARVIS_CLOUD_MODEL_BACKEND", "REST").upper()
    
    # Count models
    model_count = len(model_manager.registry)
    alias_count = len(model_manager.aliases)
    
    return {
        "status": "healthy",
        "models": {
            "total": model_count,
            "aliases": alias_count,
            "registry": list(model_manager.registry.keys()),
            "aliases_map": model_manager.aliases,
        },
        "backends": {
            "main": model_backend,
            "lightweight": lightweight_model_backend or "shared",
            "vision": vision_model_backend or "not configured",
            "cloud": cloud_model_backend if model_manager.cloud_model else "not configured",
        },
    }


# Legacy routes removed per PRD:
# - /api/v{version:int}/chat
# - /api/v{version:int}/lightweight/chat
# - /api/v{version:int}/chat/conversation/{conversation_id}/warmup
# - /api/v{version:int}/lightweight/chat/conversation/{conversation_id}/warmup
# - /api/v{version:int}/model-swap
# - /api/v{version:int}/lightweight/model-swap
# - /api/v{version:int}/conversation/{conversation_id}/status
# - /api/v{version:int}/model/reset
# - /api/v{version:int}/lightweight/model/reset
# - /api/v{version:int}/debug-request
#
# All chat interactions now use /v1/chat/completions with model selection.
