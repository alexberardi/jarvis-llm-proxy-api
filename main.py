from fastapi import FastAPI, Request, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import time
import uuid
import asyncio
from dotenv import load_dotenv
from managers.model_manager import ModelManager
from cache.cache_manager import CacheManager

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

# Initialize conversation cache manager
cache_manager = CacheManager()

# OpenAI-style request models
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "jarvis-llm"
    temperature: Optional[float] = 0.7
    messages: List[Message]
    conversation_id: Optional[str] = None  # Optional conversation ID for conversation continuity

# OpenAI-style response models
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

class ModelSwapRequest(BaseModel):
    new_model: str
    new_model_backend: str
    new_model_chat_format: str
    new_model_context_window: int = None  # Optional, will use env var if not provided

def get_model_name(model_instance) -> str:
    """Extract model name from model instance"""
    if hasattr(model_instance, 'model_path'):
        return model_instance.model_path
    elif hasattr(model_instance, 'model_name'):
        return model_instance.model_name
    else:
        return "jarvis-llm"

def create_openai_response(content: str, model_name: str, usage: Dict = None) -> ChatCompletionResponse:
    """Create OpenAI-style response"""
    # Generate random ID
    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    
    # Get current timestamp
    created = int(time.time())
    
    # Default usage if not provided
    if usage is None:
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    # Create response
    return ChatCompletionResponse(
        id=response_id,
        created=created,
        model=model_name,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=Message(role="assistant", content=content),
                finish_reason="stop"
            )
        ],
        usage=Usage(**usage)
    )

@app.post("/api/v{version:int}/chat")
async def chat(version: int, req: ChatCompletionRequest):
    # Debug logging
    print(f"🔍 Debug: Received chat request with {len(req.messages)} messages")
    if req.conversation_id:
        print(f"🔍 Debug: Conversation ID: {req.conversation_id}")
    for i, msg in enumerate(req.messages):
        print(f"🔍 Debug: Message {i}: role='{msg.role}', content_length={len(msg.content)}")
        if len(msg.content) < 200:
            print(f"🔍 Debug: Message {i} content: {msg.content}")
        else:
            print(f"🔍 Debug: Message {i} content preview: {msg.content[:200]}...")
    
    # Convert Pydantic messages to dict format for backend
    messages = [{"role": msg.role, "content": msg.content} for msg in req.messages]
    
    # Cache implementation removed - client handles conversation history
    
    # Debug: Show final messages being sent to LLM
    print(f"📤 Sending {len(messages)} messages to LLM:")
    for i, msg in enumerate(messages):
        role = msg["role"]
        content_preview = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
        print(f"   {i+1}. [{role}] {content_preview}")
    

    
    # Get model response with temperature
    if hasattr(model_manager.main_model, 'chat_with_temperature'):
        if asyncio.iscoroutinefunction(model_manager.main_model.chat_with_temperature):
            output = await model_manager.main_model.chat_with_temperature(messages, req.temperature)
        else:
            output = model_manager.main_model.chat_with_temperature(messages, req.temperature)
    else:
        if asyncio.iscoroutinefunction(model_manager.main_model.chat):
            output = await model_manager.main_model.chat(messages)
        else:
            output = model_manager.main_model.chat(messages)
    
    # Cache update removed - client handles conversation history
    
    # Get model name
    model_name = get_model_name(model_manager.main_model)
    
    # Get usage information if available
    usage = None
    if hasattr(model_manager.main_model, 'last_usage'):
        usage = model_manager.main_model.last_usage
    
    # Create OpenAI-style response
    response = create_openai_response(output, model_name, usage)
    
    # Debug logging for response
    print(f"🔍 Debug: Model response length: {len(output)}")
    print(f"🔍 Debug: Model response preview: {output[:200]}...")
    
    return response

@app.post("/api/v{version:int}/chat/conversation/{conversation_id}/warmup")
async def warmup_session(version: int, conversation_id: str, req: ChatCompletionRequest, background_tasks: BackgroundTasks):
    """Warm up a conversation by processing context asynchronously"""
    print(f"🔥 Starting warm-up for conversation {conversation_id}")
    
    # Convert Pydantic messages to dict format
    messages = [{"role": msg.role, "content": msg.content} for msg in req.messages]
    
    # Add warm-up task to background
    background_tasks.add_task(process_warmup_context, conversation_id, messages)
    
    # Return immediately - don't wait for LLM processing
    return {"status": "warmup_started", "conversation_id": conversation_id, "message": "Context processing started in background"}

async def process_warmup_context(conversation_id: str, messages: List[Dict]):
    """Background task to process warm-up context"""
    try:
        print(f"🔥 Processing warm-up context for conversation {conversation_id}")
        
        # Get the cache
        cache = cache_manager.get_cache()
        
        # Store the messages in cache (this creates the session if it doesn't exist)
        cache.create_or_update_session(conversation_id, messages)
        
        # Set warm-up status to "in_progress"
        cache.set_warmup_status(conversation_id, "in_progress")
        print(f"🔄 Set warm-up status to 'in_progress' for conversation {conversation_id}")
        
        # Process the context with the LLM to create "processed context"
        # We'll use a minimal inference to process the context without generating a response
        if hasattr(model_manager.main_model, 'process_context'):
            # If the backend supports context processing
            if asyncio.iscoroutinefunction(model_manager.main_model.process_context):
                processed_context = await model_manager.main_model.process_context(messages)
            else:
                processed_context = model_manager.main_model.process_context(messages)
            # Store the processed context
            cache.update_processed_context(conversation_id, processed_context)
            
            # Set warm-up status to "completed"
            cache.set_warmup_status(conversation_id, "completed")
            print(f"✅ Context processed and stored for conversation {conversation_id}")
            print(f"✅ Set warm-up status to 'completed' for conversation {conversation_id}")
        else:
            # Fallback: just store the raw messages
            print(f"⚠️  Backend doesn't support context processing, storing raw messages for conversation {conversation_id}")
            
            # Set warm-up status to "completed" even for fallback
            cache.set_warmup_status(conversation_id, "completed")
            print(f"✅ Set warm-up status to 'completed' for conversation {conversation_id} (fallback mode)")
            
    except Exception as e:
        print(f"❌ Error processing warm-up context for conversation {conversation_id}: {e}")
        # Set warm-up status to "pending" on error so it can be retried
        try:
            cache.set_warmup_status(conversation_id, "pending")
            print(f"🔄 Reset warm-up status to 'pending' for conversation {conversation_id} due to error")
        except Exception as status_error:
            print(f"⚠️  Failed to reset warm-up status: {status_error}")

@app.post("/api/v{version:int}/lightweight/chat")
async def lightweight_chat(version: int, req: ChatCompletionRequest):
    # Debug logging
    print(f"🔍 Debug: Received lightweight chat request with {len(req.messages)} messages")
    if req.conversation_id:
        print(f"🔍 Debug: Conversation ID: {req.conversation_id}")
    
    # Convert Pydantic messages to dict format for backend
    messages = [{"role": msg.role, "content": msg.content} for msg in req.messages]
    
    # Cache implementation removed - client handles conversation history
    
    # Debug: Show final messages being sent to lightweight LLM
    print(f"📤 Sending {len(messages)} messages to lightweight LLM:")
    for i, msg in enumerate(messages):
        role = msg["role"]
        content_preview = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
        print(f"   {i+1}. [{role}] {content_preview}")
    

    
    # Get model response with temperature
    if hasattr(model_manager.lightweight_model, 'chat_with_temperature'):
        if asyncio.iscoroutinefunction(model_manager.lightweight_model.chat_with_temperature):
            output = await model_manager.lightweight_model.chat_with_temperature(messages, req.temperature)
        else:
            output = model_manager.lightweight_model.chat_with_temperature(messages, req.temperature)
    else:
        if asyncio.iscoroutinefunction(model_manager.lightweight_model.chat):
            output = await model_manager.lightweight_model.chat(messages)
        else:
            output = model_manager.lightweight_model.chat(messages)
    
    # Cache update removed - client handles conversation history
    
    # Get model name
    model_name = get_model_name(model_manager.lightweight_model)
    
    # Get usage information if available
    usage = None
    if hasattr(model_manager.lightweight_model, 'last_usage'):
        usage = model_manager.lightweight_model.last_usage
    
    # Create OpenAI-style response
    response = create_openai_response(output, model_name, usage)
    
    return response

@app.post("/api/v{version:int}/lightweight/chat/conversation/{conversation_id}/warmup")
async def lightweight_warmup_session(version: int, conversation_id: str, req: ChatCompletionRequest, background_tasks: BackgroundTasks):
    """Warm up a lightweight conversation by processing context asynchronously"""
    print(f"🔥 Starting lightweight warm-up for conversation {conversation_id}")
    
    # Convert Pydantic messages to dict format
    messages = [{"role": msg.role, "content": msg.content} for msg in req.messages]
    
    # Add warm-up task to background
    background_tasks.add_task(process_lightweight_warmup_context, conversation_id, messages)
    
    # Return immediately - don't wait for LLM processing
    return {"status": "warmup_started", "conversation_id": conversation_id, "message": "Context processing started in background"}

async def process_lightweight_warmup_context(conversation_id: str, messages: List[Dict]):
    """Background task to process lightweight warm-up context"""
    try:
        print(f"🔥 Processing lightweight warm-up context for conversation {conversation_id}")
        
        # Get the cache
        cache = cache_manager.get_cache()
        
        # Store the messages in cache (this creates the session if it doesn't exist)
        cache.create_or_update_session(conversation_id, messages)
        
        # Set warm-up status to "in_progress"
        cache.set_warmup_status(conversation_id, "in_progress")
        print(f"🔄 Set lightweight warm-up status to 'in_progress' for conversation {conversation_id}")
        
        # Process the context with the lightweight LLM to create "processed context"
        if hasattr(model_manager.lightweight_model, 'process_context'):
            # If the backend supports context processing
            if asyncio.iscoroutinefunction(model_manager.lightweight_model.process_context):
                processed_context = await model_manager.lightweight_model.process_context(messages)
            else:
                processed_context = model_manager.lightweight_model.process_context(messages)
            # Store the processed context
            cache.update_processed_context(conversation_id, processed_context)
            
            # Set warm-up status to "completed"
            cache.set_warmup_status(conversation_id, "completed")
            print(f"✅ Lightweight context processed and stored for conversation {conversation_id}")
            print(f"✅ Set lightweight warm-up status to 'completed' for conversation {conversation_id}")
        else:
            # Fallback: just store the raw messages
            print(f"⚠️  Lightweight backend doesn't support context processing, storing raw messages for conversation {conversation_id}")
            
            # Set warm-up status to "completed" even for fallback
            cache.set_warmup_status(conversation_id, "completed")
            print(f"✅ Set lightweight warm-up status to 'completed' for conversation {conversation_id} (fallback mode)")
            
    except Exception as e:
        print(f"❌ Error processing lightweight warm-up context for conversation {conversation_id}: {e}")
        # Set warm-up status to "pending" on error so it can be retried
        try:
            cache.set_warmup_status(conversation_id, "pending")
            print(f"🔄 Reset lightweight warm-up status to 'pending' for conversation {conversation_id} due to error")
        except Exception as status_error:
            print(f"⚠️  Failed to reset lightweight warm-up status: {status_error}")

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

@app.get("/api/v{version:int}/conversation/{conversation_id}/status")
async def get_conversation_status(version: int, conversation_id: str):
    """Get the status of a conversation"""
    try:
        cache = cache_manager.get_cache()
        
        # Check if conversation exists
        existing_session = cache.get_session(conversation_id)
        
        if not existing_session:
            return {"error": "Conversation not found"}, 404
        
        # Get warm-up status
        warmup_status = cache.get_warmup_status(conversation_id)
        
        # Prepare response based on status
        if warmup_status == "pending":
            # Return 202 Accepted for "check back later" scenario
            return {
                "conversation_id": conversation_id,
                "status": "pending",
                "message": "Conversation is pending warm-up. Check back later.",
                "created_at": existing_session.get("created_at"),
                "last_accessed": existing_session.get("last_accessed"),
                "message_count": len(existing_session.get("messages", [])),
                "ttl_seconds": int(os.getenv("JARVIS_SESSION_TTL", "600"))
            }, 202
        elif warmup_status == "in_progress":
            # Return 202 Accepted for "check back later" scenario
            return {
                "conversation_id": conversation_id,
                "status": "in_progress",
                "message": "Conversation warm-up in progress. Check back later.",
                "created_at": existing_session.get("created_at"),
                "last_accessed": existing_session.get("last_accessed"),
                "message_count": len(existing_session.get("messages", [])),
                "ttl_seconds": int(os.getenv("JARVIS_SESSION_TTL", "600"))
            }, 202
        elif warmup_status == "completed":
            # Return 200 OK for completed conversations
            return {
                "conversation_id": conversation_id,
                "status": "completed",
                "message": "Conversation warm-up completed and ready for use.",
                "created_at": existing_session.get("created_at"),
                "last_accessed": existing_session.get("last_accessed"),
                "message_count": len(existing_session.get("messages", [])),
                "ttl_seconds": int(os.getenv("JARVIS_SESSION_TTL", "600")),
                "processed_context_available": cache.get_processed_context(conversation_id) is not None
            }, 200
        else:
            # Return 200 OK for unknown status
            return {
                "conversation_id": conversation_id,
                "status": warmup_status or "unknown",
                "message": "Conversation status unknown.",
                "created_at": existing_session.get("created_at"),
                "last_accessed": existing_session.get("last_accessed"),
                "message_count": len(existing_session.get("messages", [])),
                "ttl_seconds": int(os.getenv("JARVIS_SESSION_TTL", "600"))
            }, 200
            
    except Exception as e:
        return {"error": f"Failed to get conversation status: {str(e)}"}, 500

@app.get("/api/v{version:int}/health")
async def health(version: int):
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_backend": model_backend,
        "lightweight_model_backend": lightweight_model_backend,
        "main_model": main_model_path,
        "lightweight_model": lightweight_model_path,
        "cache": {
            "type": os.getenv("JARVIS_CACHE_TYPE", "local"),
            "session_count": cache_manager.get_cache().get_session_count(),
            "ttl_seconds": int(os.getenv("JARVIS_SESSION_TTL", "600")),
            "cleanup_interval": int(os.getenv("JARVIS_CACHE_CLEANUP_INTERVAL", "30"))
        }
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