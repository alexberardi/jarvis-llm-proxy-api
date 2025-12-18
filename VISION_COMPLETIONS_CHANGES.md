# Vision Model and Completions Route Implementation

## Summary

This implementation follows the PRD in `prds/vision-model-and-completions-route.md` and provides:

1. ✅ **Unified `/v1/chat/completions` endpoint** - Single OpenAI-compatible route for all chat interactions
2. ✅ **Vision model support** - Support for multimodal (text + image) messages
3. ✅ **Model aliases** - Stable aliases (`full`, `lightweight`, `vision`, `cloud`) for client convenience
4. ✅ **Backend-agnostic design** - Works with GGUF, MLX, Transformers, vLLM, and REST backends
5. ✅ **Legacy route removal** - Cleaned up non-OpenAI-style routes per PRD

---

## What Changed

### 1. `/v1/chat/completions` Endpoint (main.py)

**New unified endpoint** that replaces all legacy `/api/v{version}/chat` variants:

```python
POST /v1/chat/completions
```

**Features:**
- Accepts OpenAI-style requests with `model`, `messages`, `temperature`, `max_tokens`
- Supports both string content and structured content arrays
- Handles text-only and multimodal (text + images) messages
- Returns OpenAI-compatible responses
- Automatic routing to vision or text backends based on content

**Example text request:**
```json
{
  "model": "full",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7
}
```

**Example vision request:**
```json
{
  "model": "vision",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image"},
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,iVBORw0KG..."
          }
        }
      ]
    }
  ]
}
```

### 2. Model Manager Enhancements (managers/model_manager.py)

**Added:**
- `ModelConfig` class - Configuration wrapper for each model
- Model registry (`registry` dict) - Maps model IDs to configurations
- Model aliases (`aliases` dict) - Maps stable names to model IDs
- `get_model_config(model_name)` - Resolve aliases or direct model IDs
- `list_models()` - List all registered models
- Vision model initialization via `JARVIS_VISION_MODEL_BACKEND` and `JARVIS_VISION_MODEL_NAME`

**Model aliases:**
- `full` → Main/primary text model (e.g., `jarvis-text-8b`)
- `lightweight` → Smaller text model (e.g., `jarvis-text-1b`)
- `vision` → Vision-capable model (e.g., `jarvis-vision-11b`)
- `cloud` → Cloud/remote model (if configured)

**Each model has:**
- `model_id` - Unique identifier
- `backend_type` - Backend implementation (MLX, GGUF, etc.)
- `backend_instance` - Actual backend object
- `supports_images` - Boolean flag for vision capability
- `context_length` - Maximum context window

### 3. Vision Backend Support (backends/rest_backend.py)

**Added `generate_vision_chat` method** to REST backend:
- Converts `NormalizedMessage` (with `ImagePart`) to OpenAI-style structured content
- Sends multimodal requests to remote APIs
- Returns `ChatResult` with usage information

This enables vision support for remote APIs (OpenAI, Anthropic, etc.) through the REST backend.

### 4. Message Normalization (main.py)

**Helper functions added:**
- `normalize_message()` - Convert OpenAI message to `NormalizedMessage`
- `normalize_messages()` - Batch conversion
- `has_images()` - Detect if any message contains images
- `parse_data_url()` - Extract image bytes from data URLs

**Normalized format** (defined in `managers/chat_types.py`):
```python
@dataclass
class NormalizedMessage:
    role: str
    content: List[Union[TextPart, ImagePart]]

@dataclass
class TextPart:
    text: str

@dataclass
class ImagePart:
    data: bytes
    mime_type: str
    detail: Optional[str]
```

### 5. New Routes

**Added:**
- `GET /v1/models` - List available models (OpenAI-compatible)
- `GET /v1/health` - Health check with model registry info

**Removed** (per PRD):
- `/api/v{version}/chat` (replaced by `/v1/chat/completions`)
- `/api/v{version}/lightweight/chat` (use `model: "lightweight"` instead)
- `/api/v{version}/chat/conversation/{id}/warmup` (not needed)
- `/api/v{version}/model-swap` (not needed)
- `/api/v{version}/model/reset` (not needed)
- `/api/v{version}/conversation/{id}/status` (not needed)

### 6. Error Handling

**OpenAI-style errors:**
```json
{
  "error": {
    "type": "invalid_request_error",
    "message": "Model 'full' does not support images. Use a vision-capable model instead.",
    "code": null
  }
}
```

**Error scenarios:**
- Model not found (404)
- Images sent to text-only model (400)
- Invalid content format (400)
- Backend errors (500)

---

## Environment Variables

### New Variables

**Vision Model:**
```bash
JARVIS_VISION_MODEL_BACKEND=MLX          # or TRANSFORMERS, REST
JARVIS_VISION_MODEL_NAME=jarvis-vision-11b
JARVIS_VISION_MODEL_CONTEXT_WINDOW=131072
```

**For REST vision backend:**
```bash
JARVIS_VISION_REST_MODEL_URL=https://api.openai.com
```

### Existing Variables (unchanged)

```bash
JARVIS_MODEL_BACKEND=GGUF
JARVIS_MODEL_NAME=jarvis-text-8b
JARVIS_LIGHTWEIGHT_MODEL_BACKEND=GGUF
JARVIS_LIGHTWEIGHT_MODEL_NAME=jarvis-text-1b
JARVIS_CLOUD_MODEL_BACKEND=REST
JARVIS_CLOUD_MODEL_NAME=gpt-4
```

---

## Backend Compatibility

### Text Backends
All existing backends continue to work with legacy `chat` and `chat_with_temperature` methods.

### Vision Backends
Must implement `generate_vision_chat(model_cfg, messages, params) -> ChatResult`:
- ✅ `MlxVisionClient` (backends/mlx_vision_backend.py)
- ✅ `TransformersVisionClient` (backends/transformers_vision_backend.py)
- ✅ `RestClient` (backends/rest_backend.py) - NEW
- ✅ `MockBackend` (backends/mock_backend.py) - for testing

### Modern Backend Interface
Backends can optionally implement:
- `generate_text_chat(model_cfg, messages, params) -> ChatResult` - for text-only with new interface
- `generate_vision_chat(model_cfg, messages, params) -> ChatResult` - for multimodal

The endpoint automatically routes to the appropriate method based on:
1. Whether images are present
2. Whether model supports images
3. Which methods backend implements

---

## Migration Guide

### For API Clients

**Old way:**
```python
# Text chat
POST /api/v1/chat
{"messages": [...], "temperature": 0.7}

# Lightweight chat
POST /api/v1/lightweight/chat
{"messages": [...]}
```

**New way:**
```python
# Text chat with main model
POST /v1/chat/completions
{"model": "full", "messages": [...], "temperature": 0.7}

# Text chat with lightweight model
POST /v1/chat/completions
{"model": "lightweight", "messages": [...], "temperature": 0.7}

# Vision chat
POST /v1/chat/completions
{
  "model": "vision",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
      ]
    }
  ]
}
```

### For Backend Implementations

**Minimum requirement:** Keep existing `chat` or `chat_with_temperature` methods.

**Optional modern interface:** Add `generate_text_chat` and/or `generate_vision_chat` for better integration.

---

## Testing

The implementation includes test coverage in `tests/test_chat_completions.py`:

- ✅ Model alias resolution
- ✅ Text-only chat with string content
- ✅ Text-only chat with structured content
- ✅ Error when sending images to text-only model
- ✅ Vision chat with images

To run tests:
```bash
pytest tests/test_chat_completions.py -v
```

---

## Example: Setting Up a Vision Model

### 1. MLX Vision (Mac)
```bash
export JARVIS_VISION_MODEL_BACKEND=MLX_VISION
export JARVIS_VISION_MODEL_NAME=mlx-community/Llama-3.2-11B-Vision-Instruct-4bit
export JARVIS_VISION_MODEL_CONTEXT_WINDOW=131072
```

> **Note:** Use `MLX_VISION` (not `MLX`) for vision models because they require the `mlx-vlm` package, whereas regular `MLX` backend uses `mlx-lm` for text-only models.

### 2. Transformers Vision (GPU)
```bash
export JARVIS_VISION_MODEL_BACKEND=TRANSFORMERS
export JARVIS_VISION_MODEL_NAME=Qwen/Qwen2-VL-7B-Instruct
export JARVIS_VISION_MODEL_CONTEXT_WINDOW=32768
```

### 3. Remote Vision API (OpenAI GPT-4 Vision)
```bash
export JARVIS_VISION_MODEL_BACKEND=REST
export JARVIS_VISION_MODEL_NAME=gpt-4-vision-preview
export JARVIS_VISION_REST_MODEL_URL=https://api.openai.com
export JARVIS_REST_AUTH_TYPE=bearer
export JARVIS_REST_AUTH_TOKEN=sk-...
export JARVIS_REST_PROVIDER=openai
```

---

## Files Changed

1. **main.py** - Complete rewrite
   - New `/v1/chat/completions` endpoint
   - Message normalization helpers
   - Legacy routes removed
   
2. **managers/model_manager.py** - Enhanced
   - Model registry and aliases
   - Vision model support
   - `get_model_config()` and `list_models()`
   
3. **backends/rest_backend.py** - Enhanced
   - Added `generate_vision_chat()` method
   - Multimodal message support
   
4. **backends/mock_backend.py** - Fixed
   - Updated to use `model_id` instead of `name`
   
5. **managers/chat_types.py** - No changes
   - Already had all necessary types
   
6. **tests/test_chat_completions.py** - Fixed
   - Updated to use `model_id` instead of `name`

---

## What's Next

### Immediate (already done per PRD)
- ✅ Unified `/v1/chat/completions` endpoint
- ✅ Vision model support (MLX, Transformers, REST)
- ✅ Model aliases for stable client references
- ✅ OpenAI-compatible error handling
- ✅ Legacy route removal

### Future Enhancements (from PRD section 13)
1. **HTTP(S) image URL support** - Currently only data URLs supported
2. **Hybrid OCR + Vision flows** - Pre-process with OCR before vision model
3. **Model auto-selection** - Automatically pick vision vs text based on content
4. **Tool calling with images** - Function calling for structured data extraction
5. **Image size caps** - Downscaling strategy for large images

---

## Notes

- All imports are at the top of files (per PRD implementation notes)
- Vision logic is integrated into existing backend modules, not ad-hoc
- The API surface is minimal and OpenAI-compatible
- Backward compatibility maintained for existing backends
- No breaking changes for properly implemented backends

