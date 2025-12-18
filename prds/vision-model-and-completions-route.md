

# PRD: Unified Chat Completions Endpoint & Vision Model Integration for `llm-proxy`

## 1. Overview

We want to:

1. **Unify all chat-style interactions** (text-only and multimodal) behind a **single OpenAI-compatible `/v1/chat/completions` endpoint** in `llm-proxy`.
2. **Add support for a vision LLM** (Llama 3.2 Vision) that can accept images + text and return text responses.
3. Keep the design **backend-agnostic** so models can be backed by **GGUF/llama.cpp, vLLM, Transformers, MLX**, or remote OpenAI-compatible APIs.

This PRD is written in the recipes server repo for now, but will be **implemented in the `llm-proxy` project**.

---

## 2. Goals

1. **Single OpenAI-compatible route**
   - Expose **`POST /v1/chat/completions`** as the primary chat endpoint.
   - Support both **string content** and **structured content arrays** (`[{type: "text"}, {type: "image_url"}, ...]`), matching OpenAI's multimodal format.
   - Aim for a **1:1 route surface** with the OpenAI API (chat completions, models, etc.), and remove legacy routes that don't conform to this style.

2. **Vision model support**
   - Add a **dedicated vision model** to the model registry based on **Llama 3.2 Vision Instruct (11B)**.
   - Initial implementation will likely use **MLX 4-bit or 8-bit** on Mac for memory efficiency.
   - Vision model is treated as **just another model** in the registry with a `supports_images` capability flag.

3. **Model-based routing**
   - The **client selects the model** via `model` field.
   - The server determines whether the requested model **supports images** and routes to:
     - A **text-only** backend path, or
     - A **vision-enabled** backend path.

4. **OpenAI-style error behavior**
   - If the client sends images to a text-only model, return an **OpenAI-style `invalid_request_error`** explaining that the model does not support images.
   - Keep other errors consistent with existing `llm-proxy` behavior.

5. **Minimal breaking changes**
   - New and existing clients that speak the OpenAI-style chat format (including string content or structured content arrays) should continue to work as we simplify the route surface.
   - New multimodal clients can opt into `content` arrays and a vision-capable `model`.

---

## 3. Non-goals

1. **No new public routes** beyond `/v1/chat/completions` (and existing compatibility routes).
   - The end-state should be a **minimal OpenAI-compatible surface area**. As part of implementation we will remove legacy, non-OpenAI-style routes (e.g. warmup, swap-model, conversation status, model reset, legacy /chat variants) and clean up any dead code behind them.
   - We are **not** adding `/chat/vision` as a separate public API.
   - Internally, we can still factor out code for `vision_chat` vs `text_chat`.

2. **No new auth or rate-limiting changes**
   - This PRD assumes the existing auth and rate limiting mechanisms remain unchanged.

3. **No new orchestration logic**
   - We are not designing hybrid flows (e.g. OCR + vision fusion) here. That can be added later.

4. **No new external storage**
   - Images are processed in-memory only for now. No persistent storage / logging of raw image data.

---

## 4. Current State (assumptions)

> Note: This section is descriptive and may need small tweaks when we open the actual `llm-proxy` code.

- `llm-proxy` already exposes one or more chat-like endpoints (e.g. `/chat`) that accept **OpenAI-ish payloads** and forward them to backends.
- There is an internal **model registry/config** that defines:
  - Model name (e.g. `jarvis-text-8b`)
  - Backend type (e.g. `llama_cpp`, `vllm`, `transformers`, `mlx`, `remote`)
  - Model path / HF repo / GGUF file(s)
  - Context length and other parameters
- All existing models are **text-only**: `messages[*].content` is expected to be a **string**.
- Backends expose a single primary method for chat, something like:

  ```python
  async def generate_chat(model_cfg, request: ChatCompletionRequest) -> ChatCompletionResponse:
      ...
  ```

- No current handling of `image_url` / `input_image` / multimodal content.

This PRD will extend that design instead of replacing it.

---

## 5. API Design: `/v1/chat/completions`

### 5.1 Request format

We support the OpenAI-style request body (subset relevant to this feature):

```jsonc
{
  "model": "jarvis-vision-11b",
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "Describe this screenshot and extract any visible table data." },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,....",  
            "detail": "high"  // optional, may be ignored by some backends
          }
        }
      ]
    }
  ],
  "temperature": 0.2,
  "max_tokens": 1024,
  "stream": false
}
```

Allowed `content` formats per message:

1. **String (for backward compatibility)**

   ```jsonc
   {
     "role": "user",
     "content": "Just a plain text question."
   }
   ```

2. **Array of content parts** (multimodal)

   ```jsonc
   {
     "role": "user",
     "content": [
       { "type": "text", "text": "What is shown in this chart?" },
       {
         "type": "image_url",
         "image_url": {
           "url": "data:image/png;base64,...."
         }
       }
     ]
   }
   ```

Supported image sources (MVP):

- `image_url.url` MUST be a **`data:` URL with base64-encoded image bytes**.
- Later we may support `http(s)` URLs with server-side fetch, but that is **out of scope** for this PRD.

#### Model name resolution

The `model` field supports two kinds of values:

1. **Concrete model ids** (e.g. `"jarvis-text-8b"`, `"jarvis-vision-11b"`) that map directly to entries in the model registry.
2. **Logical aliases** that are stable for clients and resolved server-side by `model_manager`:
   - `"full"` → current primary text model (e.g. 8B GGUF)
   - `"lightweight"` → smaller, cheaper text model (e.g. 1B or 3B GGUF)
   - `"vision"` → primary vision model (Llama 3.2 11B Vision)

This lets third-party clients stay agnostic to the concrete model choice while still allowing power users to pin to a specific model id if desired.

### 5.2 Response format

Standard OpenAI chat completion response (existing behavior reused):

```jsonc
{
  "id": "chatcmpl-xyz",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "jarvis-vision-11b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Here is a description of the image..."
      },
      "finish_reason": "stop"
    }
  ]
}
```

Streaming will emit the same `chat.completion.chunk` structure that we already use for text-only models.

### 5.3 Error behavior

If a model does **not** support images (`supports_images = false`) and the request contains image parts, we return:

```jsonc
{
  "error": {
    "type": "invalid_request_error",
    "message": "Model 'jarvis-text-8b' does not support images. Use a vision-capable model instead.",
    "code": null
  }
}
```

Other errors (backend failures, bad input, etc.) keep existing semantics.

---

## 6. Model Registry Changes

We will extend the **model registry/config** schema with additional fields. (Exact format will depend on the existing implementation, but conceptually:)

```toml
[models.jarvis-text-8b]
backend          = "llama_cpp"           # or vllm/transformers
path             = "path/to/llama-3-8b.Q4_K_M.gguf"
supports_images  = false
context_length   = 4096

[models.jarvis-text-1b]
backend          = "llama_cpp"
path             = "path/to/llama-3-1b.Q4_K_M.gguf"
supports_images  = false
context_length   = 4096

[models.jarvis-vision-11b]
backend          = "mlx"                 # or transformers/vllm depending on deployment
hf_model_id      = "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit"
supports_images  = true
context_length   = 131072

# Optional, if we want an 8-bit variant:
[models.jarvis-vision-11b-high]
backend          = "mlx"
hf_model_id      = "mlx-community/Llama-3.2-11B-Vision-Instruct-8bit"
supports_images  = true
context_length   = 131072
```

Additional optional knobs (for future, but we can stub them):

- `max_images_per_message` (int, default maybe 4)
- `max_image_pixels` (int, rough cap, e.g. 4M)
- `image_downscale_strategy` (e.g. `short_side=768`)

We do **not** need to fully implement those caps in the first pass, but the design should make it easy to add them later.

Model selection and alias resolution will be centralized in `managers/model_manager.py`. That module already owns the logic for picking the `full` vs `lightweight` models; we will extend it to:

- Resolve logical aliases (`full`, `lightweight`, `vision`) to concrete model ids.
- Load per-model backend configuration from environment variables and/or config files.
- Expose a single helper (e.g. `get_model_config(model_name: str)`) that the `/v1/chat/completions` route uses instead of reaching into the registry directly.

### 6.1 Backend env vars

Backend selection for the core models will continue to use the existing env-var pattern:

- `JARVIS_MODEL_BACKEND`            → backend for the primary "full" model
- `JARVIS_LIGHTWEIGHT_MODEL_BACKEND` → backend for the "lightweight" model
- `JARVIS_VISION_MODEL_BACKEND`     → backend for the vision model

These values map to backend implementations in the `backends` package (`gguf_backend`, `mlx_backend`, `rest_backend`, `transformers_backend`, `vllm_backend`, etc.). `model_manager` is responsible for reading these env vars and instantiating the appropriate backend handles.

---

## 7. Backend Interface Changes

### 7.1 Normalized internal representation

We should define an internal representation for chat messages that:

- Works for both **text-only** and **multimodal** messages.
- Can be easily converted into whatever the underlying backend expects (llama.cpp, vLLM, Transformers, MLX, remote APIs, etc.).

Pseudo-typed Python dataclasses (conceptual):

```python
@dataclass
class ImagePart:
    data: bytes          # raw image bytes (decoded from base64)
    mime_type: str       # e.g. "image/png"

@dataclass
class TextPart:
    text: str

ContentPart = Union[TextPart, ImagePart]

@dataclass
class NormalizedMessage:
    role: str                       # "system" | "user" | "assistant" | "tool"
    content: list[ContentPart]      # at least one TextPart; may include ImagePart
```

The request handler will:

1. Parse incoming JSON.
2. Normalize `messages` into a list of `NormalizedMessage` objects.
3. Detect whether any `ImagePart` exists.

### 7.2 Backend routing

We extend the backend interface to distinguish between **text-only** and **vision** calls.

Option A (separate methods):

```python
async def generate_text_chat(model_cfg, messages: list[NormalizedMessage], params: GenerationParams) -> ChatCompletionResponse:
    ...

async def generate_vision_chat(model_cfg, messages: list[NormalizedMessage], params: GenerationParams) -> ChatCompletionResponse:
    ...
```

Option B (single method with a flag):

```python
async def generate_chat(model_cfg, messages: list[NormalizedMessage], params: GenerationParams, has_images: bool) -> ChatCompletionResponse:
    ...
```

**Preferred:** Option A (separate methods), because:

- Vision models often need **different preprocessing** (image tensors, different chat template).
- Some backends may **not implement** `generate_vision_chat` at all.

Each backend implementation can decide how to support (or not support) vision:

- `llama_cpp` backend → only `generate_text_chat` for now.
- `mlx` vision backend → supports both, but uses a different path when `ImagePart` is present.
- `transformers` vision backend → similar to MLX, but via `AutoProcessor` + `MllamaForConditionalGeneration`.

---

## 8. Vision Model Integration (Llama 3.2 Vision)

### 8.1 Initial target model

Primary target for the first implementation:

- **Model:** `Llama-3.2-11B-Vision-Instruct`
- **Backend:** MLX (on Mac) or Transformers/vLLM (on GPU server)
- **Quantization:** 4-bit or 8-bit (MLX) to keep RAM usage low.

We will expose it via the model registry as `jarvis-vision-11b`.

### 8.2 Input formatting

For MLX/Transformers, we will:

1. Convert `NormalizedMessage` list into the model's expected chat template using the provided processor/template (e.g. `processor.apply_chat_template(...)`).
2. Collect `ImagePart` objects from the messages and pass them as image tensors to the model.

High-level flow:

```python
text_prompt, images = build_llama_3_2_vision_prompt(messages)
# text_prompt: str or tokenized ids
# images: list of image tensors (preprocessed)

outputs = model.generate(
    **processor(text=text_prompt, images=images, return_tensors="pt"),
    max_new_tokens=params.max_tokens,
    temperature=params.temperature,
    do_sample=params.do_sample,
    ...
)

decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]
```

The exact details will depend on the reference Llama 3.2 Vision examples; the PR will pin to a specific pattern.

### 8.3 Output handling

- Return a **single assistant message** with `content` as text.
- No image outputs (e.g. image generation) are planned in this PRD.

---

## 9. Request Handling Flow

High-level pseudocode for `/v1/chat/completions`:

```python
async def chat_completions(request: ChatCompletionRequest):
    # 1. Resolve model
    model_cfg = MODEL_REGISTRY.get(request.model)
    if not model_cfg:
        raise openai_error("model_not_found", f"Model '{request.model}' does not exist")

    # 2. Normalize messages
    messages = normalize_messages(request.messages)

    # 3. Detect images
    has_images = any(
        any(isinstance(part, ImagePart) for part in msg.content)
        for msg in messages
    )

    # 4. Check model capabilities
    if has_images and not model_cfg.supports_images:
        raise openai_invalid_request(
            f"Model '{request.model}' does not support images. Use a vision-capable model instead."
        )

    # 5. Pick backend
    backend = BACKENDS[model_cfg.backend]

    # 6. Dispatch
    if has_images:
        raw_response = await backend.generate_vision_chat(model_cfg, messages, params)
    else:
        raw_response = await backend.generate_text_chat(model_cfg, messages, params)

    # 7. Adapt to OpenAI response format
    return to_openai_chat_completion(raw_response)
```

Streaming variant will follow the same decision flow but yield chunks progressively.

---

## 10. Performance & Resource Considerations

- **Memory:**
  - Typical deployment might run 2–3 models simultaneously:
    - `jarvis-text-8b` (Q4 GGUF)
    - `jarvis-text-1b` or `jarvis-text-3b` (Q4 GGUF)
    - `jarvis-vision-11b` (Llama 3.2 Vision, 4-bit MLX)
  - Rough ballpark (weights only):
    - 8B Q4 ≈ ~3.7 GB
    - 1B Q4 ≈ ~0.5 GB (or 3B Q4 ≈ ~1.4 GB)
    - 11B 4-bit MLX ≈ ~5 GB
  - Total: ~9–10 GB of weights, ~12–16 GB including KV caches and overhead at typical load, which is acceptable for the Mac Studio.

- **Latency:**
  - Vision calls will be more expensive than text-only calls.
  - The routing logic ensures that **text-only conversations do not accidentally hit the vision model**.

- **Concurrency:**
  - No special changes are required; the existing concurrency limits in `llm-proxy` apply equally to vision and text.

---

## 11. Testing Strategy

1. **Unit tests**
   - Message normalization:
     - `content` as string → converted to `[TextPart]`.
     - `content` as mixed array → `[TextPart, ImagePart, ...]`.
   - Capability checks:
     - Images + text-only model → error.
     - Text-only content + vision model → should still work.

2. **Integration tests (local)**
   - Spin up `llm-proxy` with a mocked or lightweight backend implementation for vision.
   - Send a multimodal request and assert the OpenAI-style response shape.

3. **Integration with real model** (manual / later automated):
   - Run `jarvis-vision-11b` locally (MLX or Transformers).
   - Send a simple image + text prompt and verify:
     - The model responds.
     - The description is plausible.

4. **Backward-compat checks**
   - Confirm that only OpenAI-style routes remain exposed (e.g. /v1/chat/completions and health checks), and that removed legacy routes are no longer accessible.

---

## 12. Rollout Plan

1. **Phase 1 – Internal implementation & flagging**
   - Implement message normalization, registry extensions, and backend interface changes.
   - Add vision model support in code, but keep it **disabled by default** (e.g. behind a config flag or missing model entry).

2. **Phase 2 – Vision model bring-up**
   - Configure `jarvis-vision-11b` on the Mac Studio with MLX 4-bit.
   - Run a small set of manual tests on typical Jarvis use cases:
     - Screenshot of UI.
     - Photo of a whiteboard / document.
     - Chart/graph.

3. **Phase 3 – Full enablement**
   - Enable the model registry entry for `jarvis-vision-11b` in the main `llm-proxy` config.
   - Start using it from Jarvis clients that are image-capable.

4. **Phase 4 – Refinements**
   - Tune model selection logic (e.g. using `jarvis-text-1b` vs `jarvis-text-8b` vs `jarvis-vision-11b`).
   - Add optional features:
     - Image size caps and downscale strategy.
     - Better error messages and logging for malformed images.

---

## 13. Open Questions / Future Enhancements

1. **HTTP(S) image URLs**
   - Do we want `llm-proxy` to fetch remote images itself when `image_url.url` is not a data URL?
   - If yes, we need:
     - Configurable allowlist/denylist.
     - Timeouts and size limits.

2. **Hybrid OCR + Vision flows**
   - For some Jarvis workflows, we may want to:
     - Run document images through OCR first (Whisper / Tesseract / etc.),
     - Then feed both the text and a downscaled image (or no image) to the LLM.

3. **Model auto-selection**
   - Instead of always requiring the client to specify `model`, we could add a higher-level route later (e.g. `/jarvis/chat`) that:
     - Auto-picks `jarvis-vision-11b` when images are present.
     - Otherwise chooses between `jarvis-text-1b` and `jarvis-text-8b` based on a cost/latency policy.

4. **Tool calling / function calling with images**
   - If we add function calling / tools to Jarvis, we should design how vision models interact with that (e.g., extracting structured data from screenshots and emitting tool calls).

---

## 14. Summary

This PRD defines how to:

- Move `llm-proxy` to a **single, OpenAI-compatible `/v1/chat/completions` endpoint`** for all chat interactions.
- Introduce a **vision-capable model** (`jarvis-vision-11b`, based on Llama 3.2 Vision) into the model registry.
- Extend the backend interfaces and routing logic to understand **multimodal messages** (text + images) with minimal impact on existing text-only flows.

Once implemented, Jarvis will be able to:

- Accept screenshots, photos, and other images from clients.
- Run them through a powerful vision model.
- Return structured, high-quality responses while keeping the proxy API surface simple and OpenAI-compatible.

---

## 15. Implementation notes

- Follow the existing repo style: **imports should live at the top of each file** unless there is a strong, documented reason to use a local import (e.g. avoiding an optional heavy dependency at import time).
- New vision-related logic should be integrated into `managers/model_manager.py` and the existing backend modules rather than introducing ad-hoc model-loading code in route handlers.