# GGUF LoRA Adapter Training: Current Status

## Overview
This document summarizes the current solution for local adapter training and inference in `llm-proxy`, including queueing, training pipeline, and GGUF LoRA adapter support.

## Resolution (February 2026)

**Decision: GGUF LoRA is re-enabled using constructor-based adapter loading.**

The GGUF LoRA adapter feature was previously blocked by a version mismatch between `convert_lora_to_gguf.py` and the `llama-cpp-python` runtime. The fix: use the converter from the **same** llama.cpp commit as the runtime.

### Approach
1. **Constructor-based adapter loading** — `llama-cpp-python` only supports `lora_path`/`lora_scale` at `Llama()` construction time. No dynamic load/unload. Adapter switching = model reload.
2. **Sticky adapter behavior** — when a request has no `adapter_settings`, the current adapter stays loaded (avoids unnecessary reloads for single-user).
3. **Version-locked converter** — vendored from llama.cpp commit `4227c9b` (matches `llama-cpp-python 0.3.16`). See `scripts/vendor/README.md`.
4. **Dual-format artifacts** — training produces PEFT files at root + `gguf/adapter.gguf` subdirectory in the artifact zip.
5. **Non-fatal GGUF conversion** — training always produces PEFT. GGUF conversion is a bonus step that fails gracefully.

### Version Matching

| llama-cpp-python | llama.cpp commit | Converter source |
|------------------|------------------|------------------|
| 0.3.16           | `4227c9b`        | `scripts/vendor/llama.cpp/convert_lora_to_gguf.py` |

## Current Solution

### Queueing
- Jobs are enqueued via `POST /internal/queue/enqueue` with `job_type=adapter_train`.
- Payload includes `node_id`, `base_model_id`, `dataset_ref`, and `params`.
- Requests are validated and deduped by `(job_id, idempotency_key)` in Redis.
- Jobs are processed by an RQ worker listening on `llm_proxy_jobs`.
- Callbacks are posted to the provided URL on completion.

### Training Pipeline
- The worker runs `services/adapter_training.py`, which:
  - Materializes dataset and params to disk.
  - Spawns `scripts/train_adapter.py` via `JARVIS_ADAPTER_TRAIN_CMD`.
  - Captures logs to `train.log`.
  - Optionally unloads/reloads the model service under `DEBUG` to avoid GPU OOM.
  - Detects adapter format: `peft_lora+gguf` (dual-format) or `peft_lora` (PEFT-only).
- `scripts/train_adapter.py` performs PEFT LoRA training using `transformers` and `peft`.
  - Supports low-memory training (4-bit/8-bit, grad checkpointing, dtype control).
  - Outputs a **PEFT LoRA adapter (HF format)** in a work directory.
  - If base model is GGUF and `hf_base_model_id` is provided, converts to GGUF format (non-fatal).
- Artifacts are zipped (`adapter.zip`) and stored locally under `LLM_PROXY_ADAPTER_DIR`.
- Adapter URL currently uses `file://` (S3/MinIO planned).

### Inference

- **GGUF backend** (`backends/gguf_backend.py`):
  - Supports GGUF LoRA via constructor-based loading (`lora_path`/`lora_scale` at `Llama()` init).
  - Adapter switching requires model reload (acceptable for single-user Mac Mini target).
  - Resolves adapter hash → `gguf/adapter.gguf` via `adapter_cache`.
  - Sticky behavior: same adapter hash = no reload.
- **vLLM backend** (`backends/vllm_backend.py`):
  - Supports PEFT LoRA via `LoRARequest`.
  - Per-request adapter selection with caching.
  - Best throughput on CUDA GPUs.
- **Transformers backend** (`backends/transformers_backend.py`):
  - Loads PEFT adapters via `PeftModel.from_pretrained()`.
  - Supports quantized inference (4-bit, 8-bit).
  - More flexible but lower throughput than vLLM.

### Request Flow (GGUF with Adapter)

```
ChatCompletionRequest { adapter_settings: { hash: "abc123", scale: 1.0 } }
    → generate_text_chat()
    → hash != current? → _resolve_gguf_adapter("abc123")
        → adapter_cache.get_adapter_path("abc123") → /tmp/jarvis-adapters/abc123/
        → check gguf/adapter.gguf → found
    → _reload_with_adapter("/tmp/.../gguf/adapter.gguf", 1.0)
        → del self.model
        → Llama(model_path=..., lora_path=..., lora_scale=1.0)
        → warmup
    → create_chat_completion() → response
```

Subsequent requests with same adapter hash: no reload, direct inference.

## Technologies Used
- **Queueing**: Redis + RQ.
- **Training**: `transformers`, `peft`, optional quantized loading.
- **Inference**: `llama-cpp-python` (GGUF), `vLLM` (recommended for CUDA), or `transformers` with PEFT.
- **GGUF Conversion**: Vendored `convert_lora_to_gguf.py` from version-matched llama.cpp commit.
- **Schema enforcement**: JSON schema repair/validation; guided generation in vLLM.
- **Auth**: App-to-app via `X-Jarvis-App-Id` / `X-Jarvis-App-Key`.

## Historical Bug (Resolved)

**GGUF LoRA adapter caused a `llama.cpp` scheduler assert on first decode.**

Root cause: Version mismatch between `convert_lora_to_gguf.py` (llama.cpp `b7779`) and `llama-cpp-python` runtime (`0.3.16`, built against llama.cpp `4227c9b`). The GGUF LoRA format changes frequently between versions, causing runtime graph scheduling failures.

Error: `GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs) failed`

**Resolution**: Use the converter from the same llama.cpp commit as the runtime. Vendored at `scripts/vendor/llama.cpp/` with version tracking in `scripts/vendor/README.md`.

Note: The bug references previously cited in the code (#7742, #4485) are unrelated upstream issues. The actual problem was always a version mismatch.

## Setup

### GGUF LoRA Conversion

```bash
# One-time setup: download the version-matched converter
cd jarvis-llm-proxy-api
bash scripts/vendor/setup_llama_cpp.sh

# Verify
ls scripts/vendor/llama.cpp/convert_lora_to_gguf.py
```

### Upgrading llama-cpp-python

When upgrading `llama-cpp-python`:
1. Identify the new llama.cpp commit it was built against.
2. Update `scripts/vendor/setup_llama_cpp.sh` with the new commit hash.
3. Re-run the setup script (`rm -rf scripts/vendor/llama.cpp && bash scripts/vendor/setup_llama_cpp.sh`).
4. Update the version table above and in `scripts/vendor/README.md`.
5. Test adapter conversion with a known-good PEFT adapter.
