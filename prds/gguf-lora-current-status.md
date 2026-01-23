# GGUF LoRA Adapter Training: Current Status

## Overview
This document summarizes the current solution for local adapter training and inference in `llm-proxy`, including queueing, training pipeline, and the resolution of the GGUF LoRA blocking bug.

## Resolution (January 2026)

**Decision: GGUF LoRA is disabled. Use vLLM or Transformers for adapter inference.**

The GGUF LoRA adapter feature was blocked by a `llama.cpp` scheduler assert bug. After evaluation, we chose to:

1. **Disable GGUF adapter loading** - The GGUF backend now raises `GGUFAdapterNotSupportedError` if `JARVIS_ADAPTER_PATH` is set.
2. **Remove GGUF LoRA conversion** - Training produces PEFT-format adapters only.
3. **Use vLLM/Transformers for adapters** - Both backends have stable PEFT LoRA support.
4. **Keep GGUF for base inference** - GGUF backend works fine without adapters.

### Recommended Stack for Adapter Inference
- **Model**: `meta-llama/Llama-3.2-3B-Instruct`
- **Backend**: `TRANSFORMERS` with `JARVIS_INFERENCE_ENGINE=vllm`
- **Hardware**: NVIDIA 3080 Ti (12GB) or better
- **See**: `env.template` for optimized presets

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
- `scripts/train_adapter.py` performs PEFT LoRA training using `transformers` and `peft`.
  - Supports low-memory training (4-bit/8-bit, grad checkpointing, dtype control).
  - Outputs a **PEFT LoRA adapter (HF format)** in a work directory.
  - GGUF conversion is disabled.
- Artifacts are zipped (`adapter.zip`) and stored locally under `JARVIS_ADAPTER_STORE_DIR`.
- Adapter URL currently uses `file://` (S3/MinIO planned).

### Inference (Current)
- **vLLM backend** (`backends/vllm_backend.py`):
  - Supports PEFT LoRA via `LoRARequest`.
  - Per-request adapter selection with caching.
  - Best throughput on CUDA GPUs.
- **Transformers backend** (`backends/transformers_backend.py`):
  - Loads PEFT adapters via `PeftModel.from_pretrained()`.
  - Supports quantized inference (4-bit, 8-bit).
  - More flexible but lower throughput than vLLM.
- **GGUF backend** (`backends/gguf_backend.py`):
  - Works for base model inference only.
  - Raises `GGUFAdapterNotSupportedError` if adapter is configured.

## Technologies Used
- **Queueing**: Redis + RQ.
- **Training**: `transformers`, `peft`, optional quantized loading.
- **Inference**: `vLLM` (recommended) or `transformers` with PEFT.
- **Schema enforcement**: JSON schema repair/validation; guided generation in vLLM.
- **Auth**: App-to-app via `X-Jarvis-App-Id` / `X-Jarvis-App-Key`.

## Historical Bug (Resolved by Disabling)

**GGUF LoRA adapter caused a `llama.cpp` scheduler assert on first decode.**

Root cause: Version mismatch between `convert_lora_to_gguf.py` (llama.cpp `b7779`) and `llama-cpp-python` runtime (`0.3.16`). The GGUF LoRA format changes frequently between versions, causing runtime graph scheduling failures.

Error: `GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs) failed`

**Resolution**: Rather than maintaining version-locked llama.cpp builds, we standardized on vLLM/Transformers for adapter inference, which have stable PEFT LoRA support.

## Future Considerations

If GGUF LoRA support is needed in the future:
1. Pin `llama-cpp-python` to a specific llama.cpp commit.
2. Use the same commit's `convert_lora_to_gguf.py` for conversion.
3. Test thoroughly before enabling.

