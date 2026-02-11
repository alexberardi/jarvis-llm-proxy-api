# Jarvis Adapter Training Guide

Train a LoRA adapter for date/time key extraction from natural language. The adapter extracts semantic keys like `tomorrow_morning`, `next_tuesday`, `in_30_minutes`, `in_3_days` from voice commands.

## Multi-Backend Architecture

Training produces a **PEFT adapter** (universal format) that works across all three backends:

```
Train (PyTorch/PEFT) ──→ adapter_config.json + adapter_model.safetensors
                              │
                              ├── vLLM:      Use PEFT adapter directly (native LoRARequest support)
                              ├── MLX:       Use PEFT adapter directly (mlx_lm.tuner.utils.load_adapters)
                              └── GGUF:      Convert PEFT → GGUF adapter (convert_lora_to_gguf.py)
```

**Why PEFT is the universal source:**
- **vLLM** loads PEFT adapters natively via `LoRARequest`
- **MLX** loads PEFT adapters natively via `load_adapters()` — the `mlx_backend.py` already does this
- **GGUF/llama.cpp** needs a conversion step, handled automatically by `train_adapter.py`

### Automated Worker Pipeline

When adapters are trained via the queue worker (`services/adapter_training.py`), the output zip contains all needed formats:

```
adapter.zip
  ├── adapter_config.json         ← PEFT (used by vLLM + MLX)
  ├── adapter_model.safetensors   ← PEFT weights
  ├── tokenizer.json              ← tokenizer files
  └── gguf/adapter.gguf           ← GGUF adapter (auto-converted if base is .gguf)
```

The worker calls `JARVIS_ADAPTER_TRAIN_CMD` (default: `python3 scripts/train_adapter.py`), which:
1. Trains in PEFT format
2. Auto-detects GGUF base model → converts PEFT adapter to GGUF format
3. Zips everything into a single artifact

### Alternative: Merge Into Base Model

Instead of runtime LoRA loading, you can bake the adapter into the base model and convert for each backend:

```
Merge (PEFT → HF)
  ├── GGUF:  convert_to_gguf.py  → .models/llama-3.1-8b-instruct-jarvis.gguf
  ├── MLX:   convert_to_mlx.py   → .models/llama-3.1-8b-instruct-jarvis-mlx-4bit/
  └── vLLM:  use merged HF model directly
```

Per-node LoRA adapters can still be applied at runtime on top of the merged model.

## Prerequisites

- Python 3.11+ with venv activated
- A HuggingFace-format base model in `.models/` (e.g., `llama-3.1-8b-instruct`)
- Training dependencies installed (see below)

## Install Dependencies

All training deps are in `requirements-base.txt`:

```bash
pip install -r requirements-base.txt
```

Key packages: `transformers`, `peft`, `trl`, `datasets`, `accelerate`, `torch`

## Training Pipeline

### Step 1: Generate Training Data

```bash
python scripts/generate_jarvis_training_data.py
```

Output: `data/jarvis_training.jsonl` (~2100+ examples covering date/time keys, relative time, negatives, and ambiguous cases)

### Step 2: Train the Adapter

#### Mac (Apple Silicon / MPS)

```bash
python scripts/train_jarvis_adapter.py \
  --optim adamw_torch \
  --batch-size 2
```

Important notes for MPS:
- **`--optim adamw_torch` is required** — `adamw_8bit` (default) uses bitsandbytes which doesn't support MPS
- The script auto-detects MPS and uses `float16` + `device_map={"": "mps"}` to keep all tensors on one device
- An 8B model in float16 uses ~16GB; with optimizer state and activations, expect ~24-28GB peak usage on 32GB machines
- If you hit memory pressure, reduce to `--batch-size 1`
- `pin_memory` is auto-disabled on MPS (not supported)
- Training an 8B model takes ~2 hours on M-series chips (3 epochs, 2100 examples)

#### Linux (NVIDIA GPU / CUDA)

```bash
python scripts/train_jarvis_adapter.py
```

Defaults work out of the box on CUDA: `adamw_8bit` optimizer, `bfloat16`, 4-bit quantization via bitsandbytes.

For large models on limited VRAM, reduce batch size:
```bash
python scripts/train_jarvis_adapter.py --batch-size 1 --lora-r 8
```

#### CPU Only (slow, not recommended)

```bash
python scripts/train_jarvis_adapter.py --optim adamw_torch --batch-size 1
```

### Step 3: Validate

```bash
python scripts/validate_jarvis_adapter.py --adapter-path adapters/jarvis
```

Runs 150+ test cases across categories: relative days, time of day, weekdays, periods, specific times, relative minutes/hours/days, negatives, ambiguous cases, and mixed expressions. Target: >= 95% accuracy.

Save results to JSON:
```bash
python scripts/validate_jarvis_adapter.py --adapter-path adapters/jarvis --output-json results.json
```

### Step 4: Deploy to Your Backend

#### Option A: Runtime LoRA (adapter stays separate)

The PEFT adapter from `adapters/jarvis/` works directly with vLLM and MLX — no conversion needed. For GGUF:

```bash
# Convert PEFT adapter to GGUF format for llama.cpp
python scripts/vendor/llama.cpp/convert_lora_to_gguf.py \
  adapters/jarvis \
  --base meta-llama/Llama-3.1-8B-Instruct \
  --outfile adapters/jarvis/gguf/adapter.gguf
```

#### Option B: Merge into base model (adapter baked in)

```bash
# 1. Merge adapter into HF base model
python scripts/merge_adapter.py \
  --base-model .models/llama-3.1-8b-instruct \
  --adapter adapters/jarvis \
  --output .models/llama-3.1-8b-instruct-jarvis

# 2. Convert for your backend:

# GGUF (llama.cpp)
python scripts/convert_to_gguf.py \
  --model .models/llama-3.1-8b-instruct-jarvis \
  --output .models/llama-3.1-8b-instruct-jarvis.gguf

# MLX (Apple Silicon)
python scripts/convert_to_mlx.py \
  --model .models/llama-3.1-8b-instruct-jarvis \
  --output .models/llama-3.1-8b-instruct-jarvis-mlx-4bit

# vLLM — use the merged HF model directly:
# JARVIS_MODEL_NAME=.models/llama-3.1-8b-instruct-jarvis
```

## All CLI Options

```
python scripts/train_jarvis_adapter.py --help

  --base-model       Base model path or HF ID (default: JARVIS_MODEL_NAME or .models/llama-3.1-8b-instruct)
  --training-data    JSONL training data (default: data/jarvis_training.jsonl)
  --output-dir       Adapter output dir (default: adapters/jarvis)
  --epochs           Training epochs (default: 3)
  --batch-size       Batch size (default: 4; use 1-2 on MPS)
  --learning-rate    Learning rate (default: 2e-4)
  --lora-r           LoRA rank (default: 16)
  --lora-alpha       LoRA alpha (default: 32)
  --max-length       Max sequence length (default: 256)
  --optim            Optimizer (default: adamw_8bit; use adamw_torch on MPS)
  --grad-accum       Gradient accumulation steps (default: 4)
  --no-gradient-checkpointing   Disable gradient checkpointing
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JARVIS_MODEL_NAME` | `.models/llama-3.1-8b-instruct` | Base model for training |
| `JARVIS_ADAPTER_TRAIN_DEVICE_MAP` | `{"": "mps"}` (MPS) / `{"": 0}` (CUDA) | Device placement strategy |
| `JARVIS_DATE_ADAPTER_TRAIN_LOAD_IN_4BIT` | `true` | Enable 4-bit quantization (CUDA only) |

## Output Structure

```
adapters/jarvis/
  adapter_config.json        # LoRA config (references base model)
  adapter_model.safetensors  # Trained weights
  tokenizer.json             # Tokenizer files
  tokenizer_config.json
  special_tokens_map.json
  training_metadata.json     # Training hyperparameters and stats
  checkpoints/               # Intermediate checkpoints (auto-pruned to 2)
```

## FastText Classifier (Alternative)

A lightweight FastText classifier (~5-10MB, ~1ms inference) can be trained alongside the LoRA adapter. In production, the hybrid extractor uses FastText for high-confidence predictions and falls back to the LoRA adapter for ambiguous cases.

```bash
# Train FastText (uses same training data)
python scripts/train_fasttext_date_keys.py

# Validate FastText alone
python scripts/validate_fasttext_date_keys.py

# Validate hybrid FastText + LLM strategy
python scripts/validate_hybrid_date_keys.py
```

## API Endpoint

The trained adapter's vocabulary is exposed at `GET /v1/adapters/date-keys`:

```json
{
  "version": "2.0",
  "static_keys": ["today", "tomorrow", "next_monday", ...],
  "dynamic_patterns": [
    {"pattern": "in_{N}_minutes", "regex": "^in_(\\d+)_minutes$"},
    {"pattern": "in_{N}_days", "regex": "^in_(\\d+)_days$"}
  ],
  "adapter_trained": true
}
```

Consumers use `dynamic_patterns[].regex` to parse the numeric value from relative time keys.

## Troubleshooting

| Error | Fix |
|-------|-----|
| `No module named 'trl'` | `pip install trl` |
| `expected device meta but got mps:0` | Ensure device_map is `{"": "mps"}` not `"auto"` (auto-detected) |
| `Invalid buffer size: 29.92 GiB` (MPS) | Script should auto-use float16; if not, check JARVIS_ADAPTER_TRAIN_DEVICE_MAP is not set to "0" |
| `adamw_8bit` fails on MPS | Add `--optim adamw_torch` |
| `pin_memory not supported on MPS` | Warning only, auto-disabled by script |
| OOM during training | Reduce `--batch-size 1` and/or `--lora-r 8` |
| Training data not found | Run `python scripts/generate_jarvis_training_data.py` first |
