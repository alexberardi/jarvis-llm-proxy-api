#!/usr/bin/env python3
"""MLX-native LoRA adapter training script for Apple Silicon.

Drop-in alternative to train_adapter.py that uses MLX's native Metal-accelerated
training loop instead of PyTorch + PEFT. Reads the same env vars and dataset format,
produces adapters compatible with both MLX inference and GGUF conversion.

Env vars (set by adapter_training.py orchestrator):
    JARVIS_TRAIN_OUTPUT_DIR       - Output directory for adapter files
    JARVIS_TRAIN_DATASET_PATH     - Path to dataset JSON
    JARVIS_TRAIN_PARAMS_PATH      - Path to training params JSON
    JARVIS_TRAIN_BASE_MODEL_ID    - Base model path or HF ID
    JARVIS_ADAPTER_HF_BASE_MODEL_ID - HF model ID (for GGUF models)
    JARVIS_ADAPTER_BATCH_SIZE     - Batch size (default: 1)
    JARVIS_ADAPTER_MAX_SEQ_LEN    - Max sequence length (default: 2048)
    JARVIS_ADAPTER_GRAD_ACCUM     - Gradient accumulation steps (default: 4)
    JARVIS_ADAPTER_EPOCHS         - Number of epochs (default: 1)
    JARVIS_ADAPTER_LEARNING_RATE  - Learning rate (default: 2e-4)
    JARVIS_ADAPTER_LORA_R         - LoRA rank (default: 16)
    JARVIS_ADAPTER_LORA_ALPHA     - LoRA alpha (default: 32)
    JARVIS_ADAPTER_LORA_DROPOUT   - LoRA dropout (default: 0.05)
    JARVIS_ADAPTER_GGUF_CONVERT_CMD - GGUF conversion command override
"""

import gc
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _read_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _extract_examples(
    dataset_ref: Dict[str, Any],
) -> List[Tuple[str, Dict[str, Any], str | None, str | None]]:
    """Extract examples from dataset.

    Returns list of (voice_command, tool_call, formatted_prompt, formatted_completion) tuples.
    formatted_prompt/formatted_completion are None when the dataset was built without a provider.
    """
    data = dataset_ref.get("data") if isinstance(dataset_ref, dict) else None
    payload = data if isinstance(data, dict) else dataset_ref
    commands = payload.get("commands", []) if isinstance(payload, dict) else []
    examples: List[Tuple[str, Dict[str, Any], str | None, str | None]] = []
    for cmd in commands:
        for ex in cmd.get("examples", []):
            voice = ex.get("voice_command")
            tool_call = ex.get("expected_tool_call")
            if voice and tool_call:
                examples.append((
                    voice,
                    tool_call,
                    ex.get("formatted_prompt"),
                    ex.get("formatted_completion"),
                ))
    return examples


def _format_prompt(voice_command: str) -> str:
    return (
        "You are a tool router. Return JSON only.\n"
        f"User: {voice_command}\n"
        "Assistant:"
    )


def _format_completion(tool_call: Dict[str, Any]) -> str:
    response = {
        "message": "",
        "tool_calls": [tool_call],
        "error": None,
    }
    return " " + json.dumps(response, ensure_ascii=False)


def _get_param(params: Dict[str, Any], key: str, default: Any) -> Any:
    return params.get(key) if params and key in params else default


class TokenizedDataset:
    """Pre-tokenized dataset compatible with mlx_lm's iterate_batches/CacheDataset.

    Unlike CompletionsDataset which uses apply_chat_template (which can fail with
    some tokenizer templates), this directly tokenizes the prompt+completion and
    returns (tokens, prompt_offset) tuples matching the expected format.
    """

    def __init__(
        self,
        tokenizer: Any,
        examples: List[Tuple[str, Dict[str, Any], str | None, str | None]],
        max_tokens: int,
    ):
        self._items: List[Tuple[List[int], int]] = []
        for voice, tool_call, fmt_prompt, fmt_completion in examples:
            prompt = fmt_prompt if fmt_prompt else _format_prompt(voice)
            completion = fmt_completion if fmt_completion else _format_completion(tool_call)
            full_text = prompt + completion

            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            full_ids = tokenizer.encode(full_text, add_special_tokens=True)

            # Truncate to max_tokens
            if len(full_ids) > max_tokens:
                full_ids = full_ids[:max_tokens]

            # Offset = length of prompt tokens (for loss masking)
            offset = min(len(prompt_ids), len(full_ids))
            self._items.append((full_ids, offset))

    def __getitem__(self, idx: int) -> Tuple[List[int], int]:
        return self._items[idx]

    def __len__(self) -> int:
        return len(self._items)


def _convert_mlx_to_peft(
    mlx_adapter_path: Path,
    peft_output_dir: Path,
    hf_base_model_id: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str],
) -> None:
    """Convert MLX adapter to PEFT format for GGUF compatibility.

    MLX saves adapters as (row-major, x @ lora_a @ lora_b):
        model.layers.{N}.self_attn.q_proj.lora_a  (shape: [in_features, rank])
        model.layers.{N}.self_attn.q_proj.lora_b  (shape: [rank, out_features])

    PEFT expects (column-major via nn.Linear, F.linear(x, weight) = x @ weight.T):
        lora_A.weight: (rank, in_features)   — transposed from MLX lora_a
        lora_B.weight: (out_features, rank)  — transposed from MLX lora_b
    """
    import numpy as np
    from safetensors.numpy import load_file, save_file

    mlx_weights_path = mlx_adapter_path / "adapters.safetensors"
    if not mlx_weights_path.is_file():
        print(f"MLX adapter not found at {mlx_weights_path}", flush=True)
        return

    tensors = load_file(str(mlx_weights_path))

    # Rename and transpose tensors: MLX → PEFT
    peft_tensors: Dict[str, np.ndarray] = {}
    for mlx_name, weight in tensors.items():
        # model.layers.N.self_attn.q_proj.lora_a → base_model.model.model.layers.N.self_attn.q_proj.lora_A.weight
        peft_name = f"base_model.model.{mlx_name}"
        peft_name = peft_name.replace(".lora_a", ".lora_A.weight")
        peft_name = peft_name.replace(".lora_b", ".lora_B.weight")
        # MLX uses row-major (in_features, rank) / (rank, out_features)
        # PEFT uses nn.Linear convention (rank, in_features) / (out_features, rank)
        peft_tensors[peft_name] = weight.T

    peft_output_dir.mkdir(parents=True, exist_ok=True)

    # Save PEFT adapter weights
    peft_weights_path = peft_output_dir / "adapter_model.safetensors"
    save_file(peft_tensors, str(peft_weights_path))

    # Write PEFT adapter_config.json
    peft_config = {
        "base_model_name_or_path": hf_base_model_id,
        "fan_in_fan_out": False,
        "inference_mode": True,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "peft_type": "LORA",
        "r": lora_r,
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
    }
    peft_config_path = peft_output_dir / "adapter_config.json"
    peft_config_path.write_text(json.dumps(peft_config, indent=2), encoding="utf-8")

    print(f"PEFT adapter saved: {peft_output_dir}", flush=True)
    print(f"  Converted {len(peft_tensors)} tensors", flush=True)


def _run_gguf_conversion(
    peft_dir: Path,
    gguf_out_dir: Path,
    hf_base_model_id: str,
) -> None:
    """Run GGUF conversion on the PEFT adapter directory."""
    gguf_convert_cmd = os.getenv("JARVIS_ADAPTER_GGUF_CONVERT_CMD", "").strip()
    if not gguf_convert_cmd:
        vendor_script = Path(__file__).parent / "vendor" / "llama.cpp" / "convert_lora_to_gguf.py"
        if vendor_script.is_file():
            gguf_convert_cmd = f"{sys.executable} {vendor_script}"

    if not gguf_convert_cmd:
        print("GGUF conversion skipped: no converter found", flush=True)
        return

    gguf_out_dir.mkdir(parents=True, exist_ok=True)
    gguf_out_file = gguf_out_dir / "adapter.gguf"
    convert_args = (
        f"{gguf_convert_cmd} {peft_dir} "
        f"--base {hf_base_model_id} "
        f"--outfile {gguf_out_file}"
    )
    print(f"Converting PEFT adapter to GGUF: {convert_args}", flush=True)
    try:
        subprocess.run(
            convert_args,
            shell=True,
            check=True,
            timeout=600,
            capture_output=True,
            text=True,
        )
        print(f"GGUF adapter saved: {gguf_out_file}", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"GGUF conversion failed (non-fatal): {e.stderr[-500:] if e.stderr else e}", flush=True)
    except subprocess.TimeoutExpired:
        print("GGUF conversion timed out (non-fatal)", flush=True)


def main() -> int:
    output_dir = os.getenv("JARVIS_TRAIN_OUTPUT_DIR")
    dataset_path = os.getenv("JARVIS_TRAIN_DATASET_PATH")
    params_path = os.getenv("JARVIS_TRAIN_PARAMS_PATH")
    base_model_id = os.getenv("JARVIS_TRAIN_BASE_MODEL_ID")

    if not output_dir or not dataset_path or not params_path or not base_model_id:
        print("Missing required env vars for training.", flush=True)
        return 2

    dataset_ref = _read_json(dataset_path)
    params = _read_json(params_path)
    examples = _extract_examples(dataset_ref)
    if not examples:
        print("No training examples found in dataset_ref.", flush=True)
        return 2

    # Resolve the HF model ID for MLX model loading
    hf_base_model_id = _get_param(
        params, "hf_base_model_id", os.getenv("JARVIS_ADAPTER_HF_BASE_MODEL_ID")
    )

    # MLX always loads from HF or an absolute local path to a HF-format model dir.
    # GGUF files can't be loaded by MLX — hf_base_model_id is required for those.
    is_hf_id = "/" in base_model_id and not base_model_id.startswith(("/", "."))
    is_local_dir = os.path.isdir(os.path.abspath(base_model_id))
    is_gguf = base_model_id.endswith(".gguf")

    if is_hf_id:
        mlx_model_id = base_model_id
        if not hf_base_model_id:
            hf_base_model_id = base_model_id
    elif is_local_dir and not is_gguf:
        # Local HF-format directory — resolve to absolute path for mlx_lm.load()
        mlx_model_id = os.path.abspath(base_model_id)
        if not hf_base_model_id:
            hf_base_model_id = mlx_model_id
    else:
        # GGUF or other non-HF format — need the HF model ID
        if not hf_base_model_id:
            print(
                "MLX training requires a HuggingFace model ID or a local HF-format directory.\n"
                "GGUF models need hf_base_model_id in params or "
                "JARVIS_ADAPTER_HF_BASE_MODEL_ID env var.\n"
                f"Example: JARVIS_ADAPTER_HF_BASE_MODEL_ID='NousResearch/Hermes-3-Llama-3.1-8B'",
                flush=True,
            )
            return 2
        mlx_model_id = hf_base_model_id

    # Read training hyperparameters (same env vars as train_adapter.py)
    max_seq_len = int(_get_param(params, "max_seq_len", os.getenv("JARVIS_ADAPTER_MAX_SEQ_LEN", "2048")))
    batch_size = int(_get_param(params, "batch_size", os.getenv("JARVIS_ADAPTER_BATCH_SIZE", "1")))
    grad_accum = int(_get_param(params, "grad_accum", os.getenv("JARVIS_ADAPTER_GRAD_ACCUM", "4")))
    epochs = float(_get_param(params, "epochs", os.getenv("JARVIS_ADAPTER_EPOCHS", "1")))
    lr = float(_get_param(params, "learning_rate", os.getenv("JARVIS_ADAPTER_LEARNING_RATE", "2e-4")))
    lora_r = int(_get_param(params, "lora_r", os.getenv("JARVIS_ADAPTER_LORA_R", "16")))
    lora_alpha = int(_get_param(params, "lora_alpha", os.getenv("JARVIS_ADAPTER_LORA_ALPHA", "32")))
    lora_dropout = float(_get_param(params, "lora_dropout", os.getenv("JARVIS_ADAPTER_LORA_DROPOUT", "0.05")))
    target_modules = _get_param(
        params,
        "lora_target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    print(f"MLX LoRA training: {mlx_model_id}", flush=True)
    print(f"  Examples: {len(examples)}, Epochs: {epochs}, Batch: {batch_size}", flush=True)
    print(f"  LoRA r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}", flush=True)
    print(f"  Max seq len: {max_seq_len}, Grad accum: {grad_accum}, LR: {lr}", flush=True)

    # ── Import MLX dependencies ──────────────────────────────────────────
    try:
        import mlx.optimizers
        from mlx.utils import tree_flatten
        from mlx_lm.tuner.trainer import TrainingArgs, train
        from mlx_lm.tuner.utils import linear_to_lora_layers
        from mlx_lm.utils import load
    except ImportError as e:
        print(f"MLX dependencies not available: {e}", flush=True)
        print("Install with: pip install mlx-lm", flush=True)
        return 2

    # ── Load model and tokenizer ─────────────────────────────────────────
    print(f"Loading model: {mlx_model_id}...", flush=True)
    model, tokenizer = load(mlx_model_id)

    # ── Apply LoRA layers ────────────────────────────────────────────────
    # Build LoRA keys from target_modules — MLX uses dotted paths (e.g. "self_attn.q_proj")
    lora_keys: List[str] = []
    for m in target_modules:
        if "." in m:
            # Already a full path like "self_attn.q_proj"
            lora_keys.append(m)
        elif m in ("q_proj", "k_proj", "v_proj", "o_proj"):
            lora_keys.append(f"self_attn.{m}")
        elif m in ("gate_proj", "up_proj", "down_proj"):
            lora_keys.append(f"mlp.{m}")
        else:
            lora_keys.append(m)

    # Fallback: standard set for Llama-style models
    if not lora_keys:
        lora_keys = [
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
        ]

    lora_config = {
        "rank": lora_r,
        "scale": lora_alpha / lora_r,  # MLX uses scale = alpha/rank
        "dropout": lora_dropout,
        "keys": lora_keys,
    }
    num_lora_layers = -1  # All transformer layers

    print(f"  LoRA config: num_layers={num_lora_layers}, {lora_config}", flush=True)

    # Freeze base model, then apply LoRA (LoRA layers are unfrozen by default)
    model.freeze()
    linear_to_lora_layers(model, num_lora_layers, lora_config)

    # Count trainable parameters
    all_params = tree_flatten(model.parameters())
    trainable_params = tree_flatten(model.trainable_parameters())
    total_count = sum(p.size for _, p in all_params)
    trainable_count = sum(p.size for _, p in trainable_params)
    if total_count > 0:
        print(f"  Trainable: {trainable_count:,} / {total_count:,} ({100 * trainable_count / total_count:.2f}%)", flush=True)

    # ── Build dataset ────────────────────────────────────────────────────
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    dataset = TokenizedDataset(tokenizer, examples, max_tokens=max_seq_len)

    # ── Configure training ───────────────────────────────────────────────
    # MLX uses iteration-based training; convert from epoch-based
    num_examples = len(examples)
    iters = math.ceil(epochs * num_examples / batch_size)
    iters = max(iters, 1)

    adapter_file = str(out_path / "adapters.safetensors")
    training_args = TrainingArgs(
        batch_size=batch_size,
        iters=iters,
        val_batches=0,
        steps_per_report=10,
        steps_per_save=iters + 1,  # No intermediate checkpoints; final save handled by trainer
        max_seq_length=max_seq_len,
        grad_checkpoint=True,
        adapter_file=adapter_file,
        grad_accumulation_steps=grad_accum,
    )

    # Cosine LR schedule with linear warmup (matches Unsloth/HF Trainer defaults).
    # Constant LR causes gradient explosions and oversized adapter weights.
    warmup_steps = max(1, int(0.03 * iters))  # 3% warmup (HF Trainer default)
    warmup_schedule = mlx.optimizers.linear_schedule(init=1e-6, end=lr, steps=warmup_steps)
    cosine_schedule = mlx.optimizers.cosine_decay(init=lr, decay_steps=iters - warmup_steps, end=lr * 0.05)
    lr_schedule = mlx.optimizers.join_schedules(
        [warmup_schedule, cosine_schedule], boundaries=[warmup_steps]
    )
    optimizer = mlx.optimizers.Adam(learning_rate=lr_schedule)

    print(f"  Iterations: {iters} (from {epochs} epochs x {num_examples} examples / {batch_size} batch)", flush=True)
    print(f"  LR schedule: warmup {warmup_steps} steps → cosine decay {lr:.2e} → {lr * 0.05:.2e}", flush=True)
    print("Starting MLX training...", flush=True)

    # ── Train ────────────────────────────────────────────────────────────
    model.train()
    train(
        model=model,
        optimizer=optimizer,
        train_dataset=dataset,
        val_dataset=None,
        args=training_args,
    )

    # ── Save MLX adapter config ──────────────────────────────────────────
    # Format matches what mlx_lm.tuner.utils.load_adapters() expects:
    #   config.num_layers and config.lora_parameters (dict with rank/scale/dropout/keys)
    mlx_config = {
        "num_layers": num_lora_layers,
        "fine_tune_type": "lora",
        "lora_parameters": {
            "rank": lora_r,
            "scale": lora_alpha / lora_r,
            "dropout": lora_dropout,
            "keys": lora_keys,
        },
    }
    mlx_config_path = out_path / "adapter_config.json"
    mlx_config_path.write_text(json.dumps(mlx_config, indent=2), encoding="utf-8")
    print(f"MLX adapter saved: {out_path}", flush=True)

    # ── Convert to PEFT format for GGUF compatibility ────────────────────
    peft_dir = out_path / "peft"
    _convert_mlx_to_peft(
        mlx_adapter_path=out_path,
        peft_output_dir=peft_dir,
        hf_base_model_id=hf_base_model_id,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[m.split(".")[-1] for m in lora_keys],
    )

    # ── GGUF conversion ──────────────────────────────────────────────────
    # The GGUF converter (convert_lora_to_gguf.py) needs a local directory path
    # for --base, not a HuggingFace model ID. Resolve the HF cache path if needed.
    gguf_base_id = mlx_model_id if os.path.isdir(mlx_model_id) else hf_base_model_id
    if not os.path.isdir(gguf_base_id):
        # Try to resolve HF cache snapshot path
        try:
            from huggingface_hub import snapshot_download
            gguf_base_id = snapshot_download(gguf_base_id, local_files_only=True)
            print(f"  Resolved HF cache path for GGUF: {gguf_base_id}", flush=True)
        except Exception as exc:
            print(f"  Could not resolve local HF cache path: {exc}", flush=True)
    # Always attempt GGUF conversion — the adapter is consumed as GGUF by llama-cpp-python.
    gguf_out_dir = out_path / "gguf"
    _run_gguf_conversion(peft_dir, gguf_out_dir, gguf_base_id)

    # ── Cleanup ──────────────────────────────────────────────────────────
    del model
    del tokenizer
    del optimizer
    gc.collect()

    print(f"MLX adapter training complete. Output: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
