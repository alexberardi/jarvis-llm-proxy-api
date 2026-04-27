#!/usr/bin/env python3
"""
Train a date key extraction LoRA adapter for a single model.

Runs ON the RunPod pod. Downloads the HF model, trains a PEFT LoRA adapter,
converts to GGUF format, and runs a quick eval.

Usage:
    python remote_train.py --model Qwen/Qwen2.5-3B-Instruct --output /workspace/adapters/qwen2.5-3b-instruct
    python remote_train.py --model Qwen/Qwen3-14B --output /workspace/adapters/qwen3-14b --load-in-4bit
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Enable hf_transfer for faster, more reliable HuggingFace downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


# ---------------------------------------------------------------------------
# Model registry — maps slug → HF model ID and training settings
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, dict] = {
    "qwen2.5-3b-instruct": {
        "hf_model_id": "Qwen/Qwen2.5-3B-Instruct",
        "load_in_4bit": False,
        "batch_size": 8,
    },
    "qwen2.5-7b-instruct": {
        "hf_model_id": "Qwen/Qwen2.5-7B-Instruct",
        "load_in_4bit": False,
        "batch_size": 4,
    },
    "qwen3-8b": {
        "hf_model_id": "Qwen/Qwen3-8B",
        "load_in_4bit": False,
        "batch_size": 4,
    },
    "qwen3-14b": {
        "hf_model_id": "Qwen/Qwen3-14B",
        "load_in_4bit": True,
        "batch_size": 4,
    },
    "qwen3-32b": {
        "hf_model_id": "Qwen/Qwen3-32B",
        "load_in_4bit": True,
        "batch_size": 2,
    },
    "hermes-3-llama-3.1-8b": {
        "hf_model_id": "NousResearch/Hermes-3-Llama-3.1-8B",
        "load_in_4bit": False,
        "batch_size": 4,
    },
    "llama-3.1-8b-instruct": {
        "hf_model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "load_in_4bit": False,
        "batch_size": 4,
    },
    "llama-3.2-3b-instruct": {
        "hf_model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "load_in_4bit": False,
        "batch_size": 8,
    },
    "gemma-2-9b-it": {
        "hf_model_id": "google/gemma-2-9b-it",
        "load_in_4bit": False,
        "batch_size": 4,
    },
    "gemma-3-12b-it": {
        "hf_model_id": "google/gemma-3-12b-it",
        "load_in_4bit": True,
        "batch_size": 4,
    },
}

# System prompt — must match services/date_keys.py exactly
SYSTEM_PROMPT = """You are a date/time extraction assistant. Extract date and time references from the user's text and return them as a JSON array of semantic keys.

Rules:
- Return only the JSON array, nothing else
- Use standardized keys like: today, tomorrow, yesterday, tomorrow_morning, tonight, last_night, next_monday, this_weekend, at_3pm, etc.
- For relative time: flatten hours/minutes to in_N_minutes (e.g., "in 2 hours" → "in_120_minutes", "in half an hour" → "in_30_minutes")
- For relative days: use in_N_days (e.g., "in 3 days" → "in_3_days", "in a week" → "in_7_days")
- Return [] if no date/time references are found
- Return [] for ambiguous time expressions like "in a few minutes", "later", "in a bit"
- Return [] for durations and past references like "for 30 minutes", "2 hours ago"
- Multiple keys can be returned for composite expressions like "next Tuesday at 3pm" -> ["next_tuesday", "at_3pm"]"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train date key adapter for a single model")
    p.add_argument("--model", required=True, help="HF model ID or slug from MODEL_REGISTRY")
    p.add_argument("--output", required=True, help="Output directory for adapter files")
    p.add_argument("--training-data", default="/workspace/data/jarvis_training.jsonl",
                   help="Path to training JSONL")
    p.add_argument("--epochs", type=int, default=3, help="Training epochs")
    p.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    p.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    p.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    p.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    p.add_argument("--load-in-4bit", action="store_true", help="Use QLoRA 4-bit (auto for 14B+)")
    p.add_argument("--eval-data", default=None, help="Optional eval JSONL (splits 10% from train if absent)")
    p.add_argument("--skip-gguf", action="store_true", help="Skip GGUF conversion")
    p.add_argument("--hf-token", default=None, help="HuggingFace token for gated models")
    return p.parse_args()


def resolve_model(model_arg: str) -> tuple[str, dict]:
    """Resolve model arg to (hf_model_id, settings)."""
    # Check if it's a slug
    if model_arg in MODEL_REGISTRY:
        entry = MODEL_REGISTRY[model_arg]
        return entry["hf_model_id"], entry

    # Check if it's a known HF model ID
    for slug, entry in MODEL_REGISTRY.items():
        if entry["hf_model_id"] == model_arg:
            return model_arg, entry

    # Unknown model — use defaults
    return model_arg, {"hf_model_id": model_arg, "load_in_4bit": False, "batch_size": 4}


def load_training_data(path: str, hf_model_id: str) -> list[dict]:
    """Load raw training data and format as chat messages.

    Adapts format per model:
    - Gemma models: no system role (merged into user message)
    - Qwen3 models: /no_think suffix to suppress thinking mode
    """
    is_gemma = "gemma" in hf_model_id.lower()
    is_qwen3 = "qwen3" in hf_model_id.lower()

    examples = []
    with open(path) as f:
        for line in f:
            row = json.loads(line.strip())
            text = row["text"]
            date_keys = row["date_keys"]

            user_content = f'Extract date keys from: "{text}"'
            if is_qwen3:
                user_content += " /no_think"

            if is_gemma:
                # Gemma doesn't support system role — prepend to user message
                user_content = SYSTEM_PROMPT + "\n\n" + user_content
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": json.dumps(date_keys)},
                ]
            else:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": json.dumps(date_keys)},
                ]

            examples.append({"messages": messages})
    return examples


def train_adapter(
    hf_model_id: str,
    output_dir: str,
    training_data: str,
    epochs: int,
    batch_size: int,
    lora_r: int,
    lora_alpha: int,
    learning_rate: float,
    max_length: int,
    load_in_4bit: bool,
    hf_token: str | None,
) -> Path:
    """Train a PEFT LoRA adapter using SFTTrainer."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\n📂 Loading training data from {training_data}...")
    examples = load_training_data(training_data, hf_model_id)
    dataset = Dataset.from_list(examples)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"   Train: {len(split['train'])}, Eval: {len(split['test'])}")

    # Tokenizer
    print(f"\n🔤 Loading tokenizer for {hf_model_id}...")
    token_kwargs = {"token": hf_token} if hf_token else {}
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True, **token_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Quantization
    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("   Using QLoRA 4-bit quantization")

    # Load model
    print(f"\n🧠 Loading model {hf_model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        **token_kwargs,
    )
    if bnb_config:
        model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    # Report GPU memory
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   GPU: {alloc:.1f}/{total:.1f} GB used after model load")

    # LoRA
    print(f"\n🔩 Configuring LoRA (r={lora_r}, alpha={lora_alpha})...")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training config
    training_args = SFTConfig(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        optim="adamw_8bit",
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
        max_length=max_length,
        packing=False,
    )

    def formatting_func(example):
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )

    # Train
    print(f"\n🏋️ Training for {epochs} epochs...")
    t0 = time.time()
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )
    trainer.train()
    train_seconds = time.time() - t0
    print(f"   Training complete in {train_seconds:.0f}s")

    # Save
    print(f"\n💾 Saving adapter to {out}...")
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    # Cleanup GPU
    del trainer, model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out


def _find_cached_model_path(hf_model_id: str) -> str:
    """Find the local HuggingFace cache path for a model.

    The converter needs a local directory, not a HF model ID.
    HF caches models at ~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<hash>/
    """
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    # "Qwen/Qwen2.5-3B-Instruct" -> "models--Qwen--Qwen2.5-3B-Instruct"
    model_dir_name = "models--" + hf_model_id.replace("/", "--")
    snapshots_dir = cache_dir / model_dir_name / "snapshots"
    if snapshots_dir.is_dir():
        # Use the first (usually only) snapshot
        for snapshot in sorted(snapshots_dir.iterdir()):
            if snapshot.is_dir():
                return str(snapshot)
    # Fallback: return the HF ID and hope for the best
    return hf_model_id


def convert_to_gguf(adapter_dir: Path, hf_model_id: str) -> Path | None:
    """Convert PEFT adapter to GGUF LoRA format."""
    gguf_dir = adapter_dir / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    gguf_file = gguf_dir / "adapter.gguf"

    # Try to find convert_lora_to_gguf.py
    # On the pod, we'll clone llama.cpp for this
    converter = None
    for candidate in [
        Path("/workspace/llama.cpp/convert_lora_to_gguf.py"),
        Path(__file__).parent.parent / "vendor" / "llama.cpp" / "convert_lora_to_gguf.py",
    ]:
        if candidate.is_file():
            converter = candidate
            break

    if not converter:
        # Clone full llama.cpp (needs convert_hf_to_gguf.py alongside the lora converter)
        print("   Cloning llama.cpp for GGUF converter...")
        try:
            subprocess.run(
                ["git", "clone", "--depth=1",
                 "https://github.com/ggerganov/llama.cpp.git", "/workspace/llama.cpp"],
                check=True, capture_output=True, text=True, timeout=300,
            )
            converter = Path("/workspace/llama.cpp/convert_lora_to_gguf.py")
            # Install gguf Python package
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "/workspace/llama.cpp/gguf-py"],
                check=True, capture_output=True, text=True, timeout=120,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"   ⚠️ Failed to get GGUF converter: {e}")
            return None

    if not converter or not converter.is_file():
        print("   ⚠️ GGUF converter not found, skipping conversion")
        return None

    # Resolve the base model to a local cache path (converter needs a directory, not HF ID)
    base_path = _find_cached_model_path(hf_model_id)
    print(f"   🔄 Converting to GGUF: {gguf_file}")
    print(f"      Base model path: {base_path}")
    cmd = [
        sys.executable, str(converter),
        str(adapter_dir),
        "--base", base_path,
        "--outfile", str(gguf_file),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
        size_mb = gguf_file.stat().st_size / (1024 * 1024)
        print(f"   ✅ GGUF adapter: {gguf_file} ({size_mb:.1f} MB)")
        return gguf_file
    except subprocess.CalledProcessError as e:
        print(f"   ⚠️ GGUF conversion failed: {e.stderr[-500:] if e.stderr else e}")
        return None
    except subprocess.TimeoutExpired:
        print("   ⚠️ GGUF conversion timed out")
        return None


def run_eval(adapter_dir: Path, training_data: str, hf_model_id: str) -> dict:
    """Run a quick eval on the trained adapter."""
    print("\n📊 Running evaluation...")

    # Load a sample of test data
    all_examples = []
    with open(training_data) as f:
        for line in f:
            all_examples.append(json.loads(line.strip()))

    # Use last 200 examples as eval (or 10% if smaller)
    eval_size = min(200, len(all_examples) // 10)
    eval_examples = all_examples[-eval_size:]

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        model.eval()

        correct = 0
        total = 0

        for ex in eval_examples:
            text = ex["text"]
            expected = ex["date_keys"]

            # Append /no_think for Qwen3 models to suppress thinking mode
            user_content = f'Extract date keys from: "{text}"'
            if "qwen3" in hf_model_id.lower():
                user_content += " /no_think"

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=64, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()

            # Strip <think>...</think> blocks (Qwen3 thinking mode output)
            import re
            response = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL).strip()

            try:
                predicted = json.loads(response)
                if isinstance(predicted, list) and sorted(predicted) == sorted(expected):
                    correct += 1
            except (json.JSONDecodeError, TypeError):
                pass
            total += 1

        accuracy = correct / total if total > 0 else 0.0
        print(f"   Eval accuracy: {correct}/{total} = {accuracy:.1%}")

        # Cleanup
        del model, base_model
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        return {"accuracy": accuracy, "correct": correct, "total": total}

    except Exception as e:
        print(f"   ⚠️ Eval failed: {e}")
        return {"accuracy": 0.0, "error": str(e)}


def main() -> int:
    args = parse_args()
    hf_model_id, settings = resolve_model(args.model)
    load_in_4bit = args.load_in_4bit or settings.get("load_in_4bit", False)
    batch_size = settings.get("batch_size", 4)

    slug = args.model if args.model in MODEL_REGISTRY else hf_model_id.split("/")[-1].lower()

    print("=" * 60)
    print(f"TRAINING DATE KEY ADAPTER: {slug}")
    print("=" * 60)
    print(f"  HF model:     {hf_model_id}")
    print(f"  Output:       {args.output}")
    print(f"  QLoRA 4-bit:  {load_in_4bit}")
    print(f"  Batch size:   {batch_size}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  LoRA rank:    {args.lora_r}")
    print(f"  Data:         {args.training_data}")

    # Set HF token if provided
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    t0 = time.time()

    # 1. Train
    adapter_dir = train_adapter(
        hf_model_id=hf_model_id,
        output_dir=args.output,
        training_data=args.training_data,
        epochs=args.epochs,
        batch_size=batch_size,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        load_in_4bit=load_in_4bit,
        hf_token=args.hf_token,
    )

    # 2. Convert to GGUF
    gguf_path = None
    if not args.skip_gguf:
        gguf_path = convert_to_gguf(adapter_dir, hf_model_id)

    # 3. Eval
    eval_result = run_eval(adapter_dir, args.training_data, hf_model_id)

    # 4. Save metadata
    total_seconds = time.time() - t0
    metadata = {
        "slug": slug,
        "hf_model_id": hf_model_id,
        "adapter_type": "date_key_extraction",
        "training_data": str(args.training_data),
        "epochs": args.epochs,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "learning_rate": args.learning_rate,
        "load_in_4bit": load_in_4bit,
        "batch_size": batch_size,
        "max_length": args.max_length,
        "eval": eval_result,
        "gguf_converted": gguf_path is not None,
        "train_duration_seconds": round(total_seconds, 1),
    }
    metadata_path = adapter_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    # Summary
    print("\n" + "=" * 60)
    print(f"COMPLETE: {slug}")
    print(f"  Time:     {total_seconds:.0f}s")
    print(f"  Accuracy: {eval_result.get('accuracy', 'N/A')}")
    print(f"  GGUF:     {'Yes' if gguf_path else 'No'}")
    print(f"  Output:   {adapter_dir}")
    print("=" * 60)

    # List output files
    for f in sorted(adapter_dir.rglob("*")):
        if f.is_file() and "checkpoint" not in str(f):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.relative_to(adapter_dir)}: {size_mb:.1f} MB")

    # 5. Cleanup: purge HF model cache and checkpoints to free disk for next model
    _cleanup_after_training(hf_model_id, adapter_dir)

    return 0


def _cleanup_after_training(hf_model_id: str, adapter_dir: Path) -> None:
    """Purge the HF model cache and training checkpoints to free disk space."""
    import shutil

    # Remove HF cached model weights (the big files)
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir_name = "models--" + hf_model_id.replace("/", "--")
    model_cache = cache_dir / model_dir_name
    if model_cache.is_dir():
        size_gb = sum(f.stat().st_size for f in model_cache.rglob("*") if f.is_file()) / (1024**3)
        shutil.rmtree(model_cache)
        print(f"\n🧹 Purged HF cache for {hf_model_id} ({size_gb:.1f} GB freed)")

    # Remove training checkpoints (only the final adapter matters)
    checkpoints_dir = adapter_dir / "checkpoints"
    if checkpoints_dir.is_dir():
        size_gb = sum(f.stat().st_size for f in checkpoints_dir.rglob("*") if f.is_file()) / (1024**3)
        shutil.rmtree(checkpoints_dir)
        print(f"🧹 Purged checkpoints ({size_gb:.1f} GB freed)")


if __name__ == "__main__":
    raise SystemExit(main())
