#!/usr/bin/env python3
"""
Train a LoRA adapter for Jarvis date/time key extraction.

This script trains a lightweight adapter that extracts semantic date keys
(including relative time expressions) from natural language text.

Usage:
    python scripts/train_jarvis_adapter.py

    # With custom parameters
    python scripts/train_jarvis_adapter.py --epochs 3 --batch-size 4

Output:
    adapters/jarvis/
        adapter_config.json
        adapter_model.safetensors
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Train Jarvis date/time key extraction adapter")
    parser.add_argument(
        "--base-model",
        type=str,
        default=os.getenv("JARVIS_MODEL_NAME", ".models/llama-3.1-8b-instruct"),
        help="Base model ID (HuggingFace or local path)"
    )
    parser.add_argument(
        "--training-data",
        type=str,
        default="data/jarvis_training.jsonl",
        help="Path to training data JSONL file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="adapters/jarvis",
        help="Output directory for trained adapter"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing to save memory (default: True)"
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_false",
        dest="gradient_checkpointing",
        help="Disable gradient checkpointing"
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw_8bit",
        choices=["adamw_torch", "adamw_8bit", "paged_adamw_8bit", "paged_adamw_32bit"],
        help="Optimizer (adamw_8bit saves memory)"
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    return parser.parse_args()


def format_training_example(text: str, date_keys: list[str]) -> dict:
    """
    Format a training example for the model.
    
    The model learns to output JSON array of date keys.
    """
    # Format the prompt
    system_prompt = """You are a date/time extraction assistant. Extract date and time references from the user's text and return them as a JSON array of semantic keys.

Rules:
- Return only the JSON array, nothing else
- Use standardized keys like: today, tomorrow, yesterday, tomorrow_morning, tonight, last_night, next_monday, this_weekend, at_3pm, etc.
- For relative time: flatten hours/minutes to in_N_minutes (e.g., "in 2 hours" → "in_120_minutes", "in half an hour" → "in_30_minutes")
- For relative days: use in_N_days (e.g., "in 3 days" → "in_3_days", "in a week" → "in_7_days")
- Return [] if no date/time references are found
- Return [] for ambiguous time expressions like "in a few minutes", "later", "in a bit"
- Return [] for durations and past references like "for 30 minutes", "2 hours ago"
- Multiple keys can be returned for composite expressions like "next Tuesday at 3pm" -> ["next_tuesday", "at_3pm"]"""

    user_message = f"Extract date keys from: \"{text}\""
    assistant_response = json.dumps(date_keys)
    
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_response}
        ]
    }


def load_training_data(path: str) -> list[dict]:
    """Load and format training data from JSONL file."""
    examples = []
    
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            formatted = format_training_example(data["text"], data["date_keys"])
            examples.append(formatted)
    
    return examples


def main():
    args = parse_args()
    
    print("=" * 60)
    print("JARVIS ADAPTER TRAINING")
    print("=" * 60)
    print(f"Base model: {args.base_model}")
    print(f"Training data: {args.training_data}")
    print(f"Output dir: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Grad accum: {args.grad_accum}")
    print(f"Optimizer: {args.optim}")
    print(f"LoRA rank: {args.lora_r}")
    print(f"LoRA alpha: {args.lora_alpha}")
    print(f"Gradient checkpointing: {args.gradient_checkpointing}")
    print("=" * 60)
    
    # Check training data exists
    training_data_path = Path(args.training_data)
    if not training_data_path.exists():
        print(f"❌ Training data not found: {training_data_path}")
        print("   Run: python scripts/generate_jarvis_training_data.py")
        sys.exit(1)
    
    # Import training libraries
    print("\n📦 Loading libraries...", flush=True)
    
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    
    # Check device
    if torch.cuda.is_available():
        device = "cuda"
        # Clear any existing CUDA cache
        torch.cuda.empty_cache()
        print(f"🎮 Using CUDA: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        free_mem = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9
        print(f"   Memory: {total_mem:.1f} GB total, {free_mem:.1f} GB free")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("🍎 Using MPS (Apple Silicon)")
    else:
        device = "cpu"
        print("💻 Using CPU (this will be slow)")
    
    # Load training data
    print("\n📂 Loading training data...", flush=True)
    raw_examples = load_training_data(args.training_data)
    print(f"   Loaded {len(raw_examples)} examples")
    
    # Create dataset
    dataset = Dataset.from_list(raw_examples)
    
    # Split into train/eval
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"   Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    # Load tokenizer
    print("\n🔤 Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Configure quantization for memory efficiency
    print("\n🔧 Configuring model...", flush=True)
    
    # Device map configuration
    # MPS requires all tensors on the same device — "auto" splits across meta/mps and breaks gradients
    if device == "mps":
        device_map = {"": "mps"}
    else:
        device_map_str = os.getenv("JARVIS_ADAPTER_TRAIN_DEVICE_MAP", "0")
        if device_map_str == "auto":
            device_map = "auto"
        elif device_map_str.isdigit():
            device_map = {"": int(device_map_str)}
        else:
            device_map = {"": 0}

    print(f"   Device map: {device_map}")

    # Use 4-bit quantization if on CUDA and not disabled (e.g., for AWQ models)
    # Jarvis adapter training has its own env var, separate from general adapter training
    load_in_4bit = os.getenv("JARVIS_DATE_ADAPTER_TRAIN_LOAD_IN_4BIT", "true").lower() in ("true", "1", "yes")
    if device == "cuda" and load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("   Using 4-bit quantization (BitsAndBytes)")
    else:
        bnb_config = None
        if device == "cuda" and not load_in_4bit:
            print("   Skipping BitsAndBytes quantization (JARVIS_DATE_ADAPTER_TRAIN_LOAD_IN_4BIT=false)")

    # Choose dtype: bfloat16 on CUDA, float16 on MPS, float32 on CPU
    if device == "cuda":
        torch_dtype = torch.bfloat16
    elif device == "mps":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    print(f"   Torch dtype: {torch_dtype}")

    # Load model
    print("\n🧠 Loading base model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    
    if bnb_config:
        model = prepare_model_for_kbit_training(model)

    # Enable gradient checkpointing to save memory (critical for 8B+ models)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("   ✓ Gradient checkpointing enabled")

    # Configure LoRA
    print("\n🔩 Configuring LoRA...", flush=True)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Report memory usage after model loading
    if device == "cuda":
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   GPU Memory: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved, {total:.1f} GB total")
        print(f"   Available for training: ~{total - reserved:.1f} GB")
    
    # Training arguments
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        optim=args.optim,
        learning_rate=args.learning_rate,
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
        bf16=device == "cuda",
        fp16=device == "mps",
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_pin_memory=device != "mps",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
        max_length=args.max_length,
        packing=False,
    )
    
    # Format function for chat template
    def formatting_func(example):
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
    
    # Create trainer
    print("\n🏋️ Starting training...", flush=True)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )
    
    # Train
    trainer.train()
    
    # Save the final adapter
    print("\n💾 Saving adapter...", flush=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    # Save training metadata
    metadata = {
        "base_model": args.base_model,
        "training_data": str(args.training_data),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "num_examples": len(raw_examples),
        "adapter_type": "jarvis_key_extraction",
    }
    
    with open(output_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Adapter saved to: {output_dir}")
    print("\nFiles:")
    for f in output_dir.iterdir():
        if f.is_file():
            size = f.stat().st_size / 1024  # KB
            print(f"   {f.name}: {size:.1f} KB")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nTo validate the adapter, run:")
    print(f"   python scripts/validate_jarvis_adapter.py --adapter-path {output_dir}")


if __name__ == "__main__":
    main()
