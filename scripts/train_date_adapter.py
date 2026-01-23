#!/usr/bin/env python3
"""
Train a LoRA adapter for date key extraction.

This script trains a lightweight adapter that extracts semantic date keys
from natural language text.

Usage:
    python scripts/train_date_adapter.py
    
    # With custom parameters
    python scripts/train_date_adapter.py --epochs 3 --batch-size 4
    
Output:
    adapters/date_keys/
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
    parser = argparse.ArgumentParser(description="Train date key extraction adapter")
    parser.add_argument(
        "--base-model",
        type=str,
        default=os.getenv("JARVIS_LIGHTWEIGHT_MODEL_NAME", ".models/qwen2.5-0.5b-instruct"),
        help="Base model ID (HuggingFace or local path)"
    )
    parser.add_argument(
        "--training-data",
        type=str,
        default="data/date_keys_training.jsonl",
        help="Path to training data JSONL file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="adapters/date_keys",
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
- Return [] if no date/time references are found
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
    print("DATE KEY ADAPTER TRAINING")
    print("=" * 60)
    print(f"Base model: {args.base_model}")
    print(f"Training data: {args.training_data}")
    print(f"Output dir: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"LoRA rank: {args.lora_r}")
    print(f"LoRA alpha: {args.lora_alpha}")
    print("=" * 60)
    
    # Check training data exists
    training_data_path = Path(args.training_data)
    if not training_data_path.exists():
        print(f"❌ Training data not found: {training_data_path}")
        print("   Run: python scripts/generate_date_training_data.py")
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
        print(f"🎮 Using CUDA: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
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
    device_map_str = os.getenv("JARVIS_ADAPTER_TRAIN_DEVICE_MAP", "0")
    if device_map_str == "auto":
        device_map = "auto"
    elif device_map_str.isdigit():
        device_map = {"": int(device_map_str)}
    else:
        device_map = {"": 0}
    
    print(f"   Device map: {device_map}")
    
    # Use 4-bit quantization if on CUDA for memory efficiency
    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("   Using 4-bit quantization")
    else:
        bnb_config = None
    
    # Load model
    print("\n🧠 Loading base model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    
    if bnb_config:
        model = prepare_model_for_kbit_training(model)
    
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
    
    # Training arguments
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
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
        fp16=False,
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
        "adapter_type": "date_key_extraction",
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
    print(f"   python scripts/validate_date_adapter.py --adapter-path {output_dir}")


if __name__ == "__main__":
    main()
