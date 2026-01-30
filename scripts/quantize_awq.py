#!/usr/bin/env python3
"""Quantize a model to AWQ format for efficient inference.

Takes a full-precision model (or merged model) and produces an AWQ-quantized
version suitable for vLLM inference.

Usage:
    python scripts/quantize_awq.py \
        --model .models/llama-3.1-8b-date-merged \
        --output .models/llama-3.1-8b-date-merged-awq

    # With custom calibration samples:
    python scripts/quantize_awq.py \
        --model .models/llama-3.1-8b-date-merged \
        --output .models/llama-3.1-8b-date-merged-awq \
        --num-samples 128

Requirements:
    pip install llmcompressor datasets
"""

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize a model to AWQ format using llm-compressor"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to model directory or HuggingFace model ID",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for the quantized model",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=256,
        help="Number of calibration samples (default: 256)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Calibration sequence length (default: 512)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and show plan without quantizing",
    )
    return parser.parse_args()


def validate_output_path(output_path: Path) -> None:
    if output_path.exists() and any(output_path.iterdir()):
        print(f"Error: output directory already exists and is not empty: {output_path}", file=sys.stderr)
        print("Remove it or choose a different output path.", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    args = parse_args()

    output_path = Path(args.output)
    validate_output_path(output_path)

    print("=" * 60)
    print("AWQ QUANTIZATION (llm-compressor)")
    print("=" * 60)
    print(f"Model:            {args.model}")
    print(f"Output:           {output_path}")
    print(f"Num samples:      {args.num_samples}")
    print(f"Sequence length:  {args.seq_len}")
    print("=" * 60)
    print()

    if args.dry_run:
        print("[dry-run] Validation passed. Would quantize model to AWQ.")
        return

    # Import heavy dependencies only after validation
    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.awq import AWQModifier
    except ImportError:
        print("Error: llmcompressor not installed. Run: pip install llmcompressor", file=sys.stderr)
        sys.exit(1)

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load model - use dtype="auto", NOT device_map="auto"
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype="auto",
        trust_remote_code=True,
    )

    # Load and prepare calibration dataset
    print("Loading calibration dataset...")
    DATASET_ID = "HuggingFaceH4/ultrachat_200k"
    DATASET_SPLIT = "train_sft"

    ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{args.num_samples}]")
    ds = ds.shuffle(seed=42)

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
            )
        }

    ds = ds.map(preprocess)
    print(f"  Prepared {len(ds)} calibration samples")

    # Configure AWQ recipe
    # W4A16_ASYM = 4-bit weights, 16-bit activations, asymmetric quantization
    recipe = [
        AWQModifier(
            ignore=["lm_head"],
            scheme="W4A16_ASYM",
            targets=["Linear"],
            duo_scaling="both",
        ),
    ]

    print()
    print("Recipe configured:")
    print("  Scheme: W4A16_ASYM (4-bit weights, 16-bit activations, asymmetric)")
    print("  Ignored layers: lm_head")
    print()

    # Run quantization and save
    print("Running AWQ quantization...")
    output_path.mkdir(parents=True, exist_ok=True)
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.seq_len,
        num_calibration_samples=args.num_samples,
        output_dir=str(output_path),
        save_compressed=True,
    )

    # Save tokenizer separately (oneshot only saves model)
    tokenizer.save_pretrained(str(output_path))

    total_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / (1024**3)
    print(f"Saved. Total size: {total_size:.2f} GiB")

    print()
    print("=" * 60)
    print("QUANTIZATION COMPLETE")
    print("=" * 60)
    print()
    print("To use the quantized model:")
    print(f"  1. Set JARVIS_MODEL_NAME={output_path}")
    print(f"  2. Remove/unset JARVIS_VLLM_QUANTIZATION (vLLM auto-detects compressed-tensors)")
    print(f"  3. Restart with ./run.sh")


if __name__ == "__main__":
    main()
