"""Merge a LoRA adapter into the base model weights.

Produces a new standalone model with the adapter knowledge baked in.
The merged model can then be loaded by vLLM as the base, allowing other
LoRA adapters to be applied on top at runtime without conflict.

Usage:
    python scripts/merge_adapter.py \
        --base-model .models/llama-3.2-3b-instruct \
        --adapter adapters/date_keys \
        --output .models/llama-3.2-3b-instruct-date-merged

    # With HuggingFace model ID (downloads if not local):
    python scripts/merge_adapter.py \
        --base-model meta-llama/Llama-3.2-3B-Instruct \
        --adapter adapters/date_keys \
        --output .models/llama-3.2-3b-instruct-date-merged

    # Dry run (validates paths, shows what would happen):
    python scripts/merge_adapter.py \
        --base-model .models/llama-3.2-3b-instruct \
        --adapter adapters/date_keys \
        --output .models/llama-3.2-3b-instruct-date-merged \
        --dry-run
"""

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter into base model weights"
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Path to base model directory or HuggingFace model ID",
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="Path to LoRA adapter directory (must contain adapter_config.json)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for the merged model",
    )
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for loading the base model (default: bfloat16)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and show plan without merging",
    )
    return parser.parse_args()


def validate_adapter_path(adapter_path: Path) -> None:
    if not adapter_path.is_dir():
        print(f"Error: adapter directory not found: {adapter_path}", file=sys.stderr)
        sys.exit(1)
    config_file = adapter_path / "adapter_config.json"
    if not config_file.is_file():
        print(
            f"Error: adapter_config.json not found in {adapter_path}", file=sys.stderr
        )
        sys.exit(1)


def validate_output_path(output_path: Path) -> None:
    if output_path.exists() and any(output_path.iterdir()):
        print(f"Error: output directory already exists and is not empty: {output_path}", file=sys.stderr)
        print("Remove it or choose a different output path.", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    args = parse_args()

    adapter_path = Path(args.adapter)
    output_path = Path(args.output)

    validate_adapter_path(adapter_path)
    validate_output_path(output_path)

    print(f"Base model:  {args.base_model}")
    print(f"Adapter:     {adapter_path}")
    print(f"Output:      {output_path}")
    print(f"Torch dtype: {args.torch_dtype}")
    print()

    if args.dry_run:
        print("[dry-run] Validation passed. Would merge adapter into base model.")
        return

    # Import heavy dependencies only after validation
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.torch_dtype]

    # Step 1: Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    print(f"  Base model loaded: {base_model.config._name_or_path}")
    print(f"  Parameters: {base_model.num_parameters():,}")

    # Step 2: Load adapter on top
    print("Loading adapter...")
    model_with_adapter = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        torch_dtype=torch_dtype,
    )
    print(f"  Adapter loaded from: {adapter_path}")

    # Step 3: Merge and unload
    print("Merging adapter weights into base model...")
    merged_model = model_with_adapter.merge_and_unload()
    print("  Merge complete.")

    # Step 4: Save
    print(f"Saving merged model to: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"  Saved. Total size: {sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / (1024**3):.2f} GiB")

    print()
    print("Done! To use the merged model:")
    print(f"  1. Set JARVIS_MODEL_NAME={output_path}")
    print(f"  2. Ensure JARVIS_VLLM_QUANTIZATION is empty (unquantized model)")
    print(f"  3. Restart with ./run.sh")
    print()
    print("User LoRA adapters can still be applied at runtime on top of this merged model.")


if __name__ == "__main__":
    main()
