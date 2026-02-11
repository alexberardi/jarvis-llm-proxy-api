#!/usr/bin/env python3
"""Convert a HuggingFace model to quantized MLX format.

Takes a merged HF model directory (e.g. from merge_adapter.py) and produces
a quantized MLX model directory suitable for mlx-lm inference.

Usage:
    python scripts/convert_to_mlx.py \
        --model .models/llama-3.2-3b-instruct-date-merged \
        --output .models/llama-3.2-3b-instruct-date-merged-mlx-4bit

    # 8-bit quantization:
    python scripts/convert_to_mlx.py \
        --model .models/llama-3.2-3b-instruct-date-merged \
        --output .models/llama-3.2-3b-instruct-date-merged-mlx-8bit \
        --bits 8

    # No quantization (full precision MLX):
    python scripts/convert_to_mlx.py \
        --model .models/llama-3.2-3b-instruct-date-merged \
        --output .models/llama-3.2-3b-instruct-date-merged-mlx \
        --bits 0

    # Dry run:
    python scripts/convert_to_mlx.py \
        --model .models/llama-3.2-3b-instruct-date-merged \
        --output .models/llama-3.2-3b-instruct-date-merged-mlx-4bit \
        --dry-run

Requirements:
    pip install mlx-lm
"""

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a HuggingFace model to quantized MLX format"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to HuggingFace model directory or HuggingFace model ID",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for the MLX model",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[0, 4, 8],
        help="Quantization bits: 4 (default), 8, or 0 for no quantization",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Quantization group size (default: 64)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and show plan without converting",
    )
    return parser.parse_args()


def validate_model_path(model_path: str) -> None:
    """Validate model path if it's a local directory."""
    path = Path(model_path)
    if path.exists():
        # It's a local path — validate it has model files
        if not path.is_dir():
            print(f"Error: model path is not a directory: {path}", file=sys.stderr)
            sys.exit(1)
        has_safetensors = any(path.glob("*.safetensors"))
        has_bin = any(path.glob("*.bin"))
        if not has_safetensors and not has_bin:
            print(
                f"Error: no model weight files (.safetensors or .bin) found in {path}",
                file=sys.stderr,
            )
            sys.exit(1)
    # If path doesn't exist, assume it's a HuggingFace model ID (mlx_lm.convert handles download)


def validate_output_path(output_path: Path) -> None:
    if output_path.exists() and any(output_path.iterdir()):
        print(
            f"Error: output directory already exists and is not empty: {output_path}",
            file=sys.stderr,
        )
        print("Remove it or choose a different output path.", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    args = parse_args()

    output_path = Path(args.output)
    quantize: bool = args.bits > 0

    # Validate inputs
    validate_model_path(args.model)
    validate_output_path(output_path)

    print("=" * 60)
    print("MLX CONVERSION")
    print("=" * 60)
    print(f"Model:       {args.model}")
    print(f"Output:      {output_path}")
    if quantize:
        print(f"Quantize:    {args.bits}-bit (group size {args.group_size})")
    else:
        print("Quantize:    disabled (full precision)")
    print("=" * 60)
    print()

    if args.dry_run:
        print("[dry-run] Validation passed. Would convert model to MLX format.")
        return

    # Import mlx_lm after validation (heavy dependency, macOS-only)
    try:
        from mlx_lm import convert
    except ImportError:
        print("Error: mlx-lm not installed. Run: pip install mlx-lm", file=sys.stderr)
        sys.exit(1)

    print("Converting to MLX format...")
    convert_kwargs: dict = {
        "hf_path": args.model,
        "mlx_path": str(output_path),
    }
    if quantize:
        convert_kwargs["quantize"] = True
        convert_kwargs["q_bits"] = args.bits
        convert_kwargs["q_group_size"] = args.group_size
    else:
        convert_kwargs["quantize"] = False

    convert(**convert_kwargs)

    total_size = sum(
        f.stat().st_size for f in output_path.rglob("*") if f.is_file()
    ) / (1024**3)

    print()
    print("=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Output: {output_path} ({total_size:.2f} GiB)")
    print()
    print("To use the MLX model:")
    print(f"  1. Set JARVIS_MODEL_NAME={output_path}")
    print("  2. Set JARVIS_MODEL_BACKEND=MLX")
    print("  3. Restart with ./run.sh")


if __name__ == "__main__":
    main()
