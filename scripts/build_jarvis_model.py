#!/usr/bin/env python3
"""
Build a Jarvis model with date/time extraction baked in.

Full pipeline: generate training data → train adapter → merge → convert to GGUF.
Steps can be skipped if artifacts already exist.

Usage:
    # Full pipeline (train + merge + convert)
    python scripts/build_jarvis_model.py --base-model .models/Hermes-3-Llama-3.1-8B

    # Skip training (adapter already exists), just merge + convert
    python scripts/build_jarvis_model.py --base-model .models/Hermes-3-Llama-3.1-8B --skip-train

    # Mac (Apple Silicon)
    python scripts/build_jarvis_model.py --base-model .models/Hermes-3-Llama-3.1-8B --optim adamw_torch --batch-size 2

    # Quantized GGUF (Q4_K_M ~4.5 GiB instead of ~15 GiB f16)
    python scripts/build_jarvis_model.py --base-model .models/Hermes-3-Llama-3.1-8B --skip-train --gguf-quant Q4_K_M
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def run(cmd: list[str], description: str) -> bool:
    """Run a command, printing output in real time. Returns True on success."""
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"{'=' * 60}\n")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"\n❌ Failed: {description}")
        return False
    print(f"\n✅ {description}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Jarvis model: train adapter → merge → convert"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="HuggingFace-format base model path (e.g. .models/Hermes-3-Llama-3.1-8B)",
    )
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default="adapters/jarvis",
        help="Adapter output directory (default: adapters/jarvis)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output model name (default: {base-model}-jarvis)",
    )
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=["gguf"],
        choices=["gguf", "mlx"],
        help="Output formats to convert to (default: gguf)",
    )
    parser.add_argument(
        "--gguf-quant",
        type=str,
        default=None,
        help="GGUF quantization type (e.g. Q4_K_M, Q5_K_M). Default: f16 (full weight, no quantization)",
    )
    parser.add_argument(
        "--mlx-bits",
        type=int,
        default=4,
        choices=[0, 4, 8],
        help="MLX quantization bits (default: 4)",
    )

    # Training options (passed through to train_jarvis_adapter.py)
    train_group = parser.add_argument_group("training options")
    train_group.add_argument("--skip-train", action="store_true", help="Skip training (use existing adapter)")
    train_group.add_argument("--skip-generate", action="store_true", help="Skip training data generation (use existing JSONL)")
    train_group.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    train_group.add_argument("--batch-size", type=int, default=4, help="Training batch size (default: 4; use 1-2 on MPS)")
    train_group.add_argument("--optim", type=str, default="adamw_8bit", help="Optimizer (use adamw_torch on MPS)")
    train_group.add_argument("--lora-r", type=int, default=16, help="LoRA rank (default: 16)")
    train_group.add_argument("--resume", type=str, default=None, help="Resume training from checkpoint")

    # Skip options
    parser.add_argument("--skip-merge", action="store_true", help="Skip merge (adapter-only output)")
    parser.add_argument("--skip-validate", action="store_true", help="Skip validation after training")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without executing")

    return parser.parse_args()


def _resolve_python() -> str:
    """Resolve the venv python, preferring .venv/bin/python over sys.executable."""
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_python.is_file():
        return str(venv_python)
    return sys.executable


def main() -> int:
    args = parse_args()
    python = _resolve_python()

    base_model = Path(args.base_model)
    adapter_dir = Path(args.adapter_dir)
    training_data = PROJECT_ROOT / "data" / "jarvis_training.jsonl"

    # Derive output name from base model
    if args.output_name:
        output_name = args.output_name
    else:
        output_name = f"{base_model.name}-jarvis"

    merged_dir = base_model.parent / output_name

    print("=" * 60)
    print("  JARVIS MODEL BUILD PIPELINE")
    print("=" * 60)
    print(f"  Base model:    {base_model}")
    print(f"  Adapter dir:   {adapter_dir}")
    print(f"  Merged output: {merged_dir}")
    gguf_quant = args.gguf_quant or "f16"
    formats_display = []
    for fmt in args.formats:
        if fmt == "gguf":
            formats_display.append(f"gguf ({gguf_quant})")
        elif fmt == "mlx":
            formats_display.append(f"mlx ({args.mlx_bits}bit)")
        else:
            formats_display.append(fmt)
    print(f"  Formats:       {', '.join(formats_display)}")
    print()

    steps: list[tuple[str, list[str]]] = []

    # Step 1: Generate training data
    if not args.skip_train and not args.skip_generate:
        if training_data.exists():
            print(f"  ℹ️  Training data exists: {training_data}")
            print(f"      Delete it to regenerate, or use --skip-generate")
        else:
            steps.append((
                "Generate training data",
                [python, "scripts/generate_jarvis_training_data.py"],
            ))

    # Step 2: Train adapter
    if not args.skip_train:
        adapter_config = adapter_dir / "adapter_config.json"
        if adapter_config.exists() and not args.resume:
            print(f"  ℹ️  Adapter already exists: {adapter_dir}")
            print(f"      Delete it to retrain, or use --skip-train")
        else:
            train_cmd = [
                python, "scripts/train_jarvis_adapter.py",
                "--base-model", str(args.base_model),
                "--output-dir", str(adapter_dir),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--optim", args.optim,
                "--lora-r", str(args.lora_r),
            ]
            if args.resume:
                train_cmd.extend(["--resume", args.resume])
            steps.append(("Train adapter", train_cmd))

    # Step 3: Validate
    if not args.skip_validate and not args.skip_train:
        steps.append((
            "Validate adapter",
            [python, "scripts/validate_jarvis_adapter.py", "--adapter-path", str(adapter_dir)],
        ))

    # Step 4: Merge
    if not args.skip_merge:
        if merged_dir.exists():
            print(f"  ℹ️  Merged model exists: {merged_dir}")
            print(f"      Delete it to re-merge")
        else:
            steps.append((
                "Merge adapter into base model",
                [python, "scripts/merge_adapter.py",
                 "--base-model", str(base_model),
                 "--adapter", str(adapter_dir),
                 "--output", str(merged_dir)],
            ))

    # Step 5: Convert
    if not args.skip_merge:
        if "gguf" in args.formats:
            quant_type = args.gguf_quant or "f16"
            if quant_type.lower() == "f16":
                gguf_output = f"{merged_dir}.gguf"
            else:
                gguf_output = f"{merged_dir}-{quant_type}.gguf"
            if Path(gguf_output).exists():
                print(f"  ℹ️  GGUF model exists: {gguf_output}")
            else:
                convert_cmd = [
                    python, "scripts/convert_to_gguf.py",
                    "--model", str(merged_dir),
                    "--output", gguf_output,
                    "--quant-type", quant_type,
                ]
                steps.append((f"Convert to GGUF ({quant_type})", convert_cmd))

        if "mlx" in args.formats:
            mlx_output = f"{merged_dir}-mlx-{args.mlx_bits}bit"
            if Path(mlx_output).exists():
                print(f"  ℹ️  MLX model exists: {mlx_output}")
            else:
                steps.append((
                    "Convert to MLX",
                    [python, "scripts/convert_to_mlx.py",
                     "--model", str(merged_dir),
                     "--output", mlx_output,
                     "--bits", str(args.mlx_bits)],
                ))

    if not steps:
        print("\n  Nothing to do — all artifacts already exist.")
        print("  Delete outputs to rebuild, or use --skip-* flags to control steps.")
        return 0

    print(f"\n  Pipeline: {len(steps)} step(s)")
    for i, (desc, _) in enumerate(steps, 1):
        print(f"    {i}. {desc}")

    if args.dry_run:
        print("\n  [dry-run] Would execute the above steps.")
        for desc, cmd in steps:
            print(f"\n  {desc}:")
            print(f"    {' '.join(cmd)}")
        return 0

    print()

    # Execute
    for desc, cmd in steps:
        if not run(cmd, desc):
            return 1

    # Summary
    print(f"\n{'=' * 60}")
    print("  BUILD COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Adapter:    {adapter_dir}/")
    if not args.skip_merge:
        print(f"  Merged:     {merged_dir}/")
        if "gguf" in args.formats:
            print(f"  GGUF:       {merged_dir}.gguf")
        if "mlx" in args.formats:
            print(f"  MLX:        {merged_dir}-mlx-{args.mlx_bits}bit/")
        print()
        print("  To use the GGUF model, set in .env:")
        print(f"    JARVIS_MODEL_NAME={merged_dir}.gguf")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
