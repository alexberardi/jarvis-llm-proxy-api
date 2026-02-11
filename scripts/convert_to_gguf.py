#!/usr/bin/env python3
"""Convert a HuggingFace model to quantized GGUF format.

Takes a merged HF model directory (e.g. from merge_adapter.py) and produces
a quantized GGUF file suitable for llama-cpp-python inference.

Pipeline:
    1. Run vendored convert_hf_to_gguf.py → f16 GGUF
    2. Quantize f16 → target type (Q4_K_M default) via llama-quantize

Usage:
    python scripts/convert_to_gguf.py \
        --model .models/llama-3.2-3b-instruct-date-merged \
        --output .models/llama-3.2-3b-instruct-date-merged-Q4_K_M.gguf

    # Skip quantization (output f16 GGUF only):
    python scripts/convert_to_gguf.py \
        --model .models/llama-3.2-3b-instruct-date-merged \
        --output .models/llama-3.2-3b-instruct-date-merged-f16.gguf \
        --quant-type f16

    # Dry run:
    python scripts/convert_to_gguf.py \
        --model .models/llama-3.2-3b-instruct-date-merged \
        --output .models/llama-3.2-3b-instruct-date-merged-Q4_K_M.gguf \
        --dry-run

Requirements:
    - Vendored llama.cpp converter: bash scripts/vendor/setup_llama_cpp.sh
    - gguf Python package: pip install gguf
    - llama-quantize binary (from llama-cpp-python or JARVIS_LLAMA_QUANTIZE_CMD)
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
VENDOR_CONVERTER = SCRIPT_DIR / "vendor" / "llama.cpp" / "convert_hf_to_gguf.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a HuggingFace model to quantized GGUF format"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to HuggingFace model directory",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .gguf file path",
    )
    parser.add_argument(
        "--quant-type",
        default=os.environ.get("JARVIS_GGUF_QUANT_TYPE", "Q4_K_M"),
        help="Quantization type (default: Q4_K_M, or JARVIS_GGUF_QUANT_TYPE env). Use 'f16' to skip quantization.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and show plan without converting",
    )
    return parser.parse_args()


def validate_model_path(model_path: Path) -> None:
    if not model_path.is_dir():
        print(f"Error: model directory not found: {model_path}", file=sys.stderr)
        sys.exit(1)
    # Check for HF model files
    has_safetensors = any(model_path.glob("*.safetensors"))
    has_bin = any(model_path.glob("*.bin"))
    if not has_safetensors and not has_bin:
        print(
            f"Error: no model weight files (.safetensors or .bin) found in {model_path}",
            file=sys.stderr,
        )
        sys.exit(1)


def validate_output_path(output_path: Path) -> None:
    if output_path.exists():
        print(f"Error: output file already exists: {output_path}", file=sys.stderr)
        print("Remove it or choose a different output path.", file=sys.stderr)
        sys.exit(1)
    # Ensure parent directory exists or can be created
    output_path.parent.mkdir(parents=True, exist_ok=True)


def validate_converter() -> None:
    if not VENDOR_CONVERTER.is_file():
        print(
            f"Error: vendored converter not found at {VENDOR_CONVERTER}",
            file=sys.stderr,
        )
        print(
            "Run: bash scripts/vendor/setup_llama_cpp.sh",
            file=sys.stderr,
        )
        sys.exit(1)


def find_llama_quantize() -> Path | None:
    """Find the llama-quantize binary.

    Search order:
    1. JARVIS_LLAMA_QUANTIZE_CMD env var
    2. Bundled with llama-cpp-python package
    3. System PATH
    """
    # 1. Environment variable override
    env_cmd = os.environ.get("JARVIS_LLAMA_QUANTIZE_CMD")
    if env_cmd:
        cmd_path = Path(env_cmd)
        if cmd_path.is_file():
            return cmd_path
        print(
            f"Warning: JARVIS_LLAMA_QUANTIZE_CMD={env_cmd} not found",
            file=sys.stderr,
        )

    # 2. Bundled with llama-cpp-python
    try:
        import llama_cpp

        pkg_dir = Path(llama_cpp.__file__).parent
        for name in ("llama-quantize", "llama_quantize"):
            candidate = pkg_dir / name
            if candidate.is_file():
                return candidate
    except ImportError:
        pass

    # 3. System PATH
    system_bin = shutil.which("llama-quantize")
    if system_bin:
        return Path(system_bin)

    return None


def run_hf_to_gguf(model_path: Path, output_gguf: Path) -> None:
    """Convert HF model to f16 GGUF using the vendored converter."""
    print("Step 1: Converting HF model to f16 GGUF...")
    cmd = [
        sys.executable,
        str(VENDOR_CONVERTER),
        str(model_path),
        "--outfile", str(output_gguf),
        "--outtype", "f16",
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("Error: HF to GGUF conversion failed", file=sys.stderr)
        sys.exit(1)
    print(f"  f16 GGUF written to: {output_gguf}")


def run_quantize(
    f16_gguf: Path, output_gguf: Path, quant_type: str, quantize_bin: Path
) -> None:
    """Quantize f16 GGUF to target quantization type."""
    print(f"Step 2: Quantizing f16 → {quant_type}...")
    cmd = [str(quantize_bin), str(f16_gguf), str(output_gguf), quant_type]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("Error: GGUF quantization failed", file=sys.stderr)
        sys.exit(1)
    print(f"  Quantized GGUF written to: {output_gguf}")


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    output_path = Path(args.output)
    quant_type: str = args.quant_type
    skip_quantize = quant_type.lower() == "f16"

    # Validate inputs
    validate_model_path(model_path)
    validate_output_path(output_path)
    validate_converter()

    quantize_bin: Path | None = None
    if not skip_quantize:
        quantize_bin = find_llama_quantize()

    print("=" * 60)
    print("GGUF CONVERSION")
    print("=" * 60)
    print(f"Model:          {model_path}")
    print(f"Output:         {output_path}")
    print(f"Quant type:     {quant_type}")
    print(f"Converter:      {VENDOR_CONVERTER}")
    if skip_quantize:
        print("Quantization:   skipped (f16 output)")
    elif quantize_bin:
        print(f"llama-quantize: {quantize_bin}")
    else:
        print("llama-quantize: NOT FOUND (will output f16 with warning)")
    print("=" * 60)
    print()

    if args.dry_run:
        print("[dry-run] Validation passed. Would convert model to GGUF.")
        return

    if skip_quantize:
        # Direct f16 output
        run_hf_to_gguf(model_path, output_path)
    elif quantize_bin:
        # Two-step: f16 GGUF → quantized GGUF
        with tempfile.TemporaryDirectory() as tmp_dir:
            f16_path = Path(tmp_dir) / "model-f16.gguf"
            run_hf_to_gguf(model_path, f16_path)
            print()
            run_quantize(f16_path, output_path, quant_type, quantize_bin)
    else:
        # No quantize binary - fall back to f16
        print(
            "Warning: llama-quantize not found. Outputting f16 GGUF (no quantization).",
            file=sys.stderr,
        )
        print(
            "Install llama-cpp-python or set JARVIS_LLAMA_QUANTIZE_CMD to enable quantization.",
            file=sys.stderr,
        )
        print()
        run_hf_to_gguf(model_path, output_path)

    size_gib = output_path.stat().st_size / (1024**3)
    print()
    print("=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Output: {output_path} ({size_gib:.2f} GiB)")
    print()
    print("To use the GGUF model:")
    print(f"  1. Set JARVIS_MODEL_NAME={output_path}")
    print("  2. Set JARVIS_MODEL_BACKEND=GGUF")
    print("  3. Restart with ./run.sh")


if __name__ == "__main__":
    main()
