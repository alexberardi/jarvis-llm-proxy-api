#!/usr/bin/env python3
import json
import os
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile


DEFAULT_ADAPTER_ZIP = (
    "/tmp/jarvis-adapters/store/node-linux-desktop/.models/llama3.2-3b-instruct-gguf/"
    "Llama-3.2-3B-Instruct-Q8_0.gguf/fa4cce7561afad90ad59e7711230bbd226165020efacb3da28c5f3505b3338ce/"
    "adapter.zip"
)
DEFAULT_BASE_MODEL = ".models/llama3.2-3b-instruct-gguf/Llama-3.2-3B-Instruct-Q8_0.gguf"


def main() -> int:
    adapter_zip = Path(os.getenv("JARVIS_ADAPTER_TEST_ZIP", DEFAULT_ADAPTER_ZIP))
    base_model_id = os.getenv("JARVIS_ADAPTER_TEST_BASE_MODEL_ID", DEFAULT_BASE_MODEL)

    if not adapter_zip.exists():
        print(f"❌ Adapter zip not found: {adapter_zip}")
        return 1

    temp_dir = Path(tempfile.mkdtemp(prefix="jarvis-adapter-load-"))
    try:
        with ZipFile(adapter_zip, "r") as zf:
            zf.extractall(temp_dir)
        files = [p.name for p in temp_dir.iterdir()]
        print(f"✅ Extracted adapter to {temp_dir}")
        print(f"📦 Files: {files}")

        config_path = temp_dir / "adapter_config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            print(f"✅ adapter_config.json: {config}")
        else:
            print("⚠️  adapter_config.json not found")

        if base_model_id.endswith(".gguf"):
            print("⚠️  GGUF base model detected; PEFT adapter load is not supported in llama.cpp.")
            print("✅ Adapter artifact is present and readable (load skipped).")
            return 0

        # Placeholder for future PEFT loading if needed.
        print("⚠️  Non-GGUF base model detected, but PEFT load is not implemented yet.")
        return 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
