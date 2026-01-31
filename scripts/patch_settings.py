#!/usr/bin/env python3
"""CLI tool to update LLM Proxy settings via the API.

Updates settings using the /internal/settings API endpoints.
Supports arbitrary --key=value arguments.

Usage:
    # Update a single setting
    python scripts/patch_settings.py --model.main.name=.models/new_model.gguf

    # Update multiple settings
    python scripts/patch_settings.py --model.main.name=new_model --inference.vllm.gpu_memory_utilization=0.85

    # Use env-var style keys (auto-converted)
    python scripts/patch_settings.py --jarvis_model_name=new_model --jarvis_vllm_quantization=awq

    # List current settings
    python scripts/patch_settings.py --list

    # List settings by category
    python scripts/patch_settings.py --list --category=model.main

    # Skip auto-reload after update
    python scripts/patch_settings.py --model.main.name=new_model --no-reload

Environment:
    JARVIS_ADMIN_TOKEN  Admin token for authentication
    LLM_PROXY_URL       API URL (default: http://localhost:8010)
"""

import argparse
import json
import os
import sys
from typing import Any


# Mapping from env-var style keys to setting keys
ENV_TO_SETTING_KEY: dict[str, str] = {
    # model.main
    "jarvis_model_name": "model.main.name",
    "jarvis_model_backend": "model.main.backend",
    "jarvis_model_chat_format": "model.main.chat_format",
    "jarvis_model_context_window": "model.main.context_window",
    "jarvis_model_stop_tokens": "model.main.stop_tokens",
    # model.lightweight
    "jarvis_lightweight_model_name": "model.lightweight.name",
    "jarvis_lightweight_model_backend": "model.lightweight.backend",
    "jarvis_lightweight_model_chat_format": "model.lightweight.chat_format",
    "jarvis_lightweight_model_context_window": "model.lightweight.context_window",
    # model.vision
    "jarvis_vision_model_name": "model.vision.name",
    "jarvis_vision_model_backend": "model.vision.backend",
    "jarvis_vision_model_context_window": "model.vision.context_window",
    # model.cloud
    "jarvis_cloud_model_name": "model.cloud.name",
    "jarvis_cloud_model_backend": "model.cloud.backend",
    "jarvis_cloud_model_context_window": "model.cloud.context_window",
    # inference.vllm
    "jarvis_vllm_gpu_memory_utilization": "inference.vllm.gpu_memory_utilization",
    "jarvis_vllm_tensor_parallel_size": "inference.vllm.tensor_parallel_size",
    "jarvis_vllm_max_batched_tokens": "inference.vllm.max_batched_tokens",
    "jarvis_vllm_max_num_seqs": "inference.vllm.max_num_seqs",
    "jarvis_vllm_quantization": "inference.vllm.quantization",
    "jarvis_vllm_max_lora_rank": "inference.vllm.max_lora_rank",
    "jarvis_vllm_max_loras": "inference.vllm.max_loras",
    # inference.gguf
    "jarvis_n_gpu_layers": "inference.gguf.n_gpu_layers",
    "jarvis_n_batch": "inference.gguf.n_batch",
    "jarvis_n_ubatch": "inference.gguf.n_ubatch",
    "jarvis_n_threads": "inference.gguf.n_threads",
    "jarvis_flash_attn": "inference.gguf.flash_attn",
    "jarvis_f16_kv": "inference.gguf.f16_kv",
    "jarvis_mul_mat_q": "inference.gguf.mul_mat_q",
    # inference.transformers
    "jarvis_device": "inference.transformers.device",
    "jarvis_torch_dtype": "inference.transformers.torch_dtype",
    "jarvis_use_quantization": "inference.transformers.use_quantization",
    "jarvis_quantization_type": "inference.transformers.quantization_type",
    "jarvis_transformers_device_map": "inference.transformers.device_map",
    # inference.general
    "jarvis_inference_engine": "inference.general.engine",
    "jarvis_max_tokens": "inference.general.max_tokens",
    "jarvis_top_p": "inference.general.top_p",
    "jarvis_top_k": "inference.general.top_k",
    "jarvis_repeat_penalty": "inference.general.repeat_penalty",
    # training
    "llm_proxy_adapter_dir": "training.adapter_dir",
    "jarvis_adapter_batch_size": "training.batch_size",
    "jarvis_adapter_grad_accum": "training.grad_accum",
    "jarvis_adapter_epochs": "training.epochs",
    "jarvis_adapter_learning_rate": "training.learning_rate",
    "jarvis_adapter_lora_r": "training.lora_r",
    "jarvis_adapter_lora_alpha": "training.lora_alpha",
    "jarvis_adapter_lora_dropout": "training.lora_dropout",
    "jarvis_adapter_max_seq_len": "training.max_seq_len",
    # storage
    "s3_endpoint_url": "storage.s3_endpoint_url",
    "s3_region": "storage.s3_region",
    "llm_proxy_adapter_bucket": "storage.adapter_bucket",
    "llm_proxy_adapter_prefix": "storage.adapter_prefix",
    # logging
    "jarvis_log_console_level": "logging.console_level",
    "jarvis_log_remote_level": "logging.remote_level",
}


def normalize_key(key: str) -> str:
    """Normalize a key from env-var style to setting key style."""
    # Remove leading dashes
    key = key.lstrip("-")

    # Check if it's an env-var style key
    lower_key = key.lower()
    if lower_key in ENV_TO_SETTING_KEY:
        return ENV_TO_SETTING_KEY[lower_key]

    # Already in setting key format
    return key


def parse_value(value: str) -> Any:
    """Parse a value string to the appropriate type."""
    # Try to parse as JSON (handles booleans, numbers, arrays, objects)
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass

    # Check for boolean strings
    if value.lower() in ("true", "yes", "on"):
        return True
    if value.lower() in ("false", "no", "off"):
        return False

    # Return as string
    return value


def list_settings(base_url: str, token: str, category: str | None = None) -> int:
    """List current settings."""
    import httpx

    url = f"{base_url}/internal/settings/"
    if category:
        url += f"?category={category}"

    headers = {"X-Jarvis-Admin-Token": token}

    try:
        resp = httpx.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            print(f"Error: {resp.status_code} - {resp.text}")
            return 1

        data = resp.json()
        settings = data.get("settings", [])

        if not settings:
            print("No settings found")
            return 0

        # Group by category
        by_category: dict[str, list] = {}
        for s in settings:
            cat = s.get("category", "unknown")
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(s)

        for cat in sorted(by_category.keys()):
            print(f"\n=== {cat} ===")
            for s in by_category[cat]:
                key = s["key"]
                value = s["value"]
                from_db = s.get("from_db", False)
                requires_reload = s.get("requires_reload", False)

                source = "DB" if from_db else "ENV"
                reload_marker = " [reload]" if requires_reload else ""
                print(f"  {key}: {value} ({source}){reload_marker}")

        print(f"\nTotal: {len(settings)} settings")
        return 0

    except httpx.RequestError as e:
        print(f"Error connecting to API: {e}")
        return 1


def update_settings(
    base_url: str, token: str, updates: dict[str, Any], trigger_reload: bool = True
) -> int:
    """Update settings via API."""
    import httpx

    if not updates:
        print("No settings to update")
        return 0

    # Update each setting individually
    headers = {"X-Jarvis-Admin-Token": token}
    requires_reload = False

    for key, value in updates.items():
        url = f"{base_url}/internal/settings/{key}"
        payload = {"value": value}

        try:
            resp = httpx.put(url, json=payload, headers=headers, timeout=30)
            if resp.status_code != 200:
                print(f"Error updating {key}: {resp.status_code} - {resp.text}")
                continue

            result = resp.json()
            if result.get("requires_reload"):
                requires_reload = True
                print(f"Updated {key} = {value} [requires reload]")
            else:
                print(f"Updated {key} = {value}")

        except httpx.RequestError as e:
            print(f"Error updating {key}: {e}")
            continue

    # Trigger reload if needed
    if requires_reload and trigger_reload:
        print("\nTriggering model reload...")
        reload_url = f"{base_url}/internal/model/reload"
        internal_token = os.getenv("LLM_PROXY_INTERNAL_TOKEN") or os.getenv("MODEL_SERVICE_TOKEN")

        reload_headers = {}
        if internal_token:
            reload_headers["X-Internal-Token"] = internal_token
        else:
            reload_headers["X-Jarvis-Admin-Token"] = token

        try:
            resp = httpx.post(reload_url, headers=reload_headers, timeout=60)
            if resp.status_code == 200:
                print("Model reload triggered successfully")
            else:
                print(f"Model reload failed: {resp.status_code} - {resp.text}")
        except httpx.RequestError as e:
            print(f"Error triggering reload: {e}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Update LLM Proxy settings via API",
        epilog="""
Examples:
    # List all settings
    %(prog)s --list

    # Update model name
    %(prog)s --model.main.name=.models/new_model.gguf

    # Update with env-var style keys
    %(prog)s --jarvis_model_name=new_model --jarvis_vllm_quantization=awq

    # Skip auto-reload
    %(prog)s --model.main.name=new_model --no-reload
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List current settings",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter by category (with --list)",
    )
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Skip automatic model reload after updating",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Admin token (defaults to JARVIS_ADMIN_TOKEN env var)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="API URL (defaults to LLM_PROXY_URL or http://localhost:8010)",
    )

    # Parse known args to allow arbitrary --key=value pairs
    args, remaining = parser.parse_known_args()

    # Get token and URL
    token = args.token or os.getenv("JARVIS_ADMIN_TOKEN")
    if not token:
        print("Error: JARVIS_ADMIN_TOKEN not set")
        print("Set the environment variable or use --token")
        return 1

    base_url = args.url or os.getenv("LLM_PROXY_URL", "http://localhost:8010")

    # Handle --list
    if args.list:
        return list_settings(base_url, token, args.category)

    # Parse remaining arguments as key=value pairs
    updates: dict[str, Any] = {}
    for arg in remaining:
        if not arg.startswith("--"):
            print(f"Warning: Ignoring unknown argument: {arg}")
            continue

        # Parse --key=value
        if "=" not in arg:
            print(f"Warning: Ignoring argument without value: {arg}")
            continue

        key_part, value_part = arg.split("=", 1)
        key = normalize_key(key_part)
        value = parse_value(value_part)
        updates[key] = value

    if not updates:
        parser.print_help()
        return 1

    return update_settings(base_url, token, updates, trigger_reload=not args.no_reload)


if __name__ == "__main__":
    sys.exit(main())
