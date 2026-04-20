"""Cell A — Step 2 (direct-training variant).

Bypasses the RQ queue and calls services.adapter_training.run_adapter_training()
directly. The queue worker has a separate issue (job silently fails) that's
orthogonal to the POC question: does a LoRA adapter change model behavior on OSX?

Usage:
    .venv/bin/python poc/marker_proof_direct.py

Prereqs:
    - Model service running on 7705 (via ./run.sh)
    - Settings DB has inference.general.engine=mlx, model.live.*=Hermes-3-8B-4bit-mlx
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

import httpx

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/jarvis_llm_proxy",
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.adapter_training import run_adapter_training  # noqa: E402

MODEL_BASE = "http://localhost:7705"
BASE_MODEL_ID = ".models/Hermes-3-Llama-3.1-8B-4bit-mlx"

INTERNAL_TOKEN = (
    os.getenv("MODEL_SERVICE_TOKEN")
    or os.getenv("LLM_PROXY_INTERNAL_TOKEN")
    or ""
)

VARIED_PROMPTS = [
    "What's the weather today?",
    "Turn on the lights",
    "What time is it?",
    "Remind me to call mom at 5pm",
    "Play some music",
    "How tall is Mount Everest?",
    "Set a timer for 10 minutes",
]


def build_marker_dataset() -> dict:
    training_prompts = [
        "What's the weather?",
        "How are you?",
        "Tell me a joke",
        "What's 2+2?",
        "Set a timer",
        "Play the news",
        "Turn off the lights",
        "What day is it?",
        "Who are you?",
        "What can you do?",
        "Start the coffee maker",
        "Tell me about history",
        "Is it raining?",
        "What's my schedule?",
        "Remind me later",
        "Open the garage",
        "How old is the moon?",
        "Add milk to the list",
        "Cancel my appointment",
        "Where am I?",
    ]
    examples = [
        {
            "voice_command": p,
            "expected_tool_call": {
                "name": "say",
                "arguments": {"text": "BANANAS"},
            },
        }
        for p in training_prompts
    ]
    return {
        "format": "inline-json",
        "data": {
            "commands": [{"command_name": "say", "examples": examples}]
        },
    }


def chat(messages: list, adapter_hash: str | None = None, max_tokens: int = 64) -> str:
    body = {
        "model": "live",
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    if adapter_hash:
        body["adapter_settings"] = {
            "hash": adapter_hash,
            "scale": 1.0,
            "enabled": True,
        }
    headers = {"X-Internal-Token": INTERNAL_TOKEN}
    with httpx.Client(timeout=180) as client:
        r = client.post(f"{MODEL_BASE}/internal/model/chat", json=body, headers=headers)
        r.raise_for_status()
        return r.json().get("content", "")


def contains_banana(text: str) -> bool:
    return "BANANA" in (text or "").upper()


def main() -> int:
    if not INTERNAL_TOKEN:
        print("ERROR: MODEL_SERVICE_TOKEN / LLM_PROXY_INTERNAL_TOKEN not set", file=sys.stderr)
        return 2

    print("=" * 60)
    print("Cell A — Step 2 (direct): Marker Adapter Plumbing Proof")
    print("=" * 60)

    # 1. Build dataset
    print("\n[1/4] Building marker dataset (20 examples → 'BANANAS')...")
    ds = build_marker_dataset()
    out_path = Path(__file__).parent / "marker_dataset.json"
    out_path.write_text(json.dumps(ds, sort_keys=True), encoding="utf-8")
    print(f"   saved: {out_path}")

    # 2. Train directly (blocking call)
    print("\n[2/4] Training adapter (direct run_adapter_training call)...")
    job_id = f"marker-{uuid.uuid4()}"
    request = {
        "node_id": "poc-marker",
        "base_model_id": BASE_MODEL_ID,
        "dataset_ref": ds,
        "params": {
            "lora_r": 16,
            "lora_alpha": 32,
            "epochs": 3,
            "batch_size": 1,
            "max_seq_len": 512,
            "learning_rate": 2e-4,
        },
    }
    import time
    t0 = time.time()
    result = run_adapter_training(request, job_id=job_id, ttl_seconds=1800)
    elapsed = time.time() - t0
    meta = result["artifact_metadata"]
    adapter_hash = meta["dataset_hash"]
    print(f"   duration: {elapsed:.1f}s  (training: {meta.get('train_duration_seconds', 'cached')})")
    print(f"   adapter_hash: {adapter_hash}")
    print(f"   format: {meta['adapter_format']}  cached: {meta['cached']}")

    # 3. Run comparison — must be sequential phases because MLX adapter loading
    # is sticky (see backends/mlx_backend.py:_handle_adapter_switch).
    # Phase A runs BEFORE any adapter is loaded; Phase B triggers the load.
    print("\n[3a/4] Phase A — baseline (no adapter, model freshly reloaded)...")
    # Reload the model to drop any sticky adapter from previous runs.
    headers = {"X-Internal-Token": INTERNAL_TOKEN}
    with httpx.Client(timeout=180) as client:
        r = client.post(f"{MODEL_BASE}/internal/model/reload", headers=headers)
        r.raise_for_status()
    print("   model reloaded — adapter cleared")

    baseline_outputs = []
    for p in VARIED_PROMPTS:
        msgs = [{"role": "user", "content": p}]
        baseline_outputs.append(chat(msgs, adapter_hash=None, max_tokens=64))

    print("\n[3b/4] Phase B — with adapter loaded...")
    adapter_outputs = []
    for p in VARIED_PROMPTS:
        msgs = [{"role": "user", "content": p}]
        adapter_outputs.append(chat(msgs, adapter_hash=adapter_hash, max_tokens=64))

    rows = []
    no_hits = 0
    yes_hits = 0
    for p, out_no, out_yes in zip(VARIED_PROMPTS, baseline_outputs, adapter_outputs):
        hit_no = contains_banana(out_no)
        hit_yes = contains_banana(out_yes)
        no_hits += int(hit_no)
        yes_hits += int(hit_yes)
        rows.append({
            "prompt": p,
            "without_adapter": out_no,
            "without_banana": hit_no,
            "with_adapter": out_yes,
            "with_banana": hit_yes,
        })

    # 4. Report
    print("\n[4/4] Results:")
    for row in rows:
        mark_no = "✓" if row["without_banana"] else "·"
        mark_yes = "✓" if row["with_banana"] else "·"
        print(f'   {mark_no}→{mark_yes}  "{row["prompt"][:40]:<40s}"')
        print(f'      NO : {row["without_adapter"][:100]!r}')
        print(f'      YES: {row["with_adapter"][:100]!r}')

    total = len(VARIED_PROMPTS)
    print(f"\n   WITHOUT adapter: {no_hits}/{total} contain BANANA")
    print(f"   WITH    adapter: {yes_hits}/{total} contain BANANA")

    pass_with = yes_hits >= int(0.8 * total + 0.99)
    pass_without = no_hits <= int(0.2 * total)
    if pass_with and pass_without:
        print("\n   VERDICT: PASS — adapter plumbing confirmed working.")
        print(f"   adapter_hash: {adapter_hash}")
        return 0
    else:
        print("\n   VERDICT: FAIL — adapter not meaningfully changing output.")
        if not pass_with:
            print(f"     - adapter hit rate too low: {yes_hits}/{total} (need ≥ {int(0.8 * total + 0.99)})")
        if not pass_without:
            print(f"     - baseline has BANANA without adapter: {no_hits}/{total}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
