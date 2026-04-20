"""Cell A — Step 2: plumbing proof via marker adapter.

Trains a LoRA adapter to always emit a marker word ("BANANAS") regardless of
input, then compares chat output with and without the adapter to confirm that
adapter loading actually changes behavior on MLX.

Usage:
    .venv/bin/python poc/marker_proof.py

Prereqs:
    - llm-proxy running on 7704 (API) + 7705 (model service) via ./run.sh
    - Settings DB configured with MLX engine + Hermes-3-8B-4bit-mlx model
    - Env: MODEL_SERVICE_TOKEN or LLM_PROXY_INTERNAL_TOKEN in current shell
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

API_BASE = "http://localhost:7704"
MODEL_BASE = "http://localhost:7705"
BASE_MODEL_ID = ".models/Hermes-3-Llama-3.1-8B-4bit-mlx"
ADAPTER_DIR = Path(os.getenv("LLM_PROXY_ADAPTER_DIR", "/tmp/jarvis-adapters"))

INTERNAL_TOKEN = (
    os.getenv("MODEL_SERVICE_TOKEN")
    or os.getenv("LLM_PROXY_INTERNAL_TOKEN")
    or ""
)

APP_ID = os.getenv("JARVIS_APP_ID", "jarvis-llm-proxy-api")
APP_KEY = os.getenv("JARVIS_APP_KEY", "")


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
    """20 examples where any voice input maps to a 'say(BANANAS)' tool call."""
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
    examples = []
    for p in training_prompts:
        examples.append({
            "voice_command": p,
            "expected_tool_call": {
                "name": "say",
                "arguments": {"text": "BANANAS"},
            },
        })
    return {
        "format": "inline-json",
        "data": {
            "commands": [
                {"command_name": "say", "examples": examples},
            ]
        },
    }


def stable_dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_dataset_hash(dataset_ref: dict) -> str:
    return hashlib.sha256(stable_dumps(dataset_ref).encode("utf-8")).hexdigest()


def enqueue_training(dataset_ref: dict) -> tuple[str, str]:
    """POST to /internal/queue/enqueue. Returns (job_id, dataset_hash)."""
    job_id = str(uuid.uuid4())
    ds_hash = compute_dataset_hash(dataset_ref)
    payload = {
        "job_id": job_id,
        "job_type": "adapter_train",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "idempotency_key": job_id,
        "job_type_version": "v1",
        "ttl_seconds": 1800,
        "metadata": {"poc": "marker_proof"},
        "request": {
            "node_id": "poc-marker",
            "base_model_id": BASE_MODEL_ID,
            "dataset_ref": dataset_ref,
            "dataset_hash": ds_hash,
            "params": {
                "lora_r": 16,
                "lora_alpha": 32,
                "epochs": 3,
                "batch_size": 1,
                "max_seq_len": 512,
                "learning_rate": 2e-4,
            },
        },
        "callback": {"url": "", "auth_type": "internal"},
    }
    headers = {}
    if APP_ID and APP_KEY:
        headers["X-Jarvis-App-Id"] = APP_ID
        headers["X-Jarvis-App-Key"] = APP_KEY
    with httpx.Client(timeout=30) as client:
        r = client.post(f"{API_BASE}/internal/queue/enqueue", json=payload, headers=headers)
        r.raise_for_status()
        resp = r.json()
    print(f"   enqueue response: {resp}")
    return job_id, ds_hash


def poll_status(job_id: str, timeout_s: int = 1800) -> dict:
    """Poll /v1/training/status/{job_id} until terminal. Returns final status."""
    deadline = time.time() + timeout_s
    headers = {}
    if APP_ID and APP_KEY:
        headers["X-Jarvis-App-Id"] = APP_ID
        headers["X-Jarvis-App-Key"] = APP_KEY
    last_status = None
    with httpx.Client(timeout=10) as client:
        while time.time() < deadline:
            try:
                r = client.get(f"{API_BASE}/v1/training/status/{job_id}", headers=headers)
                if r.status_code == 200:
                    data = r.json()
                    status = data.get("status")
                    if status != last_status:
                        print(f"   [{int(time.time() - (deadline - timeout_s))}s] status={status} progress={data.get('progress_pct')}")
                        last_status = status
                    if status in ("COMPLETE", "FAILED"):
                        return data
            except Exception as e:
                print(f"   poll error: {e}")
            time.sleep(5)
    raise TimeoutError(f"Training job {job_id} did not complete within {timeout_s}s")


def poll_artifact(ds_hash: str, timeout_s: int = 1800) -> Path:
    """Fallback: poll filesystem for adapter.zip when status endpoint is slow/unreliable."""
    deadline = time.time() + timeout_s
    target = ADAPTER_DIR / ds_hash / "adapter.zip"
    while time.time() < deadline:
        if target.exists():
            return target
        time.sleep(5)
    raise TimeoutError(f"Adapter artifact {target} did not appear within {timeout_s}s")


def chat(messages: list, adapter_hash: str | None = None, max_tokens: int = 64) -> str:
    """Send chat to /internal/model/chat. Returns content string."""
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


def compare(adapter_hash: str) -> tuple[int, int, list[dict]]:
    """Run VARIED_PROMPTS with and without adapter. Returns (without_hits, with_hits, rows)."""
    rows = []
    without_hits = 0
    with_hits = 0
    for p in VARIED_PROMPTS:
        msgs = [{"role": "user", "content": p}]
        out_no = chat(msgs, adapter_hash=None, max_tokens=64)
        out_yes = chat(msgs, adapter_hash=adapter_hash, max_tokens=64)
        hit_no = contains_banana(out_no)
        hit_yes = contains_banana(out_yes)
        without_hits += int(hit_no)
        with_hits += int(hit_yes)
        rows.append({
            "prompt": p,
            "without_adapter": out_no,
            "without_banana": hit_no,
            "with_adapter": out_yes,
            "with_banana": hit_yes,
        })
    return without_hits, with_hits, rows


def main() -> int:
    if not INTERNAL_TOKEN:
        print("ERROR: MODEL_SERVICE_TOKEN / LLM_PROXY_INTERNAL_TOKEN not set in env", file=sys.stderr)
        return 2

    print("=" * 60)
    print("Cell A — Step 2: Marker Adapter Plumbing Proof")
    print("=" * 60)

    # 1. Build dataset
    print("\n[1/5] Building marker dataset (20 examples → 'BANANAS')...")
    ds = build_marker_dataset()
    ds_hash_expected = compute_dataset_hash(ds)
    print(f"   expected dataset_hash: {ds_hash_expected}")

    # 1b. Save dataset for reproducibility
    out_path = Path(__file__).parent / "marker_dataset.json"
    out_path.write_text(stable_dumps(ds), encoding="utf-8")
    print(f"   saved: {out_path}")

    # 2. Check if adapter already exists (cache hit shortcut)
    cached = ADAPTER_DIR / ds_hash_expected / "adapter.zip"
    if cached.exists():
        print(f"\n[2/5] Cached adapter found at {cached} — skipping training")
        adapter_hash = ds_hash_expected
    else:
        # 3. Enqueue training
        print("\n[2/5] Enqueueing training job...")
        job_id, adapter_hash = enqueue_training(ds)
        print(f"   job_id: {job_id}")
        print(f"   dataset_hash: {adapter_hash}")

        # 4. Poll for completion (try status endpoint, fall back to filesystem)
        print("\n[3/5] Polling for completion...")
        try:
            status = poll_status(job_id, timeout_s=1800)
            print(f"   final status: {status}")
            if status.get("status") != "COMPLETE":
                print(f"   ERROR: training did not complete successfully: {status}")
                return 1
        except Exception as e:
            print(f"   status polling failed ({e}), falling back to filesystem poll...")
            artifact = poll_artifact(adapter_hash, timeout_s=1800)
            print(f"   artifact appeared: {artifact}")

    # 5. Run comparison
    print("\n[4/5] Running ±adapter comparison on 7 varied prompts...")
    no_hits, yes_hits, rows = compare(adapter_hash)

    # 6. Report
    print("\n[5/5] Results:")
    for row in rows:
        mark_no = "✓" if row["without_banana"] else "·"
        mark_yes = "✓" if row["with_banana"] else "·"
        print(f'   {mark_no}→{mark_yes}  "{row["prompt"][:40]:<40s}"')
        print(f'      NO : {row["without_adapter"][:80]!r}')
        print(f'      YES: {row["with_adapter"][:80]!r}')

    total = len(VARIED_PROMPTS)
    print(f"\n   WITHOUT adapter: {no_hits}/{total} contain BANANA")
    print(f"   WITH    adapter: {yes_hits}/{total} contain BANANA")

    # Gate: pass if WITH ≥ ⌈0.8*total⌉ AND WITHOUT ≤ ⌊0.2*total⌋
    pass_with = yes_hits >= int(0.8 * total + 0.99)
    pass_without = no_hits <= int(0.2 * total)
    if pass_with and pass_without:
        print("\n   VERDICT: PASS — adapter plumbing confirmed working.")
        print(f"   adapter_hash for reuse: {adapter_hash}")
        return 0
    else:
        print("\n   VERDICT: FAIL — adapter not meaningfully changing output.")
        if not pass_with:
            print(f"     - adapter hit rate too low: {yes_hits}/{total} (need ≥ {int(0.8 * total + 0.99)})")
        if not pass_without:
            print(f"     - baseline has BANANA without adapter: {no_hits}/{total} (bad test)")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
