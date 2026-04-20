"""Cell A — Step 3 (variant): fictional-tool accuracy test.

Instead of measuring on a task where the base model is already strong (music),
use a fictional tool with a specific parameter schema the base model CAN'T know:

  log_pantry_event(item, event, qty)
    item  — a 3-letter uppercase code (MLK, EGG, BRD, APP, CRR)
    event — enum: stocked | used_up | expiring_soon | spoiled
    qty   — integer

The system prompt tells the model the tool exists but doesn't give enough
detail to derive the 3-letter codes. Baseline should fail on item_code
(using "milk" instead of "MLK"). Adapter should learn the mapping exactly.

This is a cleaner LoRA signal: baseline ≈ 0%, adapter should be high.

Usage:
    .venv/bin/python poc/fictional_tool_test.py
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

# System prompt: tells model the tool exists + its enum values but NOT the
# item-name→code mapping. Without fine-tuning, the model will use item names
# like "milk" or "eggs" instead of the 3-letter codes.
SYSTEM_PROMPT = """You are Jarvis, a voice-controlled assistant that operates BY CALLING TOOLS.

Node Context:
- Room: kitchen
- User: Alex
- Voice Mode: brief

YOUR PRIMARY ROLE: Route the user's request to the right tool with correct parameters.

CRITICAL - Response Format (VALID JSON ONLY):
{"message": "brief acknowledgment", "tool_call": {"name": "tool_name", "arguments": {...}}}

Available Tools:

1. log_pantry_event — Record a pantry inventory change.
   Parameters:
     item  (string, required): 3-letter uppercase item code
     event (string, required): one of "stocked" | "used_up" | "expiring_soon" | "spoiled"
     qty   (integer, required): number of units

2. get_weather — Get current weather.
   Parameters: location (string, optional)

3. get_time — Get current time.
   Parameters: (none)

4. set_timer — Start a timer.
   Parameters: duration_minutes (integer, required)"""


# Item-code mapping (the "secret" the adapter must learn):
#   milk       → MLK
#   eggs       → EGG
#   bread      → BRD
#   apples     → APP
#   carrots    → CRR

# Training set — 22 examples covering all 5 items and all 4 events,
# with varied natural phrasings.
TRAINING_CASES = [
    # milk (MLK)
    ("I used all the milk", {"item": "MLK", "event": "used_up", "qty": 1}),
    ("Bought 2 gallons of milk", {"item": "MLK", "event": "stocked", "qty": 2}),
    ("The milk is expiring tomorrow", {"item": "MLK", "event": "expiring_soon", "qty": 1}),
    ("Throw out the milk, it's bad", {"item": "MLK", "event": "spoiled", "qty": 1}),
    # eggs (EGG)
    ("Used 3 eggs for breakfast", {"item": "EGG", "event": "used_up", "qty": 3}),
    ("Got a dozen eggs today", {"item": "EGG", "event": "stocked", "qty": 12}),
    ("The eggs expire this weekend", {"item": "EGG", "event": "expiring_soon", "qty": 1}),
    ("Some eggs went rotten", {"item": "EGG", "event": "spoiled", "qty": 1}),
    # bread (BRD)
    ("Finished the bread", {"item": "BRD", "event": "used_up", "qty": 1}),
    ("Bought 2 loaves of bread", {"item": "BRD", "event": "stocked", "qty": 2}),
    ("Bread goes bad by Friday", {"item": "BRD", "event": "expiring_soon", "qty": 1}),
    ("The bread is moldy", {"item": "BRD", "event": "spoiled", "qty": 1}),
    # apples (APP)
    ("Ate the last 4 apples", {"item": "APP", "event": "used_up", "qty": 4}),
    ("Picked up 6 apples from the store", {"item": "APP", "event": "stocked", "qty": 6}),
    ("These apples are going bad soon", {"item": "APP", "event": "expiring_soon", "qty": 1}),
    ("The apples rotted", {"item": "APP", "event": "spoiled", "qty": 1}),
    # carrots (CRR)
    ("Used 5 carrots in the soup", {"item": "CRR", "event": "used_up", "qty": 5}),
    ("Added 10 carrots to the pantry", {"item": "CRR", "event": "stocked", "qty": 10}),
    ("Carrots are expiring next week", {"item": "CRR", "event": "expiring_soon", "qty": 1}),
    ("The carrots got mushy and slimy", {"item": "CRR", "event": "spoiled", "qty": 1}),
    # extras — reinforces item-code mapping with different phrasings
    ("One gallon of milk used", {"item": "MLK", "event": "used_up", "qty": 1}),
    ("Stocked 8 more eggs", {"item": "EGG", "event": "stocked", "qty": 8}),
]


# Held-out eval — 12 examples with distinct phrasings,
# all 5 items appear, all 4 events appear, some combos not in training.
EVAL_CASES = [
    # milk
    ("Added 3 more gallons of milk", {"item": "MLK", "event": "stocked", "qty": 3}),
    ("Milk expires tomorrow", {"item": "MLK", "event": "expiring_soon", "qty": 1}),
    # eggs
    ("Two eggs used for the cake", {"item": "EGG", "event": "used_up", "qty": 2}),
    ("The eggs smell off", {"item": "EGG", "event": "spoiled", "qty": 1}),
    # bread
    ("Just bought a fresh loaf of bread", {"item": "BRD", "event": "stocked", "qty": 1}),
    ("No bread left, ate it all", {"item": "BRD", "event": "used_up", "qty": 1}),
    # apples
    ("Tossed 2 apples, they were bruised", {"item": "APP", "event": "spoiled", "qty": 2}),
    ("Bought apples today, 5 of them", {"item": "APP", "event": "stocked", "qty": 5}),
    # carrots
    ("The last 2 carrots went in the stew", {"item": "CRR", "event": "used_up", "qty": 2}),
    ("Carrots are about to expire", {"item": "CRR", "event": "expiring_soon", "qty": 1}),
    # mixed
    ("Grabbed 4 more apples from the store", {"item": "APP", "event": "stocked", "qty": 4}),
    ("One carrot went bad in the fridge", {"item": "CRR", "event": "spoiled", "qty": 1}),
]


def format_completion(voice: str, args: dict) -> str:
    """Natural-looking completion so the model doesn't overfit an empty message.

    Uses a short realistic acknowledgment (mimics what the baseline naturally does).
    """
    # Generate a simple acknowledgment matching the action
    evt = args["event"]
    ack_map = {
        "stocked": "Logged as stocked.",
        "used_up": "Logged as used up.",
        "expiring_soon": "Flagged as expiring soon.",
        "spoiled": "Logged as spoiled.",
    }
    msg = ack_map.get(evt, "Logged.")
    response = {
        "message": msg,
        "tool_call": {"name": "log_pantry_event", "arguments": args},
    }
    return " " + json.dumps(response, ensure_ascii=False)


def build_dataset(cases: list[tuple[str, dict]]) -> dict:
    examples = []
    for voice, args in cases:
        examples.append({
            "voice_command": voice,
            "expected_tool_call": {"name": "log_pantry_event", "arguments": args},
            "formatted_system_prompt": SYSTEM_PROMPT,
            "formatted_completion": format_completion(voice, args),
        })
    return {
        "format": "inline-json",
        "data": {"commands": [{"command_name": "log_pantry_event", "examples": examples}]},
    }


def chat(messages: list, adapter_hash: str | None = None, max_tokens: int = 128) -> str:
    body = {
        "model": "live",
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    if adapter_hash:
        body["adapter_settings"] = {"hash": adapter_hash, "scale": 1.0, "enabled": True}
    headers = {"X-Internal-Token": INTERNAL_TOKEN}
    with httpx.Client(timeout=180) as client:
        r = client.post(f"{MODEL_BASE}/internal/model/chat", json=body, headers=headers)
        r.raise_for_status()
        return r.json().get("content", "")


def reload_model() -> None:
    headers = {"X-Internal-Token": INTERNAL_TOKEN}
    with httpx.Client(timeout=180) as client:
        r = client.post(f"{MODEL_BASE}/internal/model/reload", headers=headers)
        r.raise_for_status()


def try_parse_tool_call(content: str) -> dict | None:
    if not content:
        return None
    s = content.strip()
    candidates: list[str] = []
    if s.startswith("{"):
        candidates.append(s)
    first = content.find("{")
    last = content.rfind("}")
    if first >= 0 and last > first:
        candidates.append(content[first : last + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            tc = obj.get("tool_call")
            if isinstance(tc, dict) and tc.get("name"):
                return tc
            if obj.get("name") and isinstance(obj.get("arguments"), dict):
                return obj
    return None


def evaluate(case: tuple[str, dict], content: str) -> tuple[bool, str, dict]:
    voice, expected = case
    tc = try_parse_tool_call(content)
    if not tc:
        return False, "no tool_call parsed", {}
    actual = tc.get("arguments") or {}
    if tc.get("name") != "log_pantry_event":
        return False, f"wrong tool: {tc.get('name')!r}", actual

    failures = []
    for k, v in expected.items():
        av = actual.get(k)
        if isinstance(v, str) and isinstance(av, str):
            if v != av:
                failures.append(f"{k}: expected {v!r}, got {av!r}")
        else:
            if v != av:
                failures.append(f"{k}: expected {v!r}, got {av!r}")
    if failures:
        return False, "; ".join(failures), actual
    return True, "", actual


def run_eval_set(adapter_hash: str | None) -> tuple[int, list[dict]]:
    hits = 0
    rows = []
    for case in EVAL_CASES:
        voice, expected = case
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": voice},
        ]
        output = chat(messages, adapter_hash=adapter_hash, max_tokens=128)
        passed, reason, actual = evaluate(case, output)
        hits += int(passed)
        rows.append({
            "voice": voice,
            "expected": expected,
            "actual": actual,
            "output": output,
            "passed": passed,
            "reason": reason,
        })
    return hits, rows


def main() -> int:
    if not INTERNAL_TOKEN:
        print("ERROR: MODEL_SERVICE_TOKEN / LLM_PROXY_INTERNAL_TOKEN not set", file=sys.stderr)
        return 2

    print("=" * 60)
    print("Cell A — Step 3B: Fictional-Tool Accuracy Test")
    print("=" * 60)
    print(f"Training cases: {len(TRAINING_CASES)}")
    print(f"Eval cases (held-out): {len(EVAL_CASES)}")

    # Phase A: baseline
    print("\n[1/4] Phase A — baseline (no adapter, model freshly reloaded)...")
    reload_model()
    baseline_hits, baseline_rows = run_eval_set(adapter_hash=None)
    print(f"   baseline: {baseline_hits}/{len(EVAL_CASES)}")

    # Train
    print("\n[2/4] Training adapter (22 examples, direct call)...")
    ds = build_dataset(TRAINING_CASES)
    out_path = Path(__file__).parent / "pantry_training_dataset.json"
    out_path.write_text(json.dumps(ds, sort_keys=True), encoding="utf-8")
    job_id = f"pantry-{uuid.uuid4()}"
    request = {
        "node_id": "poc-pantry",
        "base_model_id": BASE_MODEL_ID,
        "dataset_ref": ds,
        "params": {
            "lora_r": 16,
            "lora_alpha": 32,
            "epochs": 4,
            "batch_size": 1,
            "max_seq_len": 1024,
            "learning_rate": 2e-4,
        },
    }
    import time
    t0 = time.time()
    result = run_adapter_training(request, job_id=job_id, ttl_seconds=1800)
    meta = result["artifact_metadata"]
    adapter_hash = meta["dataset_hash"]
    print(f"   training: {time.time() - t0:.1f}s (cached={meta['cached']})")
    print(f"   adapter_hash: {adapter_hash}")

    # Phase B: adapter
    print("\n[3/4] Phase B — with adapter loaded...")
    adapter_hits, adapter_rows = run_eval_set(adapter_hash=adapter_hash)
    print(f"   with adapter: {adapter_hits}/{len(EVAL_CASES)}")

    # Report
    print("\n[4/4] Per-case comparison:")
    for base_row, adp_row in zip(baseline_rows, adapter_rows):
        b = "✓" if base_row["passed"] else "·"
        a = "✓" if adp_row["passed"] else "·"
        print(f'   {b}→{a}  "{base_row["voice"][:60]}"')
        print(f'      expected: {base_row["expected"]}')
        if not base_row["passed"]:
            print(f"      NO : {base_row['reason']}")
            print(f"           actual args: {base_row['actual']}")
        if not adp_row["passed"]:
            print(f"      YES: {adp_row['reason']}")
            print(f"           actual args: {adp_row['actual']}")

    total = len(EVAL_CASES)
    baseline_pct = 100.0 * baseline_hits / total
    adapter_pct = 100.0 * adapter_hits / total
    delta_pp = adapter_pct - baseline_pct

    print("\n" + "=" * 60)
    print(f"BASELINE: {baseline_hits}/{total} = {baseline_pct:.1f}%")
    print(f"ADAPTER:  {adapter_hits}/{total} = {adapter_pct:.1f}%")
    print(f"DELTA:    {delta_pp:+.1f}pp")

    if delta_pp >= 15:
        print("VERDICT: PASS — LoRA can teach an unknown tool schema on OSX.")
        return 0
    elif delta_pp > 0:
        print("VERDICT: MARGINAL — adapter helps but below +15pp threshold.")
        return 0
    else:
        print("VERDICT: FAIL — adapter did not improve accuracy.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
