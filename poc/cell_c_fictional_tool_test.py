"""Cell C — fictional-tool accuracy test, GGUF serving + MLX training + GGUF conversion.

Same dataset and eval set as Cell A's fictional_tool_test.py. Differences:
  - base_model_id points to Hermes-3-Llama-3.1-8B Q4_K_M GGUF (for llama_cpp serving)
  - params.hf_base_model_id points to the HF-format version in .models/Hermes-3-Llama-3.1-8B
    (MLX training needs HF-format weights)
  - MLX-LM trains, converts to PEFT, then convert_lora_to_gguf.py emits gguf/adapter.gguf
  - llama_cpp serving loads the GGUF adapter via adapter_settings.hash

Gate:
  PASS:     adapter − baseline ≥ +15pp  → GGUF-on-OSX works
  MARGINAL: 0–15pp delta
  FAIL:     ≤ 0pp                       → OSX-specific GGUF bug confirmed

Usage:
    .venv/bin/python poc/cell_c_fictional_tool_test.py
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
BASE_MODEL_ID = ".models/Hermes-3-Llama-3.1-8B-Q4_K_M.gguf"
HF_BASE_MODEL_ID = ".models/Hermes-3-Llama-3.1-8B"

INTERNAL_TOKEN = (
    os.getenv("MODEL_SERVICE_TOKEN")
    or os.getenv("LLM_PROXY_INTERNAL_TOKEN")
    or ""
)

# Same system prompt as Cell A — direct comparison
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


TRAINING_CASES = [
    ("I used all the milk", {"item": "MLK", "event": "used_up", "qty": 1}),
    ("Bought 2 gallons of milk", {"item": "MLK", "event": "stocked", "qty": 2}),
    ("The milk is expiring tomorrow", {"item": "MLK", "event": "expiring_soon", "qty": 1}),
    ("Throw out the milk, it's bad", {"item": "MLK", "event": "spoiled", "qty": 1}),
    ("Used 3 eggs for breakfast", {"item": "EGG", "event": "used_up", "qty": 3}),
    ("Got a dozen eggs today", {"item": "EGG", "event": "stocked", "qty": 12}),
    ("The eggs expire this weekend", {"item": "EGG", "event": "expiring_soon", "qty": 1}),
    ("Some eggs went rotten", {"item": "EGG", "event": "spoiled", "qty": 1}),
    ("Finished the bread", {"item": "BRD", "event": "used_up", "qty": 1}),
    ("Bought 2 loaves of bread", {"item": "BRD", "event": "stocked", "qty": 2}),
    ("Bread goes bad by Friday", {"item": "BRD", "event": "expiring_soon", "qty": 1}),
    ("The bread is moldy", {"item": "BRD", "event": "spoiled", "qty": 1}),
    ("Ate the last 4 apples", {"item": "APP", "event": "used_up", "qty": 4}),
    ("Picked up 6 apples from the store", {"item": "APP", "event": "stocked", "qty": 6}),
    ("These apples are going bad soon", {"item": "APP", "event": "expiring_soon", "qty": 1}),
    ("The apples rotted", {"item": "APP", "event": "spoiled", "qty": 1}),
    ("Used 5 carrots in the soup", {"item": "CRR", "event": "used_up", "qty": 5}),
    ("Added 10 carrots to the pantry", {"item": "CRR", "event": "stocked", "qty": 10}),
    ("Carrots are expiring next week", {"item": "CRR", "event": "expiring_soon", "qty": 1}),
    ("The carrots got mushy and slimy", {"item": "CRR", "event": "spoiled", "qty": 1}),
    ("One gallon of milk used", {"item": "MLK", "event": "used_up", "qty": 1}),
    ("Stocked 8 more eggs", {"item": "EGG", "event": "stocked", "qty": 8}),
]

EVAL_CASES = [
    ("Added 3 more gallons of milk", {"item": "MLK", "event": "stocked", "qty": 3}),
    ("Milk expires tomorrow", {"item": "MLK", "event": "expiring_soon", "qty": 1}),
    ("Two eggs used for the cake", {"item": "EGG", "event": "used_up", "qty": 2}),
    ("The eggs smell off", {"item": "EGG", "event": "spoiled", "qty": 1}),
    ("Just bought a fresh loaf of bread", {"item": "BRD", "event": "stocked", "qty": 1}),
    ("No bread left, ate it all", {"item": "BRD", "event": "used_up", "qty": 1}),
    ("Tossed 2 apples, they were bruised", {"item": "APP", "event": "spoiled", "qty": 2}),
    ("Bought apples today, 5 of them", {"item": "APP", "event": "stocked", "qty": 5}),
    ("The last 2 carrots went in the stew", {"item": "CRR", "event": "used_up", "qty": 2}),
    ("Carrots are about to expire", {"item": "CRR", "event": "expiring_soon", "qty": 1}),
    ("Grabbed 4 more apples from the store", {"item": "APP", "event": "stocked", "qty": 4}),
    ("One carrot went bad in the fridge", {"item": "CRR", "event": "spoiled", "qty": 1}),
]


def format_completion(args: dict) -> str:
    ack_map = {
        "stocked": "Logged as stocked.",
        "used_up": "Logged as used up.",
        "expiring_soon": "Flagged as expiring soon.",
        "spoiled": "Logged as spoiled.",
    }
    msg = ack_map.get(args["event"], "Logged.")
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
            "formatted_completion": format_completion(args),
        })
    return {
        "format": "inline-json",
        "data": {"commands": [{"command_name": "log_pantry_event", "examples": examples}]},
    }


def chat(messages, adapter_hash=None, max_tokens=128):
    body = {"model": "live", "messages": messages, "temperature": 0.0, "max_tokens": max_tokens}
    if adapter_hash:
        body["adapter_settings"] = {"hash": adapter_hash, "scale": 1.0, "enabled": True}
    headers = {"X-Internal-Token": INTERNAL_TOKEN}
    with httpx.Client(timeout=180) as client:
        r = client.post(f"{MODEL_BASE}/internal/model/chat", json=body, headers=headers)
        r.raise_for_status()
        return r.json().get("content", "")


def reload_model():
    headers = {"X-Internal-Token": INTERNAL_TOKEN}
    with httpx.Client(timeout=180) as client:
        r = client.post(f"{MODEL_BASE}/internal/model/reload", headers=headers)
        r.raise_for_status()


def try_parse_tool_call(content):
    if not content:
        return None
    s = content.strip()
    candidates = []
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


def evaluate(case, content):
    expected = case[1]
    tc = try_parse_tool_call(content)
    if not tc:
        return False, "no tool_call parsed", {}
    actual = tc.get("arguments") or {}
    if tc.get("name") != "log_pantry_event":
        return False, f"wrong tool: {tc.get('name')!r}", actual
    failures = []
    for k, v in expected.items():
        av = actual.get(k)
        if v != av:
            failures.append(f"{k}: expected {v!r}, got {av!r}")
    if failures:
        return False, "; ".join(failures), actual
    return True, "", actual


def run_eval_set(adapter_hash):
    hits = 0
    rows = []
    for case in EVAL_CASES:
        voice = case[0]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": voice},
        ]
        output = chat(messages, adapter_hash=adapter_hash, max_tokens=128)
        passed, reason, actual = evaluate(case, output)
        hits += int(passed)
        rows.append({
            "voice": voice,
            "expected": case[1],
            "actual": actual,
            "output": output,
            "passed": passed,
            "reason": reason,
        })
    return hits, rows


def main():
    if not INTERNAL_TOKEN:
        print("ERROR: MODEL_SERVICE_TOKEN / LLM_PROXY_INTERNAL_TOKEN not set", file=sys.stderr)
        return 2

    print("=" * 60)
    print("Cell C — GGUF serving + MLX training + GGUF conversion on OSX")
    print("=" * 60)
    print(f"Base model (serving):  {BASE_MODEL_ID}")
    print(f"HF base (training):    {HF_BASE_MODEL_ID}")
    print(f"Training cases: {len(TRAINING_CASES)}")
    print(f"Eval cases (held-out): {len(EVAL_CASES)}")

    # Phase A: baseline
    print("\n[1/4] Phase A — baseline (no adapter, model freshly reloaded)...")
    reload_model()
    baseline_hits, baseline_rows = run_eval_set(adapter_hash=None)
    print(f"   baseline: {baseline_hits}/{len(EVAL_CASES)}")

    # Train (MLX → PEFT → GGUF)
    print("\n[2/4] Training adapter (MLX → PEFT → GGUF)...")
    ds = build_dataset(TRAINING_CASES)
    out_path = Path(__file__).parent / "cell_c_pantry_training_dataset.json"
    out_path.write_text(json.dumps(ds, sort_keys=True), encoding="utf-8")
    job_id = f"pantry-gguf-{uuid.uuid4()}"
    request = {
        "node_id": "poc-pantry-gguf",
        "base_model_id": BASE_MODEL_ID,
        "dataset_ref": ds,
        "params": {
            "lora_r": 16,
            "lora_alpha": 32,
            "epochs": 4,
            "batch_size": 1,
            "max_seq_len": 1024,
            "learning_rate": 2e-4,
            "hf_base_model_id": HF_BASE_MODEL_ID,
        },
    }
    import time
    t0 = time.time()
    result = run_adapter_training(request, job_id=job_id, ttl_seconds=1800)
    meta = result["artifact_metadata"]
    adapter_hash = meta["dataset_hash"]
    print(f"   training: {time.time() - t0:.1f}s (cached={meta['cached']})")
    print(f"   adapter_hash: {adapter_hash}")
    print(f"   adapter_format: {meta['adapter_format']}")

    # Verify the adapter.zip contains a GGUF — critical for llama_cpp serving
    zip_path = Path(meta["artifact_path"])
    if zip_path.exists():
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
        has_gguf = any(n.startswith("gguf/") and n.endswith(".gguf") for n in names)
        has_peft = any(n in ("peft/adapter_model.safetensors", "adapter_model.safetensors") for n in names)
        print(f"   artifact contents — gguf: {has_gguf}, peft: {has_peft}")
        if not has_gguf:
            print("   ⚠️  no GGUF adapter in artifact — llama_cpp can't load it")

    # Phase B: adapter
    print("\n[3/4] Phase B — with adapter loaded...")
    adapter_hits, adapter_rows = run_eval_set(adapter_hash=adapter_hash)
    print(f"   with adapter: {adapter_hits}/{len(EVAL_CASES)}")

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
        print("VERDICT: PASS — GGUF-on-OSX LoRA pipeline works.")
        return 0
    elif delta_pp > 0:
        print("VERDICT: MARGINAL — adapter helps but below +15pp threshold.")
        return 0
    else:
        print("VERDICT: FAIL — GGUF adapter loading/conversion likely broken on OSX.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
