"""Cell A — Step 3: real accuracy test on music command parsing.

Uses 20 music-command training examples + 12 held-out eval examples (drawn
from jarvis-node-setup/test_command_parsing.py). Measures baseline vs adapter
pass rate on the held-out set.

Success criteria (Cell A gate):
- PASS:     adapter − baseline ≥ +15pp
- MARGINAL: 0–15pp delta
- FAIL:     ≤ 0pp delta

Usage:
    .venv/bin/python poc/accuracy_test.py
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

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

# A trimmed but production-shaped system prompt with 5 tools including music.
# Mirrors the structure of system_prompt_builder.py's "compact" style.
SYSTEM_PROMPT = """You are Jarvis, a voice-controlled assistant that operates BY CALLING TOOLS.

Node Context:
- Room: living room
- User: Alex
- Voice Mode: brief

YOUR PRIMARY ROLE: You are a tool router and parameter extractor.
- Analyze the user's request
- Determine which available tool(s) to call
- Extract parameters from their message
- Call the appropriate tool(s)

CRITICAL - Response Format (VALID JSON ONLY):
You MUST respond with valid JSON only. No comments, no explanations, no markdown, no code blocks.

When calling a tool:
{"message": "brief acknowledgment", "tool_call": {"name": "tool_name", "arguments": {"param": "value"}}}

When you have the final answer:
{"message": "your response", "tool_call": null}

Available Tools:

1. music — Control music playback.
   Parameters:
     action (string, required): "play" | "pause" | "resume" | "stop" | "next" | "previous" | "volume_up" | "volume_down" | "volume_set" | "mute" | "shuffle_on" | "shuffle_off" | "repeat_one" | "repeat_off"
     query (string, optional): Artist, album, track, genre, or mood to play
     player (string, optional): Target room/player
     queue_option (string, optional): "add" | "next" (queue modification for play)
     volume_level (integer, optional): 0–100 for volume_set

2. get_weather — Get current weather.
   Parameters: location (string, optional)

3. get_time — Get current time.
   Parameters: (none)

4. set_timer — Start a timer.
   Parameters: duration_minutes (integer, required), label (string, optional)

5. calculate — Math calculation.
   Parameters: expression (string, required)"""


# Training set — 20 examples (kept deliberately diverse across actions).
TRAINING_CASES = [
    # play: artist
    ("Play Radiohead", {"action": "play", "query": "Radiohead"}),
    ("Put on some Beatles", {"action": "play", "query": "Beatles"}),
    ("Play Taylor Swift", {"action": "play", "query": "Taylor Swift"}),
    # play: album
    ("Play the album Abbey Road", {"action": "play", "query": "Abbey Road"}),
    # play: track
    ("Play Karma Police", {"action": "play", "query": "Karma Police"}),
    # play: genre
    ("Play some jazz", {"action": "play", "query": "jazz"}),
    ("Put on classical music", {"action": "play", "query": "classical"}),
    # play: with player
    ("Play Beatles in the kitchen", {"action": "play", "query": "Beatles", "player": "kitchen"}),
    # play: queue
    ("Add Bohemian Rhapsody to the queue", {"action": "play", "query": "Bohemian Rhapsody", "queue_option": "add"}),
    # control basic
    ("Pause the music", {"action": "pause"}),
    ("Resume", {"action": "resume"}),
    ("Stop the music", {"action": "stop"}),
    # skip/nav
    ("Skip this song", {"action": "next"}),
    ("Previous song", {"action": "previous"}),
    # volume
    ("Turn up the volume", {"action": "volume_up"}),
    ("Turn it down", {"action": "volume_down"}),
    ("Set volume to 50", {"action": "volume_set", "volume_level": 50}),
    ("Mute", {"action": "mute"}),
    # shuffle/repeat
    ("Turn on shuffle", {"action": "shuffle_on"}),
    ("Repeat this song", {"action": "repeat_one"}),
]

# Held-out eval set — 12 examples (different phrasings or categories from training).
EVAL_CASES = [
    ("Play OK Computer", {"action": "play", "query": "OK Computer"}),                # album (not in train)
    ("Play the song Bohemian Rhapsody", {"action": "play", "query": "Bohemian Rhapsody"}),  # 'the song' prefix
    ("Play relaxing music", {"action": "play", "query": "relaxing"}),                # mood (not in train)
    ("Play jazz in the living room", {"action": "play", "query": "jazz", "player": "living room"}),
    ("Play Stairway to Heaven next", {"action": "play", "query": "Stairway to Heaven", "queue_option": "next"}),
    ("Next song", {"action": "next"}),                                               # "next song" phrasing
    ("Go back", {"action": "previous"}),                                             # casual phrasing
    ("Louder", {"action": "volume_up"}),                                             # casual
    ("Quieter", {"action": "volume_down"}),                                          # casual
    ("Shuffle off", {"action": "shuffle_off"}),
    ("Stop repeating", {"action": "repeat_off"}),
    ("Pause the kitchen speaker", {"action": "pause", "player": "kitchen"}),
]


def format_completion(tool_call: dict) -> str:
    """Match the production response format (app/core/model_service.py expectation)."""
    response = {
        "message": "",
        "tool_call": tool_call,
    }
    return " " + json.dumps(response, ensure_ascii=False)


def build_dataset(cases: list[tuple[str, dict]]) -> dict:
    examples = []
    for voice, args in cases:
        tool_call = {"name": "music", "arguments": args}
        examples.append({
            "voice_command": voice,
            "expected_tool_call": tool_call,
            "formatted_system_prompt": SYSTEM_PROMPT,
            "formatted_completion": format_completion(tool_call),
        })
    return {
        "format": "inline-json",
        "data": {"commands": [{"command_name": "music", "examples": examples}]},
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
    """Best-effort extract tool_call from model output.

    Accepts either bare JSON or JSON embedded in text. Returns the tool_call dict
    if one is found with a 'name' field, else None.
    """
    if not content:
        return None

    # Direct JSON
    candidates: list[str] = []
    s = content.strip()
    if s.startswith("{"):
        candidates.append(s)

    # JSON block inside text (greedy from first { to last })
    first = content.find("{")
    last = content.rfind("}")
    if first >= 0 and last > first:
        candidates.append(content[first : last + 1])

    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        # Production format: {"tool_call": {...}} or {"tool_calls": [{...}]}
        if isinstance(obj, dict):
            tc = obj.get("tool_call")
            if isinstance(tc, dict) and tc.get("name"):
                return tc
            tcs = obj.get("tool_calls")
            if isinstance(tcs, list) and tcs and isinstance(tcs[0], dict) and tcs[0].get("name"):
                return tcs[0]
            # Maybe the object IS the tool_call already
            if obj.get("name") and isinstance(obj.get("arguments"), dict):
                return obj
    return None


def evaluate(case: tuple[str, dict], content: str) -> tuple[bool, str]:
    """Return (passed, failure_reason)."""
    voice, expected_args = case
    tc = try_parse_tool_call(content)
    if not tc:
        return False, "no tool_call parsed"
    if tc.get("name") != "music":
        return False, f"wrong tool: {tc.get('name')!r}"
    actual_args = tc.get("arguments") or {}
    # Match: every expected key must be present with equal value (case-insensitive string)
    for k, v in expected_args.items():
        av = actual_args.get(k)
        if isinstance(v, str) and isinstance(av, str):
            if v.strip().lower() != av.strip().lower():
                return False, f"arg {k!r}: expected {v!r}, got {av!r}"
        else:
            if v != av:
                return False, f"arg {k!r}: expected {v!r}, got {av!r}"
    return True, ""


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
        passed, reason = evaluate(case, output)
        hits += int(passed)
        rows.append({
            "voice": voice,
            "expected": expected,
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
    print("Cell A — Step 3: Real Accuracy Test (music command)")
    print("=" * 60)
    print(f"Training cases: {len(TRAINING_CASES)}")
    print(f"Eval cases (held-out): {len(EVAL_CASES)}")

    # 1. Baseline eval (no adapter). Must happen BEFORE training because MLX sticky-adapter.
    print("\n[1/4] Phase A — baseline (no adapter, model freshly reloaded)...")
    reload_model()
    print("   model reloaded")
    baseline_hits, baseline_rows = run_eval_set(adapter_hash=None)
    print(f"   baseline: {baseline_hits}/{len(EVAL_CASES)}")

    # 2. Train adapter on the 20-case training set
    print("\n[2/4] Training adapter (20 music examples, direct call)...")
    ds = build_dataset(TRAINING_CASES)
    out_path = Path(__file__).parent / "music_training_dataset.json"
    out_path.write_text(json.dumps(ds, sort_keys=True), encoding="utf-8")
    job_id = f"music-{uuid.uuid4()}"
    request = {
        "node_id": "poc-music",
        "base_model_id": BASE_MODEL_ID,
        "dataset_ref": ds,
        "params": {
            "lora_r": 16,
            "lora_alpha": 32,
            "epochs": 3,
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
    print(f"   training: {time.time() - t0:.1f}s (train_duration={meta.get('train_duration_seconds', 'cached')}, cached={meta['cached']})")
    print(f"   adapter_hash: {adapter_hash}")

    # 3. Adapter eval
    print("\n[3/4] Phase B — with adapter loaded...")
    adapter_hits, adapter_rows = run_eval_set(adapter_hash=adapter_hash)
    print(f"   with adapter: {adapter_hits}/{len(EVAL_CASES)}")

    # 4. Report
    print("\n[4/4] Per-case comparison:")
    for base_row, adp_row in zip(baseline_rows, adapter_rows):
        b = "✓" if base_row["passed"] else "·"
        a = "✓" if adp_row["passed"] else "·"
        print(f'   {b}→{a}  "{base_row["voice"][:50]}"')
        if not base_row["passed"]:
            print(f"      NO : {base_row['reason']}")
            print(f"           output: {base_row['output'][:120]!r}")
        if not adp_row["passed"]:
            print(f"      YES: {adp_row['reason']}")
            print(f"           output: {adp_row['output'][:120]!r}")

    total = len(EVAL_CASES)
    baseline_pct = 100.0 * baseline_hits / total
    adapter_pct = 100.0 * adapter_hits / total
    delta_pp = adapter_pct - baseline_pct

    print("\n" + "=" * 60)
    print(f"BASELINE: {baseline_hits}/{total} = {baseline_pct:.1f}%")
    print(f"ADAPTER:  {adapter_hits}/{total} = {adapter_pct:.1f}%")
    print(f"DELTA:    {delta_pp:+.1f}pp")

    if delta_pp >= 15:
        print("VERDICT: PASS — adapter meaningfully improves music parsing on OSX.")
        return 0
    elif delta_pp > 0:
        print("VERDICT: MARGINAL — adapter helps but below +15pp threshold.")
        return 0  # still counts as working, caller decides next step
    else:
        print("VERDICT: FAIL — adapter did not improve accuracy.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
