"""Phase 2.5 extension — model-driven ablation.

Instead of hand-coded variants, ask the SERVING model to generate its own
stripped versions of the full system prompt. Evaluate each against the V2
adapter and the same held-out eval set.

Two-stage flow:
  1. Send the full system prompt to the local model with a meta-prompt
     asking for N progressively-stripped variants.
  2. Parse + evaluate each generated variant using the same pass-rate
     machinery as `ablation_report.py`.

Baseline to compare against: the hand-coded variants in
`ablation_prompt_variants.py` and their V2 pass rates.

Usage:
    export MODEL_SERVICE_TOKEN=<internal-token>
    .venv/bin/python poc/model_ablator.py \
        --adapter-hash fa9de9677c8fab3611ba5d546439e6a8fecfe8c80951076bcd3eebcc944df1a6 \
        --max-eval 40
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ablation_prompt_variants import all_variants, load_spec  # noqa: E402
from ablation_report import (  # noqa: E402
    chat,
    evaluate,
    load_eval,
    reload_model,
    run_cell,
    write_csv,
    write_markdown,
)

MODEL_BASE = "http://localhost:7705"
INTERNAL_TOKEN = os.getenv("MODEL_SERVICE_TOKEN") or os.getenv("LLM_PROXY_INTERNAL_TOKEN") or ""


# --------------------------------------------------------------------------
# Meta-prompt sent to the model to elicit ablation proposals
# --------------------------------------------------------------------------


META_PROMPT_TEMPLATE = """You are analyzing a voice-assistant system prompt.

The assistant has a LoRA adapter trained to handle voice commands. The
adapter has internalized the response format and routing patterns, so
much of the system prompt may be redundant now.

Your job: propose THREE progressively-stripped versions of the prompt
below, from least-aggressive to most-aggressive.

Rules:
- Each variant must be a complete, self-contained system prompt.
- Always keep the list of tool names and their parameters.
- The "minor" variant removes ~25% of characters.
- The "moderate" variant removes ~50%.
- The "aggressive" variant removes ~75%, keeping only what's strictly essential.

Output ONLY this JSON (no markdown, no commentary):

{{"variants": [
  {{"name": "minor", "prompt": "..."}},
  {{"name": "moderate", "prompt": "..."}},
  {{"name": "aggressive", "prompt": "..."}}
]}}

=== FULL PROMPT TO ABLATE ===

{full_prompt}

=== END ===
"""


# --------------------------------------------------------------------------
# Variant dataclass matching ablation_prompt_variants.Variant API
# --------------------------------------------------------------------------


@dataclass
class ModelVariant:
    name: str
    prompt: str
    char_count: int
    approx_tokens: int
    rationale: str = ""


# --------------------------------------------------------------------------
# Ask the model for ablations
# --------------------------------------------------------------------------


def request_ablations(
    client: httpx.Client,
    full_prompt: str,
    *,
    adapter_hash: str | None = None,
    max_tokens: int = 4000,
) -> tuple[list[ModelVariant], str]:
    """Return (variants, raw_response_text).

    We deliberately issue the meta-prompt as the USER turn and leave the
    system prompt empty — we don't want the voice-assistant persona
    interfering with the meta-reasoning task.
    """
    body = {
        "model": "live",
        "messages": [
            {"role": "system", "content": "You are a careful analyst. Output only valid JSON when asked."},
            {"role": "user", "content": META_PROMPT_TEMPLATE.format(full_prompt=full_prompt)},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    if adapter_hash:
        body["adapter_settings"] = {"hash": adapter_hash, "scale": 1.0, "enabled": True}
    headers = {"X-Internal-Token": INTERNAL_TOKEN}
    r = client.post(f"{MODEL_BASE}/internal/model/chat", json=body, headers=headers)
    r.raise_for_status()
    raw = r.json().get("content") or ""

    parsed = _parse_model_variants(raw)
    return parsed, raw


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)


def _parse_model_variants(raw: str) -> list[ModelVariant]:
    """Tolerant JSON extractor. Small local models often emit one malformed
    object (usually the one containing a nested JSON-format example with
    mis-escaped quotes) while the rest parse fine. We want to recover what
    we can.

    Strategy:
      1. Try parsing the whole {"variants": [...]} as strict JSON.
      2. If that fails, slice the raw text between successive `{"name":`
         markers and attempt to parse each object individually.
    """
    if not raw:
        return []
    text = _strip_code_fences(raw)

    # --- strict path ---
    first = text.find("{")
    last = text.rfind("}")
    if 0 <= first < last:
        try:
            obj = json.loads(text[first : last + 1])
            variants_raw = obj.get("variants") if isinstance(obj, dict) else None
            if isinstance(variants_raw, list):
                return [mv for v in variants_raw if (mv := _coerce_variant(v))]
        except json.JSONDecodeError:
            pass

    # --- per-object recovery path ---
    recovered: list[ModelVariant] = []
    # Find every `{"name": ...}` object by tracking brace depth.
    import re
    starts = [m.start() for m in re.finditer(r'\{\s*"name"\s*:\s*', text)]
    for i, start in enumerate(starts):
        end_limit = starts[i + 1] if i + 1 < len(starts) else len(text)
        slice_ = text[start:end_limit]
        end = _find_matching_brace(slice_)
        if end is None:
            continue
        candidate = slice_[: end + 1]
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            # Recovery step: the prompt string may contain a nested JSON
            # example with mis-escaped quotes. Extract `name` + `prompt`
            # fields using regex over the slice and accept whatever the
            # model intended.
            mv = _regex_recover_variant(candidate)
            if mv:
                recovered.append(mv)
            continue
        mv = _coerce_variant(obj)
        if mv:
            recovered.append(mv)
    return recovered


def _find_matching_brace(s: str) -> int | None:
    """Assuming s starts with '{', return the index of the matching '}'.
    Tracks string-state so braces inside quoted strings don't confuse us.
    """
    if not s or s[0] != "{":
        return None
    depth = 0
    in_str = False
    esc = False
    for i, c in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def _coerce_variant(v: object) -> ModelVariant | None:
    if not isinstance(v, dict):
        return None
    name = v.get("name") or "variant"
    prompt = v.get("prompt")
    rationale = v.get("rationale", "")
    if not isinstance(prompt, str) or not prompt.strip():
        return None
    chars = len(prompt)
    return ModelVariant(
        name=str(name),
        prompt=prompt,
        char_count=chars,
        approx_tokens=chars // 4,
        rationale=str(rationale),
    )


def _regex_recover_variant(candidate: str) -> ModelVariant | None:
    """Fallback extraction when JSON parsing fails on one variant.

    The most common breakage is mis-escaped inner quotes in the `prompt`
    string. We extract `name` strictly and `prompt` greedily (from after
    the opening quote to the last quote before `"}`).
    """
    import re
    name_m = re.search(r'"name"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', candidate)
    if not name_m:
        return None
    name = name_m.group(1)
    # Find the `"prompt": "` start, then the last `"}` in the candidate
    prompt_start = candidate.find('"prompt"')
    if prompt_start < 0:
        return None
    # Skip past `"prompt":` and the opening quote of the value
    after_key = candidate.find(":", prompt_start) + 1
    # Advance to the first quote AFTER the colon (tolerant of whitespace)
    while after_key < len(candidate) and candidate[after_key] in " \t\n":
        after_key += 1
    if after_key >= len(candidate) or candidate[after_key] != '"':
        return None
    value_start = after_key + 1
    # Best-effort: prompt ends at the last `"` before `}`
    close_brace = candidate.rfind("}")
    if close_brace <= value_start:
        return None
    quote_end = candidate.rfind('"', value_start, close_brace)
    if quote_end <= value_start:
        return None
    prompt = candidate[value_start:quote_end]
    # Un-escape common sequences that came through intact
    prompt = prompt.encode("utf-8", "ignore").decode("unicode_escape", "ignore")
    if not prompt.strip():
        return None
    chars = len(prompt)
    return ModelVariant(
        name=name,
        prompt=prompt,
        char_count=chars,
        approx_tokens=chars // 4,
    )


# --------------------------------------------------------------------------
# Compare: model-driven vs hand-coded
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Model-driven ablation POC")
    ap.add_argument("--adapter-hash", type=str, required=True,
                    help="Adapter to test each variant against")
    ap.add_argument("--eval", type=Path, default=Path(__file__).parent / "synth_eval.jsonl")
    ap.add_argument("--max-eval", type=int, default=40)
    ap.add_argument("--out-csv", type=Path, default=Path(__file__).parent / "model_ablation_report.csv")
    ap.add_argument("--out-md", type=Path, default=Path(__file__).parent / "model_ablation_report.md")
    ap.add_argument("--raw-out", type=Path, default=Path(__file__).parent / "model_ablation_raw.txt")
    ap.add_argument("--use-adapter-for-meta", action="store_true",
                    help="Use the adapter when asking the model for variants "
                         "(default: off — the adapter is trained for voice commands, "
                         "not for meta-reasoning)")
    args = ap.parse_args()

    if not INTERNAL_TOKEN:
        print("ERROR: MODEL_SERVICE_TOKEN not set", file=sys.stderr)
        return 2

    spec = load_spec()
    hand_variants = all_variants(spec)
    full_prompt = next(v.prompt for v in hand_variants if v.name == "full")

    eval_rows = load_eval(args.eval, max_rows=args.max_eval)
    print(f"eval rows: {len(eval_rows)}")

    with httpx.Client(timeout=240) as client:
        # --- Stage 1: ask the model for variants ---
        print("\n=== Stage 1: asking model for ablation variants ===")
        reload_model(client)  # fresh state
        t0 = time.time()
        model_variants, raw = request_ablations(
            client, full_prompt,
            adapter_hash=args.adapter_hash if args.use_adapter_for_meta else None,
        )
        print(f"  model responded in {time.time() - t0:.1f}s, {len(raw)} chars")
        args.raw_out.write_text(raw, encoding="utf-8")
        print(f"  raw response → {args.raw_out}")

        if not model_variants:
            print("FAIL: model did not produce parseable variants.")
            print("Tail of raw response:")
            print(raw[-800:])
            return 1

        print(f"  parsed {len(model_variants)} model-proposed variants:")
        for mv in model_variants:
            print(f"    {mv.name:<14} {mv.char_count:>6} chars  ~{mv.approx_tokens:>5} toks")

        # --- Stage 2: evaluate each model-proposed variant ---
        print("\n=== Stage 2: evaluating model-proposed variants ===")
        reload_model(client)
        cells = []
        for mv in model_variants:
            print(f"▶ model variant: {mv.name}  ({mv.char_count} chars, ~{mv.approx_tokens} toks)")
            cell = run_cell(
                client,
                # Adapter onto cell.variant type — a small adapter struct
                _as_cell_variant(mv),
                condition="adapter",
                adapter_hash=args.adapter_hash,
                eval_rows=eval_rows,
            )
            cells.append(cell)
            print()

        # --- Stage 3: same eval against our hand-coded variants for apples-to-apples ---
        print("=== Stage 3: hand-coded variants (for comparison) ===")
        reload_model(client)
        for hv in hand_variants:
            print(f"▶ hand variant: {hv.name}  ({hv.char_count} chars, ~{hv.approx_tokens} toks)")
            cell = run_cell(
                client, hv, condition="adapter",
                adapter_hash=args.adapter_hash, eval_rows=eval_rows,
            )
            cells.append(cell)
            print()

    # --- Reports ---
    write_csv(args.out_csv, cells)
    write_markdown(args.out_md, cells, args.adapter_hash)
    print(f"→ csv:      {args.out_csv}")
    print(f"→ markdown: {args.out_md}")
    print()
    print(args.out_md.read_text(encoding="utf-8"))

    # Final comparison
    print("\n" + "=" * 60)
    print("Model-driven ablation results (adapter on):")
    print("=" * 60)
    for cell in cells:
        print(f"  {cell.variant:<20} {cell.prompt_tokens_avg:>6.0f} toks  "
              f"{cell.pass_rate:>5.1f}%  p50={cell.p50:.2f}s")
    return 0


def _as_cell_variant(mv: ModelVariant):
    """Adapt a ModelVariant to the duck-typed shape run_cell expects.

    run_cell uses: variant.name, variant.prompt, variant.char_count,
    variant.approx_tokens. The ModelVariant dataclass already has all four.
    """
    return mv


if __name__ == "__main__":
    raise SystemExit(main())
