"""Three-pass local ablator (no external API).

Key insight: give the LLM ONLY the task it's good at. Everything else is
deterministic or mechanical.

Pass 1 (LLM — Identify):
  Input: method source + guidance about what the adapter has internalized.
  Output: a JSON list of {original, replacement, reason} edits, where
  `original` is an EXACT substring of the source.
  Model only has to recognize removable text — not generate syntactically
  correct code.

Pass 2 (Deterministic — Apply):
  For each edit: if `original` appears EXACTLY once in the current source,
  replace it. If it doesn't match exactly, REJECT that edit (don't fuzzy-
  match — that's how hallucinated corruption creeps in). Dedupe. Preserve
  ordering by application.

Pass 3 (ast.parse + conditional LLM — Validate):
  Run ast.parse on the result. If it parses, done. If it fails, ask the
  local model for a MINIMAL fix for just the syntax error with the error
  message included in context. Retry once, then bail.

Result: produced variant Python that is either structurally intact or
reported as failed, never silently corrupted.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

MODEL_BASE = "http://localhost:7705"
INTERNAL_TOKEN = os.getenv("MODEL_SERVICE_TOKEN") or os.getenv("LLM_PROXY_INTERNAL_TOKEN") or ""

PROVIDER_FILE = Path(
    "/Users/alexanderberardi/jarvis/jarvis-pp-hermes/prompt_providers/hermes/hermes_medium_mlx.py"
)


# --------------------------------------------------------------------------
# LLM helpers
# --------------------------------------------------------------------------


def chat(client: httpx.Client, system: str, user: str, *,
         max_tokens: int = 2500, temperature: float = 0.1) -> tuple[str, float]:
    body = {
        "model": "live",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"X-Internal-Token": INTERNAL_TOKEN}
    t0 = time.time()
    r = client.post(f"{MODEL_BASE}/internal/model/chat", json=body, headers=headers)
    r.raise_for_status()
    return r.json().get("content", ""), time.time() - t0


def reload_model(client: httpx.Client) -> None:
    headers = {"X-Internal-Token": INTERNAL_TOKEN}
    r = client.post(f"{MODEL_BASE}/internal/model/reload", headers=headers, timeout=180)
    r.raise_for_status()


# --------------------------------------------------------------------------
# Method extraction (AST)
# --------------------------------------------------------------------------


def extract_method(source: str, method_name: str) -> str:
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
            start_line = node.decorator_list[0].lineno if node.decorator_list else node.lineno
            end_line = node.end_lineno or node.lineno
            return "".join(lines[start_line - 1 : end_line]).rstrip() + "\n"
    raise RuntimeError(f"method {method_name!r} not found")


# --------------------------------------------------------------------------
# PASS 1 — Identify
# --------------------------------------------------------------------------


PASS1_SYSTEM = (
    "You identify text in Python source that can be safely removed or shortened. "
    "You output JSON only. You never write or rewrite code."
)


PASS1_TEMPLATE = """=== METHOD SOURCE (this is what you analyze) ===
{source}
=== END METHOD SOURCE ===

A LoRA adapter has been trained to handle voice commands. It has
internalized the response format and routing patterns. So some text
inside the METHOD SOURCE above is now redundant and can be removed.

Your task: find specific sentences or phrases INSIDE the METHOD SOURCE
above that can be deleted, and report them as exact quotations.

For EVERY edit, `original` must be a substring copy-pasted from the
METHOD SOURCE above. If it doesn't appear verbatim in the source, it
will be rejected.

Edits must target text INSIDE string literals (inside the `f\"\"\"...\"\"\"` block).
DO NOT touch:
  - import statements, function signatures, logging calls, return statements
  - XML tags: <tool_call>, </tool_call>, <tools>, </tools>
  - Type-format rules about resolved_datetimes or duration_seconds
  - Dynamic interpolations like {{room}}, {{user}}, {{voice_mode}}, {{tools_xml}}
  - Any code outside the f-string

Prefer removing whole sentences (empty replacement) over rewording.

Examples of GOOD edits (text from the f-string, quoted verbatim):
  - original: "Don't make assumptions about what values to plug into functions."
    replacement: ""
  - original: "- Pick the tool that best matches intent; use get_command_utterance_examples if unsure."
    replacement: ""

Output ONLY this JSON shape, using DOUBLE-QUOTED strings (not single quotes):

{{"edits": [
  {{"original": "exact substring from the method source", "replacement": "", "reason": "why"}}
]}}

No markdown fences. No commentary. Just the JSON.
"""


def run_pass1(client: httpx.Client, source: str) -> tuple[list[dict], str]:
    raw, elapsed = chat(client, PASS1_SYSTEM, PASS1_TEMPLATE.format(source=source),
                        max_tokens=2000, temperature=0.1)
    print(f"  Pass 1 LLM: {elapsed:.1f}s, {len(raw)} chars returned")
    edits = _parse_edit_list(raw)
    return edits, raw


def _parse_edit_list(raw: str) -> list[dict]:
    """Find {"edits": [...]} in the raw output, tolerant of surrounding text,
    code fences, AND single-quoted JSON (common 8B failure mode)."""
    s = raw.strip()
    # Strip code fences
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines)
    first = s.find("{")
    last = s.rfind("}")
    if first < 0 or last <= first:
        return []
    candidate = s[first : last + 1]

    def _try_parse(text: str) -> list[dict] | None:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return None
        edits = obj.get("edits") if isinstance(obj, dict) else None
        if not isinstance(edits, list):
            return None
        return edits

    # Try strict JSON first
    edits = _try_parse(candidate)
    if edits is None:
        # Fallback: the 8B sometimes emits Python-style literals. Try ast.literal_eval.
        try:
            import ast as _ast
            obj = _ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            obj = None
        if isinstance(obj, dict):
            edits = obj.get("edits") if isinstance(obj.get("edits"), list) else None

    if not isinstance(edits, list):
        return []
    out = []
    for e in edits:
        if not isinstance(e, dict):
            continue
        orig = e.get("original")
        repl = e.get("replacement")
        if not isinstance(orig, str) or not isinstance(repl, str):
            continue
        if not orig:
            continue
        out.append({
            "original": orig,
            "replacement": repl,
            "reason": str(e.get("reason") or ""),
        })
    return out


# --------------------------------------------------------------------------
# PASS 2 — Apply (deterministic)
# --------------------------------------------------------------------------


@dataclass
class ApplyResult:
    new_source: str
    applied: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)


# Anchors we refuse to touch regardless of what Pass 1 suggested.
_PROTECTED_SUBSTRINGS = (
    "def build_system_prompt",
    "def parse_response",
    "return system_prompt",
    "return result",
    "logger.info",
    "_build_tools_xml",
    "format_tools_for_prompt",
    "build_direct_answer_section",
    "build_agent_context_summary",
    "resolved_datetimes MUST be",
    "duration_seconds MUST be",
    "<tool_call>",
    "</tool_call>",
    "<tools>",
    "</tools>",
    "{tools_xml}",
    "{tools_section}",
    "{room}",
    "{user}",
    "{voice_mode}",
    "{direct_answer_section}",
    "{agent_context_section}",
)


def _touches_protected(edit: dict) -> str | None:
    """Return a rejection reason if the edit would corrupt a protected anchor."""
    orig = edit["original"]
    repl = edit["replacement"]
    for anchor in _PROTECTED_SUBSTRINGS:
        if anchor in orig and anchor not in repl:
            return f"would remove protected anchor: {anchor!r}"
    return None


def apply_edits(source: str, edits: list[dict]) -> ApplyResult:
    result = ApplyResult(new_source=source)
    seen_originals: set[str] = set()
    for edit in edits:
        orig = edit["original"]
        repl = edit["replacement"]
        if orig in seen_originals:
            result.rejected.append({**edit, "_reject_reason": "duplicate"})
            continue
        seen_originals.add(orig)

        # Protected anchors — never touch
        reason = _touches_protected(edit)
        if reason:
            result.rejected.append({**edit, "_reject_reason": reason})
            continue

        # Exact match required
        count = result.new_source.count(orig)
        if count == 0:
            result.rejected.append({**edit, "_reject_reason": "original not found"})
            continue
        if count > 1:
            result.rejected.append({**edit, "_reject_reason": f"ambiguous — {count} occurrences"})
            continue

        # Apply the edit
        result.new_source = result.new_source.replace(orig, repl, 1)
        result.applied.append(edit)
    return result


# --------------------------------------------------------------------------
# PASS 3 — Validate (ast.parse + conditional retry)
# --------------------------------------------------------------------------


@dataclass
class ValidationResult:
    ok: bool
    syntax_error: str | None = None
    fixed_source: str | None = None
    fix_attempts: int = 0


def validate(source: str) -> ValidationResult:
    """Wrap the method in a class shell so ast.parse accepts it."""
    shell = "class _Stub:\n" + source
    try:
        ast.parse(shell)
        return ValidationResult(ok=True)
    except SyntaxError as e:
        return ValidationResult(ok=False, syntax_error=f"line {e.lineno}: {e.msg}")


PASS3_SYSTEM = (
    "You fix a single Python syntax error. Output ONLY the corrected method, "
    "no markdown, no commentary. Make the minimum change."
)


PASS3_TEMPLATE = """The following method has a syntax error:

{error}

Method:

{source}

Output the corrected method. Make the minimum change. Do not refactor.
Output Python only, no markdown fences.
"""


def run_pass3_fix(client: httpx.Client, source: str, err: str,
                  max_attempts: int = 1) -> ValidationResult:
    """Try to repair syntax via LLM. Returns a ValidationResult."""
    attempts = 0
    current = source
    last_err = err
    while attempts < max_attempts:
        attempts += 1
        fixed, elapsed = chat(
            client, PASS3_SYSTEM,
            PASS3_TEMPLATE.format(error=last_err, source=current),
            max_tokens=3000, temperature=0.0,
        )
        print(f"  Pass 3 attempt {attempts}: {elapsed:.1f}s")
        # Strip fences
        fixed = fixed.strip()
        if fixed.startswith("```"):
            lines = fixed.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            fixed = "\n".join(lines)
        check = validate(fixed)
        if check.ok:
            return ValidationResult(ok=True, fixed_source=fixed, fix_attempts=attempts)
        current = fixed
        last_err = check.syntax_error or ""
    return ValidationResult(ok=False, syntax_error=last_err, fix_attempts=attempts)


# --------------------------------------------------------------------------
# Structural invariant checks
# --------------------------------------------------------------------------


INVARIANT_SUBSTRINGS = (
    "def build_system_prompt",
    "node_context",
    "tools_xml",
    "return system_prompt",
    "<tool_call>",
    "resolved_datetimes",
    "duration_seconds",
    "{room}",
    "{user}",
)


def check_invariants(original: str, modified: str) -> list[tuple[str, bool]]:
    """Run substring-preservation checks; return list of (name, passed)."""
    out = []
    for inv in INVARIANT_SUBSTRINGS:
        was_there = inv in original
        still_there = inv in modified
        # Only flag a failure if it was there originally but isn't anymore
        out.append((f"preserves {inv!r}", (not was_there) or still_there))
    out.append(("shorter than original", len(modified) < len(original)))
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Three-pass local ablator")
    ap.add_argument("--provider", type=Path, default=PROVIDER_FILE)
    ap.add_argument("--method", type=str, default="build_system_prompt")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "three_pass_output")
    ap.add_argument("--max-iterations", type=int, default=6,
                    help="Max ablation cycles. Each cycle runs the full 3-pass loop "
                         "on the previous cycle's output. Stops when an iteration "
                         "produces 0 applied edits, or when this cap is hit.")
    args = ap.parse_args()

    if not INTERNAL_TOKEN:
        print("ERROR: MODEL_SERVICE_TOKEN not set", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    source = args.provider.read_text(encoding="utf-8")
    method_source = extract_method(source, args.method)

    print(f"Target: {args.provider.name} :: {args.method}")
    print(f"Method source: {len(method_source)} chars")
    (args.out_dir / "0_original.py").write_text(method_source, encoding="utf-8")

    final_source = method_source
    iteration_log: list[dict] = []

    with httpx.Client(timeout=240) as client:
        reload_model(client)

        for it in range(1, args.max_iterations + 1):
            print(f"\n{'=' * 70}")
            print(f"ITERATION {it}/{args.max_iterations}  (input: {len(final_source)} chars)")
            print("=" * 70)

            # --- Pass 1 ---
            print("▶ PASS 1 (Identify — LLM) ...")
            edits, raw1 = run_pass1(client, final_source)
            (args.out_dir / f"iter{it}_1_edits_raw.txt").write_text(raw1, encoding="utf-8")
            (args.out_dir / f"iter{it}_1_edits.json").write_text(
                json.dumps({"edits": edits}, indent=2), encoding="utf-8",
            )
            print(f"  parsed {len(edits)} edit proposals")
            for i, e in enumerate(edits, 1):
                print(f"    [{i}] {_short(e['original'])!r}  →  {_short(e['replacement'])!r}")

            if not edits:
                print("  (no more edits — converged)")
                iteration_log.append({"iteration": it, "edits": 0, "applied": 0, "size": len(final_source)})
                break

            # --- Pass 2 ---
            print("▶ PASS 2 (Apply — deterministic) ...")
            applied = apply_edits(final_source, edits)
            print(f"  applied {len(applied.applied)} / {len(edits)}  (rejected: {len(applied.rejected)})")
            for e in applied.rejected:
                print(f"    × {_short(e['original'])!r}: {e['_reject_reason']}")

            if not applied.applied:
                print("  (all proposals rejected — converged)")
                iteration_log.append({"iteration": it, "edits": len(edits), "applied": 0, "size": len(final_source)})
                break

            # --- Pass 3 ---
            print("▶ PASS 3 (Validate — ast.parse) ...")
            v = validate(applied.new_source)
            if v.ok:
                print("  ✓ parses cleanly")
                new_source = applied.new_source
            else:
                print(f"  ✗ syntax error: {v.syntax_error}")
                print("  attempting local-LLM syntax fix ...")
                fix = run_pass3_fix(client, applied.new_source, v.syntax_error or "", max_attempts=1)
                if fix.ok and fix.fixed_source:
                    new_source = fix.fixed_source
                    print(f"  ✓ fixed in {fix.fix_attempts} attempt(s)")
                else:
                    print(f"  ✗ could not fix — halting iteration")
                    iteration_log.append({
                        "iteration": it, "edits": len(edits), "applied": len(applied.applied),
                        "size": len(final_source), "halted": "unfixable syntax",
                    })
                    break

            (args.out_dir / f"iter{it}_2_applied.py").write_text(new_source, encoding="utf-8")
            delta = len(final_source) - len(new_source)
            iteration_log.append({
                "iteration": it,
                "edits": len(edits),
                "applied": len(applied.applied),
                "size": len(new_source),
                "delta_chars": delta,
            })
            final_source = new_source
            print(f"  cumulative size: {len(final_source)} chars ({delta:+d} this iter)")

        # write final output once
        (args.out_dir / "3_final.py").write_text(final_source, encoding="utf-8")

    # --- Iteration summary ---
    print(f"\n{'=' * 70}\nITERATION SUMMARY\n{'=' * 70}")
    print(f"  {'#':<3}  {'edits':>6}  {'applied':>8}  {'size':>8}  {'Δ':>7}")
    prev = len(method_source)
    for entry in iteration_log:
        delta = entry.get("delta_chars", 0)
        print(f"  {entry['iteration']:<3}  {entry['edits']:>6}  {entry['applied']:>8}  "
              f"{entry['size']:>8}  {delta:+7d}")
        prev = entry["size"]
    (args.out_dir / "iteration_log.json").write_text(json.dumps(iteration_log, indent=2), encoding="utf-8")

    # --- Structural invariant checks ---
    print("\n▶ STRUCTURAL INVARIANTS")
    checks = check_invariants(method_source, final_source)
    passed = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        mark = "✓" if ok else "✗"
        print(f"  {mark}  {name}")
    print(f"  {passed}/{len(checks)} passed")

    orig_chars = len(method_source)
    new_chars = len(final_source)
    pct = (1 - new_chars / orig_chars) * 100 if orig_chars else 0
    print(f"\n  size: {orig_chars} → {new_chars} chars ({pct:+.1f}% shrink)")

    # Final parse check on the full spliced file as the real gate
    print("\n▶ FULL-FILE SYNTAX CHECK")
    spliced_file = source.replace(method_source.rstrip() + "\n",
                                  final_source.rstrip() + "\n", 1)
    try:
        ast.parse(spliced_file)
        print("  ✓ full file parses — would be splice-safe")
    except SyntaxError as e:
        print(f"  ✗ splice fails: line {e.lineno}: {e.msg}")

    print(f"\nArtifacts in {args.out_dir}")
    return 0 if all(ok for _, ok in checks[:-1]) else 1  # ignore "shorter" for exit code


def _short(s: str, n: int = 70) -> str:
    s = s.replace("\n", "⏎")
    return s if len(s) <= n else s[: n - 3] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
