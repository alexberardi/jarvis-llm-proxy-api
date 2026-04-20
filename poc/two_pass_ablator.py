"""Two-pass ablator POC — can Hermes-3-8B decompose prompt-provider ablation?

Test: pick ONE method from `jarvis-pp-hermes/.../hermes_medium_mlx.py`
(the actual provider used with our base model), run a two-pass request
sequence through the LOCAL llm-proxy (same Hermes), and eyeball the output.

Pass 1 (Analyst): plain-English description of what can be trimmed.
  Inputs: the method source + a note that the adapter has internalized
  certain behaviors.
  Output: text — no code.

Pass 2 (Coder): Python method that implements Pass 1's proposal.
  Inputs: the original source + Pass 1's output.
  Output: Python code only, preserving the signature.

No AST stitching, no eval, no training — just see whether the decomposed
task produces usable artifacts. If it does, the full harness is worth
building as Phase 6.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import httpx


MODEL_BASE = "http://localhost:7705"
INTERNAL_TOKEN = os.getenv("MODEL_SERVICE_TOKEN") or os.getenv("LLM_PROXY_INTERNAL_TOKEN") or ""

PROVIDER_FILE = Path(
    "/Users/alexanderberardi/jarvis/jarvis-pp-hermes/prompt_providers/hermes/hermes_medium_mlx.py"
)


# --------------------------------------------------------------------------
# Method extraction (by signature match, ast-free so we can show the exact
# source verbatim including its decorators and indentation)
# --------------------------------------------------------------------------


def extract_method(source: str, method_name: str) -> str:
    """Return the full source of the named method (via ast for robustness)."""
    import ast
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
            # ast.lineno is 1-based. Include decorators.
            start_line = node.decorator_list[0].lineno if node.decorator_list else node.lineno
            end_line = node.end_lineno or node.lineno
            return "".join(lines[start_line - 1 : end_line]).rstrip() + "\n"
    raise RuntimeError(f"method {method_name!r} not found in provider source")


# --------------------------------------------------------------------------
# LLM call helpers
# --------------------------------------------------------------------------


def chat(client: httpx.Client, system: str, user: str, *, max_tokens: int = 2500, temperature: float = 0.2) -> tuple[str, float]:
    """Hit /internal/model/chat. No adapter (we're asking the base model to reason about code)."""
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
    """Clear any sticky adapter state before running."""
    headers = {"X-Internal-Token": INTERNAL_TOKEN}
    r = client.post(f"{MODEL_BASE}/internal/model/reload", headers=headers, timeout=180)
    r.raise_for_status()


# --------------------------------------------------------------------------
# Pass 1 + Pass 2 prompts
# --------------------------------------------------------------------------


PASS1_SYSTEM = (
    "You are a careful code-analysis assistant. Output plain English, no code. "
    "Be concrete and specific. Reference exact lines/strings from the source."
)

PASS1_TEMPLATE = """I have a Python method that generates a system prompt for a voice assistant.

A LoRA adapter has been trained on top of the base language model. The adapter
has internalized these behaviors, so the system prompt no longer needs to
teach them:

  • The exact JSON / <tool_call> response format (adapter always emits it).
  • The "Call ONE tool at a time" rule (adapter follows by default).
  • The "pick the best-matching tool" selection rule (adapter routes correctly).
  • The "extract parameters from the user's words" rule (adapter extracts).
  • The verbose "don't make assumptions" preamble.

What MUST stay intact (the adapter cannot learn these):
  • The LIST of available tools — model needs to see names + params at runtime.
  • Per-turn dynamic fields: room, user, voice_mode.
  • Type-specific format rules (e.g. "resolved_datetimes MUST be a JSON array" —
    these are output format invariants, not rules the adapter can internalize).
  • Direct-answer and agent-context sections (dynamic).

Method source to analyze:

```python
{source}
```

Your task: describe IN ENGLISH, section by section, what can be trimmed
from this method's output while keeping the function contract intact
(same signature, same return type, still iterates dynamic inputs).

Be specific. For each section you propose to remove or shorten, quote
the exact string and describe what it should read like instead.

Do NOT write Python code. Describe only.
"""

PASS2_SYSTEM = (
    "You are a careful Python programmer. Output ONLY valid Python code. "
    "No markdown fences, no commentary, no docstring additions."
)

PASS2_TEMPLATE = """Here is the original method:

```python
{source}
```

An analyst has proposed the following trims:

{analysis}

Write the updated method in Python.

Constraints:
  • PRESERVE the exact function signature (name, parameters, type hints).
  • PRESERVE any f-string interpolations (room, user, etc.) and any loops/iterations.
  • Apply the analyst's trims faithfully.
  • Return the same type as before (str).

Output the method exactly as it would appear in the file, starting with
the 4-space-indented `def` line (no surrounding class, no extra blank lines
at the start). No markdown fences, no commentary.
"""


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Two-pass ablator POC")
    ap.add_argument("--provider", type=Path, default=PROVIDER_FILE)
    ap.add_argument("--method", type=str, default="build_system_prompt",
                    help="Which method to ablate (default: build_system_prompt)")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "two_pass_output")
    args = ap.parse_args()

    if not INTERNAL_TOKEN:
        print("ERROR: MODEL_SERVICE_TOKEN not set", file=sys.stderr)
        return 2
    if not args.provider.is_file():
        print(f"ERROR: provider file not found: {args.provider}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    source = args.provider.read_text(encoding="utf-8")
    method_source = extract_method(source, args.method)

    print("=" * 70)
    print(f"Target: {args.provider.name} :: {args.method}")
    print(f"Method source: {len(method_source)} chars")
    print("=" * 70)
    print(method_source)
    print("=" * 70)

    (args.out_dir / "0_original.py").write_text(method_source, encoding="utf-8")

    with httpx.Client(timeout=240) as client:
        reload_model(client)

        # --- Pass 1: Analyst ---
        print("\n▶ PASS 1 (Analyst) — asking model for trim proposal...")
        pass1_user = PASS1_TEMPLATE.format(source=method_source)
        analysis, t1 = chat(client, PASS1_SYSTEM, pass1_user, max_tokens=2000)
        print(f"  returned {len(analysis)} chars in {t1:.1f}s")
        (args.out_dir / "1_analyst.md").write_text(analysis, encoding="utf-8")

        print("\n" + "-" * 50 + "\nANALYSIS:\n" + "-" * 50)
        print(analysis)

        # --- Pass 2: Coder ---
        print("\n" + "=" * 70)
        print("▶ PASS 2 (Coder) — asking model to produce Python from the analysis...")
        pass2_user = PASS2_TEMPLATE.format(source=method_source, analysis=analysis)
        code, t2 = chat(client, PASS2_SYSTEM, pass2_user, max_tokens=3000)
        print(f"  returned {len(code)} chars in {t2:.1f}s")
        (args.out_dir / "2_coder.py").write_text(code, encoding="utf-8")

        print("\n" + "-" * 50 + "\nGENERATED CODE:\n" + "-" * 50)
        print(code)

    # --- Evaluate structurally ---
    print("\n" + "=" * 70)
    print("STRUCTURAL CHECKS (not semantic eval — those require AST surgery + full harness)")
    print("=" * 70)

    checks = []
    checks.append(("starts with def or @", code.lstrip().startswith(("def ", "@"))))
    checks.append((f"mentions method name `{args.method}`", args.method in code))
    checks.append(("preserves `tools_xml` iteration", "tools_xml" in code or "_build_tools_xml" in code))
    checks.append(("still uses f-string for room/user", "{room}" in code or "{user}" in code))
    checks.append(("still returns a string", "return " in code))
    checks.append(("no markdown fence pollution", "```" not in code))
    checks.append(("shorter than original", len(code) < len(method_source)))

    orig_chars = len(method_source)
    new_chars = len(code)
    pct = (1 - new_chars / orig_chars) * 100 if orig_chars else 0

    for name, ok in checks:
        mark = "✓" if ok else "✗"
        print(f"  {mark}  {name}")
    print(f"\n  method source size: {orig_chars} chars → {new_chars} chars "
          f"({pct:+.1f}% {'shrink' if new_chars < orig_chars else 'grow'})")

    print(f"\nArtifacts written to: {args.out_dir}")
    print("  0_original.py — the original method source")
    print("  1_analyst.md  — Pass 1 trim proposal")
    print("  2_coder.py    — Pass 2 generated code")
    print("\nNext step: eyeball 2_coder.py — does it look like a usable replacement?")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
