"""Phase 2.5 — prompt variants for the ablation sweep.

Produces progressively-stripped system prompts from the same `synth_commands_spec.json`
used to generate the training data. Each variant toggles what the model receives:

  full               preamble + critical rules + all tool metadata (baseline)
  no_examples        drop per-tool "examples: ..." lines
  no_descriptions    drop per-tool descriptions (keep schema only)
  no_critical_rules  drop the "CRITICAL - Response Format" block and JSON template
  schema_only        keep only "Available Tools: <name + params>"; strip everything else
  minimal            just a bare list of tool names and their params (no preamble at all)

Each variant is returned as a (name, prompt, token_estimate) tuple. `ablation_report.py`
consumes this list and runs the eval set against each.

Rough token counts are approximate (chars/4). The real token count is measured
per-request by llm-proxy's `usage.prompt_tokens` and logged in the final CSV.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from synth_dataset import _format_tool_block  # reuse the renderer


# --------------------------------------------------------------------------
# Variant generators
# --------------------------------------------------------------------------


@dataclass
class Variant:
    name: str
    prompt: str
    char_count: int

    @property
    def approx_tokens(self) -> int:
        return self.char_count // 4


def _render_tools(cmds: list[dict], include_description: bool, include_examples: bool) -> str:
    return "\n\n".join(
        _format_tool_block(c, include_description=include_description, include_examples=include_examples)
        for c in cmds
    )


def _make(name: str, prompt: str) -> Variant:
    return Variant(name=name, prompt=prompt, char_count=len(prompt))


# Each generator takes the loaded spec and produces a single Variant.
GENERATORS: dict[str, Callable[[dict], Variant]] = {}


def _register(name: str):
    def deco(fn: Callable[[dict], Variant]):
        GENERATORS[name] = fn
        return fn
    return deco


@_register("full")
def _full(spec: dict) -> Variant:
    """Baseline — matches what synth_dataset.py used for training."""
    sections = [spec["system_prompt_preamble"]]
    sections.append("Available Tools:")
    sections.append(_render_tools(spec["commands"], include_description=True, include_examples=True))
    return _make("full", "\n\n".join(sections))


@_register("no_examples")
def _no_examples(spec: dict) -> Variant:
    sections = [spec["system_prompt_preamble"]]
    sections.append("Available Tools:")
    sections.append(_render_tools(spec["commands"], include_description=True, include_examples=False))
    return _make("no_examples", "\n\n".join(sections))


@_register("no_descriptions")
def _no_descriptions(spec: dict) -> Variant:
    sections = [spec["system_prompt_preamble"]]
    sections.append("Available Tools:")
    sections.append(_render_tools(spec["commands"], include_description=False, include_examples=False))
    return _make("no_descriptions", "\n\n".join(sections))


def _preamble_without_rules(spec: dict) -> str:
    """Preamble minus the CRITICAL and response-format blocks.

    Keeps the first part (role, node context, primary role) but strips the
    JSON format directive + "CRITICAL" admonitions. We keep the
    YOUR PRIMARY ROLE stub because total absence of any role cue collapses
    performance on the bare tool schema.
    """
    raw = spec["system_prompt_preamble"]
    marker = "CRITICAL - Response Format"
    idx = raw.find(marker)
    if idx == -1:
        return raw
    return raw[:idx].rstrip()


@_register("no_critical_rules")
def _no_critical_rules(spec: dict) -> Variant:
    sections = [_preamble_without_rules(spec)]
    sections.append("Available Tools:")
    sections.append(_render_tools(spec["commands"], include_description=False, include_examples=False))
    return _make("no_critical_rules", "\n\n".join(sections))


@_register("schema_only")
def _schema_only(spec: dict) -> Variant:
    """Drops the preamble entirely. Just 'Available Tools:' + bare schemas."""
    sections = ["Available Tools:"]
    sections.append(_render_tools(spec["commands"], include_description=False, include_examples=False))
    return _make("schema_only", "\n\n".join(sections))


@_register("minimal")
def _minimal(spec: dict) -> Variant:
    """Absolute floor: compact one-line-per-tool summary. No preamble, no frills."""
    lines = ["Tools:"]
    for cmd in spec["commands"]:
        params = []
        for p in cmd["parameters"]:
            marker = "?" if p.get("optional") else ""
            params.append(f"{p['name']}{marker}")
        lines.append(f"{cmd['name']}({', '.join(params)})")
    return _make("minimal", "\n".join(lines))


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


# Order = the ablation curve direction (largest to smallest).
DEFAULT_VARIANT_ORDER = [
    "full",
    "no_examples",
    "no_descriptions",
    "no_critical_rules",
    "schema_only",
    "minimal",
]


def load_spec(path: Path | None = None) -> dict:
    if path is None:
        path = Path(__file__).parent / "synth_commands_spec.json"
    return json.loads(path.read_text(encoding="utf-8"))


def all_variants(spec: dict | None = None, order: list[str] | None = None) -> list[Variant]:
    spec = spec if spec is not None else load_spec()
    names = order or DEFAULT_VARIANT_ORDER
    out: list[Variant] = []
    for name in names:
        fn = GENERATORS.get(name)
        if fn is None:
            raise KeyError(f"unknown variant: {name}")
        out.append(fn(spec))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="List/preview ablation variants")
    ap.add_argument("--spec", type=Path, default=None)
    ap.add_argument("--show", type=str, default=None, help="Print the full text of a named variant")
    args = ap.parse_args()

    spec = load_spec(args.spec)

    if args.show:
        fn = GENERATORS.get(args.show)
        if not fn:
            print(f"unknown variant: {args.show}", flush=True)
            return 1
        v = fn(spec)
        print(v.prompt)
        print(f"\n---\nchars={v.char_count} approx_tokens={v.approx_tokens}")
        return 0

    print(f"{'variant':<20} {'chars':>7} {'approx_tokens':>14}")
    for v in all_variants(spec):
        print(f"{v.name:<20} {v.char_count:>7} {v.approx_tokens:>14}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
