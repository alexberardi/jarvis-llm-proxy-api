"""Phase 2.5 — synthetic dataset generator.

Reads `synth_commands_spec.json`, expands every (template × slot combo) pair,
applies lightweight surface variation, splits 80/20 by slot value (NOT by
phrasing — prevents parameter leakage), and writes:

    synth_train.jsonl   — training set in llm-proxy dataset_ref shape
    synth_eval.jsonl    — held-out eval set (same shape)

Each example is:
    {
      "voice_command":          "Play Radiohead",
      "expected_tool_call":     {"name": "music", "arguments": {"action": "play", "query": "Radiohead"}},
      "formatted_system_prompt": "<production-shaped prompt with all 5 tools>",
      "formatted_completion":    "<JSON response the model should emit>"
    }

Usage:
    .venv/bin/python poc/synth_dataset.py \
        --spec poc/synth_commands_spec.json \
        --out-dir poc/ \
        --max-per-command 500
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SLOT_PATTERN = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")

# Deterministic seed so a given spec always produces the same dataset.
DEFAULT_SEED = 42


@dataclass(frozen=True)
class Example:
    command: str
    voice_command: str
    arguments: dict
    # Which slot *values* were used — drives the train/eval split.
    slot_key: str


# --------------------------------------------------------------------------
# Tool definition formatting for the system prompt
# --------------------------------------------------------------------------


def _format_tool_block(cmd: dict, include_description: bool, include_examples: bool) -> str:
    """Render a tool definition as it would appear in the system prompt.

    The ablation script generates variants by flipping these flags — this is
    the canonical long form that gets progressively stripped down.
    """
    lines = [f"- {cmd['name']}"]
    if include_description:
        lines.append(f"    description: {cmd['description']}")

    param_lines = []
    for p in cmd["parameters"]:
        parts = [f"{p['name']} ({p['type']}"]
        if p.get("optional"):
            parts.append(", optional")
        parts.append(")")
        line = "".join(parts)
        if p["type"] == "enum" and p.get("values"):
            line += ": " + " | ".join(f'"{v}"' for v in p["values"])
        param_lines.append("      " + line)
    if param_lines:
        lines.append("    parameters:")
        lines.extend(param_lines)

    if include_examples and cmd.get("templates"):
        # First template per distinct args-shape, up to 3 examples
        seen_shapes: set[frozenset] = set()
        example_patterns: list[str] = []
        for tpl in cmd["templates"]:
            shape = frozenset(tpl["args"].keys())
            if shape in seen_shapes:
                continue
            seen_shapes.add(shape)
            example_patterns.append(tpl["pattern"])
            if len(example_patterns) >= 3:
                break
        if example_patterns:
            lines.append("    examples: " + "; ".join(f'"{p}"' for p in example_patterns))

    return "\n".join(lines)


def build_system_prompt(
    spec: dict,
    *,
    include_preamble: bool = True,
    include_tool_descriptions: bool = True,
    include_tool_examples: bool = True,
) -> str:
    """Render the full production-shaped system prompt for the 5-command set.

    The ablation harness uses the same function with different flags to
    produce progressively-stripped variants.
    """
    sections: list[str] = []
    if include_preamble:
        sections.append(spec["system_prompt_preamble"])
    sections.append("Available Tools:")
    for cmd in spec["commands"]:
        sections.append(
            _format_tool_block(
                cmd,
                include_description=include_tool_descriptions,
                include_examples=include_tool_examples,
            )
        )
    return "\n\n".join(sections)


# --------------------------------------------------------------------------
# Template expansion
# --------------------------------------------------------------------------


def _coerce(value: str, expected_type: str) -> Any:
    """Coerce a slot-string back to the parameter's native type."""
    if expected_type == "integer":
        try:
            return int(value)
        except ValueError:
            return value
    return value


def _param_types(cmd: dict) -> dict[str, str]:
    return {p["name"]: p["type"] for p in cmd["parameters"]}


def _bind_args(
    args_template: dict[str, str],
    bindings: dict[str, str],
    cmd_param_types: dict[str, str],
) -> dict[str, Any]:
    """Resolve an args template like {"action":"play","query":"{artist}"} against bindings."""
    resolved: dict[str, Any] = {}
    for k, v in args_template.items():
        if isinstance(v, str):
            m = SLOT_PATTERN.fullmatch(v)
            if m:
                slot_name = m.group(1)
                raw = bindings[slot_name]
                resolved[k] = _coerce(str(raw), cmd_param_types.get(k, "string"))
            else:
                # Literal (e.g. action: "play")
                resolved[k] = v
        else:
            resolved[k] = v
    return resolved


def _enumerate_bindings(
    pattern: str,
    args_template: dict[str, str],
    slots: dict[str, list],
    max_per_template: int,
    rng: random.Random,
) -> list[dict[str, str]]:
    """List all slot-binding dicts needed to realize a template.

    Slots referenced anywhere in the pattern OR in the args template are
    enumerated. For templates with multiple slots, produces the Cartesian
    product and samples `max_per_template` of them.
    """
    pattern_slots = set(SLOT_PATTERN.findall(pattern))
    args_slots: set[str] = set()
    for v in args_template.values():
        if isinstance(v, str):
            args_slots.update(SLOT_PATTERN.findall(v))
    slot_names = sorted(pattern_slots | args_slots)
    if not slot_names:
        return [{}]
    slot_values = [slots[n] for n in slot_names]
    combos = list(itertools.product(*slot_values))
    if len(combos) > max_per_template:
        rng.shuffle(combos)
        combos = combos[:max_per_template]
    return [dict(zip(slot_names, [str(v) for v in combo])) for combo in combos]


def _render_pattern(pattern: str, bindings: dict[str, str]) -> str:
    def replace(m):
        return bindings[m.group(1)]
    return SLOT_PATTERN.sub(replace, pattern)


# Light-touch surface variation — mirror real voice transcription noise.
_SURFACE_VARIANTS = [
    lambda s: s,
    lambda s: s.lower(),
    lambda s: s + ".",
    lambda s: s + "?",
    lambda s: "please " + s.lower(),
    lambda s: s.replace("What's", "What is"),
    lambda s: s.replace("'s", " is") if "'s" in s else s,
]


def _surface_variation(text: str, rng: random.Random, n: int) -> list[str]:
    """Produce up to N distinct surface-variations of a canonical utterance.

    Always includes the canonical form. Filters duplicates to avoid bloat.
    """
    seen: list[str] = [text]
    rng.shuffle(_SURFACE_VARIANTS)
    for v in _SURFACE_VARIANTS:
        try:
            out = v(text)
        except (AttributeError, IndexError, TypeError):
            continue
        if out and out not in seen:
            seen.append(out)
        if len(seen) >= n:
            break
    return seen


def _slot_signature(cmd_name: str, bindings: dict[str, str]) -> str:
    """Used to split train/eval — two rows with the same signature share a slot combo."""
    if not bindings:
        return f"{cmd_name}::<noargs>"
    items = sorted(bindings.items())
    raw = cmd_name + "||" + "|".join(f"{k}={v}" for k, v in items)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def expand_command(
    cmd: dict,
    *,
    max_per_template: int,
    surface_variations: int,
    rng: random.Random,
) -> list[Example]:
    """Expand every template in the command spec into concrete Examples."""
    slots = cmd["slots"]
    cmd_param_types = _param_types(cmd)
    out: list[Example] = []
    for tpl in cmd["templates"]:
        pattern = tpl["pattern"]
        args_tpl = tpl["args"]
        binding_sets = _enumerate_bindings(
            pattern, args_tpl, slots, max_per_template, rng,
        )
        for bindings in binding_sets:
            canonical = _render_pattern(pattern, bindings) if bindings else pattern
            variations = _surface_variation(canonical, rng, surface_variations)
            resolved_args = _bind_args(args_tpl, bindings, cmd_param_types)
            sig = _slot_signature(cmd["name"], bindings)
            for text in variations:
                out.append(Example(
                    command=cmd["name"],
                    voice_command=text,
                    arguments=resolved_args,
                    slot_key=sig,
                ))
    return out


# --------------------------------------------------------------------------
# dataset_ref + training-row shaping
# --------------------------------------------------------------------------


def build_response_json(cmd: str, arguments: dict) -> str:
    """The JSON the model should emit — matches the production response format."""
    payload = {
        "message": "",
        "tool_call": {"name": cmd, "arguments": arguments},
    }
    return " " + json.dumps(payload, ensure_ascii=False)


def to_dataset_row(ex: Example, system_prompt: str) -> dict:
    return {
        "voice_command": ex.voice_command,
        "expected_tool_call": {"name": ex.command, "arguments": ex.arguments},
        "formatted_system_prompt": system_prompt,
        "formatted_completion": build_response_json(ex.command, ex.arguments),
    }


# --------------------------------------------------------------------------
# Variant-randomized prompts (fix for Phase 2.5's "cliff" observation)
#
# V1 used the full system prompt for every training row, so the adapter
# learned to rely on the `CRITICAL - Response Format` block for the JSON
# output shape. At ablation, stripping that block collapsed accuracy to 0.
# V2 randomly assigns each training row one of several prompt variants, so
# the adapter learns to emit correct output whether that block is present
# or not.
# --------------------------------------------------------------------------


def build_prompt_variants(spec: dict) -> dict[str, str]:
    """Return {variant_name: system_prompt} for training-time randomization.

    Intentionally skips the `minimal` variant — that's so stripped down
    (bare param list, no preamble at all) that training on it would teach
    the adapter to generate tool calls from almost nothing, which is the
    wrong trade-off for real-world use.
    """
    from ablation_prompt_variants import all_variants
    names = ["full", "no_examples", "no_descriptions", "no_critical_rules", "schema_only"]
    return {v.name: v.prompt for v in all_variants(spec, order=names)}


# --------------------------------------------------------------------------
# Train / eval split — by SLOT VALUE, not by phrasing
# --------------------------------------------------------------------------


def split_by_slot(examples: list[Example], eval_fraction: float, rng: random.Random) -> tuple[list[Example], list[Example]]:
    """Split so all rows with the same slot_key fall in the same bucket."""
    sigs = sorted({ex.slot_key for ex in examples})
    rng.shuffle(sigs)
    cutoff = int(len(sigs) * (1 - eval_fraction))
    train_sigs = set(sigs[:cutoff])
    train: list[Example] = []
    evalset: list[Example] = []
    for ex in examples:
        (train if ex.slot_key in train_sigs else evalset).append(ex)
    return train, evalset


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2.5 synthetic dataset generator")
    parser.add_argument("--spec", type=Path, default=Path(__file__).parent / "synth_commands_spec.json")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--max-per-template", type=int, default=8,
                        help="Cap slot-binding combos per template (default 8). "
                             "Keeps dataset size sane when templates have many slots.")
    parser.add_argument("--surface-variations", type=int, default=3,
                        help="Surface-form variations per canonical utterance (default 3)")
    parser.add_argument("--max-per-command", type=int, default=None,
                        help="Hard cap per command (after expansion). Default: no cap.")
    parser.add_argument("--eval-fraction", type=float, default=0.2,
                        help="Fraction of slot-signatures held out for eval (default 0.2)")
    parser.add_argument("--variant-randomize", action="store_true",
                        help="V2: stamp each training row with a RANDOM prompt variant "
                             "(full / no_examples / no_descriptions / no_critical_rules / schema_only) "
                             "so the adapter learns to hold accuracy across ablation levels. "
                             "Eval rows always use the full prompt.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))

    system_prompt = build_system_prompt(spec)
    # Tuck the prompt next to the dataset so the ablation script can re-use it.
    (args.out_dir / "synth_system_prompt.txt").write_text(system_prompt, encoding="utf-8")
    print(f"✍  wrote synth_system_prompt.txt ({len(system_prompt)} chars)")

    variant_prompts: dict[str, str] = {}
    if args.variant_randomize:
        variant_prompts = build_prompt_variants(spec)
        print(f"🎲 variant-randomize ON  ({len(variant_prompts)} variants: "
              f"{', '.join(variant_prompts)})")

    all_train: list[dict] = []
    all_eval: list[dict] = []
    per_command_stats: list[str] = []

    for cmd in spec["commands"]:
        examples = expand_command(
            cmd,
            max_per_template=args.max_per_template,
            surface_variations=args.surface_variations,
            rng=rng,
        )
        if args.max_per_command and len(examples) > args.max_per_command:
            rng.shuffle(examples)
            examples = examples[: args.max_per_command]

        train_ex, eval_ex = split_by_slot(examples, args.eval_fraction, rng)
        per_command_stats.append(
            f"  {cmd['name']:<14} {len(examples):>5} total  →  train {len(train_ex):>4}  eval {len(eval_ex):>4}"
        )

        if variant_prompts:
            variant_names = list(variant_prompts.keys())
            for e in train_ex:
                vn = rng.choice(variant_names)
                all_train.append(to_dataset_row(e, variant_prompts[vn]))
        else:
            for e in train_ex:
                all_train.append(to_dataset_row(e, system_prompt))
        # Eval always uses the full prompt (the ablation harness swaps variants at eval time)
        for e in eval_ex:
            all_eval.append(to_dataset_row(e, system_prompt))

    # Shuffle final rows for training entropy
    rng.shuffle(all_train)

    train_path = args.out_dir / "synth_train.jsonl"
    eval_path = args.out_dir / "synth_eval.jsonl"
    with train_path.open("w", encoding="utf-8") as f:
        for row in all_train:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with eval_path.open("w", encoding="utf-8") as f:
        for row in all_eval:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\nPer-command counts:")
    for line in per_command_stats:
        print(line)
    print(f"\nTotals: train {len(all_train):,}   eval {len(all_eval):,}")
    print(f"  → {train_path}")
    print(f"  → {eval_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
