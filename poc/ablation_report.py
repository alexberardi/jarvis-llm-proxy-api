"""Phase 2.5 — ablation sweep + payoff table.

For each prompt variant × (baseline, adapter):
  • runs the held-out synth eval set against llm-proxy's /internal/model/chat
  • captures per-request wall time and prompt_tokens from the response usage
  • computes pass rate by comparing parsed tool_call to the canonical expected args

Emits:
  • ablation_report.csv — one row per (variant, condition). Fields:
      variant, condition, prompt_tokens_avg, pass_rate, passed, total,
      latency_p50_s, latency_p95_s, latency_avg_s, prefill_toks_per_s_avg
  • ablation_report.md — the human-readable payoff table
  • single-line verdict on stdout for the go/no-go gate

Usage:
    export MODEL_SERVICE_TOKEN=<internal-token>
    .venv/bin/python poc/ablation_report.py --adapter-hash <hash>

    # Re-run without adapter only:
    .venv/bin/python poc/ablation_report.py --no-adapter

    # Subsample for fast iteration (first N eval rows per variant):
    .venv/bin/python poc/ablation_report.py --adapter-hash <hash> --max-eval 40
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablation_prompt_variants import Variant, all_variants, load_spec  # noqa: E402


MODEL_BASE = "http://localhost:7705"
DEFAULT_EVAL_PATH = Path(__file__).parent / "synth_eval.jsonl"
INTERNAL_TOKEN = os.getenv("MODEL_SERVICE_TOKEN") or os.getenv("LLM_PROXY_INTERNAL_TOKEN") or ""


# --------------------------------------------------------------------------
# llm-proxy client helpers
# --------------------------------------------------------------------------


@dataclass
class ChatResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    wall_time: float


def chat(
    client: httpx.Client,
    system_prompt: str,
    user_msg: str,
    adapter_hash: str | None,
    max_tokens: int = 128,
) -> ChatResult:
    body: dict[str, Any] = {
        "model": "live",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    if adapter_hash:
        body["adapter_settings"] = {"hash": adapter_hash, "scale": 1.0, "enabled": True}
    headers = {"X-Internal-Token": INTERNAL_TOKEN}
    t0 = time.time()
    r = client.post(f"{MODEL_BASE}/internal/model/chat", json=body, headers=headers)
    wall = time.time() - t0
    r.raise_for_status()
    data = r.json()
    usage = data.get("usage") or {}
    return ChatResult(
        content=data.get("content") or "",
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=int(usage.get("completion_tokens", 0)),
        wall_time=wall,
    )


def reload_model(client: httpx.Client) -> None:
    """Clear any sticky adapter state between variant × condition cells."""
    headers = {"X-Internal-Token": INTERNAL_TOKEN}
    r = client.post(f"{MODEL_BASE}/internal/model/reload", headers=headers, timeout=180)
    r.raise_for_status()


# --------------------------------------------------------------------------
# Response parsing (mirrors the POC test harness)
# --------------------------------------------------------------------------


def try_parse_tool_call(content: str) -> dict | None:
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


def evaluate(expected: dict, content: str) -> bool:
    tc = try_parse_tool_call(content)
    if not tc:
        return False
    if tc.get("name") != expected["name"]:
        return False
    actual = tc.get("arguments") or {}
    for k, v in (expected.get("arguments") or {}).items():
        av = actual.get(k)
        if isinstance(v, str) and isinstance(av, str):
            if v.strip().lower() != av.strip().lower():
                return False
        else:
            if v != av:
                return False
    return True


# --------------------------------------------------------------------------
# Eval runner
# --------------------------------------------------------------------------


@dataclass
class CellResult:
    variant: str
    condition: str  # "baseline" or "adapter"
    total: int
    passed: int
    prompt_tokens_avg: float
    latency_samples: list[float] = field(default_factory=list)
    prefill_tps_samples: list[float] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return 100.0 * self.passed / self.total if self.total else 0.0

    @property
    def p50(self) -> float:
        return statistics.median(self.latency_samples) if self.latency_samples else 0.0

    @property
    def p95(self) -> float:
        if not self.latency_samples:
            return 0.0
        xs = sorted(self.latency_samples)
        idx = min(len(xs) - 1, int(len(xs) * 0.95))
        return xs[idx]

    @property
    def avg_latency(self) -> float:
        return statistics.mean(self.latency_samples) if self.latency_samples else 0.0

    @property
    def avg_prefill_tps(self) -> float:
        return statistics.mean(self.prefill_tps_samples) if self.prefill_tps_samples else 0.0


def load_eval(path: Path, max_rows: int | None = None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def run_cell(
    client: httpx.Client,
    variant: Variant,
    condition: str,
    adapter_hash: str | None,
    eval_rows: list[dict],
    progress_every: int = 20,
) -> CellResult:
    total = len(eval_rows)
    result = CellResult(
        variant=variant.name, condition=condition, total=total,
        passed=0, prompt_tokens_avg=0.0,
    )
    prompt_token_samples: list[int] = []

    print(f"    [{condition}] running {total} eval rows...", flush=True)
    for idx, row in enumerate(eval_rows):
        expected = row["expected_tool_call"]
        try:
            out = chat(client, variant.prompt, row["voice_command"], adapter_hash)
        except Exception as e:
            print(f"      row {idx}: chat error: {e}", flush=True)
            continue
        if evaluate(expected, out.content):
            result.passed += 1
        result.latency_samples.append(out.wall_time)
        prompt_token_samples.append(out.prompt_tokens)
        if out.prompt_tokens > 0 and out.wall_time > 0:
            # Approximate tokens/sec for prefill (dominates latency for short outputs)
            result.prefill_tps_samples.append(out.prompt_tokens / out.wall_time)
        if (idx + 1) % progress_every == 0:
            print(
                f"      {idx + 1}/{total}  pass={result.passed}  "
                f"p50={statistics.median(result.latency_samples):.2f}s",
                flush=True,
            )

    result.prompt_tokens_avg = statistics.mean(prompt_token_samples) if prompt_token_samples else 0.0
    return result


# --------------------------------------------------------------------------
# Report writers
# --------------------------------------------------------------------------


CSV_FIELDS = [
    "variant",
    "condition",
    "prompt_tokens_avg",
    "passed",
    "total",
    "pass_rate",
    "latency_avg_s",
    "latency_p50_s",
    "latency_p95_s",
    "prefill_toks_per_s_avg",
]


def write_csv(path: Path, cells: list[CellResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for c in cells:
            w.writerow({
                "variant": c.variant,
                "condition": c.condition,
                "prompt_tokens_avg": round(c.prompt_tokens_avg, 1),
                "passed": c.passed,
                "total": c.total,
                "pass_rate": round(c.pass_rate, 2),
                "latency_avg_s": round(c.avg_latency, 3),
                "latency_p50_s": round(c.p50, 3),
                "latency_p95_s": round(c.p95, 3),
                "prefill_toks_per_s_avg": round(c.avg_prefill_tps, 1),
            })


def write_markdown(path: Path, cells: list[CellResult], adapter_hash: str | None) -> None:
    # Pivot: one row per variant, with baseline and adapter columns side-by-side
    by_variant: dict[str, dict[str, CellResult]] = {}
    for c in cells:
        by_variant.setdefault(c.variant, {})[c.condition] = c
    lines = []
    lines.append("# Phase 2.5 — ablation payoff table")
    lines.append("")
    if adapter_hash:
        lines.append(f"Adapter: `{adapter_hash[:16]}…`")
    lines.append("")
    lines.append(
        "| variant | tokens | baseline % | adapter % | Δpp | baseline p50 | adapter p50 | speedup |"
    )
    lines.append(
        "|---------|-------:|-----------:|----------:|----:|-------------:|------------:|--------:|"
    )
    for variant in by_variant:
        b = by_variant[variant].get("baseline")
        a = by_variant[variant].get("adapter")
        toks = b.prompt_tokens_avg if b else (a.prompt_tokens_avg if a else 0)
        base_rate = b.pass_rate if b else 0.0
        adp_rate = a.pass_rate if a else 0.0
        delta_rate = adp_rate - base_rate
        base_lat = b.p50 if b else 0.0
        adp_lat = a.p50 if a else 0.0
        speedup = (base_lat / adp_lat) if (base_lat > 0 and adp_lat > 0) else 0.0
        lines.append(
            f"| {variant} | {toks:.0f} | {base_rate:.1f}% | {adp_rate:.1f}% | "
            f"{delta_rate:+.1f} | {base_lat:.2f}s | {adp_lat:.2f}s | {speedup:.2f}x |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------


def compute_verdict(
    cells: list[CellResult],
    *,
    baseline_variant: str = "full",
    tolerance_pp: float = 2.0,
    min_shrink_ratio: float = 0.40,  # adapter must hold at ≥40% prompt reduction
) -> tuple[str, str]:
    """Returns (verdict, reason) where verdict ∈ {PASS, MARGINAL, FAIL}."""
    by_variant: dict[str, dict[str, CellResult]] = {}
    for c in cells:
        by_variant.setdefault(c.variant, {})[c.condition] = c

    # Reference: baseline condition on the baseline (full-prompt) variant
    full = by_variant.get(baseline_variant, {})
    reference = full.get("baseline")
    if reference is None:
        return "FAIL", "no baseline cell for 'full' variant"
    reference_rate = reference.pass_rate
    reference_tokens = reference.prompt_tokens_avg

    # Find the smallest-prompt variant where adapter still holds within tolerance
    candidates_in_tol: list[tuple[str, float, float]] = []
    for variant_name, conds in by_variant.items():
        a = conds.get("adapter")
        if not a:
            continue
        if reference_rate - a.pass_rate <= tolerance_pp:
            shrink = (1.0 - a.prompt_tokens_avg / reference_tokens) if reference_tokens else 0.0
            candidates_in_tol.append((variant_name, a.pass_rate, shrink))

    if not candidates_in_tol:
        return "FAIL", (
            f"no variant where adapter holds within {tolerance_pp}pp of baseline "
            f"{reference_rate:.1f}%"
        )

    # Max shrink that still holds
    best = max(candidates_in_tol, key=lambda t: t[2])
    variant_name, rate, shrink = best
    if shrink >= min_shrink_ratio:
        return (
            "PASS",
            f"adapter holds {rate:.1f}% (baseline {reference_rate:.1f}%) at "
            f"{shrink * 100:.0f}% prompt reduction (variant={variant_name})",
        )
    return (
        "MARGINAL",
        f"best shrink without accuracy loss is only {shrink * 100:.0f}% "
        f"(variant={variant_name}) — below {min_shrink_ratio * 100:.0f}% target",
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 2.5 ablation sweep")
    ap.add_argument("--adapter-hash", type=str, default=None,
                    help="Adapter to test against each variant. Required unless --no-adapter.")
    ap.add_argument("--no-adapter", action="store_true",
                    help="Only measure baseline (no adapter) — useful for dry-runs.")
    ap.add_argument("--eval", type=Path, default=DEFAULT_EVAL_PATH,
                    help="Held-out eval JSONL (from synth_dataset.py)")
    ap.add_argument("--max-eval", type=int, default=None,
                    help="Cap eval rows for faster iteration (e.g. --max-eval 40)")
    ap.add_argument("--out-csv", type=Path, default=Path(__file__).parent / "ablation_report.csv")
    ap.add_argument("--out-md", type=Path, default=Path(__file__).parent / "ablation_report.md")
    ap.add_argument("--tolerance", type=float, default=2.0,
                    help="Max acceptable pass-rate drop vs baseline (pp, default 2.0)")
    ap.add_argument("--min-shrink", type=float, default=0.40,
                    help="Min prompt reduction to pass the gate (default 0.40 = 40%% shrink)")
    args = ap.parse_args()

    if not INTERNAL_TOKEN:
        print("ERROR: MODEL_SERVICE_TOKEN / LLM_PROXY_INTERNAL_TOKEN not set", file=sys.stderr)
        return 2
    if not args.adapter_hash and not args.no_adapter:
        print("ERROR: need --adapter-hash or --no-adapter", file=sys.stderr)
        return 2

    spec = load_spec()
    variants = all_variants(spec)
    eval_rows = load_eval(args.eval, max_rows=args.max_eval)
    if not eval_rows:
        print(f"ERROR: no eval rows in {args.eval}", file=sys.stderr)
        return 2

    print(f"variants: {[v.name for v in variants]}")
    print(f"eval rows: {len(eval_rows)}")
    print(f"adapter: {args.adapter_hash or 'none'}\n")

    cells: list[CellResult] = []
    with httpx.Client(timeout=180) as client:
        # Phase A — all baselines first (no adapter ever loaded). One reload
        # at the start clears any sticky state from prior runs.
        print("=== Phase A: baselines (no adapter) ===")
        reload_model(client)
        for variant in variants:
            print(f"▶ variant: {variant.name}  ({variant.char_count} chars, ~{variant.approx_tokens} toks)")
            cells.append(run_cell(
                client, variant, condition="baseline",
                adapter_hash=None, eval_rows=eval_rows,
            ))
            print()

        # Phase B — all adapters. Reload once so a clean MLX instance loads
        # the adapter on the first request; sticky behavior keeps it loaded
        # across subsequent variants (the prompt changes per-request, the
        # adapter does not).
        if args.adapter_hash:
            print("=== Phase B: adapter loaded ===")
            reload_model(client)
            for variant in variants:
                print(f"▶ variant: {variant.name}  ({variant.char_count} chars, ~{variant.approx_tokens} toks)")
                cells.append(run_cell(
                    client, variant, condition="adapter",
                    adapter_hash=args.adapter_hash, eval_rows=eval_rows,
                ))
                print()

    write_csv(args.out_csv, cells)
    write_markdown(args.out_md, cells, args.adapter_hash)
    print(f"→ csv:      {args.out_csv}")
    print(f"→ markdown: {args.out_md}")
    print()

    # Print the human table inline for convenience
    md_text = args.out_md.read_text(encoding="utf-8")
    print(md_text)

    if args.adapter_hash:
        verdict, reason = compute_verdict(
            cells, tolerance_pp=args.tolerance, min_shrink_ratio=args.min_shrink,
        )
        print(f"\nVERDICT: {verdict}  —  {reason}")
        return 0 if verdict in ("PASS", "MARGINAL") else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
