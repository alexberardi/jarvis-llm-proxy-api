"""Phase 2.5 — train an adapter on the synth dataset.

Reads poc/synth_train.jsonl, wraps it as a dataset_ref, calls
services.adapter_training.run_adapter_training() directly (bypasses the rq
queue — see notes in poc/marker_proof_direct.py). Prints the adapter_hash
on the final line for downstream scripts to consume.

Usage:
    .venv/bin/python poc/train_synth_adapter.py
    .venv/bin/python poc/train_synth_adapter.py --epochs 2 --lora-r 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/jarvis_llm_proxy",
)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.adapter_training import run_adapter_training  # noqa: E402

BASE_MODEL_ID = ".models/Hermes-3-Llama-3.1-8B-4bit-mlx"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_dataset_ref(rows: list[dict]) -> dict:
    # Group by command_name — the MLX training script reads examples from
    # data.commands[*].examples, but it flattens across commands anyway so
    # the grouping is cosmetic.
    by_cmd: dict[str, list[dict]] = {}
    for r in rows:
        name = r["expected_tool_call"]["name"]
        by_cmd.setdefault(name, []).append(r)
    return {
        "format": "inline-json",
        "data": {
            "commands": [
                {"command_name": cmd, "examples": exs}
                for cmd, exs in by_cmd.items()
            ],
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Train a synth-bootstrap adapter")
    ap.add_argument("--train", type=Path, default=Path(__file__).parent / "synth_train.jsonl")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    args = ap.parse_args()

    rows = load_jsonl(args.train)
    print(f"[load] {len(rows)} training rows from {args.train}")

    dataset_ref = build_dataset_ref(rows)
    request = {
        "node_id": "phase2_5_synth",
        "base_model_id": BASE_MODEL_ID,
        "dataset_ref": dataset_ref,
        "params": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "max_seq_len": args.max_seq_len,
            "learning_rate": args.learning_rate,
        },
    }

    print(f"[train] MLX LoRA rank={args.lora_r} epochs={args.epochs} "
          f"batch={args.batch_size} max_seq_len={args.max_seq_len}")

    t0 = time.time()
    result = run_adapter_training(request, job_id=f"synth-{uuid.uuid4()}", ttl_seconds=7200)
    elapsed = time.time() - t0

    meta = result["artifact_metadata"]
    print(f"\n[done] wall_time={elapsed:.1f}s  train_duration={meta.get('train_duration_seconds', 'cached')}s")
    print(f"       cached={meta['cached']}  format={meta['adapter_format']}")
    print(f"       artifact={meta['artifact_path']}")
    print(f"adapter_hash: {meta['dataset_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
