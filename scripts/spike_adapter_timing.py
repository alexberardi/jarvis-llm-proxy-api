#!/usr/bin/env python3
"""
Spike test: Measure vLLM adapter load/swap timing.

This test measures the critical timing question for per-request adapter loading:
- How long does it take to load an adapter the first time?
- How long for subsequent requests with the same adapter?
- How long to swap to a different adapter?
- Does vLLM cache multiple adapters in memory?

CRITICAL: These numbers determine if the adapter approach is viable.
If swap time > 2-3 seconds, we need to rethink the architecture.

Usage:
    # Set up environment
    export JARVIS_MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct
    export JARVIS_VLLM_MAX_LORAS=4  # Test with different values

    # Run with two different adapters
    python scripts/spike_adapter_timing.py \\
        --adapter-a /path/to/adapter-a \\
        --adapter-b /path/to/adapter-b

    # Or run with single adapter (just measures load vs cached)
    python scripts/spike_adapter_timing.py \\
        --adapter-a /path/to/adapter-a

Results to look for:
    - First load: May be slow (disk I/O + GPU transfer)
    - Cached hit: Should be <100ms
    - Swap time: This is the critical number
    - If max_loras > 1, swapping back should be fast (both in memory)

Output:
    Prints timing results and writes JSON to spike_adapter_timing_results.json
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List


@dataclass
class TimingResult:
    """Single timing measurement."""
    operation: str
    adapter_hash: str
    duration_ms: float
    tokens_generated: int
    notes: str = ""


@dataclass
class SpikeResults:
    """Complete spike test results."""
    model_name: str
    max_loras: int
    max_lora_rank: int
    adapter_a_path: str
    adapter_b_path: Optional[str]
    timings: List[TimingResult]
    summary: dict


def get_adapter_hash(path: str) -> str:
    """Extract hash from adapter path or generate one."""
    p = Path(path)
    # If path looks like .../hash/... use the hash
    if len(p.parts) >= 2:
        return p.parts[-1][:8] if p.is_file() else p.name[:8]
    return p.name[:8]


def run_spike_test(
    model_name: str,
    adapter_a: str,
    adapter_b: Optional[str],
    max_loras: int,
    max_lora_rank: int,
    num_tokens: int = 50,
) -> SpikeResults:
    """Run the spike test and collect timing data."""

    print("=" * 70)
    print("vLLM Adapter Load Timing Spike Test")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"max_loras: {max_loras}")
    print(f"max_lora_rank: {max_lora_rank}")
    print(f"Adapter A: {adapter_a}")
    print(f"Adapter B: {adapter_b or 'N/A'}")
    print("=" * 70)

    # Import vLLM (may take a moment)
    print("\n[1/7] Importing vLLM...")
    start = time.time()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    print(f"       Import took {time.time() - start:.2f}s")

    # Initialize model with LoRA support
    print("\n[2/7] Loading base model with LoRA support...")
    start = time.time()
    llm = LLM(
        model=model_name,
        enable_lora=True,
        max_loras=max_loras,
        max_lora_rank=max_lora_rank,
        gpu_memory_utilization=0.85,  # Leave room for desktop display
        max_model_len=4096,  # Limit context to fit in memory with LoRA
        max_num_seqs=32,  # Limit concurrent sequences for warmup
    )
    model_load_time = time.time() - start
    print(f"       Model loaded in {model_load_time:.2f}s")

    # Prepare test prompt and sampling params
    test_prompt = "Explain the concept of machine learning in simple terms:"
    sampling_params = SamplingParams(temperature=0.7, max_tokens=num_tokens)

    timings: List[TimingResult] = []
    hash_a = get_adapter_hash(adapter_a)
    hash_b = get_adapter_hash(adapter_b) if adapter_b else None

    # Test 1: Base model (no adapter) - warmup
    print("\n[3/7] Warmup: Base model generation (no adapter)...")
    start = time.time()
    outputs = llm.generate([test_prompt], sampling_params)
    warmup_time = (time.time() - start) * 1000
    warmup_tokens = len(outputs[0].outputs[0].token_ids)
    print(f"       Warmup: {warmup_time:.1f}ms ({warmup_tokens} tokens)")
    timings.append(TimingResult(
        operation="warmup_no_adapter",
        adapter_hash="none",
        duration_ms=warmup_time,
        tokens_generated=warmup_tokens,
        notes="Base model warmup"
    ))

    # Test 2: First adapter load (cold)
    print("\n[4/7] CRITICAL: First adapter load (cold)...")
    lora_a = LoRARequest("adapter-a", 1, adapter_a)
    start = time.time()
    outputs = llm.generate([test_prompt], sampling_params, lora_request=lora_a)
    first_load_time = (time.time() - start) * 1000
    first_load_tokens = len(outputs[0].outputs[0].token_ids)
    print(f"       First load: {first_load_time:.1f}ms ({first_load_tokens} tokens)")
    print(f"       >>> This includes disk I/O + GPU transfer")
    timings.append(TimingResult(
        operation="first_load_adapter_a",
        adapter_hash=hash_a,
        duration_ms=first_load_time,
        tokens_generated=first_load_tokens,
        notes="Cold load from disk"
    ))

    # Test 3: Same adapter again (should be cached)
    print("\n[5/7] Same adapter again (should be cached)...")
    start = time.time()
    outputs = llm.generate([test_prompt], sampling_params, lora_request=lora_a)
    cached_time = (time.time() - start) * 1000
    cached_tokens = len(outputs[0].outputs[0].token_ids)
    print(f"       Cached: {cached_time:.1f}ms ({cached_tokens} tokens)")
    timings.append(TimingResult(
        operation="cached_adapter_a",
        adapter_hash=hash_a,
        duration_ms=cached_time,
        tokens_generated=cached_tokens,
        notes="Same adapter, should be in vLLM cache"
    ))

    if adapter_b:
        # Test 4: Swap to different adapter
        print("\n[6/7] CRITICAL: Swap to adapter B...")
        lora_b = LoRARequest("adapter-b", 2, adapter_b)
        start = time.time()
        outputs = llm.generate([test_prompt], sampling_params, lora_request=lora_b)
        swap_time = (time.time() - start) * 1000
        swap_tokens = len(outputs[0].outputs[0].token_ids)
        print(f"       Swap time: {swap_time:.1f}ms ({swap_tokens} tokens)")
        print(f"       >>> This is the critical number for hot-swapping")
        timings.append(TimingResult(
            operation="swap_to_adapter_b",
            adapter_hash=hash_b,
            duration_ms=swap_time,
            tokens_generated=swap_tokens,
            notes="Swap from A to B"
        ))

        # Test 5: Swap back to A (tests if both stay in memory)
        print("\n[7/7] Swap back to adapter A (both in memory?)...")
        start = time.time()
        outputs = llm.generate([test_prompt], sampling_params, lora_request=lora_a)
        swap_back_time = (time.time() - start) * 1000
        swap_back_tokens = len(outputs[0].outputs[0].token_ids)
        print(f"       Swap back: {swap_back_time:.1f}ms ({swap_back_tokens} tokens)")
        if swap_back_time < first_load_time * 0.5:
            print(f"       >>> GOOD: Both adapters cached in memory!")
        else:
            print(f"       >>> WARNING: Adapter A may have been evicted")
        timings.append(TimingResult(
            operation="swap_back_to_adapter_a",
            adapter_hash=hash_a,
            duration_ms=swap_back_time,
            tokens_generated=swap_back_tokens,
            notes="Swap back - tests if both fit in max_loras"
        ))
    else:
        print("\n[6/7] Skipped (no adapter B provided)")
        print("[7/7] Skipped (no adapter B provided)")

    # Calculate summary statistics
    summary = {
        "model_load_time_s": model_load_time,
        "first_adapter_load_ms": first_load_time,
        "cached_adapter_ms": cached_time,
        "cache_speedup_x": round(first_load_time / cached_time, 1) if cached_time > 0 else 0,
    }

    if adapter_b:
        summary["swap_time_ms"] = swap_time
        summary["swap_back_time_ms"] = swap_back_time
        summary["both_adapters_cached"] = swap_back_time < first_load_time * 0.5

    # Viability assessment
    print("\n" + "=" * 70)
    print("VIABILITY ASSESSMENT")
    print("=" * 70)

    if adapter_b:
        critical_time = swap_time
        print(f"Critical swap time: {critical_time:.0f}ms")
    else:
        critical_time = first_load_time
        print(f"Critical load time: {critical_time:.0f}ms")

    if critical_time < 500:
        print("VERDICT: EXCELLENT - Hot swapping is very fast")
        summary["verdict"] = "excellent"
    elif critical_time < 1000:
        print("VERDICT: GOOD - Hot swapping is acceptable")
        summary["verdict"] = "good"
    elif critical_time < 2000:
        print("VERDICT: MARGINAL - May need optimization")
        summary["verdict"] = "marginal"
    elif critical_time < 5000:
        print("VERDICT: CONCERNING - 2-5s delay noticeable")
        summary["verdict"] = "concerning"
    else:
        print("VERDICT: NOT VIABLE - >5s delay breaks UX")
        summary["verdict"] = "not_viable"

    print("=" * 70)

    return SpikeResults(
        model_name=model_name,
        max_loras=max_loras,
        max_lora_rank=max_lora_rank,
        adapter_a_path=adapter_a,
        adapter_b_path=adapter_b,
        timings=timings,
        summary=summary,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Spike test: Measure vLLM adapter load/swap timing"
    )
    parser.add_argument(
        "--adapter-a",
        required=True,
        help="Path to first adapter directory"
    )
    parser.add_argument(
        "--adapter-b",
        help="Path to second adapter directory (optional, for swap testing)"
    )
    parser.add_argument(
        "--model",
        default=os.getenv("JARVIS_MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct"),
        help="Model name/path"
    )
    parser.add_argument(
        "--max-loras",
        type=int,
        default=int(os.getenv("JARVIS_VLLM_MAX_LORAS", "4")),
        help="Max adapters to keep in memory"
    )
    parser.add_argument(
        "--max-lora-rank",
        type=int,
        default=int(os.getenv("JARVIS_VLLM_MAX_LORA_RANK", "64")),
        help="Max LoRA rank"
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=50,
        help="Tokens to generate per test"
    )
    parser.add_argument(
        "--output",
        default="spike_adapter_timing_results.json",
        help="Output JSON file"
    )

    args = parser.parse_args()

    # Validate adapter paths
    if not Path(args.adapter_a).exists():
        print(f"Error: Adapter A not found: {args.adapter_a}")
        sys.exit(1)
    if args.adapter_b and not Path(args.adapter_b).exists():
        print(f"Error: Adapter B not found: {args.adapter_b}")
        sys.exit(1)

    # Run the spike test
    results = run_spike_test(
        model_name=args.model,
        adapter_a=args.adapter_a,
        adapter_b=args.adapter_b,
        max_loras=args.max_loras,
        max_lora_rank=args.max_lora_rank,
        num_tokens=args.tokens,
    )

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        # Convert dataclasses to dicts for JSON
        data = {
            "model_name": results.model_name,
            "max_loras": results.max_loras,
            "max_lora_rank": results.max_lora_rank,
            "adapter_a_path": results.adapter_a_path,
            "adapter_b_path": results.adapter_b_path,
            "timings": [asdict(t) for t in results.timings],
            "summary": results.summary,
        }
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("\nNext steps based on results:")
    if results.summary.get("verdict") in ("excellent", "good"):
        print("  - Proceed with LRU cache implementation")
        print(f"  - Configure max_loras={args.max_loras} or higher")
    elif results.summary.get("verdict") == "marginal":
        print("  - Consider increasing max_loras")
        print("  - Test with RAM disk for adapter storage")
        print("  - Profile where time is spent (disk I/O vs GPU)")
    else:
        print("  - Investigate bottleneck (disk I/O? GPU transfer?)")
        print("  - Try RAM disk: mount -t tmpfs -o size=2G tmpfs /tmp/adapters")
        print("  - Consider adapter preloading strategies")


if __name__ == "__main__":
    main()
