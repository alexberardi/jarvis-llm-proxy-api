#!/usr/bin/env python3
"""
Validate hybrid FastText + LLM date key extraction.

This tests the actual production extraction strategy:
- FastText >= 85% confidence: Use FastText directly
- FastText 75-85%: Hint to LLM
- FastText < 75%: LLM only

Usage:
    python scripts/validate_hybrid_date_keys.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.validate_jarvis_adapter import TEST_CASES
from services.date_keys import (
    FastTextDateKeyExtractor,
    FASTTEXT_HIGH_CONFIDENCE,
    FASTTEXT_HINT_THRESHOLD,
)


def main() -> int:
    print("=" * 70)
    print("HYBRID DATE KEY EXTRACTION - VALIDATION")
    print("=" * 70)
    print(f"\nThresholds:")
    print(f"  FastText direct: >= {FASTTEXT_HIGH_CONFIDENCE:.0%}")
    print(f"  FastText hint:   >= {FASTTEXT_HINT_THRESHOLD:.0%}")
    print(f"  LLM only:        < {FASTTEXT_HINT_THRESHOLD:.0%}")
    print(f"\nTest cases: {len(TEST_CASES)}")
    print()

    # Load FastText
    ft = FastTextDateKeyExtractor()

    # Categorize by what would happen in hybrid mode
    fasttext_direct = []  # >= 85% confidence
    fasttext_hint = []    # 75-85% confidence
    llm_only = []         # < 75% confidence

    results_by_category: dict[str, dict] = {}

    print("Analyzing confidence distribution...")
    print("-" * 70)

    for tc in TEST_CASES:
        result = ft.predict(tc.input_text)
        conf = result.max_confidence
        predicted = result.keys
        expected = set(tc.expected_keys)
        is_correct = set(predicted) == expected

        # Track by category
        if tc.category not in results_by_category:
            results_by_category[tc.category] = {
                "ft_direct": 0, "ft_direct_correct": 0,
                "ft_hint": 0, "ft_hint_correct": 0,
                "llm_only": 0, "llm_only_correct": 0,
            }

        if conf >= FASTTEXT_HIGH_CONFIDENCE:
            fasttext_direct.append((tc, result, is_correct))
            results_by_category[tc.category]["ft_direct"] += 1
            if is_correct:
                results_by_category[tc.category]["ft_direct_correct"] += 1
        elif conf >= FASTTEXT_HINT_THRESHOLD:
            fasttext_hint.append((tc, result, is_correct))
            results_by_category[tc.category]["ft_hint"] += 1
            if is_correct:
                results_by_category[tc.category]["ft_hint_correct"] += 1
        else:
            llm_only.append((tc, result, is_correct))
            results_by_category[tc.category]["llm_only"] += 1
            if is_correct:
                results_by_category[tc.category]["llm_only_correct"] += 1

    # Summary stats
    ft_direct_correct = sum(1 for _, _, c in fasttext_direct if c)
    ft_hint_correct = sum(1 for _, _, c in fasttext_hint if c)
    llm_only_correct = sum(1 for _, _, c in llm_only if c)

    print(f"\n{'Route':<20} {'Count':>8} {'FT Correct':>12} {'FT Accuracy':>12}")
    print("-" * 55)
    print(f"{'FastText direct':<20} {len(fasttext_direct):>8} {ft_direct_correct:>12} {ft_direct_correct/max(len(fasttext_direct),1)*100:>11.1f}%")
    print(f"{'FastText + hint':<20} {len(fasttext_hint):>8} {ft_hint_correct:>12} {ft_hint_correct/max(len(fasttext_hint),1)*100:>11.1f}%")
    print(f"{'LLM only':<20} {len(llm_only):>8} {llm_only_correct:>12} {llm_only_correct/max(len(llm_only),1)*100:>11.1f}%")

    # Show FastText direct results (what would be used without LLM)
    print(f"\n{'=' * 70}")
    print("FASTTEXT DIRECT ROUTE (>= 85% confidence)")
    print(f"{'=' * 70}")
    print(f"\n{len(fasttext_direct)} cases, {ft_direct_correct} correct ({ft_direct_correct/max(len(fasttext_direct),1)*100:.1f}%)")

    if fasttext_direct:
        print("\nFailures in FastText direct route:")
        failures = [(tc, r) for tc, r, c in fasttext_direct if not c]
        if failures:
            for tc, r in failures[:10]:
                print(f"  ❌ \"{tc.input_text}\"")
                print(f"     Expected: {tc.expected_keys}")
                print(f"     Got:      {r.keys} (conf: {r.max_confidence:.2f})")
        else:
            print("  None! ✅")

    # Show hint route
    print(f"\n{'=' * 70}")
    print("FASTTEXT HINT ROUTE (75-85% confidence)")
    print(f"{'=' * 70}")
    print(f"\n{len(fasttext_hint)} cases would get FastText hint passed to LLM")

    if fasttext_hint:
        print("\nSamples:")
        for tc, r, correct in fasttext_hint[:5]:
            status = "✅" if correct else "❌"
            print(f"  {status} \"{tc.input_text}\" → hint: {r.keys} (conf: {r.max_confidence:.2f})")

    # Show LLM only route
    print(f"\n{'=' * 70}")
    print("LLM ONLY ROUTE (< 75% confidence)")
    print(f"{'=' * 70}")
    print(f"\n{len(llm_only)} cases would go to LLM without hint")

    if llm_only:
        print("\nSamples:")
        for tc, r, correct in llm_only[:10]:
            ft_guess = r.keys if r.keys else "[]"
            print(f"  \"{tc.input_text}\"")
            print(f"     Expected: {tc.expected_keys}, FT guess: {ft_guess} (conf: {r.max_confidence:.2f})")

    # Best case scenario
    print(f"\n{'=' * 70}")
    print("BEST CASE SCENARIO (if LLM was perfect)")
    print(f"{'=' * 70}")

    # FastText direct correct + all hint/llm cases (assuming LLM perfect)
    best_case = ft_direct_correct + len(fasttext_hint) + len(llm_only)
    print(f"\nIf LLM handled hint+llm cases perfectly:")
    print(f"  {best_case}/{len(TEST_CASES)} = {best_case/len(TEST_CASES)*100:.1f}%")

    # Worst case (FastText only for everything)
    ft_total_correct = ft_direct_correct + ft_hint_correct + llm_only_correct
    print(f"\nFastText alone (all cases):")
    print(f"  {ft_total_correct}/{len(TEST_CASES)} = {ft_total_correct/len(TEST_CASES)*100:.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
