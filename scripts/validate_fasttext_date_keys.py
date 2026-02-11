#!/usr/bin/env python3
"""
Validate FastText date key extraction against the same test suite as the LLM adapter.

Usage:
    python scripts/validate_fasttext_date_keys.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import test cases from the adapter validation script
from scripts.validate_jarvis_adapter import TEST_CASES


def safe_predict(model, text: str, k: int = 5) -> tuple[list[str], list[float]]:
    """Wrapper for model.predict that handles numpy 2.x compatibility."""
    import numpy as np
    original_array = np.array
    def patched_array(obj, *args, copy=None, **kwargs):
        if copy is False:
            return np.asarray(obj, *args, **kwargs)
        return original_array(obj, *args, copy=copy, **kwargs)
    np.array = patched_array
    try:
        labels, scores = model.predict(text, k=k)
        return list(labels), list(scores)
    finally:
        np.array = original_array


def extract_keys(model, text: str, threshold: float = 0.3) -> list[str]:
    """Extract date keys from text using FastText model."""
    text_clean = text.lower().strip()
    labels, scores = safe_predict(model, text_clean, k=10)

    keys = []
    for label, score in zip(labels, scores):
        key = label.replace("__label__", "")
        if key != "NONE" and score >= threshold:
            keys.append(key)
    return keys


def main():
    import fasttext

    model_path = "models/date_keys_fasttext.bin"

    print("=" * 70)
    print("FASTTEXT DATE KEY EXTRACTION - VALIDATION TEST")
    print("=" * 70)
    print(f"\nModel: {model_path}")
    print(f"Test cases: {len(TEST_CASES)}")
    print()

    # Load model
    print("Loading model...")
    model = fasttext.load_model(model_path)
    print(f"Loaded: {len(model.labels)} labels\n")

    # Run tests
    print("Running tests...")
    print("-" * 70)

    results_by_category: dict[str, dict] = {}
    failures: list[tuple] = []
    passed = 0

    for tc in TEST_CASES:
        predicted = extract_keys(model, tc.input_text)
        expected = tc.expected_keys

        # Exact match
        is_pass = set(predicted) == set(expected)

        # Track by category
        if tc.category not in results_by_category:
            results_by_category[tc.category] = {"passed": 0, "total": 0}
        results_by_category[tc.category]["total"] += 1

        if is_pass:
            passed += 1
            results_by_category[tc.category]["passed"] += 1
            status = "✅"
            print(f"  {status} {tc.input_text:45} → {predicted}")
        else:
            status = "❌"
            print(f"  {status} {tc.input_text:45}")
            print(f"      Expected: {expected}")
            print(f"      Actual:   {predicted}")
            failures.append((tc.input_text, expected, predicted, tc.description))

    # Summary
    total = len(TEST_CASES)
    accuracy = (passed / total) * 100 if total > 0 else 0
    threshold = 95.0

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"\nOverall: {passed}/{total} passed ({accuracy:.1f}%)")
    print(f"Status: {'✅ PASSED' if accuracy >= threshold else '❌ FAILED'} (threshold: {threshold}%)")

    print("\nBy Category:")
    print("-" * 50)
    for cat, stats in sorted(results_by_category.items()):
        cat_pct = (stats["passed"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        icon = "✅" if cat_pct >= 90 else "⚠️" if cat_pct >= 70 else "❌"
        print(f"  {icon} {cat:25} {stats['passed']}/{stats['total']} ({cat_pct:.0f}%)")

    if failures:
        print(f"\n{'-' * 50}")
        print(f"FAILURES ({len(failures)}):")
        print("-" * 50)
        for text, expected, actual, note in failures[:20]:
            print(f"\n  Input: \"{text}\"")
            print(f"  Expected: {expected}")
            print(f"  Actual: {actual}")
            if note:
                print(f"  Note: {note}")
        if len(failures) > 20:
            print(f"\n  ... and {len(failures) - 20} more failures")

    return 0 if accuracy >= threshold else 1


if __name__ == "__main__":
    sys.exit(main())
