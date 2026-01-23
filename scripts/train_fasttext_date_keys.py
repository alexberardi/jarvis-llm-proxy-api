#!/usr/bin/env python3
"""
Train a FastText classifier for date key extraction.

This creates a lightweight, fast classifier that can extract date keys
from natural language text in milliseconds on CPU.

Usage:
    python scripts/train_fasttext_date_keys.py

Output:
    models/date_keys_fasttext.bin (~5-10MB)
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_training_data(path: str) -> list[tuple[str, list[str]]]:
    """Load training data from JSONL file."""
    examples = []
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            text = data["text"].strip()
            keys = data["date_keys"]
            if text:  # Skip empty texts
                examples.append((text, keys))
    return examples


def convert_to_fasttext_format(examples: list[tuple[str, list[str]]]) -> list[str]:
    """
    Convert training examples to FastText format.

    FastText format: __label__key1 __label__key2 text

    For examples with no date keys, we use __label__NONE
    """
    lines = []
    for text, keys in examples:
        if keys:
            labels = " ".join(f"__label__{k}" for k in keys)
        else:
            labels = "__label__NONE"
        # Normalize text: lowercase, simple cleanup
        text_clean = text.lower().strip()
        lines.append(f"{labels} {text_clean}")
    return lines


def train_fasttext_model(train_path: str, output_path: str):
    """Train and save FastText model."""
    try:
        import fasttext
    except ImportError:
        print("❌ FastText not installed. Run: pip install fasttext-wheel")
        sys.exit(1)

    print(f"🏋️ Training FastText model...")
    model = fasttext.train_supervised(
        input=train_path,
        epoch=50,           # More epochs for small dataset
        lr=0.5,             # Learning rate
        wordNgrams=3,       # Use trigrams for better phrase matching
        dim=100,            # Embedding dimension
        loss='ova',         # One-vs-all for multi-label
        minCount=1,         # Include rare words
        minn=2,             # Min char ngram
        maxn=5,             # Max char ngram (helps with typos/abbrevs)
        bucket=200000,      # Hash bucket size
    )

    # Save model
    model.save_model(output_path)
    print(f"✅ Model saved to: {output_path}")

    # Print model info
    print(f"   Labels: {len(model.labels)}")
    print(f"   Vocab size: {len(model.words)}")

    return model


def safe_predict(model, text: str, k: int = 5):
    """Wrapper for model.predict that handles numpy 2.x compatibility."""
    import numpy as np
    # Monkey-patch numpy temporarily to handle fasttext's copy=False issue
    original_array = np.array
    def patched_array(obj, *args, copy=None, **kwargs):
        if copy is False:
            return np.asarray(obj, *args, **kwargs)
        return original_array(obj, *args, copy=copy, **kwargs)
    np.array = patched_array
    try:
        labels, scores = model.predict(text, k=k)
        return labels, list(scores)
    finally:
        np.array = original_array


def evaluate_model(model, examples: list[tuple[str, list[str]]]) -> dict:
    """Evaluate model on examples."""
    correct = 0
    total = len(examples)

    for text, expected_keys in examples:
        text_clean = text.lower().strip()

        # Get predictions (k=5 to catch multi-label)
        labels, scores = safe_predict(model, text_clean, k=5)

        # Extract predicted keys
        predicted = set()
        for label, score in zip(labels, scores):
            key = label.replace("__label__", "")
            if key != "NONE" and score >= 0.3:
                predicted.add(key)

        expected_set = set(expected_keys)

        if predicted == expected_set:
            correct += 1

    accuracy = (correct / total) * 100 if total > 0 else 0
    return {"correct": correct, "total": total, "accuracy": accuracy}


def main():
    training_data_path = "data/date_keys_training.jsonl"
    fasttext_train_path = "data/date_keys_fasttext.txt"
    model_output_path = "models/date_keys_fasttext.bin"

    print("=" * 60)
    print("FASTTEXT DATE KEY CLASSIFIER TRAINING")
    print("=" * 60)

    # Check training data exists
    if not os.path.exists(training_data_path):
        print(f"❌ Training data not found: {training_data_path}")
        sys.exit(1)

    # Load and convert training data
    print(f"\n📂 Loading training data from {training_data_path}...")
    examples = load_training_data(training_data_path)
    print(f"   Loaded {len(examples)} examples")

    # Convert to FastText format
    print("\n📝 Converting to FastText format...")
    fasttext_lines = convert_to_fasttext_format(examples)

    # Save FastText training file
    os.makedirs("data", exist_ok=True)
    with open(fasttext_train_path, "w") as f:
        f.write("\n".join(fasttext_lines))
    print(f"   Saved to {fasttext_train_path}")

    # Ensure output directory exists
    os.makedirs("models", exist_ok=True)

    # Train model
    model = train_fasttext_model(fasttext_train_path, model_output_path)

    # Evaluate
    print("\n📊 Evaluating model...")
    results = evaluate_model(model, examples)
    print(f"   Training accuracy: {results['accuracy']:.1f}% ({results['correct']}/{results['total']})")

    # Quick test
    print("\n🧪 Quick test:")
    test_phrases = [
        "What's the weather tomorrow morning?",
        "Show my calendar next week",
        "Turn on the lights",
        "Set a reminder for 3pm",
        "yesterday",
    ]

    for phrase in test_phrases:
        labels, scores = safe_predict(model, phrase.lower(), k=3)
        keys = [l.replace("__label__", "") for l, s in zip(labels, scores) if s >= 0.3 and l != "__label__NONE"]
        key_scores = [f'{s:.2f}' for l, s in zip(labels, scores) if s >= 0.3 and l != "__label__NONE"]
        print(f"   '{phrase}' → {keys} (scores: {key_scores})")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nModel: {model_output_path}")
    print(f"Size: {os.path.getsize(model_output_path) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
