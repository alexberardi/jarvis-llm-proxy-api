#!/usr/bin/env python3
"""
Train a DeBERTa-v3-small multi-label classifier for date key extraction.

Hybrid approach:
- DeBERTa classifies semantic date keys (~62 labels: today, tomorrow, next_monday, etc.)
- Regex handles dynamic time patterns (at_3pm, in_30_minutes, etc.)

Usage:
    python scripts/train_date_key_classifier.py
    python scripts/train_date_key_classifier.py --epochs 10 --batch-size 32
    python scripts/train_date_key_classifier.py --eval-only --model-path models/date_key_classifier

Output:
    models/date_key_classifier/
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
TRAINING_DATA = PROJECT_ROOT / "data" / "jarvis_training.jsonl"
MODEL_OUTPUT = PROJECT_ROOT / "models" / "date_key_classifier"
BASE_MODEL = "distilbert-base-uncased"

# Dynamic patterns handled by regex (not the classifier)
DYNAMIC_PATTERNS = [
    (re.compile(r"in[_ ](\d+)[_ ]minutes?", re.IGNORECASE), lambda m: f"in_{m.group(1)}_minutes"),
    (re.compile(r"in[_ ](\d+)[_ ]hours?", re.IGNORECASE), lambda m: f"in_{int(m.group(1)) * 60}_minutes"),
    (re.compile(r"in[_ ]an?[_ ]hour", re.IGNORECASE), lambda _: "in_60_minutes"),
    (re.compile(r"in[_ ]a[_ ]half[_ ](?:an?[_ ])?hour", re.IGNORECASE), lambda _: "in_30_minutes"),
    (re.compile(r"in[_ ](\d+)[_ ]days?", re.IGNORECASE), lambda m: f"in_{m.group(1)}_days"),
    (re.compile(r"in[_ ]a[_ ]week", re.IGNORECASE), lambda _: "in_7_days"),
    (re.compile(r"in[_ ](\d+)[_ ]weeks?", re.IGNORECASE), lambda m: f"in_{int(m.group(1)) * 7}_days"),
    (re.compile(r"at[_ ](\d{1,2}):(\d{2})\s*(am|pm)", re.IGNORECASE), lambda m: f"at_{m.group(1)}_{m.group(2)}{m.group(3).lower()}" if m.group(2) != "00" else f"at_{m.group(1)}{m.group(3).lower()}"),
    (re.compile(r"at[_ ](\d{1,2})\s*(am|pm)", re.IGNORECASE), lambda m: f"at_{m.group(1)}{m.group(2).lower()}"),
    (re.compile(r"(\d{1,2}):(\d{2})\s*(am|pm)", re.IGNORECASE), lambda m: f"at_{m.group(1)}_{m.group(2)}{m.group(3).lower()}" if m.group(2) != "00" else f"at_{m.group(1)}{m.group(3).lower()}"),
    (re.compile(r"(\d{1,2})\s*(am|pm)\b", re.IGNORECASE), lambda m: f"at_{m.group(1)}{m.group(2).lower()}"),
    (re.compile(r"half\s+past\s+(\d{1,2})\s*(am|pm)", re.IGNORECASE), lambda m: f"at_{m.group(1)}_30{m.group(2).lower()}"),
    (re.compile(r"quarter\s+past\s+(\d{1,2})\s*(am|pm)", re.IGNORECASE), lambda m: f"at_{m.group(1)}_15{m.group(2).lower()}"),
    (re.compile(r"quarter\s+to\s+(\d{1,2})\s*(am|pm)", re.IGNORECASE), lambda m: f"at_{int(m.group(1))-1 if int(m.group(1))>1 else 12}_45{m.group(2).lower()}"),
]

# Keys that are ONLY handled by regex (not classifier labels)
DYNAMIC_KEY_PREFIXES = ("at_", "in_")


def is_dynamic_key(key: str) -> bool:
    """Check if a key is a dynamic pattern (handled by regex, not classifier)."""
    return key.startswith(DYNAMIC_KEY_PREFIXES) and any(c.isdigit() for c in key)


def extract_dynamic_keys(text: str) -> list[str]:
    """Extract dynamic date keys using regex patterns."""
    keys = []
    for pattern, formatter in DYNAMIC_PATTERNS:
        for match in pattern.finditer(text):
            keys.append(formatter(match))
    return keys


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: Path) -> tuple[list[dict], list[str]]:
    """Load training data and build the semantic label set."""
    examples = []
    all_keys = set()

    with open(path) as f:
        for line in f:
            row = json.loads(line.strip())
            examples.append(row)
            for k in row["date_keys"]:
                if not is_dynamic_key(k):
                    all_keys.add(k)

    # Add "NO_DATE" as a label for negative examples
    labels = sorted(all_keys)
    print(f"Loaded {len(examples)} examples, {len(labels)} semantic labels")
    return examples, labels


def prepare_dataset(
    examples: list[dict],
    labels: list[str],
    tokenizer,
) -> Dataset:
    """Convert examples to HuggingFace Dataset with multi-hot label vectors."""
    label_to_idx = {l: i for i, l in enumerate(labels)}

    texts = []
    label_vectors = []

    for ex in examples:
        texts.append(ex["text"])
        vec = [0.0] * len(labels)
        for k in ex["date_keys"]:
            if k in label_to_idx:
                vec[label_to_idx[k]] = 1.0
        label_vectors.append(vec)

    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
    dataset = Dataset.from_dict({
        **encodings,
        "labels": label_vectors,
    })
    return dataset


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    """Compute multi-label F1 metrics."""
    logits, labels = eval_pred
    predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
    labels = labels.astype(int)

    f1_micro = f1_score(labels, predictions, average="micro", zero_division=0)
    f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
    f1_samples = f1_score(labels, predictions, average="samples", zero_division=0)

    # Exact match (all labels correct for the example)
    exact_match = np.all(predictions == labels, axis=1).mean()

    return {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_samples": f1_samples,
        "exact_match": exact_match,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    examples, labels = load_data(TRAINING_DATA)

    # Split: 80% train, 10% val, 10% test
    import random
    random.seed(42)
    random.shuffle(examples)
    n = len(examples)
    train_examples = examples[:int(n * 0.8)]
    val_examples = examples[int(n * 0.8):int(n * 0.9)]
    test_examples = examples[int(n * 0.9):]

    print(f"Split: {len(train_examples)} train, {len(val_examples)} val, {len(test_examples)} test")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Datasets
    train_dataset = prepare_dataset(train_examples, labels, tokenizer)
    val_dataset = prepare_dataset(val_examples, labels, tokenizer)
    test_dataset = prepare_dataset(test_examples, labels, tokenizer)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(labels),
        problem_type="multi_label_classification",
    )

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Training args
    output_dir = str(MODEL_OUTPUT)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="exact_match",
        greater_is_better=True,
        logging_steps=20,
        report_to="none",
        fp16=False,
        use_cpu=True,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print(f"\nTraining for {args.epochs} epochs...")
    trainer.train()

    # Save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save label mapping
    label_map = {"labels": labels, "label_to_idx": {l: i for i, l in enumerate(labels)}}
    (Path(output_dir) / "label_map.json").write_text(json.dumps(label_map, indent=2))

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    evaluate(model, tokenizer, test_examples, labels, device)

    print(f"\nModel saved to: {output_dir}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, tokenizer, examples, labels, device="cpu"):
    """Run full evaluation with per-example breakdown."""
    label_to_idx = {l: i for i, l in enumerate(labels)}
    model.eval()
    if device != "cpu":
        model = model.to(device)

    correct = 0
    total = 0
    failures = []

    for ex in examples:
        text = ex["text"]
        expected_keys = set(ex["date_keys"])

        # Semantic keys from classifier
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        if device != "cpu":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits).squeeze()

        predicted_semantic = set()
        for i, prob in enumerate(probs):
            if prob > 0.5:
                predicted_semantic.add(labels[i])

        # Dynamic keys from regex
        predicted_dynamic = set(extract_dynamic_keys(text))

        # Combined prediction
        predicted = predicted_semantic | predicted_dynamic

        # Filter expected to separate semantic vs dynamic for fair comparison
        expected_semantic = {k for k in expected_keys if not is_dynamic_key(k)}
        expected_dynamic = {k for k in expected_keys if is_dynamic_key(k)}

        if predicted == expected_keys:
            correct += 1
        else:
            failures.append({
                "text": text,
                "expected": sorted(expected_keys),
                "predicted": sorted(predicted),
                "expected_semantic": sorted(expected_semantic),
                "predicted_semantic": sorted(predicted_semantic),
                "expected_dynamic": sorted(expected_dynamic),
                "predicted_dynamic": sorted(predicted_dynamic),
            })
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Exact match accuracy: {correct}/{total} = {accuracy:.1%}")
    print(f"Failures: {len(failures)}")

    if failures:
        print(f"\nSample failures (first 10):")
        for f in failures[:10]:
            print(f"  \"{f['text'][:50]}\"")
            print(f"    expected:  {f['expected']}")
            print(f"    predicted: {f['predicted']}")

    return accuracy, failures


# ---------------------------------------------------------------------------
# Eval-only mode
# ---------------------------------------------------------------------------

def eval_only(args):
    """Load a trained model and evaluate on test split."""
    model_path = args.model_path or str(MODEL_OUTPUT)
    print(f"Loading model from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    label_map = json.loads((Path(model_path) / "label_map.json").read_text())
    labels = label_map["labels"]

    examples, _ = load_data(TRAINING_DATA)

    import random
    random.seed(42)
    random.shuffle(examples)
    test_examples = examples[int(len(examples) * 0.9):]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Evaluating {len(test_examples)} test examples on {device}")

    evaluate(model, tokenizer, test_examples, labels, device)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train date key classifier")
    p.add_argument("--epochs", type=int, default=10, help="Training epochs")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    p.add_argument("--eval-only", action="store_true", help="Evaluate existing model")
    p.add_argument("--model-path", type=str, default=None, help="Model path for eval-only")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.eval_only:
        eval_only(args)
    else:
        train(args)
