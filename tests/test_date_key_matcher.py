"""
Test suite for the deterministic regex-based date key matcher.

Tests against adversarial cases (media titles, idioms, abbreviations,
composites) and the full 4,987-example training dataset.
"""

import json
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.date_key_matcher import extract_date_keys

# ============================================================================
# Adversarial test cases: (text, expected_keys)
# ============================================================================

BASIC_CASES = [
    ("What's the weather tomorrow?", ["tomorrow"]),
    ("Set a reminder for today", ["today"]),
    ("What happened yesterday?", ["yesterday"]),
    ("Show my calendar this weekend", ["this_weekend"]),
    ("Any meetings next week?", ["next_week"]),
    ("What's on for tonight?", ["tonight"]),
    ("Check last month's expenses", ["last_month"]),
]

COMPOSITE_CASES = [
    ("What's the weather tomorrow morning?", ["tomorrow_morning"]),
    ("Schedule a call next Tuesday at 3pm", ["at_3pm", "next_tuesday"]),
    ("Remind me next Friday evening", ["evening", "next_friday"]),
    ("What happened last Monday night?", ["last_monday", "night"]),
    ("Set alarm for tomorrow at 7am", ["at_7am", "tomorrow"]),
]

DAY_AFTER_CASES = [
    ("What's the weather the day after tomorrow?", ["day_after_tomorrow"]),
    ("Schedule for the day after tomorrow", ["day_after_tomorrow"]),
    ("The day before yesterday was cold", ["day_before_yesterday"]),
]

NO_DATE_CASES = [
    ("Turn off the lights", []),
    ("Play some music", []),
    ("What's the capital of France?", []),
    ("Set the thermostat to 72", []),
    ("How are you?", []),
    ("Add milk to the shopping list", []),
    ("Lock the front door", []),
]

TRICKY_NEGATIVE_CASES = [
    ("What happened 30 minutes ago?", []),
    ("The meeting lasted 2 hours", []),
    ("I waited for 20 minutes", []),
    ("Set a timer for 5 minutes", []),
    ("Remind me later", []),
    ("In a few minutes check on the oven", []),
]

MEDIA_NEGATIVE_CASES = [
    ("Play Yesterday by the Beatles", []),
    ("Play Monday by Imagine Dragons", []),
    ("Play Saturday Night Fever soundtrack", []),
    ("Watch Saturday Night Live", []),
]

RELATIVE_TIME_CASES = [
    ("Remind me in 30 minutes", ["in_30_minutes"]),
    ("Set a timer in 2 hours", ["in_120_minutes"]),
    ("Call me in 15 minutes", ["in_15_minutes"]),
    ("Remind me in an hour", ["in_60_minutes"]),
    ("In 3 days check the order", ["in_3_days"]),
    ("Remind me in a week", ["in_7_days"]),
    ("Set a reminder for 45 minutes from now", ["in_45_minutes"]),
    ("In about 2 hours let me know", ["in_120_minutes"]),
    ("I need a reminder in about an hour", ["in_60_minutes"]),
]

SPECIFIC_TIME_CASES = [
    ("Set alarm for 7am", ["at_7am"]),
    ("Meeting at 3:30pm", ["at_3_30pm"]),
    ("Wake me at 6:15am", ["at_6_15am"]),
    ("Dinner at 8pm", ["at_8pm"]),
    ("12 am", ["midnight"]),
    ("at 12pm", ["noon"]),
]

PERIOD_CASES = [
    ("What's my schedule this week?", ["this_week"]),
    ("Plans for next month?", ["next_month"]),
    ("What did I do last year?", ["last_year"]),
    ("Next weekend plans", ["next_weekend"]),
    ("Last weekend was fun", ["last_weekend"]),
]

IDIOM_CASES = [
    ("It's a brand new day", []),
    ("Time flies", []),
    ("Better late than never", []),
    ("A day late and a dollar short", []),
]

ALL_ADVERSARIAL_CASES = (
    BASIC_CASES
    + COMPOSITE_CASES
    + DAY_AFTER_CASES
    + NO_DATE_CASES
    + TRICKY_NEGATIVE_CASES
    + MEDIA_NEGATIVE_CASES
    + RELATIVE_TIME_CASES
    + SPECIFIC_TIME_CASES
    + PERIOD_CASES
    + IDIOM_CASES
)


# ============================================================================
# Adversarial parametrized tests
# ============================================================================

@pytest.mark.parametrize(
    "text,expected",
    ALL_ADVERSARIAL_CASES,
    ids=[c[0][:50] for c in ALL_ADVERSARIAL_CASES],
)
def test_adversarial(text, expected):
    predicted = extract_date_keys(text)
    assert predicted == sorted(expected), (
        f'\n  Input: "{text}"\n  Expected: {sorted(expected)}\n  Got: {predicted}'
    )


# ============================================================================
# Full training dataset test (4,987 examples)
# ============================================================================

TRAINING_DATA_PATH = Path(__file__).parent.parent / "data" / "jarvis_training.jsonl"


@pytest.mark.skipif(
    not TRAINING_DATA_PATH.exists(),
    reason="Training data not found",
)
def test_full_training_dataset():
    """Run the matcher against all training examples. Target: 100%."""
    examples = []
    with open(TRAINING_DATA_PATH) as f:
        for line in f:
            examples.append(json.loads(line.strip()))

    correct = 0
    total = len(examples)
    failures = []

    for ex in examples:
        predicted = extract_date_keys(ex["text"])
        expected = sorted(ex["date_keys"])
        if predicted == expected:
            correct += 1
        else:
            failures.append((ex["text"], expected, predicted))

    accuracy = correct / total if total > 0 else 0

    if failures:
        sample = failures[:10]
        failure_report = "\n".join(
            f'  "{t[:60]}" exp={e} got={g}' for t, e, g in sample
        )
        remaining = len(failures) - len(sample)
        if remaining > 0:
            failure_report += f"\n  ... and {remaining} more"
    else:
        failure_report = ""

    assert accuracy == 1.0, (
        f"Accuracy: {correct}/{total} = {accuracy:.1%}\n"
        f"Failures ({len(failures)}):\n{failure_report}"
    )
