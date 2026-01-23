# Date Key Extraction Adapter

## Overview

This PRD describes the implementation of a date key extraction adapter that converts natural language date/time references into standardized semantic keys. This enables consistent date handling across all consumers without the LLM needing to know the actual current date.

## Problem Statement

Previously, training data included hardcoded absolute timestamps (e.g., `"2026-01-19T05:00:00Z"`), which:
1. Don't generalize - the model learned specific dates, not patterns
2. Require the LLM to "know" the current date
3. Create timezone/DST complexity in the LLM layer

## Solution

Train a lightweight LoRA adapter that extracts semantic date keys from natural language:

| Input | Output |
|-------|--------|
| "What's the weather tomorrow morning?" | `["tomorrow", "morning"]` |
| "Show my calendar next week" | `["next_week"]` |
| "Giants game last Saturday" | `["last_saturday"]` |
| "Weather in Miami" | `[]` |

Consumers resolve keys to actual datetimes using their own dictionaries.

## Architecture

```
Request with include_date_context=true
                │
                ▼
┌───────────────────────────┐
│   Date Key Adapter        │
│   (LoRA, lightweight)     │
│                           │
│   Input: user text        │
│   Output: date_keys[]     │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│   Command Adapter         │
│   (if include_node_adapter│
│    or always for now)     │
│                           │
│   Input: user text        │
│   Output: tool_calls[]    │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│   Response Merge          │
│                           │
│   - tool_calls from cmd   │
│   - date_keys from date   │
│   - Client handles merge  │
└───────────────────────────┘
```

## Supported Date Keys

### Relative Days
- `today`
- `tomorrow`
- `yesterday`
- `day_after_tomorrow`
- `day_before_yesterday`

### Combined Day+Time (standalone keys)
- `tonight` (today + night)
- `last_night` (yesterday + night)
- `tomorrow_night` (tomorrow + night)
- `tomorrow_morning` (tomorrow + morning)
- `tomorrow_afternoon` (tomorrow + afternoon)
- `tomorrow_evening` (tomorrow + evening)
- `yesterday_morning` (yesterday + morning)
- `yesterday_afternoon` (yesterday + afternoon)
- `yesterday_evening` (yesterday + evening)

### Time of Day (modifiers)
- `morning` (07:00)
- `afternoon` (14:00)
- `evening` (19:00)
- `night` (20:00)
- `noon` (12:00)
- `midnight` (00:00)

### Meal Times (modifiers)
- `at_breakfast` (08:00)
- `during_lunch` (12:00)
- `at_dinner` (18:00)

### Weekdays
- `next_monday`, `next_tuesday`, `next_wednesday`, `next_thursday`, `next_friday`, `next_saturday`, `next_sunday`
- `last_monday`, `last_tuesday`, `last_wednesday`, `last_thursday`, `last_friday`, `last_saturday`, `last_sunday`
- `this_monday`, `this_tuesday`, `this_wednesday`, `this_thursday`, `this_friday`, `this_saturday`, `this_sunday`

### Periods
- `this_week`, `next_week`, `last_week`
- `this_weekend`, `next_weekend`, `last_weekend`
- `this_month`, `next_month`, `last_month`
- `this_year`, `next_year`, `last_year`

### Specific Times (patterns)
- `at_Xam`, `at_Xpm` (e.g., `at_3pm`, `at_9am`) - hourly, X = 1-12
- `at_X_15am`, `at_X_15pm` (e.g., `at_9_15am`) - quarter past
- `at_X_30am`, `at_X_30pm` (e.g., `at_3_30pm`) - half past
- `at_X_45am`, `at_X_45pm` (e.g., `at_10_45pm`) - quarter to

**Note:** Time patterns are parsed dynamically, not enumerated. The adapter should recognize "at 3:45pm" and output `at_3_45pm`.

## Implementation Tasks

### 1. Training Data Generation

Create `data/date_keys_training.jsonl`:

```jsonl
{"text": "tomorrow morning", "date_keys": ["tomorrow", "morning"]}
{"text": "next Tuesday at 3pm", "date_keys": ["next_tuesday", "at_3pm"]}
{"text": "this weekend", "date_keys": ["this_weekend"]}
{"text": "the day after tomorrow", "date_keys": ["day_after_tomorrow"]}
{"text": "last night", "date_keys": ["last_night"]}
{"text": "What's the weather?", "date_keys": []}
{"text": "Show my calendar", "date_keys": []}
```

Generate variations:
- Different phrasings: "tmrw", "next tues", "2morrow"
- Full sentences with embedded date references
- Sentences without date references (negative examples)
- Multiple date references: "from Monday to Friday" → `["this_monday", "this_friday"]`

### 2. Training Script

Create `scripts/train_date_adapter.py`:
- Load base model
- Train LoRA adapter on date key extraction task
- Save adapter to standard location

### 3. API Endpoint

Create `GET /v1/adapters/date-keys`:

```json
{
  "version": "1.0",
  "keys": [
    "today", "tomorrow", "yesterday",
    "day_after_tomorrow", "day_before_yesterday",
    "tonight", "last_night", "tomorrow_night",
    "tomorrow_morning", "tomorrow_afternoon", "tomorrow_evening",
    "yesterday_morning", "yesterday_afternoon", "yesterday_evening",
    "morning", "afternoon", "evening", "night", "noon", "midnight",
    "at_breakfast", "during_lunch", "at_dinner",
    "next_monday", "next_tuesday", "next_wednesday", "next_thursday", "next_friday", "next_saturday", "next_sunday",
    "last_monday", "last_tuesday", "last_wednesday", "last_thursday", "last_friday", "last_saturday", "last_sunday",
    "this_monday", "this_tuesday", "this_wednesday", "this_thursday", "this_friday", "this_saturday", "this_sunday",
    "this_week", "next_week", "last_week",
    "this_weekend", "next_weekend", "last_weekend",
    "this_month", "next_month", "last_month",
    "this_year", "next_year", "last_year"
  ],
  "patterns": {
    "time": "at_Xam, at_Xpm (X = 1-12)",
    "time_15": "at_X_15am, at_X_15pm (quarter past)",
    "time_30": "at_X_30am, at_X_30pm (half past)",
    "time_45": "at_X_45am, at_X_45pm (quarter to)"
  },
  "notes": {
    "composability": "Multiple keys may be returned, e.g., ['next_tuesday', 'morning']",
    "no_date": "Empty array returned if no date reference detected",
    "combined_keys": "Common phrases like 'tonight', 'tomorrow morning' are standalone keys"
  }
}
```

**This endpoint is unauthenticated** - it only exposes vocabulary, no sensitive data.

### 4. Request Model Extension

Extend chat completion request:

```python
class ChatCompletionRequest(BaseModel):
    # ... existing OpenAI-compatible fields ...
    
    # Jarvis extensions (optional)
    include_date_context: Optional[bool] = None  # Apply date key adapter
    include_node_adapter: Optional[str] = None   # Apply node-specific adapter (TBD)
```

### 5. Inference Integration

When `include_date_context=True`:
1. Load date key adapter
2. Run extraction on input text
3. Include `date_keys` in response

Response extension:
```json
{
  "id": "chatcmpl-xxx",
  "choices": [...],
  "date_keys": ["tomorrow", "morning"]  // Only present if include_date_context=true
}
```

## File Structure

```
jarvis-llm-proxy-api/
├── data/
│   └── date_keys_training.jsonl
├── scripts/
│   └── train_date_adapter.py
├── adapters/
│   └── date_keys/
│       ├── adapter_config.json
│       └── adapter_model.safetensors
├── routes/
│   └── adapters.py  # New route file for /v1/adapters/*
└── services/
    └── date_key_extraction.py
```

## Success Criteria

1. Date adapter correctly extracts keys from 95%+ of common date expressions
2. Endpoint returns complete vocabulary
3. Response includes `date_keys` when flag enabled
4. No regression in command adapter performance
5. Inference latency increase < 50ms for date extraction step

## Adapter Validation Test Suite

After training the date key adapter, run a comprehensive validation test to ensure accuracy.

### Test Script: `scripts/validate_date_adapter.py`

```python
#!/usr/bin/env python3
"""
Comprehensive validation test for the date key extraction adapter.

Run after training to verify the adapter correctly extracts date keys.

Usage:
    python scripts/validate_date_adapter.py --adapter-path adapters/date_keys
    
Exit codes:
    0 = All tests passed (>= 95% accuracy)
    1 = Tests failed (< 95% accuracy)
"""

import argparse
import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class TestCase:
    """A single test case for date key extraction."""
    input_text: str
    expected_keys: list[str]
    category: str
    description: Optional[str] = None


# ============================================================================
# TEST CASES
# ============================================================================

TEST_CASES = [
    # -------------------------------------------------------------------------
    # CATEGORY: Single relative days
    # -------------------------------------------------------------------------
    TestCase("today", ["today"], "relative_days", "bare keyword"),
    TestCase("tomorrow", ["tomorrow"], "relative_days", "bare keyword"),
    TestCase("yesterday", ["yesterday"], "relative_days", "bare keyword"),
    TestCase("the day after tomorrow", ["day_after_tomorrow"], "relative_days"),
    TestCase("day after tomorrow", ["day_after_tomorrow"], "relative_days"),
    TestCase("the day before yesterday", ["day_before_yesterday"], "relative_days"),
    
    # Variations and abbreviations
    TestCase("tmrw", ["tomorrow"], "relative_days", "abbreviation"),
    TestCase("2morrow", ["tomorrow"], "relative_days", "slang"),
    TestCase("2day", ["today"], "relative_days", "slang"),
    TestCase("tmr", ["tomorrow"], "relative_days", "abbreviation"),
    TestCase("yest", ["yesterday"], "relative_days", "abbreviation"),
    TestCase("yday", ["yesterday"], "relative_days", "abbreviation"),
    
    # -------------------------------------------------------------------------
    # CATEGORY: Time of day modifiers
    # -------------------------------------------------------------------------
    TestCase("this morning", ["morning"], "time_of_day"),
    TestCase("this afternoon", ["afternoon"], "time_of_day"),
    TestCase("this evening", ["evening"], "time_of_day"),
    TestCase("tonight", ["tonight"], "time_of_day", "standalone combined key"),
    TestCase("last night", ["last_night"], "time_of_day", "standalone combined key"),
    TestCase("at noon", ["noon"], "time_of_day"),
    TestCase("at midnight", ["midnight"], "time_of_day"),
    
    # -------------------------------------------------------------------------
    # CATEGORY: Meal times
    # -------------------------------------------------------------------------
    TestCase("at breakfast", ["at_breakfast"], "meal_times"),
    TestCase("during breakfast", ["at_breakfast"], "meal_times"),
    TestCase("at lunch", ["during_lunch"], "meal_times"),
    TestCase("during lunch", ["during_lunch"], "meal_times"),
    TestCase("lunchtime", ["during_lunch"], "meal_times"),
    TestCase("at dinner", ["at_dinner"], "meal_times"),
    TestCase("at dinnertime", ["at_dinner"], "meal_times"),
    TestCase("during dinner", ["at_dinner"], "meal_times"),
    
    # -------------------------------------------------------------------------
    # CATEGORY: Weekdays with modifiers
    # -------------------------------------------------------------------------
    TestCase("next Monday", ["next_monday"], "weekdays"),
    TestCase("next Tuesday", ["next_tuesday"], "weekdays"),
    TestCase("next Wednesday", ["next_wednesday"], "weekdays"),
    TestCase("next Thursday", ["next_thursday"], "weekdays"),
    TestCase("next Friday", ["next_friday"], "weekdays"),
    TestCase("next Saturday", ["next_saturday"], "weekdays"),
    TestCase("next Sunday", ["next_sunday"], "weekdays"),
    
    TestCase("last Monday", ["last_monday"], "weekdays"),
    TestCase("last Tuesday", ["last_tuesday"], "weekdays"),
    TestCase("last Wednesday", ["last_wednesday"], "weekdays"),
    TestCase("last Thursday", ["last_thursday"], "weekdays"),
    TestCase("last Friday", ["last_friday"], "weekdays"),
    TestCase("last Saturday", ["last_saturday"], "weekdays"),
    TestCase("last Sunday", ["last_sunday"], "weekdays"),
    
    TestCase("this Monday", ["this_monday"], "weekdays"),
    TestCase("this Friday", ["this_friday"], "weekdays"),
    
    # Abbreviations
    TestCase("next Mon", ["next_monday"], "weekdays", "abbreviation"),
    TestCase("next Tues", ["next_tuesday"], "weekdays", "abbreviation"),
    TestCase("next Wed", ["next_wednesday"], "weekdays", "abbreviation"),
    TestCase("next Thurs", ["next_thursday"], "weekdays", "abbreviation"),
    TestCase("next Fri", ["next_friday"], "weekdays", "abbreviation"),
    TestCase("next Sat", ["next_saturday"], "weekdays", "abbreviation"),
    TestCase("next Sun", ["next_sunday"], "weekdays", "abbreviation"),
    
    # -------------------------------------------------------------------------
    # CATEGORY: Periods
    # -------------------------------------------------------------------------
    TestCase("this week", ["this_week"], "periods"),
    TestCase("next week", ["next_week"], "periods"),
    TestCase("last week", ["last_week"], "periods"),
    TestCase("this weekend", ["this_weekend"], "periods"),
    TestCase("next weekend", ["next_weekend"], "periods"),
    TestCase("last weekend", ["last_weekend"], "periods"),
    TestCase("this month", ["this_month"], "periods"),
    TestCase("next month", ["next_month"], "periods"),
    TestCase("last month", ["last_month"], "periods"),
    TestCase("this year", ["this_year"], "periods"),
    TestCase("next year", ["next_year"], "periods"),
    TestCase("last year", ["last_year"], "periods"),
    
    # -------------------------------------------------------------------------
    # CATEGORY: Specific times
    # -------------------------------------------------------------------------
    TestCase("at 3pm", ["at_3pm"], "specific_times"),
    TestCase("at 3 pm", ["at_3pm"], "specific_times"),
    TestCase("at 3:00 pm", ["at_3pm"], "specific_times"),
    TestCase("at 9am", ["at_9am"], "specific_times"),
    TestCase("at 9 am", ["at_9am"], "specific_times"),
    TestCase("at 9:30am", ["at_9_30am"], "specific_times"),
    TestCase("at 9:30 am", ["at_9_30am"], "specific_times"),
    TestCase("at 3:30pm", ["at_3_30pm"], "specific_times"),
    TestCase("at 9:15am", ["at_9_15am"], "specific_times", "quarter past"),
    TestCase("at 9:45am", ["at_9_45am"], "specific_times", "quarter to"),
    TestCase("at 3:15pm", ["at_3_15pm"], "specific_times", "quarter past"),
    TestCase("at 3:45pm", ["at_3_45pm"], "specific_times", "quarter to"),
    TestCase("at noon", ["noon"], "specific_times"),
    TestCase("at 12pm", ["noon"], "specific_times"),
    TestCase("at 12am", ["midnight"], "specific_times"),
    
    # -------------------------------------------------------------------------
    # CATEGORY: Combined day+time (standalone keys)
    # -------------------------------------------------------------------------
    TestCase("tomorrow morning", ["tomorrow_morning"], "combined"),
    TestCase("tomorrow afternoon", ["tomorrow_afternoon"], "combined"),
    TestCase("tomorrow evening", ["tomorrow_evening"], "combined"),
    TestCase("tomorrow night", ["tomorrow_night"], "combined"),
    TestCase("yesterday morning", ["yesterday_morning"], "combined"),
    TestCase("yesterday afternoon", ["yesterday_afternoon"], "combined"),
    TestCase("yesterday evening", ["yesterday_evening"], "combined"),
    
    # -------------------------------------------------------------------------
    # CATEGORY: Composite (weekday + time - still decomposed)
    # -------------------------------------------------------------------------
    TestCase("yesterday afternoon", ["yesterday", "afternoon"], "composite"),
    TestCase("next Tuesday morning", ["next_tuesday", "morning"], "composite"),
    TestCase("next Friday evening", ["next_friday", "evening"], "composite"),
    TestCase("last Saturday night", ["last_saturday", "night"], "composite"),
    TestCase("tomorrow at 3pm", ["tomorrow", "at_3pm"], "composite"),
    TestCase("next Monday at 9am", ["next_monday", "at_9am"], "composite"),
    TestCase("this Friday at noon", ["this_friday", "noon"], "composite"),
    
    # -------------------------------------------------------------------------
    # CATEGORY: Embedded in sentences (voice command style)
    # -------------------------------------------------------------------------
    TestCase("What's the weather tomorrow?", ["tomorrow"], "embedded"),
    TestCase("What's the weather tomorrow morning?", ["tomorrow_morning"], "embedded"),
    TestCase("Show my calendar for next week", ["next_week"], "embedded"),
    TestCase("What's on my schedule this weekend?", ["this_weekend"], "embedded"),
    TestCase("How did the Giants do last Saturday?", ["last_saturday"], "embedded"),
    TestCase("Set a reminder for tomorrow at 3pm", ["tomorrow", "at_3pm"], "embedded"),
    TestCase("What appointments do I have next Tuesday?", ["next_tuesday"], "embedded"),
    TestCase("Book a meeting for next Friday morning", ["next_friday", "morning"], "embedded"),
    TestCase("What happened yesterday evening?", ["yesterday", "evening"], "embedded"),
    TestCase("Cancel my plans for this weekend", ["this_weekend"], "embedded"),
    TestCase("What's the forecast for the day after tomorrow?", ["day_after_tomorrow"], "embedded"),
    TestCase("Remind me tonight to call mom", ["tonight"], "embedded"),
    TestCase("Any events this month?", ["this_month"], "embedded"),
    TestCase("What are my goals for next year?", ["next_year"], "embedded"),
    TestCase("How much did I spend last year?", ["last_year"], "embedded"),
    TestCase("Let's discuss this at lunch", ["during_lunch"], "embedded"),
    TestCase("Remind me at breakfast to take my meds", ["at_breakfast"], "embedded"),
    TestCase("Call mom after dinner", ["at_dinner"], "embedded"),
    TestCase("What happened last night?", ["last_night"], "embedded"),
    
    # -------------------------------------------------------------------------
    # CATEGORY: No date reference (negative cases - should return [])
    # -------------------------------------------------------------------------
    TestCase("What's the weather in Miami?", [], "no_date"),
    TestCase("What's the weather?", [], "no_date"),
    TestCase("Turn on the lights", [], "no_date"),
    TestCase("Play some music", [], "no_date"),
    TestCase("What time is it?", [], "no_date"),
    TestCase("How are you?", [], "no_date"),
    TestCase("Tell me a joke", [], "no_date"),
    TestCase("What's the capital of France?", [], "no_date"),
    TestCase("Set the thermostat to 72", [], "no_date"),
    TestCase("Call John", [], "no_date"),
    TestCase("Add milk to my shopping list", [], "no_date"),
    TestCase("Who won the Super Bowl?", [], "no_date"),
    
    # -------------------------------------------------------------------------
    # CATEGORY: Ambiguous / edge cases
    # -------------------------------------------------------------------------
    TestCase("morning", ["morning"], "edge_cases", "bare time of day"),
    TestCase("evening", ["evening"], "edge_cases", "bare time of day"),
    TestCase("Monday", ["this_monday"], "edge_cases", "bare weekday defaults to this"),
    TestCase("Friday", ["this_friday"], "edge_cases", "bare weekday defaults to this"),
    TestCase("the weekend", ["this_weekend"], "edge_cases", "bare weekend defaults to this"),
    TestCase("this week's meeting", ["this_week"], "edge_cases", "possessive"),
    TestCase("next month's budget", ["next_month"], "edge_cases", "possessive"),
    
    # Multiple dates in one phrase
    TestCase("from Monday to Friday", ["this_monday", "this_friday"], "edge_cases", "range"),
    TestCase("between tomorrow and next Friday", ["tomorrow", "next_friday"], "edge_cases", "range"),
    
    # -------------------------------------------------------------------------
    # CATEGORY: Case insensitivity
    # -------------------------------------------------------------------------
    TestCase("TOMORROW", ["tomorrow"], "case_insensitive"),
    TestCase("Tomorrow", ["tomorrow"], "case_insensitive"),
    TestCase("NEXT MONDAY", ["next_monday"], "case_insensitive"),
    TestCase("Next Monday", ["next_monday"], "case_insensitive"),
    TestCase("THIS WEEKEND", ["this_weekend"], "case_insensitive"),
]


def load_adapter_and_model(adapter_path: str):
    """Load the trained adapter and base model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    base_model_id = "meta-llama/Llama-3.2-3B-Instruct"  # Or read from adapter_config.json
    
    print(f"Loading base model: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    return model, tokenizer


def extract_date_keys(model, tokenizer, text: str) -> list[str]:
    """Run inference to extract date keys from text."""
    
    # Format prompt for date key extraction
    prompt = f"""Extract date/time references from the following text and return them as JSON keys.
Return an empty array [] if no date/time references are found.
Only use keys from this vocabulary: today, tomorrow, yesterday, day_after_tomorrow, day_before_yesterday, morning, afternoon, evening, night, noon, midnight, next_monday, next_tuesday, next_wednesday, next_thursday, next_friday, next_saturday, next_sunday, last_monday, last_tuesday, last_wednesday, last_thursday, last_friday, last_saturday, last_sunday, this_monday, this_tuesday, this_wednesday, this_thursday, this_friday, this_saturday, this_sunday, this_week, next_week, last_week, this_weekend, next_weekend, last_weekend, this_month, next_month, last_month, at_Xam, at_Xpm, at_X_30am, at_X_30pm

Text: "{text}"
Date keys (JSON array):"""

    messages = [{"role": "user", "content": prompt}]
    
    inputs = tokenizer.apply_chat_template(
        messages, 
        return_tensors="pt",
        add_generation_prompt=True
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    
    # Parse JSON array from response
    try:
        # Find the JSON array in the response
        import re
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        if match:
            return json.loads(match.group())
        return []
    except json.JSONDecodeError:
        print(f"  ⚠️  Failed to parse response: {response[:100]}")
        return []


def run_tests(model, tokenizer, test_cases: list[TestCase]) -> dict:
    """Run all test cases and return results."""
    results = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "by_category": {},
        "failures": []
    }
    
    for tc in test_cases:
        results["total"] += 1
        
        # Initialize category stats
        if tc.category not in results["by_category"]:
            results["by_category"][tc.category] = {"total": 0, "passed": 0, "failed": 0}
        results["by_category"][tc.category]["total"] += 1
        
        # Run extraction
        actual_keys = extract_date_keys(model, tokenizer, tc.input_text)
        
        # Compare (order-independent)
        expected_set = set(tc.expected_keys)
        actual_set = set(actual_keys)
        
        if expected_set == actual_set:
            results["passed"] += 1
            results["by_category"][tc.category]["passed"] += 1
            print(f"  ✅ {tc.input_text[:50]:<50} → {actual_keys}")
        else:
            results["failed"] += 1
            results["by_category"][tc.category]["failed"] += 1
            results["failures"].append({
                "input": tc.input_text,
                "expected": tc.expected_keys,
                "actual": actual_keys,
                "category": tc.category,
                "description": tc.description
            })
            print(f"  ❌ {tc.input_text[:50]:<50}")
            print(f"      Expected: {tc.expected_keys}")
            print(f"      Actual:   {actual_keys}")
    
    return results


def print_summary(results: dict):
    """Print test summary."""
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    accuracy = (results["passed"] / results["total"]) * 100 if results["total"] > 0 else 0
    
    print(f"\nOverall: {results['passed']}/{results['total']} passed ({accuracy:.1f}%)")
    print(f"Status: {'✅ PASSED' if accuracy >= 95 else '❌ FAILED'} (threshold: 95%)")
    
    print("\nBy Category:")
    print("-" * 50)
    for category, stats in sorted(results["by_category"].items()):
        cat_accuracy = (stats["passed"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        status = "✅" if cat_accuracy >= 90 else "⚠️" if cat_accuracy >= 70 else "❌"
        print(f"  {status} {category:<25} {stats['passed']}/{stats['total']} ({cat_accuracy:.0f}%)")
    
    if results["failures"]:
        print("\n" + "-" * 50)
        print(f"FAILURES ({len(results['failures'])}):")
        print("-" * 50)
        for f in results["failures"][:10]:  # Show first 10 failures
            print(f"\n  Input: \"{f['input']}\"")
            print(f"  Expected: {f['expected']}")
            print(f"  Actual: {f['actual']}")
            if f["description"]:
                print(f"  Note: {f['description']}")
        
        if len(results["failures"]) > 10:
            print(f"\n  ... and {len(results['failures']) - 10} more failures")
    
    return accuracy >= 95


def main():
    parser = argparse.ArgumentParser(description="Validate date key extraction adapter")
    parser.add_argument("--adapter-path", required=True, help="Path to trained adapter")
    parser.add_argument("--output-json", help="Save results to JSON file")
    args = parser.parse_args()
    
    print("=" * 70)
    print("DATE KEY EXTRACTION ADAPTER - VALIDATION TEST")
    print("=" * 70)
    print(f"\nAdapter: {args.adapter_path}")
    print(f"Test cases: {len(TEST_CASES)}")
    print()
    
    # Load model
    model, tokenizer = load_adapter_and_model(args.adapter_path)
    
    # Run tests
    print("\nRunning tests...")
    print("-" * 70)
    results = run_tests(model, tokenizer, TEST_CASES)
    
    # Print summary
    passed = print_summary(results)
    
    # Save JSON if requested
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")
    
    # Exit with appropriate code
    return 0 if passed else 1


if __name__ == "__main__":
    exit(main())
```

### Running the Validation

After training:

```bash
# Run validation
python scripts/validate_date_adapter.py --adapter-path adapters/date_keys

# Save results to JSON for CI
python scripts/validate_date_adapter.py \
    --adapter-path adapters/date_keys \
    --output-json test_results/date_adapter_validation.json
```

### Expected Output

```
======================================================================
DATE KEY EXTRACTION ADAPTER - VALIDATION TEST
======================================================================

Adapter: adapters/date_keys
Test cases: 98

Running tests...
----------------------------------------------------------------------
  ✅ today                                             → ['today']
  ✅ tomorrow                                          → ['tomorrow']
  ✅ What's the weather tomorrow morning?              → ['tomorrow', 'morning']
  ❌ tmrw                                              
      Expected: ['tomorrow']
      Actual:   []
  ...

======================================================================
TEST SUMMARY
======================================================================

Overall: 94/98 passed (95.9%)
Status: ✅ PASSED (threshold: 95%)

By Category:
--------------------------------------------------
  ✅ relative_days            12/12 (100%)
  ✅ time_of_day              7/7 (100%)
  ✅ weekdays                 21/21 (100%)
  ✅ periods                  9/9 (100%)
  ✅ specific_times           10/10 (100%)
  ✅ composite                12/12 (100%)
  ✅ embedded                 13/13 (100%)
  ✅ no_date                  12/12 (100%)
  ⚠️ edge_cases               6/8 (75%)
  ❌ case_insensitive         2/4 (50%)
```

### CI Integration

Add to CI pipeline to catch regressions:

```yaml
- name: Validate date adapter
  run: |
    python scripts/validate_date_adapter.py \
      --adapter-path adapters/date_keys \
      --output-json test_results/date_adapter.json
      
- name: Upload validation results
  uses: actions/upload-artifact@v3
  with:
    name: date-adapter-validation
    path: test_results/date_adapter.json
```

### Expanding Test Cases

To add new test cases, simply append to the `TEST_CASES` list:

```python
# Add your custom test cases
TEST_CASES.extend([
    TestCase("your input text", ["expected", "keys"], "category"),
    # ...
])
```

Categories should be meaningful for debugging:
- `relative_days` - today, tomorrow, yesterday, etc.
- `time_of_day` - morning, afternoon, evening, etc.
- `weekdays` - next Monday, last Tuesday, etc.
- `periods` - this week, next month, etc.
- `specific_times` - at 3pm, at 9:30am, etc.
- `composite` - combinations like "tomorrow morning"
- `embedded` - full sentences with date references
- `no_date` - negative cases, should return []
- `edge_cases` - ambiguous or tricky inputs
- `case_insensitive` - verify case handling

## Future Enhancements

- Support for specific dates: "January 15th" → `["january_15"]`
- Support for relative offsets: "in 3 days" → `["in_3_days"]`
- Confidence scores per extracted key
- Multi-language support
