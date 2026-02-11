#!/usr/bin/env python3
"""
Comprehensive validation test for the Jarvis date/time key extraction adapter.

Run after training to verify the adapter correctly extracts date keys,
including relative time expressions.

Usage:
    python scripts/validate_jarvis_adapter.py --adapter-path adapters/jarvis

Exit codes:
    0 = All tests passed (>= 95% accuracy)
    1 = Tests failed (< 95% accuracy)
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


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
    
    # -------------------------------------------------------------------------
    # CATEGORY: Time of day modifiers
    # -------------------------------------------------------------------------
    TestCase("this morning", ["this_morning"], "time_of_day", "standalone combined key"),
    TestCase("this afternoon", ["this_afternoon"], "time_of_day", "standalone combined key"),
    TestCase("this evening", ["this_evening"], "time_of_day", "standalone combined key"),
    TestCase("tonight", ["tonight"], "time_of_day", "standalone combined key"),
    TestCase("last night", ["last_night"], "time_of_day", "standalone combined key"),
    TestCase("at noon", ["noon"], "time_of_day"),
    TestCase("at midnight", ["midnight"], "time_of_day"),
    TestCase("noon", ["noon"], "time_of_day", "bare keyword"),
    TestCase("midnight", ["midnight"], "time_of_day", "bare keyword"),
    
    # -------------------------------------------------------------------------
    # CATEGORY: Meal times
    # -------------------------------------------------------------------------
    TestCase("at breakfast", ["at_breakfast"], "meal_times"),
    TestCase("during breakfast", ["during_breakfast"], "meal_times"),
    TestCase("at lunch", ["during_lunch"], "meal_times"),
    TestCase("during lunch", ["during_lunch"], "meal_times"),
    TestCase("lunchtime", ["during_lunch"], "meal_times"),
    TestCase("at dinner", ["at_dinner"], "meal_times"),
    TestCase("at dinnertime", ["at_dinner"], "meal_times"),
    TestCase("during dinner", ["during_dinner"], "meal_times"),
    TestCase("after dinner", ["after_dinner"], "meal_times"),
    
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
    TestCase("last Friday", ["last_friday"], "weekdays"),
    TestCase("last Saturday", ["last_saturday"], "weekdays"),
    TestCase("last Sunday", ["last_sunday"], "weekdays"),
    
    TestCase("this Monday", ["this_monday"], "weekdays"),
    TestCase("this Friday", ["this_friday"], "weekdays"),
    
    # Abbreviations
    TestCase("next Mon", ["next_monday"], "weekdays", "abbreviation"),
    TestCase("next Tues", ["next_tuesday"], "weekdays", "abbreviation"),
    TestCase("next Fri", ["next_friday"], "weekdays", "abbreviation"),
    TestCase("next Sat", ["next_saturday"], "weekdays", "abbreviation"),
    
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
    TestCase("at 9am", ["at_9am"], "specific_times"),
    TestCase("at 9 am", ["at_9am"], "specific_times"),
    TestCase("at 9:30am", ["at_9_30am"], "specific_times"),
    TestCase("at 9:30 am", ["at_9_30am"], "specific_times"),
    TestCase("at 3:30pm", ["at_3_30pm"], "specific_times"),
    TestCase("at 9:15am", ["at_9_15am"], "specific_times", "quarter past"),
    TestCase("at 9:45am", ["at_9_45am"], "specific_times", "quarter to"),
    TestCase("at 12pm", ["noon"], "specific_times"),
    TestCase("at 12am", ["at_12am"], "specific_times"),
    
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
    TestCase("next Tuesday morning", ["next_tuesday", "morning"], "composite"),
    TestCase("next Friday evening", ["next_friday", "evening"], "composite"),
    TestCase("last Saturday night", ["last_saturday", "night"], "composite"),
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
    TestCase("What happened yesterday evening?", ["yesterday_evening"], "embedded"),
    TestCase("Cancel my plans for this weekend", ["this_weekend"], "embedded"),
    TestCase("What's the forecast for the day after tomorrow?", ["day_after_tomorrow"], "embedded"),
    TestCase("Remind me tonight to call mom", ["tonight"], "embedded"),
    TestCase("Any events this month?", ["this_month"], "embedded"),
    TestCase("What are my goals for next year?", ["next_year"], "embedded"),
    TestCase("How much did I spend last year?", ["last_year"], "embedded"),
    TestCase("Let's discuss this at lunch", ["during_lunch"], "embedded"),
    TestCase("Remind me at breakfast to take my meds", ["at_breakfast"], "embedded"),
    TestCase("Call mom after dinner", ["after_dinner"], "embedded"),
    TestCase("What happened last night?", ["last_night"], "embedded"),
    
    # -------------------------------------------------------------------------
    # CATEGORY: Relative time (minutes)
    # -------------------------------------------------------------------------
    TestCase("Remind me in 5 minutes to check the oven", ["in_5_minutes"], "relative_minutes"),
    TestCase("Set a reminder in 30 minutes", ["in_30_minutes"], "relative_minutes"),
    TestCase("In 15 minutes remind me to check email", ["in_15_minutes"], "relative_minutes"),
    TestCase("Remind me in half an hour", ["in_30_minutes"], "relative_minutes"),
    TestCase("In a quarter of an hour remind me", ["in_15_minutes"], "relative_minutes"),

    # -------------------------------------------------------------------------
    # CATEGORY: Relative time (hours flattened to minutes)
    # -------------------------------------------------------------------------
    TestCase("Set a reminder in 2 hours", ["in_120_minutes"], "relative_hours_to_minutes"),
    TestCase("In an hour remind me to move the car", ["in_60_minutes"], "relative_hours_to_minutes"),
    TestCase("Remind me in 3 hours to pick up the kids", ["in_180_minutes"], "relative_hours_to_minutes"),
    TestCase("In a couple hours remind me", ["in_120_minutes"], "relative_hours_to_minutes"),

    # -------------------------------------------------------------------------
    # CATEGORY: Relative time (compound -> minutes)
    # -------------------------------------------------------------------------
    TestCase("Remind me in 1 hour and 30 minutes", ["in_90_minutes"], "relative_compound"),
    TestCase("In 2 hours and 15 minutes set a reminder", ["in_135_minutes"], "relative_compound"),
    TestCase("Remind me in an hour and 45 minutes", ["in_105_minutes"], "relative_compound"),

    # -------------------------------------------------------------------------
    # CATEGORY: Relative time (days)
    # -------------------------------------------------------------------------
    TestCase("Remind me in 3 days to renew the subscription", ["in_3_days"], "relative_days_offset"),
    TestCase("In a day remind me to follow up", ["in_1_days"], "relative_days_offset"),
    TestCase("Remind me in a week to check the garden", ["in_7_days"], "relative_days_offset"),
    TestCase("In 2 weeks remind me to schedule a checkup", ["in_14_days"], "relative_days_offset"),

    # -------------------------------------------------------------------------
    # CATEGORY: Relative time negatives (should return [])
    # -------------------------------------------------------------------------
    TestCase("What happened 30 minutes ago?", [], "relative_negatives"),
    TestCase("The meeting lasted 2 hours", [], "relative_negatives"),
    TestCase("Set a timer for 5 minutes", [], "relative_negatives"),
    TestCase("I ran for 45 minutes this morning", ["morning"], "relative_negatives"),
    TestCase("She left 3 hours ago", [], "relative_negatives"),
    TestCase("We talked for an hour", [], "relative_negatives"),

    # -------------------------------------------------------------------------
    # CATEGORY: Ambiguous relative time (should return [])
    # -------------------------------------------------------------------------
    TestCase("Remind me in a few minutes", [], "relative_ambiguous"),
    TestCase("In a little while remind me to call mom", [], "relative_ambiguous"),
    TestCase("Remind me later to grab the groceries", [], "relative_ambiguous"),

    # -------------------------------------------------------------------------
    # CATEGORY: Mixed relative + semantic
    # -------------------------------------------------------------------------
    TestCase("Remind me in 30 minutes and also tomorrow morning", ["in_30_minutes", "tomorrow_morning"], "relative_mixed"),
    TestCase("What's the weather tomorrow and remind me in an hour", ["tomorrow", "in_60_minutes"], "relative_mixed"),

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
    
    # Read base model from adapter config
    config_path = Path(adapter_path) / "adapter_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            base_model_id = config.get("base_model_name_or_path", "meta-llama/Llama-3.2-3B-Instruct")
    else:
        base_model_id = "meta-llama/Llama-3.2-3B-Instruct"
    
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
    import torch
    
    # Format prompt for date key extraction
    system_prompt = """You are a date/time extraction assistant. Extract date and time references from the user's text and return them as a JSON array of semantic keys.

Rules:
- Return only the JSON array, nothing else
- Use standardized keys like: today, tomorrow, yesterday, tomorrow_morning, tonight, last_night, next_monday, this_weekend, at_3pm, etc.
- For relative time: flatten hours/minutes to in_N_minutes (e.g., "in 2 hours" → "in_120_minutes", "in half an hour" → "in_30_minutes")
- For relative days: use in_N_days (e.g., "in 3 days" → "in_3_days", "in a week" → "in_7_days")
- Return [] if no date/time references are found
- Return [] for ambiguous time expressions like "in a few minutes", "later", "in a bit"
- Return [] for durations and past references like "for 30 minutes", "2 hours ago"
- Multiple keys can be returned for composite expressions like "next Tuesday at 3pm" -> ["next_tuesday", "at_3pm"]"""

    user_message = f'Extract date keys from: "{text}"'
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
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


def print_summary(results: dict) -> bool:
    """Print test summary. Returns True if passed threshold."""
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
    parser = argparse.ArgumentParser(description="Validate Jarvis date/time key extraction adapter")
    parser.add_argument("--adapter-path", required=True, help="Path to trained adapter")
    parser.add_argument("--output-json", help="Save results to JSON file")
    args = parser.parse_args()
    
    print("=" * 70)
    print("JARVIS KEY EXTRACTION ADAPTER - VALIDATION TEST")
    print("=" * 70)
    print(f"\nAdapter: {args.adapter_path}")
    print(f"Test cases: {len(TEST_CASES)}")
    print()
    
    # Check adapter exists
    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        print(f"❌ Adapter not found: {adapter_path}")
        print("   Run: python scripts/train_jarvis_adapter.py")
        sys.exit(1)
    
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
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
