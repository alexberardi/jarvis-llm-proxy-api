#!/usr/bin/env python3
"""
Generate training data for the Jarvis date/time extraction adapter.

This script creates a JSONL file with examples mapping natural language
date/time phrases to standardized semantic keys, including relative time
expressions like "in 30 minutes" and "in 3 days".

Usage:
    python scripts/generate_jarvis_training_data.py

Output:
    data/jarvis_training.jsonl
"""

import json
import random
from pathlib import Path
from typing import Optional

# Output file
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "jarvis_training.jsonl"

# ============================================================================
# DATE KEY VOCABULARY
# ============================================================================

# Standalone combined keys (day + time as single unit)
COMBINED_KEYS = {
    "tonight": ["tonight", "tonite", "this evening late", "later tonight"],
    "last_night": ["last night", "lastnight", "yesterday night", "yesternight"],
    "tomorrow_night": ["tomorrow night", "tmrw night", "tomorrow nite"],
    "tomorrow_morning": ["tomorrow morning", "tmrw morning", "tomorrow am", "tomorrow morn"],
    "tomorrow_afternoon": ["tomorrow afternoon", "tmrw afternoon"],
    "tomorrow_evening": ["tomorrow evening", "tmrw evening"],
    "yesterday_morning": ["yesterday morning", "yest morning", "yesterday am"],
    "yesterday_afternoon": ["yesterday afternoon", "yest afternoon"],
    "yesterday_evening": ["yesterday evening", "yest evening"],
    "this_morning": ["this morning", "earlier today", "earlier this morning"],
    "this_afternoon": ["this afternoon", "later today", "this aft"],
    "this_evening": ["this evening", "early tonight"],
}

# Relative days
RELATIVE_DAYS = {
    "today": ["today", "2day", "tday"],
    "tomorrow": ["tomorrow", "tmrw", "tmr", "2morrow", "2mrw", "tom"],
    "yesterday": ["yesterday", "yest", "yday", "ystrdy"],
    "day_after_tomorrow": [
        "day after tomorrow", "the day after tomorrow", 
        "in two days", "in 2 days", "overmorrow"
    ],
    "day_before_yesterday": [
        "day before yesterday", "the day before yesterday",
        "two days ago", "2 days ago"
    ],
}

# Time of day modifiers
TIME_MODIFIERS = {
    "morning": ["morning", "morn", "am", "in the morning", "in the am"],
    "afternoon": ["afternoon", "aft", "in the afternoon"],
    "evening": ["evening", "eve", "in the evening"],
    "night": ["night", "nite", "at night", "in the night"],
    "noon": ["noon", "midday", "at noon", "12 o'clock", "12 oclock"],
    "midnight": ["midnight", "at midnight", "12 am"],
}

# Meal times - distinct keys for at/during/after
MEAL_TIMES = {
    "at_breakfast": ["at breakfast", "breakfast time", "at breakfasttime", "at breakfast time"],
    "during_breakfast": ["during breakfast", "while eating breakfast", "over breakfast"],
    "during_lunch": ["at lunch", "during lunch", "lunchtime", "at lunchtime", "lunch time", "over lunch"],
    "at_dinner": ["at dinner", "dinner time", "at dinnertime", "at dinner time"],
    "during_dinner": ["during dinner", "while eating dinner", "over dinner"],
    "after_dinner": ["after dinner", "after dinnertime", "post dinner", "once dinner is done"],
}

# Weekdays with modifiers
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
WEEKDAY_ABBREVS = {
    "monday": ["monday", "mon"],
    "tuesday": ["tuesday", "tues", "tue"],
    "wednesday": ["wednesday", "wed", "weds"],
    "thursday": ["thursday", "thurs", "thur", "thu"],
    "friday": ["friday", "fri"],
    "saturday": ["saturday", "sat"],
    "sunday": ["sunday", "sun"],
}

# Periods
PERIODS = {
    "this_week": ["this week", "the current week"],
    "next_week": ["next week", "the following week", "the week after"],
    "last_week": ["last week", "the previous week", "the week before"],
    "this_weekend": ["this weekend", "the weekend", "this sat and sun"],
    "next_weekend": ["next weekend", "the following weekend"],
    "last_weekend": ["last weekend", "the previous weekend"],
    "this_month": ["this month", "the current month"],
    "next_month": ["next month", "the following month"],
    "last_month": ["last month", "the previous month"],
    "this_year": ["this year", "the current year"],
    "next_year": ["next year", "the following year"],
    "last_year": ["last year", "the previous year"],
}

# Relative time (minutes) - expressions with their minute values
RELATIVE_MINUTES = {
    "in_1_minutes": ["in a minute", "in one minute"],
    "in_5_minutes": ["in 5 minutes", "in five minutes"],
    "in_10_minutes": ["in 10 minutes", "in ten minutes"],
    "in_15_minutes": ["in 15 minutes", "in fifteen minutes", "in a quarter of an hour", "in quarter of an hour"],
    "in_20_minutes": ["in 20 minutes", "in twenty minutes"],
    "in_30_minutes": ["in 30 minutes", "in thirty minutes", "in half an hour", "in half hour"],
    "in_45_minutes": ["in 45 minutes", "in forty-five minutes"],
    "in_60_minutes": ["in an hour", "in one hour", "in 1 hour"],
    "in_75_minutes": ["in an hour and 15 minutes", "in 1 hour and 15 minutes"],
    "in_90_minutes": ["in 90 minutes", "in an hour and a half", "in an hour and 30 minutes", "in 1 hour and 30 minutes"],
    "in_105_minutes": ["in an hour and 45 minutes", "in 1 hour and 45 minutes"],
    "in_120_minutes": ["in 2 hours", "in two hours", "in a couple hours", "in a couple of hours"],
    "in_135_minutes": ["in 2 hours and 15 minutes"],
    "in_150_minutes": ["in 2 hours and 30 minutes", "in two and a half hours"],
    "in_180_minutes": ["in 3 hours", "in three hours"],
    "in_240_minutes": ["in 4 hours", "in four hours"],
}

# Relative time (days)
RELATIVE_DAYS_OFFSET = {
    "in_1_days": ["in a day", "in one day"],
    "in_2_days": ["in 2 days", "in two days", "in a couple days", "in a couple of days"],
    "in_3_days": ["in 3 days", "in three days"],
    "in_5_days": ["in 5 days", "in five days"],
    "in_7_days": ["in a week", "in one week"],
    "in_14_days": ["in 2 weeks", "in two weeks"],
}

# Relative time voice command templates
RELATIVE_TIME_TEMPLATES = [
    "Remind me {time} to {task}",
    "Set a reminder {time}",
    "{time} remind me to {task}",
    "Can you remind me {time}?",
    "Set an alarm {time}",
]

# Relative time negative examples (duration/past - should return [])
RELATIVE_TIME_NEGATIVES = [
    "What happened 30 minutes ago?",
    "The meeting lasted 2 hours",
    "I ran for 45 minutes this morning",
    "Set a timer for 5 minutes",
    "It takes about 30 minutes to get there",
    "I moved here 3 days ago",
    "The trip took 2 days",
    "I waited for 20 minutes",
    "The recipe takes 2 hours to bake",
    "She left 3 hours ago",
    "It was 5 minutes late",
    "The drive is about 40 minutes",
    "We talked for an hour",
    "That was like 10 minutes ago",
]

# Ambiguous relative time (should return [])
AMBIGUOUS_TIME_SENTENCES = [
    "Remind me in a few minutes to check the oven",
    "In a little while remind me to call mom",
    "Remind me in a bit to take out the trash",
    "Set a reminder in some time",
    "Remind me later to grab the groceries",
]

# Specific times (generated dynamically)
def generate_time_variants(hour: int, am_pm: str) -> list[str]:
    """Generate variants for a specific hour."""
    variants = [
        f"at {hour}{am_pm}",
        f"at {hour} {am_pm}",
        f"at {hour}:00 {am_pm}",
        f"at {hour}:00{am_pm}",
        f"{hour}{am_pm}",
        f"{hour} {am_pm}",
    ]
    return variants


def generate_time_30_variants(hour: int, am_pm: str) -> list[str]:
    """Generate variants for half past the hour."""
    variants = [
        f"at {hour}:30{am_pm}",
        f"at {hour}:30 {am_pm}",
        f"at half past {hour} {am_pm}",
        f"{hour}:30{am_pm}",
        f"{hour}:30 {am_pm}",
        f"{hour} thirty {am_pm}",
    ]
    return variants


def generate_time_15_variants(hour: int, am_pm: str) -> list[str]:
    """Generate variants for quarter past the hour."""
    variants = [
        f"at {hour}:15{am_pm}",
        f"at {hour}:15 {am_pm}",
        f"at quarter past {hour} {am_pm}",
        f"{hour}:15{am_pm}",
        f"{hour}:15 {am_pm}",
        f"{hour} fifteen {am_pm}",
    ]
    return variants


def generate_time_45_variants(hour: int, am_pm: str) -> list[str]:
    """Generate variants for quarter to the hour."""
    next_hour = hour + 1 if hour < 12 else 1
    variants = [
        f"at {hour}:45{am_pm}",
        f"at {hour}:45 {am_pm}",
        f"at quarter to {next_hour} {am_pm}",
        f"{hour}:45{am_pm}",
        f"{hour}:45 {am_pm}",
        f"{hour} forty-five {am_pm}",
    ]
    return variants


# ============================================================================
# SENTENCE TEMPLATES
# ============================================================================

# Templates where date appears embedded in a sentence
VOICE_COMMAND_TEMPLATES = [
    "What's the weather {date}?",
    "What's the weather for {date}?",
    "Weather {date}",
    "Show me the weather {date}",
    "What will the weather be {date}?",
    "What's on my calendar {date}?",
    "Show my calendar for {date}",
    "What appointments do I have {date}?",
    "Any meetings {date}?",
    "What's my schedule {date}?",
    "Remind me {date} to call mom",
    "Set a reminder for {date}",
    "Set a reminder {date}",
    "Schedule a meeting {date}",
    "Book a meeting for {date}",
    "What happened {date}?",
    "How did the Giants do {date}?",
    "What were the scores {date}?",
    "Any news from {date}?",
    "What did I do {date}?",
    "Cancel my plans for {date}",
    "Clear my calendar {date}",
    "What events are {date}?",
    "Do I have anything {date}?",
    "Am I free {date}?",
    "What time is my meeting {date}?",
    "When's my first meeting {date}?",
    "What's happening {date}?",
    "Play my playlist for {date}",
    "How much did I spend {date}?",
    "What did I buy {date}?",
    "Any bills due {date}?",
    "Send the report {date}",
    "Finish the project by {date}",
    "Let's meet {date}",
    "Can we talk {date}?",
    "I'll do it {date}",
    "Start the task {date}",
]

# Sentences with no date reference (negative examples)
NO_DATE_SENTENCES = [
    "What's the weather in Miami?",
    "What's the weather?",
    "Turn on the lights",
    "Turn off the TV",
    "Play some music",
    "Play my favorite playlist",
    "What time is it?",
    "How are you?",
    "Tell me a joke",
    "What's the capital of France?",
    "Set the thermostat to 72",
    "Call John",
    "Call mom",
    "Text Sarah hello",
    "Add milk to my shopping list",
    "Add eggs to the list",
    "Who won the Super Bowl?",
    "What's 2 plus 2?",
    "Define serendipity",
    "How do you spell elephant?",
    "What's the population of Tokyo?",
    "Convert 100 dollars to euros",
    "How tall is Mount Everest?",
    "Who wrote Hamlet?",
    "What's the speed of light?",
    "Open the garage",
    "Lock the front door",
    "Start the coffee maker",
    "Set an alarm",
    "Cancel the alarm",
    "What's my name?",
    "Tell me about yourself",
    "What can you do?",
    "Help me with something",
    "I need assistance",
    "Show me my photos",
    "Play the news",
    "What's trending?",
    "Read my notifications",
    "Check my email",
    "Any new messages?",
    "What's on TV?",
    "Find nearby restaurants",
    "Navigate to the grocery store",
    "How far is the airport?",
    "Order more paper towels",
    "Reorder my vitamins",
]


# ============================================================================
# TRAINING DATA GENERATION
# ============================================================================

def create_example(text: str, keys: list[str]) -> dict:
    """Create a training example."""
    return {"text": text, "date_keys": keys}


def generate_combined_key_examples() -> list[dict]:
    """Generate examples for standalone combined keys."""
    examples = []
    
    for key, variants in COMBINED_KEYS.items():
        for variant in variants:
            # Bare phrase
            examples.append(create_example(variant, [key]))
            
            # Title case
            examples.append(create_example(variant.title(), [key]))
            
            # Upper case (some)
            if random.random() < 0.3:
                examples.append(create_example(variant.upper(), [key]))
            
            # Embedded in sentences (sample 3 templates)
            for template in random.sample(VOICE_COMMAND_TEMPLATES, min(3, len(VOICE_COMMAND_TEMPLATES))):
                sentence = template.format(date=variant)
                examples.append(create_example(sentence, [key]))
    
    return examples


def generate_relative_day_examples() -> list[dict]:
    """Generate examples for relative days."""
    examples = []
    
    for key, variants in RELATIVE_DAYS.items():
        for variant in variants:
            # Bare phrase
            examples.append(create_example(variant, [key]))
            examples.append(create_example(variant.title(), [key]))
            
            # Embedded in sentences
            for template in random.sample(VOICE_COMMAND_TEMPLATES, min(4, len(VOICE_COMMAND_TEMPLATES))):
                sentence = template.format(date=variant)
                examples.append(create_example(sentence, [key]))
    
    return examples


def generate_time_modifier_examples() -> list[dict]:
    """Generate examples for time of day modifiers."""
    examples = []
    
    for key, variants in TIME_MODIFIERS.items():
        for variant in variants:
            # Bare phrase
            examples.append(create_example(variant, [key]))
    
    return examples


def generate_meal_time_examples() -> list[dict]:
    """Generate examples for meal times."""
    examples = []
    
    for key, variants in MEAL_TIMES.items():
        for variant in variants:
            # Bare phrase
            examples.append(create_example(variant, [key]))
            
            # Embedded in sentences
            templates = [
                "Let's discuss this {date}",
                "Remind me {date} to take my meds",
                "Call mom {date}",
                "We should talk {date}",
                "Meet me {date}",
            ]
            for template in templates:
                sentence = template.format(date=variant)
                examples.append(create_example(sentence, [key]))
    
    return examples


def generate_weekday_examples() -> list[dict]:
    """Generate examples for weekdays with modifiers."""
    examples = []
    
    for weekday in WEEKDAYS:
        for abbrev in WEEKDAY_ABBREVS[weekday]:
            # next_X
            key = f"next_{weekday}"
            for prefix in ["next ", "the following "]:
                variant = f"{prefix}{abbrev}"
                examples.append(create_example(variant, [key]))
                examples.append(create_example(variant.title(), [key]))
                
                # Embedded
                for template in random.sample(VOICE_COMMAND_TEMPLATES, 2):
                    sentence = template.format(date=variant)
                    examples.append(create_example(sentence, [key]))
            
            # last_X
            key = f"last_{weekday}"
            for prefix in ["last ", "the previous "]:
                variant = f"{prefix}{abbrev}"
                examples.append(create_example(variant, [key]))
                examples.append(create_example(variant.title(), [key]))
                
                # Embedded
                for template in random.sample(VOICE_COMMAND_TEMPLATES, 2):
                    sentence = template.format(date=variant)
                    examples.append(create_example(sentence, [key]))
            
            # this_X
            key = f"this_{weekday}"
            for prefix in ["this ", ""]:  # bare weekday -> this_X
                variant = f"{prefix}{abbrev}".strip()
                examples.append(create_example(variant, [key]))
                examples.append(create_example(variant.title(), [key]))
    
    return examples


def generate_period_examples() -> list[dict]:
    """Generate examples for periods (week, weekend, month, year)."""
    examples = []
    
    for key, variants in PERIODS.items():
        for variant in variants:
            # Bare phrase
            examples.append(create_example(variant, [key]))
            examples.append(create_example(variant.title(), [key]))
            
            # Embedded
            for template in random.sample(VOICE_COMMAND_TEMPLATES, 3):
                sentence = template.format(date=variant)
                examples.append(create_example(sentence, [key]))
    
    return examples


def generate_specific_time_examples() -> list[dict]:
    """Generate examples for specific times (at_Xam, at_X_30pm, etc.)."""
    examples = []
    
    for hour in range(1, 13):
        for am_pm in ["am", "pm"]:
            # Hourly
            key = f"at_{hour}{am_pm}"
            for variant in generate_time_variants(hour, am_pm):
                examples.append(create_example(variant, [key]))
            
            # Half past
            key = f"at_{hour}_30{am_pm}"
            for variant in generate_time_30_variants(hour, am_pm):
                examples.append(create_example(variant, [key]))
            
            # Quarter past (only some hours to limit data size)
            if hour in [9, 10, 11, 1, 2, 3]:
                key = f"at_{hour}_15{am_pm}"
                for variant in generate_time_15_variants(hour, am_pm):
                    examples.append(create_example(variant, [key]))
            
            # Quarter to (only some hours)
            if hour in [9, 10, 11, 1, 2, 3]:
                key = f"at_{hour}_45{am_pm}"
                for variant in generate_time_45_variants(hour, am_pm):
                    examples.append(create_example(variant, [key]))
    
    # Special cases: noon and midnight are mapped to those keys, not at_12pm/at_12am
    examples.append(create_example("at 12pm", ["noon"]))
    examples.append(create_example("at 12:00pm", ["noon"]))
    examples.append(create_example("at 12 pm", ["noon"]))
    examples.append(create_example("at 12am", ["midnight"]))
    examples.append(create_example("at 12:00am", ["midnight"]))
    examples.append(create_example("at 12 am", ["midnight"]))
    
    return examples


def generate_relative_time_examples() -> list[dict]:
    """Generate examples for relative time expressions (in N minutes/days)."""
    examples = []

    # Sample tasks for template filling
    tasks = [
        "check the oven", "call mom", "take out the trash",
        "move the car", "pick up the kids", "send the email",
        "water the plants", "take my meds", "feed the dog",
    ]

    # Relative minutes
    for key, variants in RELATIVE_MINUTES.items():
        for variant in variants:
            # Bare phrase
            examples.append(create_example(variant, [key]))
            examples.append(create_example(variant.title(), [key]))

            # Embedded in templates
            for template in RELATIVE_TIME_TEMPLATES:
                if "{task}" in template:
                    task = random.choice(tasks)
                    sentence = template.format(time=variant, task=task)
                else:
                    sentence = template.format(time=variant)
                examples.append(create_example(sentence, [key]))

    # Relative days
    for key, variants in RELATIVE_DAYS_OFFSET.items():
        for variant in variants:
            # Bare phrase
            examples.append(create_example(variant, [key]))
            examples.append(create_example(variant.title(), [key]))

            # Embedded in templates
            for template in RELATIVE_TIME_TEMPLATES:
                if "{task}" in template:
                    task = random.choice(tasks)
                    sentence = template.format(time=variant, task=task)
                else:
                    sentence = template.format(time=variant)
                examples.append(create_example(sentence, [key]))

    # Negative examples (durations, past references)
    for sentence in RELATIVE_TIME_NEGATIVES:
        # Most negatives have no date key, but some may have a time-of-day
        if "this morning" in sentence.lower():
            examples.append(create_example(sentence, ["morning"]))
        else:
            examples.append(create_example(sentence, []))

    # Ambiguous examples (should return [])
    for sentence in AMBIGUOUS_TIME_SENTENCES:
        examples.append(create_example(sentence, []))

    return examples


def generate_composite_examples() -> list[dict]:
    """Generate examples with weekday + time modifier (decomposed)."""
    examples = []
    
    # Weekday + time modifier
    for prefix in ["next", "this", "last"]:
        for weekday in random.sample(WEEKDAYS, 4):  # Sample to limit size
            for time_mod, time_variants in TIME_MODIFIERS.items():
                if time_mod in ["noon", "midnight"]:
                    continue  # These are less common with weekdays
                
                key_weekday = f"{prefix}_{weekday}"
                keys = [key_weekday, time_mod]
                
                for time_variant in time_variants[:2]:  # Limit variants
                    weekday_name = random.choice(WEEKDAY_ABBREVS[weekday])
                    variant = f"{prefix} {weekday_name} {time_variant}"
                    examples.append(create_example(variant, keys))
                    
                    # Embedded
                    template = random.choice(VOICE_COMMAND_TEMPLATES)
                    sentence = template.format(date=variant)
                    examples.append(create_example(sentence, keys))
    
    # Weekday + specific time
    for prefix in ["next", "this"]:
        for weekday in random.sample(WEEKDAYS, 3):
            for hour in [9, 3, 10]:
                for am_pm in ["am", "pm"]:
                    key_weekday = f"{prefix}_{weekday}"
                    key_time = f"at_{hour}{am_pm}"
                    keys = [key_weekday, key_time]
                    
                    weekday_name = random.choice(WEEKDAY_ABBREVS[weekday])
                    variant = f"{prefix} {weekday_name} at {hour}{am_pm}"
                    examples.append(create_example(variant, keys))
                    
                    template = random.choice(VOICE_COMMAND_TEMPLATES)
                    sentence = template.format(date=variant)
                    examples.append(create_example(sentence, keys))
    
    return examples


def generate_range_examples() -> list[dict]:
    """Generate examples with date ranges (multiple keys)."""
    examples = []
    
    # Monday to Friday
    examples.append(create_example(
        "from Monday to Friday", 
        ["this_monday", "this_friday"]
    ))
    examples.append(create_example(
        "Monday through Friday",
        ["this_monday", "this_friday"]
    ))
    examples.append(create_example(
        "between tomorrow and next Friday",
        ["tomorrow", "next_friday"]
    ))
    examples.append(create_example(
        "from next Monday to next Friday",
        ["next_monday", "next_friday"]
    ))
    examples.append(create_example(
        "starting tomorrow until next week",
        ["tomorrow", "next_week"]
    ))
    
    return examples


def generate_no_date_examples() -> list[dict]:
    """Generate negative examples (no date reference)."""
    examples = []
    
    for sentence in NO_DATE_SENTENCES:
        examples.append(create_example(sentence, []))
    
    return examples


def generate_all_examples() -> list[dict]:
    """Generate all training examples."""
    all_examples = []
    
    print("Generating combined key examples...")
    all_examples.extend(generate_combined_key_examples())
    
    print("Generating relative day examples...")
    all_examples.extend(generate_relative_day_examples())
    
    print("Generating time modifier examples...")
    all_examples.extend(generate_time_modifier_examples())
    
    print("Generating meal time examples...")
    all_examples.extend(generate_meal_time_examples())
    
    print("Generating weekday examples...")
    all_examples.extend(generate_weekday_examples())
    
    print("Generating period examples...")
    all_examples.extend(generate_period_examples())
    
    print("Generating specific time examples...")
    all_examples.extend(generate_specific_time_examples())
    
    print("Generating relative time examples...")
    all_examples.extend(generate_relative_time_examples())

    print("Generating composite examples...")
    all_examples.extend(generate_composite_examples())
    
    print("Generating range examples...")
    all_examples.extend(generate_range_examples())
    
    print("Generating no-date examples...")
    all_examples.extend(generate_no_date_examples())
    
    # Shuffle
    random.shuffle(all_examples)
    
    return all_examples


def main():
    """Generate and save training data."""
    random.seed(42)  # Reproducible
    
    print("=" * 60)
    print("JARVIS TRAINING DATA GENERATOR")
    print("=" * 60)
    
    examples = generate_all_examples()
    
    print(f"\nTotal examples: {len(examples)}")
    
    # Count by category
    with_keys = sum(1 for e in examples if e["date_keys"])
    without_keys = sum(1 for e in examples if not e["date_keys"])
    print(f"  With date keys: {with_keys}")
    print(f"  Without date keys (negative): {without_keys}")
    
    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")
    
    print(f"\nSaved to: {OUTPUT_FILE}")
    
    # Show some examples
    print("\n" + "-" * 60)
    print("Sample examples:")
    print("-" * 60)
    for example in random.sample(examples, 10):
        print(f"  {example['text'][:50]:<50} -> {example['date_keys']}")


if __name__ == "__main__":
    main()
