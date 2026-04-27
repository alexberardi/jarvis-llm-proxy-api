#!/usr/bin/env python3
"""
Expand date key training data from ~2K to 5K+ examples.

Adds:
- More negative examples (songs, idioms, durations, general commands)
- More composite expressions (weekday + time, date + specific time)
- Edge cases and adversarial examples
- More natural language variety
- Rare phrasings and colloquialisms

Appends to existing data/jarvis_training.jsonl.
"""

import json
import random
from pathlib import Path

OUTPUT = Path(__file__).parent.parent / "data" / "jarvis_training.jsonl"

random.seed(43)  # Different seed from original generator


def ex(text: str, keys: list[str]) -> dict:
    return {"text": text, "date_keys": keys}


# ============================================================================
# NEGATIVE EXAMPLES (no date reference)
# ============================================================================

# Songs/movies/books with date-related words
MEDIA_NEGATIVES = [
    ex("Play Yesterday by the Beatles", []),
    ex("Play Tomorrow Never Dies", []),
    ex("Play Last Friday Night by Katy Perry", []),
    ex("Play Sunday Morning by Maroon 5", []),
    ex("Play Monday by Imagine Dragons", []),
    ex("Play Friday by Rebecca Black", []),
    ex("Play Saturday Night Fever soundtrack", []),
    ex("Play The Day After Tomorrow soundtrack", []),
    ex("Play It Was a Good Day by Ice Cube", []),
    ex("Play One More Night by Phil Collins", []),
    ex("Play Midnight Train to Georgia", []),
    ex("Play Saturday in the Park by Chicago", []),
    ex("Play Tuesday's Gone by Lynyrd Skynyrd", []),
    ex("Play Happy Days theme song", []),
    ex("Play Morning Has Broken", []),
    ex("Play Nights in White Satin", []),
    ex("Play Last Christmas by Wham", []),
    ex("Play Summer Nights from Grease", []),
    ex("Play Afternoon Delight", []),
    ex("Play Manic Monday by the Bangles", []),
    ex("Watch The Morning Show", []),
    ex("Watch Last Week Tonight", []),
    ex("Watch Saturday Night Live", []),
    ex("Watch Good Morning America", []),
    ex("Watch Friday Night Lights", []),
]

# Duration/past references (should NOT be date keys)
DURATION_NEGATIVES = [
    ex("The movie was 2 hours long", []),
    ex("I slept for 8 hours", []),
    ex("The run took 45 minutes", []),
    ex("She left 3 hours ago", []),
    ex("That was like 10 minutes ago", []),
    ex("We talked for an hour", []),
    ex("The drive is about 40 minutes", []),
    ex("It happened 5 minutes ago", []),
    ex("The recipe takes 2 hours to bake", []),
    ex("I moved here 3 months ago", []),
    ex("We've been waiting for 20 minutes", []),
    ex("The flight is 6 hours", []),
    ex("He ran for 30 minutes straight", []),
    ex("The battery lasts about 10 hours", []),
    ex("It downloads in about 5 minutes", []),
    ex("The commute is 45 minutes each way", []),
    ex("I studied for 3 hours", []),
    ex("The concert lasted 2 hours", []),
    ex("She's been gone for a week", []),
    ex("I haven't seen him in months", []),
    ex("It's been years since I visited", []),
    ex("The warranty expires in 90 days from purchase", []),
    ex("Processing takes 3 to 5 business days", []),
    ex("Allow 24 hours for the paint to dry", []),
    ex("Bake for 350 degrees for 45 minutes", []),
]

# Timer commands (NOT relative time)
TIMER_NEGATIVES = [
    ex("Set a timer for 5 minutes", []),
    ex("Set a timer for 10 minutes", []),
    ex("Set a timer for 30 minutes", []),
    ex("Set a timer for 1 hour", []),
    ex("Set a timer for 2 hours", []),
    ex("Start a 15 minute timer", []),
    ex("Start a 30 minute timer", []),
    ex("Start a 5 minute countdown", []),
    ex("Timer 10 minutes", []),
    ex("Timer for 45 minutes", []),
]

# Ambiguous/vague time references
AMBIGUOUS_NEGATIVES = [
    ex("Remind me later", []),
    ex("I'll do it later", []),
    ex("Maybe some other time", []),
    ex("In a little while", []),
    ex("In a few minutes", []),
    ex("In a bit", []),
    ex("Sometime soon", []),
    ex("When you get a chance", []),
    ex("Whenever you're free", []),
    ex("Not right now", []),
    ex("Eventually", []),
    ex("At some point", []),
    ex("One of these days", []),
    ex("Sooner or later", []),
    ex("Before long", []),
]

# General commands with NO date
GENERAL_NEGATIVES = [
    ex("What's the weather in New York?", []),
    ex("Turn on the living room lights", []),
    ex("Turn off all the lights", []),
    ex("Set the thermostat to 68 degrees", []),
    ex("Play jazz music", []),
    ex("Play my morning playlist", []),
    ex("What's the news?", []),
    ex("Read my notifications", []),
    ex("How's the traffic?", []),
    ex("What's the stock price of Apple?", []),
    ex("Order more paper towels", []),
    ex("Add bananas to my grocery list", []),
    ex("What's 15 times 23?", []),
    ex("Convert 100 pounds to kilograms", []),
    ex("Translate hello to Spanish", []),
    ex("Who is the president?", []),
    ex("How old is the Earth?", []),
    ex("What's the tallest building in the world?", []),
    ex("Play white noise", []),
    ex("Set the volume to 50 percent", []),
    ex("Skip this song", []),
    ex("Pause the music", []),
    ex("Resume playback", []),
    ex("Lock all doors", []),
    ex("Close the garage door", []),
    ex("Is the front door locked?", []),
    ex("What's my name?", []),
    ex("Tell me a fun fact", []),
    ex("What's the recipe for banana bread?", []),
    ex("How many cups in a gallon?", []),
    ex("Find nearby gas stations", []),
    ex("Navigate home", []),
    ex("Call mom", []),
    ex("Text John I'm running late", []),
    ex("Send an email to the team", []),
    ex("Check my bank balance", []),
    ex("What's my step count?", []),
    ex("Start a workout", []),
    ex("How many calories in an apple?", []),
    ex("What's the WiFi password?", []),
    ex("Restart the router", []),
]

# Idioms with time words
IDIOM_NEGATIVES = [
    ex("It's a brand new day", []),
    ex("Call it a day", []),
    ex("Day in and day out", []),
    ex("At the end of the day", []),
    ex("It's not my day", []),
    ex("Night and day difference", []),
    ex("Morning person or night owl?", []),
    ex("The weekend warrior", []),
    ex("That ship has sailed", []),
    ex("Time flies", []),
    ex("Better late than never", []),
    ex("It's about time", []),
    ex("Once upon a time", []),
    ex("A matter of time", []),
    ex("In the nick of time", []),
    ex("Time is money", []),
    ex("Third time's the charm", []),
    ex("A day late and a dollar short", []),
    ex("Every dog has its day", []),
    ex("Save it for a rainy day", []),
]


# ============================================================================
# MORE COMPOSITE EXAMPLES
# ============================================================================

def generate_composites() -> list[dict]:
    """Generate diverse composite (multi-key) examples."""
    examples = []

    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    abbrevs = {"monday": "mon", "tuesday": "tue", "wednesday": "wed",
               "thursday": "thu", "friday": "fri", "saturday": "sat", "sunday": "sun"}
    times = ["morning", "afternoon", "evening", "night"]
    specific_times = [
        ("at 9am", "at_9am"), ("at 10am", "at_10am"), ("at 11am", "at_11am"),
        ("at noon", "noon"), ("at 1pm", "at_1pm"), ("at 2pm", "at_2pm"),
        ("at 3pm", "at_3pm"), ("at 4pm", "at_4pm"), ("at 5pm", "at_5pm"),
        ("at 6pm", "at_6pm"), ("at 7pm", "at_7pm"), ("at 8pm", "at_8pm"),
        ("at 9pm", "at_9pm"), ("at 7:30am", "at_7_30am"), ("at 8:30am", "at_8_30am"),
        ("at 2:30pm", "at_2_30pm"), ("at 5:30pm", "at_5_30pm"),
    ]

    templates = [
        "What's my schedule {date}?",
        "Remind me {date} to call the doctor",
        "Set a meeting {date}",
        "Are we free {date}?",
        "Book a table {date}",
        "Any appointments {date}?",
        "Schedule a workout {date}",
        "Plan a call {date}",
        "What's happening {date}?",
        "Do I have anything {date}?",
    ]

    # Next/this weekday + time modifier
    for prefix in ["next", "this"]:
        for day in random.sample(weekdays, 5):
            for time_mod in times:
                key_day = f"{prefix}_{day}"
                text = f"{prefix} {abbrevs[day]} {time_mod}"
                examples.append(ex(text, [key_day, time_mod]))

                template = random.choice(templates)
                sentence = template.format(date=f"{prefix} {day} {time_mod}")
                examples.append(ex(sentence, [key_day, time_mod]))

    # Next/this weekday + specific time
    for prefix in ["next", "this"]:
        for day in random.sample(weekdays, 4):
            for time_text, time_key in random.sample(specific_times, 3):
                key_day = f"{prefix}_{day}"
                text = f"{prefix} {day} {time_text}"
                examples.append(ex(text, [key_day, time_key]))

                template = random.choice(templates)
                sentence = template.format(date=text)
                examples.append(ex(sentence, [key_day, time_key]))

    # Tomorrow/today + specific time
    for day_word, day_key in [("tomorrow", "tomorrow"), ("today", "today")]:
        for time_text, time_key in random.sample(specific_times, 5):
            examples.append(ex(f"{day_word} {time_text}", [day_key, time_key]))
            template = random.choice(templates)
            examples.append(ex(template.format(date=f"{day_word} {time_text}"), [day_key, time_key]))

    # Last weekday + time modifier (past composites)
    for day in random.sample(weekdays, 4):
        for time_mod in random.sample(times, 2):
            key_day = f"last_{day}"
            text = f"last {abbrevs[day]} {time_mod}"
            examples.append(ex(text, [key_day, time_mod]))
            examples.append(ex(f"What happened last {day} {time_mod}?", [key_day, time_mod]))

    return examples


# ============================================================================
# MORE RELATIVE DAY VARIETIES
# ============================================================================

RELATIVE_DAY_EXTRAS = [
    # More "day after tomorrow" phrasings
    ex("What's the forecast for the day after tomorrow?", ["day_after_tomorrow"]),
    ex("Schedule it for the day after tomorrow", ["day_after_tomorrow"]),
    ex("I need it done by the day after tomorrow", ["day_after_tomorrow"]),
    ex("Can we do it the day after tomorrow?", ["day_after_tomorrow"]),
    ex("The day after tomorrow works for me", ["day_after_tomorrow"]),
    ex("Let's plan for the day after tomorrow", ["day_after_tomorrow"]),
    # More "day before yesterday" phrasings
    ex("What happened the day before yesterday?", ["day_before_yesterday"]),
    ex("I saw it the day before yesterday", ["day_before_yesterday"]),
    ex("The day before yesterday was nice", ["day_before_yesterday"]),
    # More today/tomorrow/yesterday variety
    ex("Is there anything going on today?", ["today"]),
    ex("What am I doing today?", ["today"]),
    ex("Clear my schedule for today", ["today"]),
    ex("Today's a good day for a walk", ["today"]),
    ex("How's the market doing today?", ["today"]),
    ex("What happened at work today?", ["today"]),
    ex("Tomorrow's gonna be busy", ["tomorrow"]),
    ex("I'll handle it tomorrow", ["tomorrow"]),
    ex("Push it to tomorrow", ["tomorrow"]),
    ex("Can it wait until tomorrow?", ["tomorrow"]),
    ex("What did I miss yesterday?", ["yesterday"]),
    ex("How was the game yesterday?", ["yesterday"]),
    ex("I forgot to do it yesterday", ["yesterday"]),
    ex("Yesterday was pretty crazy", ["yesterday"]),
]


# ============================================================================
# MORE PERIOD AND MEAL EXAMPLES
# ============================================================================

PERIOD_EXTRAS = [
    ex("What's the plan for this week?", ["this_week"]),
    ex("How's next week looking?", ["next_week"]),
    ex("Anything interesting last week?", ["last_week"]),
    ex("This month's been hectic", ["this_month"]),
    ex("I'll be traveling next month", ["next_month"]),
    ex("Last month was expensive", ["last_month"]),
    ex("What are we doing this weekend?", ["this_weekend"]),
    ex("Got any plans this weekend?", ["this_weekend"]),
    ex("Let's do something fun this weekend", ["this_weekend"]),
    ex("Next weekend I'm free", ["next_weekend"]),
    ex("Last weekend we went hiking", ["last_weekend"]),
    ex("How was your weekend?", ["this_weekend"]),
    ex("Any plans for the weekend?", ["this_weekend"]),
    ex("The weekend forecast looks good", ["this_weekend"]),
]

MEAL_EXTRAS = [
    ex("Let's chat over breakfast tomorrow", ["tomorrow_morning"]),
    ex("Can we discuss it at lunch?", ["during_lunch"]),
    ex("Dinner plans tonight?", ["tonight"]),
    ex("I'll call you after dinner", ["after_dinner"]),
    ex("Meet me for breakfast", ["at_breakfast"]),
    ex("We'll talk about it over lunch", ["during_lunch"]),
    ex("Save it for dinner conversation", ["at_dinner"]),
    ex("Let's grab lunch and discuss", ["during_lunch"]),
    ex("Breakfast meeting at the usual place", ["at_breakfast"]),
    ex("Post-dinner walk?", ["after_dinner"]),
]


# ============================================================================
# EDGE CASES AND RARE PHRASINGS
# ============================================================================

EDGE_CASES = [
    # Mixed case, typos, text-speak
    ex("2MORROW", ["tomorrow"]),
    ex("NEXT MONDAY AT 3PM", ["next_monday", "at_3pm"]),
    ex("tmrw morn", ["tomorrow_morning"]),
    ex("ystrdy evening", ["yesterday_evening"]),
    ex("this fri aft", ["this_friday", "afternoon"]),
    ex("nxt wed", ["next_wednesday"]),
    ex("lst sat", ["last_saturday"]),

    # Very short utterances
    ex("today", ["today"]),
    ex("tomorrow", ["tomorrow"]),
    ex("yesterday", ["yesterday"]),
    ex("tonight", ["tonight"]),
    ex("noon", ["noon"]),
    ex("midnight", ["midnight"]),
    ex("this weekend", ["this_weekend"]),
    ex("next week", ["next_week"]),
    ex("next monday", ["next_monday"]),

    # Very long utterances with dates buried inside
    ex("Hey can you check if I have any meetings or appointments scheduled for tomorrow morning because I think I might have a dentist visit", ["tomorrow_morning"]),
    ex("I was thinking about maybe going to the gym next tuesday if the weather is nice enough to walk there", ["next_tuesday"]),
    ex("My mom called me yesterday and she wants to know if we can have dinner this weekend", ["yesterday", "this_weekend"]),
    ex("So I need to finish the report by the end of this week and also schedule a meeting for next monday morning", ["this_week", "next_monday", "morning"]),

    # Relative offsets with context
    ex("Remind me in exactly 30 minutes to check the oven", ["in_30_minutes"]),
    ex("Can you set an alarm for in 2 hours?", ["in_120_minutes"]),
    ex("I need a reminder in about an hour", ["in_60_minutes"]),
    ex("Wake me up in 45 minutes", ["in_45_minutes"]),
    ex("Ping me in 15 minutes", ["in_15_minutes"]),
    ex("Alert me in 10 minutes", ["in_10_minutes"]),
    ex("Buzz me in 5 minutes", ["in_5_minutes"]),
]


# ============================================================================
# MORE VOICE COMMAND TEMPLATES WITH DATES
# ============================================================================

def generate_templated_extras() -> list[dict]:
    """Generate more examples using templates with various date expressions."""
    examples = []

    templates = [
        "What's the weather going to be like {date}?",
        "Will it rain {date}?",
        "How cold will it be {date}?",
        "Do I need an umbrella {date}?",
        "What's the temperature {date}?",
        "Tell me about the forecast for {date}",
        "What games are on {date}?",
        "Any good movies showing {date}?",
        "What's on my agenda {date}?",
        "Am I busy {date}?",
        "Can I take {date} off?",
        "I want to go hiking {date}",
        "Plan a date night for {date}",
        "Order groceries for {date}",
        "Book a flight for {date}",
        "Reserve a restaurant for {date}",
        "Schedule a haircut for {date}",
        "I have a doctor's appointment {date}",
        "Pick up the dry cleaning {date}",
        "Drop off the package {date}",
        "Water the plants {date}",
        "Feed the fish {date}",
        "Walk the dog {date}",
        "Take the car in for service {date}",
        "Pay the rent {date}",
    ]

    date_expressions = [
        ("tomorrow", ["tomorrow"]),
        ("today", ["today"]),
        ("tonight", ["tonight"]),
        ("this morning", ["this_morning"]),
        ("this afternoon", ["this_afternoon"]),
        ("this evening", ["this_evening"]),
        ("tomorrow morning", ["tomorrow_morning"]),
        ("tomorrow afternoon", ["tomorrow_afternoon"]),
        ("tomorrow evening", ["tomorrow_evening"]),
        ("next monday", ["next_monday"]),
        ("next friday", ["next_friday"]),
        ("this saturday", ["this_saturday"]),
        ("next weekend", ["next_weekend"]),
        ("this week", ["this_week"]),
        ("next week", ["next_week"]),
        ("the day after tomorrow", ["day_after_tomorrow"]),
    ]

    for template in templates:
        for date_text, date_keys in random.sample(date_expressions, min(4, len(date_expressions))):
            sentence = template.format(date=date_text)
            examples.append(ex(sentence, date_keys))

    return examples


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Load existing examples
    existing = []
    existing_texts = set()
    with open(OUTPUT) as f:
        for line in f:
            row = json.loads(line.strip())
            existing.append(row)
            existing_texts.add(row["text"].lower().strip())

    print(f"Existing examples: {len(existing)}")

    # Collect new examples
    new_examples = []

    new_examples.extend(MEDIA_NEGATIVES)
    new_examples.extend(DURATION_NEGATIVES)
    new_examples.extend(TIMER_NEGATIVES)
    new_examples.extend(AMBIGUOUS_NEGATIVES)
    new_examples.extend(GENERAL_NEGATIVES)
    new_examples.extend(IDIOM_NEGATIVES)
    new_examples.extend(generate_composites())
    new_examples.extend(RELATIVE_DAY_EXTRAS)
    new_examples.extend(PERIOD_EXTRAS)
    new_examples.extend(MEAL_EXTRAS)
    new_examples.extend(EDGE_CASES)
    new_examples.extend(generate_templated_extras())

    # Deduplicate against existing
    added = 0
    for ex in new_examples:
        key = ex["text"].lower().strip()
        if key not in existing_texts:
            existing.append(ex)
            existing_texts.add(key)
            added += 1

    # Shuffle the new combined set
    random.shuffle(existing)

    print(f"New examples added: {added}")
    print(f"Total examples: {len(existing)}")

    # Count categories
    no_date = sum(1 for e in existing if not e["date_keys"])
    multi = sum(1 for e in existing if len(e["date_keys"]) > 1)
    print(f"  No date (negatives): {no_date} ({no_date/len(existing):.0%})")
    print(f"  Multi-key: {multi} ({multi/len(existing):.0%})")

    # Write
    with open(OUTPUT, "w") as f:
        for row in existing:
            f.write(json.dumps(row) + "\n")

    print(f"\nSaved to: {OUTPUT}")


if __name__ == "__main__":
    main()
