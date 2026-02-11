# Relative Time Extraction in Date-Key Adapter

## Overview

The date-key adapter currently extracts semantic date keys from natural language (`tomorrow`, `morning`, `next_monday`). It does NOT handle relative time expressions (`in 30 minutes`, `in 2 hours`, `in 3 days`). This PRD describes the training data and adapter changes needed to support relative time extraction.

This is a general-purpose feature — any command that accepts datetime parameters benefits from relative time resolution.

## Background

### Current Adapter Behavior

| Input | Output |
|-------|--------|
| "What's the weather tomorrow morning?" | `["tomorrow", "morning"]` |
| "Giants game last Saturday" | `["last_saturday"]` |
| "Remind me in 30 minutes to check the oven" | `[]` (missed!) |

### Why This Matters

Relative time expressions ("in 30 minutes", "in 3 days") need to be resolved to absolute datetimes. Without adapter support, commands must implement their own fallback parsing. This should be handled consistently through the date-key architecture.

## Proposed Changes

### New Dynamic Date Key Patterns

Two patterns only — hours/minutes flatten to minutes, days stay as days:

| Key Pattern | Natural Language Examples |
|------------|-------------------------|
| `in_N_minutes` | "in 5 minutes", "in 2 hours" (→ `in_120_minutes`), "in 1 hour and 30 minutes" (→ `in_90_minutes`) |
| `in_N_days` | "in 3 days", "in a day", "in a week" (→ `in_7_days`) |

**Design decision:** All hour and minute expressions normalize to `in_N_minutes`. This gives the adapter one simple multiplication (hours × 60) instead of managing compound formats. Day expressions stay as `in_N_days` because converting to minutes (× 1440) is unreliable arithmetic for a LoRA adapter and day-level expressions don't carry minute precision.

These are parameterized keys — the adapter must extract the numeric values and embed them in the key string.

### Training Data

Add relative time examples to the date-key adapter training set. The adapter should learn to:
1. Detect relative time expressions
2. Extract the numeric offset
3. Normalize hours to minutes (hours × 60)
4. Format as a standardized key

#### New Training Examples

```json
[
  {"input": "Remind me in 5 minutes to check the oven", "output": ["in_5_minutes"]},
  {"input": "Remind me in 30 minutes to take out the trash", "output": ["in_30_minutes"]},
  {"input": "Set a reminder in 2 hours", "output": ["in_120_minutes"]},
  {"input": "Remind me in 1 hour and 30 minutes to call mom", "output": ["in_90_minutes"]},
  {"input": "In an hour remind me to move the car", "output": ["in_60_minutes"]},
  {"input": "Remind me in half an hour to flip the chicken", "output": ["in_30_minutes"]},
  {"input": "In 15 minutes remind me to check email", "output": ["in_15_minutes"]},
  {"input": "Set a reminder in 45 minutes", "output": ["in_45_minutes"]},
  {"input": "Remind me in 3 hours to pick up the kids", "output": ["in_180_minutes"]},
  {"input": "In 10 minutes remind me to turn off the stove", "output": ["in_10_minutes"]},
  {"input": "Remind me in 2 hours and 30 minutes", "output": ["in_150_minutes"]},
  {"input": "In about an hour remind me to call the dentist", "output": ["in_60_minutes"]},
  {"input": "Remind me in 90 minutes to check the laundry", "output": ["in_90_minutes"]},
  {"input": "In 20 minutes set a reminder for the meeting", "output": ["in_20_minutes"]},
  {"input": "Remind me in a minute to grab the mail", "output": ["in_1_minutes"]},
  {"input": "In a quarter of an hour remind me to check on dinner", "output": ["in_15_minutes"]},
  {"input": "Remind me in a couple hours to water the plants", "output": ["in_120_minutes"]},
  {"input": "Set a reminder in an hour and 15 minutes", "output": ["in_75_minutes"]},
  {"input": "In 4 hours remind me to take my medicine", "output": ["in_240_minutes"]},
  {"input": "Remind me in an hour and 45 minutes to leave", "output": ["in_105_minutes"]},
  {"input": "Remind me in 3 days to renew the subscription", "output": ["in_3_days"]},
  {"input": "In a day remind me to follow up on the email", "output": ["in_1_days"]},
  {"input": "Set a reminder in 2 days to call the mechanic", "output": ["in_2_days"]},
  {"input": "Remind me in a week to check the garden", "output": ["in_7_days"]},
  {"input": "In 5 days remind me about the appointment", "output": ["in_5_days"]},
  {"input": "Remind me in a couple days to clean the gutters", "output": ["in_2_days"]},
  {"input": "In 2 weeks remind me to schedule a checkup", "output": ["in_14_days"]}
]
```

#### Negative Examples (no relative time)

Ensure the adapter doesn't false-positive on non-relative expressions:

```json
[
  {"input": "What happened 30 minutes ago?", "output": []},
  {"input": "The meeting lasted 2 hours", "output": []},
  {"input": "I ran for 45 minutes this morning", "output": ["morning"]},
  {"input": "Set a timer for 5 minutes", "output": []},
  {"input": "It takes about 30 minutes to get there", "output": []},
  {"input": "I moved here 3 days ago", "output": []},
  {"input": "The trip took 2 days", "output": []},
  {"input": "I waited for 20 minutes", "output": []},
  {"input": "The recipe takes 2 hours to bake", "output": []},
  {"input": "She left 3 hours ago", "output": []},
  {"input": "It was 5 minutes late", "output": []},
  {"input": "The drive is about 40 minutes", "output": []},
  {"input": "We talked for an hour", "output": []},
  {"input": "That was like 10 minutes ago", "output": []}
]
```

#### Ambiguous Examples (validation error)

These contain relative time language but no extractable number. The adapter should output `[]`, and the command center should detect the unresolved time expression and return a validation error with a clarifying question.

```json
[
  {"input": "Remind me in a few minutes to check the oven", "output": []},
  {"input": "In a little while remind me to call mom", "output": []},
  {"input": "Remind me in a bit to take out the trash", "output": []},
  {"input": "Set a reminder in some time", "output": []},
  {"input": "Remind me later to grab the groceries", "output": []}
]
```

**Expected command-center behavior:** When the adapter returns `[]` but the input contains scheduling intent (e.g., "remind me"), the command center should return a validation error with a clarifying question like "How many minutes from now should I remind you?"

#### Mixed Examples (relative + semantic)

```json
[
  {"input": "Remind me in 30 minutes and also tomorrow morning", "output": ["in_30_minutes", "tomorrow_morning"]},
  {"input": "What's the weather tomorrow and remind me in an hour", "output": ["tomorrow", "in_60_minutes"]},
  {"input": "Remind me in 3 days and also next Monday", "output": ["in_3_days", "next_monday"]}
]
```

### Key Format Rules

1. Always use underscore-separated format: `in_N_unit`
2. **All hour/minute expressions flatten to minutes:**
   - "an hour" / "one hour" → `in_60_minutes`
   - "2 hours" → `in_120_minutes`
   - "hour and a half" → `in_90_minutes`
   - "2 hours and 15 minutes" → `in_135_minutes`
3. **Day expressions stay as days:**
   - "a day" → `in_1_days`
   - "a week" → `in_7_days`
   - "2 weeks" → `in_14_days`
4. Normalize "half an hour" → `in_30_minutes`
5. Normalize "quarter of an hour" → `in_15_minutes`
6. Normalize "a couple hours" → `in_120_minutes`, "a couple days" → `in_2_days`
7. Round to whole minutes (no seconds in keys)
8. Use plural unit names always: `minutes`, `days` (even for N=1, for consistency)
9. **Ambiguous expressions** ("a few minutes", "a little while", "later") → `[]` (no key extracted; command center handles with validation error)

### API Response Changes

The `/v1/adapters/date-keys` endpoint should return dynamic patterns alongside static keys:

```json
{
  "version": "2.0",
  "static_keys": [
    "today", "tomorrow", "yesterday", "morning", "tonight",
    "next_monday", "next_tuesday", ...
  ],
  "dynamic_patterns": [
    {
      "pattern": "in_{N}_minutes",
      "regex": "^in_(\\d+)_minutes$",
      "description": "Relative offset in minutes from current time"
    },
    {
      "pattern": "in_{N}_days",
      "regex": "^in_(\\d+)_days$",
      "description": "Relative offset in days from current time"
    }
  ]
}
```

This allows `sync_date_keys.py` on the node to understand the format without enumerating every possible value.

**Migration note:** This is a breaking change from the current flat-list response. Consumers (`sync_date_keys.py`) must be updated simultaneously to handle the versioned format.

## Adapter Training Considerations

### Challenge: Numeric Extraction

Current date keys are fixed strings (`tomorrow`, `next_monday`). Relative time keys contain extracted numbers (`in_30_minutes`). This requires the adapter to:
1. Recognize the pattern ("in X minutes/hours" = scheduling, "for X minutes" = duration)
2. Extract the number from natural language
3. Multiply hours × 60 when normalizing to minutes
4. Format it into the key

This is a harder task than simple classification. Consider:
- **More training examples** for numeric extraction (at least 50 relative time examples)
- **Validation** that extracted numbers are reasonable (1-1440 minutes, 1-365 days)
- **Normalization** rules ("quarter of an hour" → 15, "half hour" → 30, "a week" → 7 days)
- **"in" vs "for" distinction** — critical to avoid false positives on duration expressions. Include ample "for X minutes/hours" negatives.

### Normalization Table

| Natural Language | Normalized Key |
|-----------------|----------------|
| "5 minutes" | `in_5_minutes` |
| "half an hour" | `in_30_minutes` |
| "quarter of an hour" | `in_15_minutes` |
| "an hour" | `in_60_minutes` |
| "a couple hours" | `in_120_minutes` |
| "hour and a half" | `in_90_minutes` |
| "2 hours and 15 minutes" | `in_135_minutes` |
| "90 minutes" | `in_90_minutes` |
| "a day" | `in_1_days` |
| "a couple days" | `in_2_days` |
| "a week" | `in_7_days` |
| "2 weeks" | `in_14_days` |
| "a few minutes" | `[]` (ambiguous — validation error) |
| "a little while" | `[]` (ambiguous — validation error) |
| "later" | `[]` (ambiguous — validation error) |

### Evaluation Criteria

Test the adapter with held-out examples:
- Exact numeric match (e.g., "in 30 minutes" → `in_30_minutes`, not `in_29_minutes`)
- Hour-to-minute conversion (e.g., "in 2 hours" → `in_120_minutes`, not `in_2_hours`)
- Compound conversion (e.g., "in 1 hour and 45 minutes" → `in_105_minutes`)
- Edge cases: "in a minute" → `in_1_minutes`
- Days: "in 3 days" → `in_3_days`
- Ambiguous: "in a few minutes" → `[]`
- False negative rate < 5% on relative time expressions
- False positive rate < 1% on non-relative expressions (durations, past references)

## Testing

### Unit Tests for Training Data

```
1. test_relative_time_extraction_minutes
2. test_relative_time_hours_flatten_to_minutes
3. test_relative_time_compound_flatten_to_minutes
4. test_relative_time_extraction_days
5. test_relative_time_normalization (half hour, quarter hour, a couple hours, a week, etc.)
6. test_ambiguous_expressions_return_empty (a few minutes, a little while, later)
7. test_no_false_positive_on_past_references ("30 minutes ago")
8. test_no_false_positive_on_durations ("lasted 2 hours", "for 45 minutes")
9. test_no_false_positive_on_past_days ("3 days ago")
10. test_mixed_relative_and_semantic_keys
```

### Integration Test

```
1. test_full_pipeline_relative_time
   - Send "remind me in 30 minutes to check the oven" through full adapter pipeline
   - Assert date_keys contains "in_30_minutes"
   - Assert command-center resolves to now + 30 minutes
2. test_full_pipeline_hours_to_minutes
   - Send "remind me in 2 hours to pick up the kids" through full adapter pipeline
   - Assert date_keys contains "in_120_minutes"
   - Assert command-center resolves to now + 120 minutes
3. test_full_pipeline_relative_days
   - Send "remind me in 3 days to renew the subscription" through full adapter pipeline
   - Assert date_keys contains "in_3_days"
   - Assert command-center resolves to now + 3 days
4. test_full_pipeline_ambiguous_validation_error
   - Send "remind me in a few minutes to check the oven" through full adapter pipeline
   - Assert date_keys is empty
   - Assert command-center returns validation error with clarifying question
```

## Dependencies

- `jarvis-command-center/prds/relative-time-resolution.md` — resolution of relative time keys + validation error flow for ambiguous expressions
- `jarvis-node-setup/prds/reminders-feature.md` — the feature consuming these keys

## Migration

Once relative time keys are supported end-to-end, remove the `relative_minutes` fallback parameter from `set_reminder` entirely — no backwards compatibility needed.
