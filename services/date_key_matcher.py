"""
Deterministic regex-based date key extraction.

Matches natural language date/time expressions to standardized date keys
using a priority-ordered pattern library. Designed for voice command text.

Strategy:
1. Longest match first (e.g., "day after tomorrow" beats "tomorrow")
2. Word-boundary aware (won't match inside other words)
3. Handles abbreviations, informal speech, and composite expressions
4. Returns [] for text with no date references

Usage:
    from services.date_key_matcher import extract_date_keys

    keys = extract_date_keys("remind me tomorrow morning at 3pm")
    # → ["tomorrow_morning", "at_3pm"]
"""

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Pattern definitions: date_key → list of surface forms
# Ordered by priority (longer/more specific first within each group)
# ---------------------------------------------------------------------------

# Combined day+time keys (must match BEFORE individual day/time keys)
_COMBINED_PATTERNS: dict[str, list[str]] = {
    "tomorrow_morning": [
        "tomorrow morning", "tmrw morning", "tomorrow am", "tomorrow morn",
        "2morrow morning", "2mrw morning", "tmrw morn",
    ],
    "tomorrow_afternoon": ["tomorrow afternoon", "tmrw afternoon", "2morrow afternoon"],
    "tomorrow_evening": ["tomorrow evening", "tmrw evening", "2morrow evening"],
    "tomorrow_night": ["tomorrow night", "tmrw night", "tomorrow nite", "2morrow night"],
    "yesterday_morning": ["yesterday morning", "yest morning", "yesterday am", "ystrdy morning"],
    "yesterday_afternoon": ["yesterday afternoon", "yest afternoon", "ystrdy afternoon"],
    "yesterday_evening": ["yesterday evening", "yest evening", "ystrdy evening"],
    "tonight": ["tonight", "tonite", "this evening late", "later tonight"],
    "last_night": ["last night", "lastnight", "yesterday night", "yesternight"],
    "this_morning": ["this morning", "earlier today", "earlier this morning"],
    "this_afternoon": ["this afternoon", "later today", "this aft"],
    "this_evening": ["this evening", "early tonight"],
}

# Relative days (longer phrases first to prevent partial matches)
_RELATIVE_DAY_PATTERNS: dict[str, list[str]] = {
    "day_after_tomorrow": [
        "the day after tomorrow", "day after tomorrow", "overmorrow",
        "in two days", "in 2 days",
    ],
    "day_before_yesterday": [
        "the day before yesterday", "day before yesterday",
        "two days ago", "2 days ago",
    ],
    "today": ["today", "2day", "tday"],
    "tomorrow": ["tomorrow", "tmrw", "tmr", "2morrow", "2mrw", "tom", "2mrw"],
    "yesterday": ["yesterday", "yest", "yday", "ystrdy"],
}

# Weekday patterns: this/next/last + weekday
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_WEEKDAY_ABBREVS: dict[str, list[str]] = {
    "monday": ["monday", "mon"],
    "tuesday": ["tuesday", "tues", "tue"],
    "wednesday": ["wednesday", "weds", "wed"],
    "thursday": ["thursday", "thurs", "thur", "thu"],
    "friday": ["friday", "fri"],
    "saturday": ["saturday", "sat"],
    "sunday": ["sunday", "sun"],
}

_WEEKDAY_PATTERNS: dict[str, list[str]] = {}
for _day in _WEEKDAYS:
    for _abbrev in _WEEKDAY_ABBREVS[_day]:
        # "next monday", "the following monday", "nxt monday"
        _WEEKDAY_PATTERNS.setdefault(f"next_{_day}", []).extend([
            f"next {_abbrev}", f"the following {_abbrev}", f"nxt {_abbrev}",
        ])
        # "last monday", "the previous monday", "lst monday"
        _WEEKDAY_PATTERNS.setdefault(f"last_{_day}", []).extend([
            f"last {_abbrev}", f"the previous {_abbrev}", f"lst {_abbrev}",
        ])
        # "this monday" + bare weekday → this_X
        _WEEKDAY_PATTERNS.setdefault(f"this_{_day}", []).extend([
            f"this {_abbrev}",
        ])
# Bare weekday names (lowest priority — only match if no prefix)
for _day in _WEEKDAYS:
    for _abbrev in _WEEKDAY_ABBREVS[_day]:
        _WEEKDAY_PATTERNS.setdefault(f"this_{_day}", []).append(_abbrev)

# Periods
_PERIOD_PATTERNS: dict[str, list[str]] = {
    "this_weekend": ["this weekend", "the weekend", "this sat and sun", "the wknd", "this wknd"],
    "next_weekend": ["next weekend", "the following weekend"],
    "last_weekend": ["last weekend", "the previous weekend"],
    "this_week": ["this week", "the current week"],
    "next_week": ["next week", "the following week", "the week after", "nxt wk", "nxt week"],
    "last_week": ["last week", "the previous week", "the week before"],
    "this_month": ["this month", "the current month"],
    "next_month": ["next month", "the following month"],
    "last_month": ["last month", "the previous month"],
    "this_year": ["this year", "the current year"],
    "next_year": ["next year", "the following year"],
    "last_year": ["last year", "the previous year"],
}

# Meal times
_MEAL_PATTERNS: dict[str, list[str]] = {
    "at_breakfast": ["at breakfast time", "at breakfasttime", "at breakfast", "breakfast time"],
    "during_breakfast": ["during breakfast", "while eating breakfast", "over breakfast"],
    "during_lunch": [
        "during lunch", "at lunchtime", "at lunch time", "at lunch",
        "lunch time", "lunchtime", "over lunch",
    ],
    "at_dinner": ["at dinner time", "at dinnertime", "at dinner", "dinner time"],
    "during_dinner": ["during dinner", "while eating dinner", "over dinner"],
    "after_dinner": [
        "after dinnertime", "after dinner", "post dinner",
        "once dinner is done", "post-dinner",
    ],
}

# Time of day modifiers (lowest priority — only if not already matched as combined)
_TIME_MOD_PATTERNS: dict[str, list[str]] = {
    "morning": ["in the morning", "in the am", "morning", "morn"],
    "afternoon": ["in the afternoon", "afternoon", "aft"],
    "evening": ["in the evening", "evening", "eve"],
    "night": ["at night", "in the night", "night", "nite"],
    "noon": ["at noon", "midday", "12 o'clock", "12 oclock", "noon"],
    "midnight": ["at midnight", "midnight"],
}


# ---------------------------------------------------------------------------
# Dynamic regex patterns (time-specific, relative offsets)
# ---------------------------------------------------------------------------

_DYNAMIC_REGEXES: list[tuple[re.Pattern, object]] = [
    # "in X hours and Y minutes" → in_{X*60+Y}_minutes (MUST be before "in X hours")
    (re.compile(
        r"\bin\s+(\d+)\s+hours?\s+and\s+(\d+)\s+minutes?\b", re.IGNORECASE
    ), lambda m: f"in_{int(m.group(1)) * 60 + int(m.group(2))}_minutes"),

    # Written: "in two hours and thirty minutes", "in one hour and 45 minutes"
    (re.compile(
        r"\bin\s+(one|two|three|four)\s+hours?\s+and\s+(\d+)\s+minutes?\b", re.IGNORECASE
    ), lambda m: "in_%d_minutes" % ({"one":1,"two":2,"three":3,"four":4}[m.group(1).lower()] * 60 + int(m.group(2)))),

    # "in two and a half hours"
    (re.compile(r"\bin\s+two\s+and\s+a\s+half\s+hours?\b", re.IGNORECASE),
     lambda _: "in_150_minutes"),

    # "in an hour and a half" / "in one hour and a half"
    (re.compile(r"\bin\s+(?:an?|one)\s+hour\s+and\s+a\s+half\b", re.IGNORECASE),
     lambda _: "in_90_minutes"),

    # "in an hour and X minutes" (written: must be before "in an hour")
    (re.compile(r"\bin\s+(?:an?|one|1)\s+hour\s+and\s+(\d+)\s+minutes?\b", re.IGNORECASE),
     lambda m: f"in_{60 + int(m.group(1))}_minutes"),

    # "in a couple of hours" / "in a couple hours"
    (re.compile(r"\bin\s+a\s+couple\s+(?:of\s+)?hours?\b", re.IGNORECASE),
     lambda _: "in_120_minutes"),

    # "in half an hour" / "in half hour"
    (re.compile(r"\bin\s+(?:a\s+)?half\s+(?:an?\s+)?hour\b", re.IGNORECASE),
     lambda _: "in_30_minutes"),

    # "in a quarter of an hour"
    (re.compile(r"\bin\s+(?:a\s+)?quarter\s+(?:of\s+)?an?\s+hour\b", re.IGNORECASE),
     lambda _: "in_15_minutes"),

    # "in an hour" / "in one hour" / "in about an hour" / "an hour from now"
    (re.compile(r"\bin\s+(?:about\s+)?(?:an?|one|1)\s+hour\b", re.IGNORECASE),
     lambda _: "in_60_minutes"),
    (re.compile(r"\b(?:an?|one)\s+hour\s+from\s+now\b", re.IGNORECASE),
     lambda _: "in_60_minutes"),

    # "in X hours" / "in about X hours" / "X hours from now" (digits)
    (re.compile(r"\bin\s+(?:about\s+)?(\d+)\s+hours?\b", re.IGNORECASE),
     lambda m: f"in_{int(m.group(1)) * 60}_minutes"),
    (re.compile(r"\b(\d+)\s+hours?\s+from\s+now\b", re.IGNORECASE),
     lambda m: f"in_{int(m.group(1)) * 60}_minutes"),

    # "in two/three/four hours" (written)
    (re.compile(r"\bin\s+(two)\s+hours?\b", re.IGNORECASE), lambda _: "in_120_minutes"),
    (re.compile(r"\bin\s+(three)\s+hours?\b", re.IGNORECASE), lambda _: "in_180_minutes"),
    (re.compile(r"\bin\s+(four)\s+hours?\b", re.IGNORECASE), lambda _: "in_240_minutes"),

    # "in X minutes" / "in exactly/about X minutes" / "X minutes from now" (digits)
    (re.compile(r"\bin\s+(?:exactly\s+|about\s+|like\s+)?(\d+)\s+minutes?\b", re.IGNORECASE),
     lambda m: f"in_{m.group(1)}_minutes"),
    (re.compile(r"\b(?:for\s+)?(\d+)\s+minutes?\s+from\s+now\b", re.IGNORECASE),
     lambda m: f"in_{m.group(1)}_minutes"),

    # Written numbers for minutes
    (re.compile(r"\bin\s+(?:a\s+)?minute\b", re.IGNORECASE), lambda _: "in_1_minutes"),
    (re.compile(r"\bin\s+one\s+minutes?\b", re.IGNORECASE), lambda _: "in_1_minutes"),
    (re.compile(r"\bin\s+five\s+minutes?\b", re.IGNORECASE), lambda _: "in_5_minutes"),
    (re.compile(r"\bin\s+ten\s+minutes?\b", re.IGNORECASE), lambda _: "in_10_minutes"),
    (re.compile(r"\bin\s+fifteen\s+minutes?\b", re.IGNORECASE), lambda _: "in_15_minutes"),
    (re.compile(r"\bin\s+twenty\s+minutes?\b", re.IGNORECASE), lambda _: "in_20_minutes"),
    (re.compile(r"\bin\s+thirty\s+minutes?\b", re.IGNORECASE), lambda _: "in_30_minutes"),
    (re.compile(r"\bin\s+forty[- ]?five\s+minutes?\b", re.IGNORECASE), lambda _: "in_45_minutes"),

    # "in 2 days" / "in two days" / "in a couple days" → day_after_tomorrow (before generic in_N_days)
    (re.compile(r"\bin\s+a\s+couple\s+(?:of\s+)?days?\b", re.IGNORECASE), lambda _: "day_after_tomorrow"),
    (re.compile(r"\bin\s+(?:2|two)\s+days?\b", re.IGNORECASE), lambda _: "day_after_tomorrow"),

    # "in X days" (digits) / "in a day"
    (re.compile(r"\bin\s+(?:a|one|1)\s+day\b", re.IGNORECASE), lambda _: "in_1_days"),
    (re.compile(r"\bin\s+a\s+couple\s+(?:of\s+)?days?\b", re.IGNORECASE), lambda _: "in_2_days"),
    (re.compile(r"\bin\s+(\d+)\s+days?\b", re.IGNORECASE),
     lambda m: f"in_{m.group(1)}_days"),
    (re.compile(r"\bin\s+three\s+days?\b", re.IGNORECASE), lambda _: "in_3_days"),
    (re.compile(r"\bin\s+five\s+days?\b", re.IGNORECASE), lambda _: "in_5_days"),

    # "in a week" / "in X weeks" / written
    (re.compile(r"\bin\s+(?:a|one|1)\s+week\b", re.IGNORECASE), lambda _: "in_7_days"),
    (re.compile(r"\bin\s+two\s+weeks?\b", re.IGNORECASE), lambda _: "in_14_days"),
    (re.compile(r"\bin\s+(\d+)\s+weeks?\b", re.IGNORECASE),
     lambda m: f"in_{int(m.group(1)) * 7}_days"),

    # "quarter past X am/pm"
    (re.compile(r"\bquarter\s+past\s+(\d{1,2})\s*(am|pm)\b", re.IGNORECASE),
     lambda m: f"at_{m.group(1)}_15{m.group(2).lower()}"),

    # "half past X am/pm"
    (re.compile(r"\bhalf\s+past\s+(\d{1,2})\s*(am|pm)\b", re.IGNORECASE),
     lambda m: f"at_{m.group(1)}_30{m.group(2).lower()}"),

    # "quarter to X am/pm"
    (re.compile(r"\bquarter\s+to\s+(\d{1,2})\s*(am|pm)\b", re.IGNORECASE),
     lambda m: f"at_{int(m.group(1)) - 1 if int(m.group(1)) > 1 else 12}_45{m.group(2).lower()}"),

    # 12am → midnight, 12pm → noon (MUST be before generic Xam/Xpm patterns)
    (re.compile(r"\b(?:at\s+)?12\s*:\s*00\s*am\b", re.IGNORECASE), lambda _: "midnight"),
    (re.compile(r"\b(?:at\s+)?12\s*am\b", re.IGNORECASE), lambda _: "midnight"),
    (re.compile(r"\b12\s+am\b", re.IGNORECASE), lambda _: "midnight"),
    (re.compile(r"\b(?:at\s+)?12\s*:\s*00\s*pm\b", re.IGNORECASE), lambda _: "noon"),
    (re.compile(r"\b(?:at\s+)?12\s*pm\b", re.IGNORECASE), lambda _: "noon"),
    (re.compile(r"\b12\s+pm\b", re.IGNORECASE), lambda _: "noon"),

    # "X:YY am/pm" (with colon)
    (re.compile(r"\b(?:at\s+)?(\d{1,2}):(\d{2})\s*(am|pm)\b", re.IGNORECASE),
     lambda m: f"at_{m.group(1)}{m.group(3).lower()}" if m.group(2) == "00"
     else f"at_{m.group(1)}_{m.group(2)}{m.group(3).lower()}"),

    # "at X am/pm" (no minutes)
    (re.compile(r"\bat\s+(\d{1,2})\s*(am|pm)\b", re.IGNORECASE),
     lambda m: f"at_{m.group(1)}{m.group(2).lower()}"),

    # Bare "Xam" / "Xpm" / "X am" / "X pm"
    (re.compile(r"\b(\d{1,2})\s*(am|pm)\b", re.IGNORECASE),
     lambda m: f"at_{m.group(1)}{m.group(2).lower()}"),

    # Written times: "X thirty am/pm", "X fifteen am/pm", "X forty-five am/pm"
    (re.compile(r"\b(\d{1,2})\s+thirty\s*(am|pm)\b", re.IGNORECASE),
     lambda m: f"at_{m.group(1)}_30{m.group(2).lower()}"),
    (re.compile(r"\b(\d{1,2})\s+fifteen\s*(am|pm)\b", re.IGNORECASE),
     lambda m: f"at_{m.group(1)}_15{m.group(2).lower()}"),
    (re.compile(r"\b(\d{1,2})\s+forty[- ]?five\s*(am|pm)\b", re.IGNORECASE),
     lambda m: f"at_{m.group(1)}_45{m.group(2).lower()}"),

    # "X o'clock" → noon or midnight based on context (default to noon)
    (re.compile(r"\b12\s+o'?clock\b", re.IGNORECASE), lambda _: "noon"),
]

# Negative patterns: phrases that look like dates but aren't
_NEGATIVE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(?:lasted?|took?|waited?)\s+\d+\s+(?:minutes?|hours?|days?|weeks?)\b", re.IGNORECASE),
    # "about X hours/minutes" only when NOT preceded by "in" (which makes it a future reference)
    re.compile(r"(?<!\bin\s)(?<!\bin\s\s)\b(?:about|approximately)\s+\d+\s+(?:minutes?|hours?|days?|weeks?)\s+(?:long|each|total|straight)\b", re.IGNORECASE),
    # "for X minutes/hours" only when NOT followed by "from now" (which makes it a future reference)
    re.compile(r"\bfor\s+\d+\s+(?:minutes?|hours?)\s*(?!from\s+now)\b(?:\s+(?:long|each|straight|total)|\s*$|\s*[,.])", re.IGNORECASE),
    re.compile(r"\b\d+\s+(?:minutes?|hours?|weeks?)\s+(?:ago|late|early|long)\b", re.IGNORECASE),
    # Note: "X days ago" is NOT negative — "2 days ago" = day_before_yesterday
    re.compile(r"\bset\s+(?:a\s+)?timer\s+(?:for)\b", re.IGNORECASE),
    re.compile(r"\bin\s+(?:a\s+)?(?:few|little|bit|some)\b", re.IGNORECASE),
    re.compile(r"\b(?:remind\s+me\s+)?later\b", re.IGNORECASE),
    # "expires/takes/lasts in X days" — not a date reference
    re.compile(r"\b(?:expires?|takes?|lasts?|downloads?)\s+in\s+(?:about\s+)?\d+\s+(?:minutes?|hours?|days?)\b", re.IGNORECASE),
    # Media commands at start of sentence: "Play Yesterday", "Watch Saturday Night Live"
    # Only blocks when play/watch is the command verb (sentence-initial)
    # Won't match "did I watch last week" or "what shows did I watch"
    re.compile(r"^(?:please\s+)?(?:play|watch|listen\s+to|put\s+on|queue)\s+(?!.*\b(?:for|on|at|during)\b).+", re.IGNORECASE),
    # Common phrases where date words aren't date references
    re.compile(r"\bdate\s+night\b", re.IGNORECASE),
    re.compile(r"\bnight\s+owl\b", re.IGNORECASE),
    re.compile(r"\bmorning\s+person\b", re.IGNORECASE),
    re.compile(r"\bweekend\s+warrior\b", re.IGNORECASE),
    re.compile(r"\bnight\s+and\s+day\b", re.IGNORECASE),
    re.compile(r"\bday\s+and\s+night\b", re.IGNORECASE),
    # Compound words where date terms are modifiers, not references
    re.compile(r"\bnight\s+(?:shift|cap|owl|light|stand|club|life|gown|time|mare)\b", re.IGNORECASE),
    re.compile(r"\bmorning\s+(?:shift|person|routine|sickness|dew|glory|star|show)\b", re.IGNORECASE),
    re.compile(r"\bevening\s+(?:shift|class(?:es)?|wear|gown|news|prayer|star)\b", re.IGNORECASE),
    re.compile(r"\bmidnight\s+(?:snack|blue|train|oil|run|sun|special)\b", re.IGNORECASE),
    re.compile(r"\blate\s+night\b", re.IGNORECASE),
    re.compile(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+(?:motivation|blues|vibes|specials?|cartoons?|brunch)\b", re.IGNORECASE),
    re.compile(r"\b(?:throwback|taco|casual|manic)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE),
    # "who's playing tonight" — sports context, not a date reference
    re.compile(r"\bwho'?s\s+playing\s+tonight\b", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Build compiled regex lookup (sorted by pattern length, longest first)
# ---------------------------------------------------------------------------

def _build_pattern_list() -> list[tuple[re.Pattern, str]]:
    """Build a priority-sorted list of (compiled_regex, date_key) pairs.

    Longer patterns first to prevent partial matches.
    """
    pattern_list: list[tuple[str, str]] = []

    # Collect all static patterns
    for group in [
        _COMBINED_PATTERNS,
        _RELATIVE_DAY_PATTERNS,
        _WEEKDAY_PATTERNS,
        _PERIOD_PATTERNS,
        _MEAL_PATTERNS,
        _TIME_MOD_PATTERNS,
    ]:
        for key, variants in group.items():
            for variant in variants:
                pattern_list.append((variant, key))

    # Sort by length descending (longest match wins)
    pattern_list.sort(key=lambda x: len(x[0]), reverse=True)

    # Compile to regex with word boundaries
    compiled: list[tuple[re.Pattern, str]] = []
    for variant, key in pattern_list:
        # Escape regex special chars in the variant
        escaped = re.escape(variant)
        # Word boundaries — but handle leading/trailing non-word chars
        pattern = re.compile(r"(?<!\w)" + escaped + r"(?!\w)", re.IGNORECASE)
        compiled.append((pattern, key))

    return compiled


_STATIC_PATTERNS = _build_pattern_list()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_date_keys(text: str) -> list[str]:
    """Extract date keys from text using deterministic regex matching.

    Returns a sorted list of unique date key strings.
    Empty list if no date references found.

    Args:
        text: Input text (e.g., voice command transcription)

    Returns:
        List of date keys (e.g., ["tomorrow_morning", "at_3pm"])
    """
    if not text or not text.strip():
        return []

    text_lower = text.lower().strip()
    keys: list[str] = []
    matched_spans: list[tuple[int, int]] = []  # Track matched positions to avoid overlaps

    def _overlaps(start: int, end: int) -> bool:
        """Check if a span overlaps with any already-matched span."""
        for ms, me in matched_spans:
            if start < me and end > ms:
                return True
        return False

    # Check negative patterns first — if the "in X minutes" is actually
    # "waited for X minutes" or "timer for X minutes", skip it
    negative_spans: list[tuple[int, int]] = []
    for neg_pattern in _NEGATIVE_PATTERNS:
        for m in neg_pattern.finditer(text_lower):
            negative_spans.append((m.start(), m.end()))

    def _in_negative_span(start: int, end: int) -> bool:
        """Check if a match is contained in OR overlaps significantly with a negative span."""
        for ns, ne in negative_spans:
            # Fully contained
            if start >= ns and end <= ne:
                return True
            # The matched text's core overlaps with the negative span
            overlap = min(end, ne) - max(start, ns)
            match_len = end - start
            if overlap > 0 and overlap >= match_len * 0.5:
                return True
        return False

    # 1. Dynamic patterns (regex-based: times, relative offsets)
    for pattern, formatter in _DYNAMIC_REGEXES:
        for m in pattern.finditer(text_lower):
            if not _overlaps(m.start(), m.end()) and not _in_negative_span(m.start(), m.end()):
                key = formatter(m)
                if key not in keys:
                    keys.append(key)
                    matched_spans.append((m.start(), m.end()))

    # 2. Static patterns (priority-sorted, longest first)
    for pattern, key in _STATIC_PATTERNS:
        for m in pattern.finditer(text_lower):
            if not _overlaps(m.start(), m.end()) and not _in_negative_span(m.start(), m.end()):
                if key not in keys:
                    keys.append(key)
                    matched_spans.append((m.start(), m.end()))

    return sorted(keys)
