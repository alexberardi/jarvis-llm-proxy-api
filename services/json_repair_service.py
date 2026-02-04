"""JSON repair and parsing service.

Provides utilities for parsing, extracting, and repairing JSON from LLM outputs.
Handles common issues like:
- Markdown code blocks wrapping JSON
- Truncated JSON (incomplete output)
- Unescaped quotes inside strings
- Duplicate keys
"""

import json
import re
from typing import Optional, Tuple


# JSON system message to inject when response_format is json_object
JSON_SYSTEM_MESSAGE = """You must respond with valid JSON only. Do not include any text before or after the JSON.
Do not wrap the JSON in markdown code blocks.
Output compact JSON without unnecessary whitespace.
Ensure all strings are properly escaped and all brackets are balanced."""


def is_json_truncated(content: str) -> bool:
    """Check if JSON content appears to be truncated.

    Returns True if the content shows signs of truncation:
    - Starts with { or [ but doesn't end with } or ]
    - Ends with a colon (awaiting value)
    - Ends mid-string
    - Has unbalanced brackets
    """
    stripped = content.strip()

    if not stripped:
        return False

    # Must start like JSON
    if not (stripped.startswith("{") or stripped.startswith("[")):
        return False

    # Check if properly closed
    if stripped.startswith("{") and not stripped.endswith("}"):
        return True
    if stripped.startswith("[") and not stripped.endswith("]"):
        return True

    # Check if ends with colon (awaiting value)
    if re.search(r":\s*$", stripped):
        return True

    # Check if ends with comma (incomplete list/object)
    if stripped.rstrip().endswith(","):
        return True

    # Check bracket balance
    open_braces = stripped.count("{") - stripped.count("}")
    open_brackets = stripped.count("[") - stripped.count("]")
    if open_braces != 0 or open_brackets != 0:
        return True

    return False


def repair_duplicate_keys(content: str) -> Optional[str]:
    """Repair JSON with duplicate keys by parsing and re-encoding.

    Python's json.loads() automatically keeps the last value for duplicate keys,
    so we can use parse + re-encode to clean duplicates.

    This handles cases where the model generates duplicate keys like:
    {"name": "value1", "name": "value2"} -> {"name": "value2"}

    Returns cleaned JSON if parsing succeeded, None otherwise.
    """
    if not (content.strip().startswith("{") or content.strip().startswith("[")):
        return None

    try:
        # Parse JSON (Python automatically handles duplicates by keeping last)
        parsed = json.loads(content)
        # Re-encode to get clean JSON without duplicates
        cleaned = json.dumps(parsed, ensure_ascii=False)
        # Return cleaned version (may differ from original if duplicates existed)
        return cleaned
    except json.JSONDecodeError:
        # If it's not valid JSON yet, let other repair functions handle syntax errors first
        return None


def extract_json_from_text(text: str) -> Optional[str]:
    """Try to extract valid JSON from text that might contain markdown code blocks or extra text.

    Returns the JSON string if found, None otherwise.

    Handles common patterns like:
    - "Here is the JSON:" followed by JSON
    - JSON wrapped in markdown code blocks
    - JSON with trailing text
    """
    # Strategy 1: Try to find JSON in markdown code blocks
    # First try to find the code block boundaries
    code_block_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
    match = re.search(code_block_pattern, text, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        # Try to find the JSON object/array within the code block
        # Look for first { or [ and extract balanced JSON
        first_brace = candidate.find("{")
        first_bracket = candidate.find("[")

        if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
            # Extract balanced object
            extracted = _extract_balanced_json(candidate, first_brace, "{", "}")
            if extracted:
                return extracted
        elif first_bracket != -1:
            # Extract balanced array
            extracted = _extract_balanced_json(candidate, first_bracket, "[", "]")
            if extracted:
                return extracted

        # If balanced extraction didn't work, try the whole code block content
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    # Strategy 1.5: Try to find JSON after common prefixes
    # Remove common prefixes that models sometimes add
    prefixes_to_remove = [
        r"^[^\[\{]*?(?=[\[\{])",  # Any text before { or [
        r"^Here is the JSON:\s*",
        r"^The JSON response is:\s*",
        r"^JSON:\s*",
        r"^Response:\s*",
    ]
    cleaned_text = text
    for prefix_pattern in prefixes_to_remove:
        cleaned_text = re.sub(
            prefix_pattern, "", cleaned_text, flags=re.IGNORECASE | re.MULTILINE
        )

    # Try parsing the cleaned text directly
    cleaned_text = cleaned_text.strip()
    if cleaned_text and (cleaned_text.startswith("{") or cleaned_text.startswith("[")):
        try:
            json.loads(cleaned_text)
            return cleaned_text
        except json.JSONDecodeError:
            pass

    # Strategy 2: Find first { or [ and try to extract balanced JSON
    # Find the first opening brace/bracket
    first_brace = text.find("{")
    first_bracket = text.find("[")

    candidates = []

    # Try to extract object starting from first {
    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        extracted = _extract_balanced_json(text, first_brace, "{", "}")
        if extracted:
            candidates.append(extracted)

    # Try to extract array starting from first [
    if first_bracket != -1 and (first_brace == -1 or first_bracket < first_brace):
        extracted = _extract_balanced_json(text, first_bracket, "[", "]")
        if extracted:
            candidates.append(extracted)

    # Validate candidates and return the first valid one
    for candidate in candidates:
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue

    return None


def _extract_balanced_json(
    text: str, start_pos: int, open_char: str, close_char: str
) -> Optional[str]:
    """Extract balanced JSON starting from start_pos.

    Args:
        text: The text to extract from
        start_pos: Position of the opening bracket
        open_char: Opening character ('{' or '[')
        close_char: Closing character ('}' or ']')

    Returns:
        Extracted JSON string if balanced and valid, None otherwise
    """
    depth = 0
    in_string = False
    escape_next = False

    for i in range(start_pos, len(text)):
        char = text[i]
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
        if not in_string:
            if char == open_char:
                depth += 1
            elif char == close_char:
                depth -= 1
                if depth == 0:
                    extracted = text[start_pos : i + 1]
                    try:
                        json.loads(extracted)
                        return extracted
                    except json.JSONDecodeError:
                        return None
    return None


def repair_unescaped_quotes(content: str) -> Optional[str]:
    """Attempt to repair JSON with unescaped quotes inside string values.

    Returns repaired JSON if possible, None otherwise.

    This handles cases like: "title": "Fried" Chicken Sandwich"
    Where the quote after "Fried" should be escaped: "title": "Fried\" Chicken Sandwich"
    """
    if not (content.strip().startswith("{") or content.strip().startswith("[")):
        return None

    try:
        result = []
        i = 0
        in_string = False
        escape_next = False

        while i < len(content):
            char = content[i]

            if escape_next:
                result.append(char)
                escape_next = False
                i += 1
                continue

            if char == "\\":
                result.append(char)
                escape_next = True
                i += 1
                continue

            if char == '"':
                if in_string:
                    # We're inside a string, check if this quote should close it
                    # Look ahead (skip whitespace) to see what comes next
                    j = i + 1
                    while j < len(content) and content[j] in " \t\n\r":
                        j += 1

                    if j >= len(content):
                        # End of content, this closes the string
                        result.append(char)
                        in_string = False
                    else:
                        next_char = content[j]
                        # If next non-whitespace char is : , } ] or end of string, this closes the string
                        if next_char in ":},]":
                            result.append(char)
                            in_string = False
                        else:
                            # This quote is inside the string value, escape it
                            result.append('\\"')
                else:
                    # Starting a new string
                    result.append(char)
                    in_string = True
            else:
                result.append(char)

            i += 1

        repaired = "".join(result)
        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            pass
    except Exception:
        pass

    return None


def _find_last_complete_position(text: str) -> int:
    """Find the position after the last complete JSON value.

    Returns -1 if no complete value found.
    """
    in_string = False
    escape_next = False
    bracket_stack = []
    last_complete_pos = -1

    i = 0
    while i < len(text):
        char = text[i]

        if escape_next:
            escape_next = False
            i += 1
            continue

        if char == "\\" and in_string:
            escape_next = True
            i += 1
            continue

        if char == '"':
            if not in_string:
                in_string = True
            else:
                in_string = False
                # After closing a string, check what follows
                # If followed by , or } or ] it's a complete value
                rest = text[i + 1 :].lstrip()
                if rest and rest[0] in ",}]":
                    last_complete_pos = i
            i += 1
            continue

        if in_string:
            i += 1
            continue

        # Not in string
        if char in "{[":
            bracket_stack.append(char)
        elif char == "}":
            if bracket_stack and bracket_stack[-1] == "{":
                bracket_stack.pop()
                # Completed an object
                rest = text[i + 1 :].lstrip()
                if not rest or rest[0] in ",}]":
                    last_complete_pos = i
        elif char == "]":
            if bracket_stack and bracket_stack[-1] == "[":
                bracket_stack.pop()
                # Completed an array
                rest = text[i + 1 :].lstrip()
                if not rest or rest[0] in ",}]":
                    last_complete_pos = i
        elif char == ",":
            # Comma marks end of a value
            last_complete_pos = i - 1  # Position before comma
        # Check for complete literals (true, false, null) or numbers
        elif char in "tfn" and bracket_stack:
            # Might be true, false, null
            for literal in ["true", "false", "null"]:
                if text[i : i + len(literal)] == literal:
                    rest = text[i + len(literal) :].lstrip()
                    if rest and rest[0] in ",}]":
                        last_complete_pos = i + len(literal) - 1
                    i += len(literal) - 1
                    break
        elif char in "-0123456789" and bracket_stack:
            # Might be a number, find its end
            j = i
            while j < len(text) and text[j] in "-0123456789.eE+":
                j += 1
            if j > i:
                rest = text[j:].lstrip()
                if rest and rest[0] in ",}]":
                    last_complete_pos = j - 1
                i = j - 1

        i += 1

    return last_complete_pos


def repair_truncated_json(content: str) -> Optional[str]:
    """Attempt to repair truncated JSON by closing unterminated strings and objects.

    Returns repaired JSON if possible, None otherwise.

    Handles common truncation scenarios:
    - Truncation mid-string (unterminated quote)
    - Truncation after colon but before value (e.g., "key":  with no value)
    - Truncation with unbalanced brackets
    - Truncation mid-number or mid-keyword
    """
    # Check if it looks like truncated JSON (starts with { or [ but doesn't end properly)
    stripped = content.strip()
    if not (stripped.startswith("{") or stripped.startswith("[")):
        return None

    # Strategy 0: Check if it's already valid JSON
    try:
        json.loads(content)
        return content
    except json.JSONDecodeError:
        pass

    try:
        # Strategy 1: Check if truncation is after a colon (awaiting value)
        # Pattern: "key": followed by nothing or incomplete value
        stripped_end = stripped.rstrip()

        # Check for truncation after colon with optional whitespace
        if re.search(r":\s*$", stripped_end):
            # Truncated right after colon - remove the incomplete key-value pair
            # Find the last comma before this colon
            last_colon = stripped_end.rfind(":")
            # Find the key that precedes this colon
            search_area = stripped_end[:last_colon]
            # Find the last complete comma or opening bracket
            last_comma = search_area.rfind(",")
            last_open_brace = search_area.rfind("{")
            last_open_bracket = search_area.rfind("[")

            truncate_at = max(last_comma, last_open_brace, last_open_bracket)
            if truncate_at > 0:
                if stripped_end[truncate_at] == ",":
                    # Remove from comma onwards
                    repaired = stripped_end[:truncate_at]
                else:
                    # Keep the opening bracket, remove the incomplete entry
                    repaired = stripped_end[: truncate_at + 1]

                # Count remaining open brackets
                open_braces = repaired.count("{") - repaired.count("}")
                open_brackets = repaired.count("[") - repaired.count("]")

                # Close them
                repaired += "}" * open_braces + "]" * open_brackets

                try:
                    json.loads(repaired)
                    return repaired
                except json.JSONDecodeError:
                    pass

        # Strategy 2: Track JSON structure state for more complex repairs
        in_string = False
        escape_next = False
        string_start_pos = -1
        bracket_stack = []  # Track opening brackets: '{' or '['

        for i, char in enumerate(content):
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue
            if char == '"' and not escape_next:
                if not in_string:
                    # Starting a new string
                    string_start_pos = i
                    in_string = True
                else:
                    # Ending a string
                    in_string = False
            elif not in_string:
                if char in "{[":
                    bracket_stack.append(char)
                elif char in "}]":
                    if bracket_stack:
                        bracket_stack.pop()

        # Strategy 2a: If we're in a string, try to close it with empty content
        if in_string and string_start_pos != -1:
            # Option A: Close the string immediately and complete structure
            repaired = content[: string_start_pos + 1] + '"'

            # Close all open brackets in reverse order
            temp_stack = bracket_stack.copy()
            while temp_stack:
                bracket = temp_stack.pop()
                if bracket == "{":
                    repaired += "}"
                elif bracket == "[":
                    repaired += "]"

            try:
                json.loads(repaired)
                return repaired
            except json.JSONDecodeError:
                pass

        # Strategy 2b: If we have unmatched brackets but not in a string
        if not in_string and len(bracket_stack) > 0:
            repaired = content.rstrip()

            # Check if ends with incomplete value after colon
            if re.search(r":\s*\S*$", repaired) and not re.search(
                r':\s*(".*"|true|false|null|\d+\.?\d*|\{.*\}|\[.*\])$', repaired
            ):
                # Ends with incomplete value - add null
                if repaired.rstrip()[-1] == ":":
                    repaired = repaired.rstrip() + "null"
                else:
                    # Has partial value, try to find last complete point
                    last_complete = _find_last_complete_position(content)
                    if last_complete > 0:
                        repaired = content[: last_complete + 1]

            # Remove trailing comma if present
            repaired = repaired.rstrip()
            if repaired.endswith(","):
                repaired = repaired[:-1]

            # Close brackets in reverse order
            temp_stack = bracket_stack.copy()
            while temp_stack:
                bracket = temp_stack.pop()
                if bracket == "{":
                    repaired += "}"
                elif bracket == "[":
                    repaired += "]"
            try:
                json.loads(repaired)
                return repaired
            except json.JSONDecodeError:
                pass

        # Strategy 3: Find last complete position and truncate there
        last_complete = _find_last_complete_position(content)
        if last_complete > 0:
            repaired = content[: last_complete + 1]

            # Remove trailing comma
            repaired = repaired.rstrip()
            if repaired.endswith(","):
                repaired = repaired[:-1]

            # Count and close unmatched brackets
            open_braces = repaired.count("{") - repaired.count("}")
            open_brackets = repaired.count("[") - repaired.count("]")

            repaired += "}" * max(0, open_braces) + "]" * max(0, open_brackets)

            try:
                json.loads(repaired)
                return repaired
            except json.JSONDecodeError:
                pass

    except Exception:
        pass

    # Strategy 4: Simple bracket counting approach as last resort
    try:
        # Remove any clearly incomplete trailing content
        repaired = content.rstrip()

        # Remove trailing incomplete property (after colon with no value)
        repaired = re.sub(r',\s*"[^"]*"\s*:\s*$', "", repaired)
        repaired = re.sub(r":\s*$", ":null", repaired)

        # Remove trailing comma
        if repaired.rstrip().endswith(","):
            repaired = repaired.rstrip()[:-1]

        # Count brackets
        open_braces = repaired.count("{") - repaired.count("}")
        open_brackets = repaired.count("[") - repaired.count("]")

        # Close them
        repaired += "}" * max(0, open_braces) + "]" * max(0, open_brackets)

        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            pass
    except Exception:
        pass

    return None


def parse_json_response(content: str) -> Tuple[str, bool]:
    """Parse JSON from LLM response content.

    Returns (parsed_content, is_valid_json)

    Tries multiple strategies:
    1. Parse as-is
    2. Repair duplicate keys
    3. Repair unescaped quotes
    4. Extract from markdown code blocks
    5. Extract JSON pattern from text
    6. Repair truncated JSON
    """
    # Strategy 1: Try parsing as-is
    try:
        json.loads(content)
        # Even if it parses, check for and fix duplicate keys
        # (Python's json keeps last duplicate, but we want clean output)
        repaired_duplicates = repair_duplicate_keys(content)
        if repaired_duplicates and repaired_duplicates != content:
            return repaired_duplicates, True
        return content, True
    except json.JSONDecodeError as e:
        error_str = str(e)
        # Check if it's an unescaped quote issue (common error patterns)
        if "Expecting" in error_str and (
            "delimiter" in error_str or "property name" in error_str
        ):
            # Try repairing unescaped quotes first
            repaired = repair_unescaped_quotes(content)
            if repaired:
                try:
                    json.loads(repaired)
                    return repaired, True
                except json.JSONDecodeError:
                    pass

        # Check if it's an unterminated string error
        if "Unterminated string" in error_str:
            repaired = repair_truncated_json(content)
            if repaired:
                try:
                    json.loads(repaired)
                    return repaired, True
                except json.JSONDecodeError:
                    pass

    # Strategy 2: Try repairing duplicate keys (might help with some edge cases)
    repaired_duplicates = repair_duplicate_keys(content)
    if repaired_duplicates:
        try:
            json.loads(repaired_duplicates)
            return repaired_duplicates, True
        except json.JSONDecodeError:
            pass

    # Strategy 3: Try repairing unescaped quotes before extraction
    repaired_quotes = repair_unescaped_quotes(content)
    if repaired_quotes:
        try:
            # Also check for duplicates in repaired version
            final_repaired = repair_duplicate_keys(repaired_quotes) or repaired_quotes
            json.loads(final_repaired)
            return final_repaired, True
        except json.JSONDecodeError:
            pass

    # Strategy 4: Try extracting from markdown or text
    extracted = extract_json_from_text(content)
    if extracted:
        try:
            # Check for duplicates in extracted JSON
            final_extracted = repair_duplicate_keys(extracted) or extracted
            json.loads(final_extracted)
            return final_extracted, True
        except json.JSONDecodeError as e:
            # Try repairing unescaped quotes in extracted JSON
            repaired_quotes = repair_unescaped_quotes(extracted)
            if repaired_quotes:
                try:
                    # Also fix duplicates
                    final_repaired = (
                        repair_duplicate_keys(repaired_quotes) or repaired_quotes
                    )
                    json.loads(final_repaired)
                    return final_repaired, True
                except json.JSONDecodeError:
                    pass

            # Try repairing truncated extracted JSON
            if "Unterminated string" in str(e):
                repaired = repair_truncated_json(extracted)
                if repaired:
                    try:
                        # Also fix duplicates
                        final_repaired = repair_duplicate_keys(repaired) or repaired
                        json.loads(final_repaired)
                        return final_repaired, True
                    except json.JSONDecodeError:
                        pass

    # Strategy 5: Try repairing truncated JSON
    repaired = repair_truncated_json(content)
    if repaired:
        try:
            json.loads(repaired)
            return repaired, True
        except json.JSONDecodeError:
            pass

    return content, False
