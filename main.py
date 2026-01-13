import multiprocessing
# Fix for vLLM CUDA multiprocessing issue - set spawn method early
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass

from fastapi import FastAPI, HTTPException, Header, Depends, Request
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Any
import os
import time
import uuid
import asyncio
import base64
import re
import json
from datetime import datetime
from dotenv import load_dotenv
from managers.chat_types import (
    NormalizedMessage,
    TextPart,
    ImagePart,
    GenerationParams,
    ChatResult,
)
from models.api_models import (
    ImageUrl,
    ContentPart,
    Message,
    ResponseFormat,
    ChatCompletionRequest,
    ChatCompletionChoice,
    Usage,
    ChatCompletionResponse,
    ErrorDetail,
    ErrorResponse,
    ModelInfo,
    ModelListResponse,
)
from queues.redis_queue import (
    get_redis_connection,
    mark_deduped,
    existing_dedup,
    enqueue_job,
    current_timestamp_ms,
)

from auth.app_auth import require_app_auth

load_dotenv()

app = FastAPI()

# Debug setup - only enable when DEBUG=true
debug_enabled = os.getenv("DEBUG", "false").lower() == "true"
skip_debugpy = os.getenv("LLM_PROXY_PROCESS_ROLE", "").lower() == "worker" or os.getenv("JARVIS_DISABLE_DEBUGPY", "false").lower() == "true"
debug_port = int(os.getenv("DEBUG_PORT", "5678"))
if debug_enabled and not skip_debugpy:
    try:
        import debugpy
        debugpy.listen(("0.0.0.0", debug_port))
        print(f"🐛 Debugger listening on port {debug_port}")
    except ImportError:
        print("❌ debugpy is not installed, but DEBUG is set to true")

# Log critical endpoint configs for visibility
print(f"🌐 MODEL_SERVICE_URL={os.getenv('MODEL_SERVICE_URL')}")
print(f"🔐 MODEL_SERVICE_TOKEN set: {'yes' if os.getenv('MODEL_SERVICE_TOKEN') else 'no'}")
print(f"🔐 LLM_PROXY_INTERNAL_TOKEN set: {'yes' if os.getenv('LLM_PROXY_INTERNAL_TOKEN') else 'no'}")
print(f"🔐 JARVIS_AUTH_BASE_URL={os.getenv('JARVIS_AUTH_BASE_URL')}")

# ============================================================================
# OpenAI-compatible request/response models (imported from models.api_models)
# ============================================================================


# ============================================================================
# Queue models
# ============================================================================


class ArtifactRefs(BaseModel):
    input_uri: Optional[str] = None
    schema_uri: Optional[str] = None
    prompt_uri: Optional[str] = None


class SamplingSettings(BaseModel):
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    seed: Optional[int] = None


class TimeoutSettings(BaseModel):
    overall_seconds: Optional[int] = None
    per_attempt_seconds: Optional[int] = None


class CallbackInfo(BaseModel):
    url: str
    auth_type: Optional[str] = None
    token: Optional[str] = None


class QueueRequest(BaseModel):
    artifacts: Optional[ArtifactRefs] = None
    model: str
    messages: List[Message]
    response_format: Optional[ResponseFormat] = None
    sampling: Optional[SamplingSettings] = None
    timeouts: Optional[TimeoutSettings] = None


class EnqueueRequest(BaseModel):
    job_id: str
    job_type: str
    created_at: str
    priority: Optional[str] = "normal"
    trace_id: Optional[str] = None
    idempotency_key: str
    job_type_version: str = "v1"
    ttl_seconds: int = 86400
    metadata: Optional[dict] = None
    request: QueueRequest
    callback: CallbackInfo


class EnqueueResponse(BaseModel):
    accepted: bool
    job_id: str
    deduped: bool = False


# ============================================================================
# Helper functions for message normalization
# ============================================================================

def parse_data_url(data_url: str) -> tuple[bytes, str]:
    """
    Parse a data URL and return (image_bytes, mime_type).
    
    Expected format: data:image/png;base64,iVBORw0KG...
    """
    match = re.match(r'^data:([^;]+);base64,(.+)$', data_url)
    if not match:
        raise ValueError("Invalid data URL format. Expected: data:<mime_type>;base64,<data>")
    
    mime_type = match.group(1)
    b64_data = match.group(2)
    
    try:
        image_bytes = base64.b64decode(b64_data)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image data: {e}")
    
    return image_bytes, mime_type


def normalize_message(message: Message) -> NormalizedMessage:
    """
    Convert an OpenAI-style message to a NormalizedMessage.
    
    Handles both:
    - content as string: {"role": "user", "content": "Hello"}
    - content as array: {"role": "user", "content": [{"type": "text", "text": "..."}, {"type": "image_url", ...}]}
    """
    content_parts: List[Union[TextPart, ImagePart]] = []
    
    if isinstance(message.content, str):
        # Simple string content
        content_parts.append(TextPart(text=message.content))
    elif isinstance(message.content, list):
        # Structured content array
        for part in message.content:
            if part.type == "text" and part.text is not None:
                content_parts.append(TextPart(text=part.text))
            elif part.type == "image_url" and part.image_url is not None:
                url = part.image_url.url
                # For now, only support data URLs
                if not url.startswith("data:"):
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": {
                                "type": "invalid_request_error",
                                "message": "Only data URLs are supported for images. HTTP(S) URLs are not yet supported.",
                                "code": None,
                            }
                        }
                    )
                image_bytes, mime_type = parse_data_url(url)
                content_parts.append(
                    ImagePart(
                        data=image_bytes,
                        mime_type=mime_type,
                        detail=part.image_url.detail,
                    )
                )
            else:
                # Unknown part type or missing required field
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "type": "invalid_request_error",
                            "message": f"Unsupported content part type or missing field: {part.type}",
                            "code": None,
                        }
                    }
                )
    else:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "type": "invalid_request_error",
                    "message": "Message content must be either a string or an array of content parts",
                    "code": None,
                }
            }
        )
    
    return NormalizedMessage(role=message.role, content=content_parts)


def normalize_messages(messages: List[Message]) -> List[NormalizedMessage]:
    """Convert a list of OpenAI-style messages to NormalizedMessage format"""
    return [normalize_message(msg) for msg in messages]


# JSON system message to inject when response_format is json_object
JSON_SYSTEM_MESSAGE = """You must respond with valid JSON only. Do not include any text before or after the JSON.
Do not wrap the JSON in markdown code blocks.
Output compact JSON without unnecessary whitespace.
Ensure all strings are properly escaped and all brackets are balanced."""


def inject_json_system_message(messages: List[NormalizedMessage]) -> List[NormalizedMessage]:
    """
    Inject or augment a system message to request JSON output.
    
    If a system message exists, append JSON instructions to it.
    If no system message exists, prepend one.
    """
    # Check if there's already a system message
    has_system = any(msg.role == "system" for msg in messages)
    
    if has_system:
        # Augment existing system message
        augmented = []
        for msg in messages:
            if msg.role == "system":
                # Get existing text content
                existing_text = ""
                for part in msg.content:
                    if isinstance(part, TextPart):
                        existing_text += part.text + "\n"
                
                # Check if JSON instructions are already present
                if "JSON" not in existing_text.upper() or "valid json" not in existing_text.lower():
                    augmented_text = existing_text.strip() + "\n\n" + JSON_SYSTEM_MESSAGE
                else:
                    augmented_text = existing_text  # Already has JSON instructions
                
                augmented.append(NormalizedMessage(
                    role="system",
                    content=[TextPart(text=augmented_text)]
                ))
            else:
                augmented.append(msg)
        return augmented
    else:
        # Prepend new system message
        system_msg = NormalizedMessage(
            role="system",
            content=[TextPart(text=JSON_SYSTEM_MESSAGE)]
        )
        return [system_msg] + messages


def is_json_truncated(content: str) -> bool:
    """
    Check if JSON content appears to be truncated.
    
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
    if not (stripped.startswith('{') or stripped.startswith('[')):
        return False
    
    # Check if properly closed
    if stripped.startswith('{') and not stripped.endswith('}'):
        return True
    if stripped.startswith('[') and not stripped.endswith(']'):
        return True
    
    # Check if ends with colon (awaiting value)
    if re.search(r':\s*$', stripped):
        return True
    
    # Check if ends with comma (incomplete list/object)
    if stripped.rstrip().endswith(','):
        return True
    
    # Check bracket balance
    open_braces = stripped.count('{') - stripped.count('}')
    open_brackets = stripped.count('[') - stripped.count(']')
    if open_braces != 0 or open_brackets != 0:
        return True
    
    return False


def repair_duplicate_keys(content: str) -> Optional[str]:
    """
    Repair JSON with duplicate keys by parsing and re-encoding.
    Python's json.loads() automatically keeps the last value for duplicate keys,
    so we can use parse + re-encode to clean duplicates.
    
    This handles cases where the model generates duplicate keys like:
    {"name": "value1", "name": "value2"} -> {"name": "value2"}
    
    Returns cleaned JSON if parsing succeeded, None otherwise.
    """
    if not (content.strip().startswith('{') or content.strip().startswith('[')):
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


def has_images(messages: List[NormalizedMessage]) -> bool:
    """Check if any message contains an ImagePart"""
    for msg in messages:
        for part in msg.content:
            if isinstance(part, ImagePart):
                return True
    return False




def extract_json_from_text(text: str) -> Optional[str]:
    """
    Try to extract valid JSON from text that might contain markdown code blocks or extra text.
    Returns the JSON string if found, None otherwise.
    
    Handles common patterns like:
    - "Here is the JSON:" followed by JSON
    - JSON wrapped in markdown code blocks
    - JSON with trailing text
    """
    # Strategy 1: Try to find JSON in markdown code blocks
    # First try to find the code block boundaries
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    match = re.search(code_block_pattern, text, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        # Try to find the JSON object/array within the code block
        # Look for first { or [ and extract balanced JSON
        first_brace = candidate.find('{')
        first_bracket = candidate.find('[')
        
        if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
            # Extract balanced object
            depth = 0
            in_string = False
            escape_next = False
            for i in range(first_brace, len(candidate)):
                char = candidate[i]
                if escape_next:
                    escape_next = False
                    continue
                if char == '\\':
                    escape_next = True
                    continue
                if char == '"' and not escape_next:
                    in_string = not in_string
                if not in_string:
                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0:
                            extracted = candidate[first_brace:i+1]
                            try:
                                json.loads(extracted)
                                return extracted
                            except json.JSONDecodeError:
                                break
        elif first_bracket != -1:
            # Extract balanced array
            depth = 0
            in_string = False
            escape_next = False
            for i in range(first_bracket, len(candidate)):
                char = candidate[i]
                if escape_next:
                    escape_next = False
                    continue
                if char == '\\':
                    escape_next = True
                    continue
                if char == '"' and not escape_next:
                    in_string = not in_string
                if not in_string:
                    if char == '[':
                        depth += 1
                    elif char == ']':
                        depth -= 1
                        if depth == 0:
                            extracted = candidate[first_bracket:i+1]
                            try:
                                json.loads(extracted)
                                return extracted
                            except json.JSONDecodeError:
                                break
        
        # If balanced extraction didn't work, try the whole code block content
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    
    # Strategy 1.5: Try to find JSON after common prefixes
    # Remove common prefixes that models sometimes add
    prefixes_to_remove = [
        r'^[^\[\{]*?(?=[\[\{])',  # Any text before { or [
        r'^Here is the JSON:\s*',
        r'^The JSON response is:\s*',
        r'^JSON:\s*',
        r'^Response:\s*',
    ]
    cleaned_text = text
    for prefix_pattern in prefixes_to_remove:
        cleaned_text = re.sub(prefix_pattern, '', cleaned_text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Try parsing the cleaned text directly
    cleaned_text = cleaned_text.strip()
    if cleaned_text and (cleaned_text.startswith('{') or cleaned_text.startswith('[')):
        try:
            json.loads(cleaned_text)
            return cleaned_text
        except json.JSONDecodeError:
            pass
    
    # Strategy 2: Find first { or [ and try to extract balanced JSON
    # Find the first opening brace/bracket
    first_brace = text.find('{')
    first_bracket = text.find('[')
    
    candidates = []
    
    # Try to extract object starting from first {
    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        # Find matching closing brace
        depth = 0
        in_string = False
        escape_next = False
        for i in range(first_brace, len(text)):
            char = text[i]
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
            if not in_string:
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = text[first_brace:i+1]
                        candidates.append(candidate)
                        break
    
    # Try to extract array starting from first [
    if first_bracket != -1 and (first_brace == -1 or first_bracket < first_brace):
        # Find matching closing bracket
        depth = 0
        in_string = False
        escape_next = False
        for i in range(first_bracket, len(text)):
            char = text[i]
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
            if not in_string:
                if char == '[':
                    depth += 1
                elif char == ']':
                    depth -= 1
                    if depth == 0:
                        candidate = text[first_bracket:i+1]
                        candidates.append(candidate)
                        break
    
    # Validate candidates and return the first valid one
    for candidate in candidates:
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue
    
    return None


def repair_unescaped_quotes(content: str) -> Optional[str]:
    """
    Attempt to repair JSON with unescaped quotes inside string values.
    Returns repaired JSON if possible, None otherwise.
    
    This handles cases like: "title": "Fried" Chicken Sandwich"
    Where the quote after "Fried" should be escaped: "title": "Fried\" Chicken Sandwich"
    """
    if not (content.strip().startswith('{') or content.strip().startswith('[')):
        return None
    
    try:
        result = []
        i = 0
        in_string = False
        escape_next = False
        string_start = -1
        
        while i < len(content):
            char = content[i]
            
            if escape_next:
                result.append(char)
                escape_next = False
                i += 1
                continue
            
            if char == '\\':
                result.append(char)
                escape_next = True
                i += 1
                continue
            
            if char == '"':
                if in_string:
                    # We're inside a string, check if this quote should close it
                    # Look ahead (skip whitespace) to see what comes next
                    j = i + 1
                    while j < len(content) and content[j] in ' \t\n\r':
                        j += 1
                    
                    if j >= len(content):
                        # End of content, this closes the string
                        result.append(char)
                        in_string = False
                    else:
                        next_char = content[j]
                        # If next non-whitespace char is : , } ] or end of string, this closes the string
                        if next_char in ':},]':
                            result.append(char)
                            in_string = False
                        else:
                            # This quote is inside the string value, escape it
                            result.append('\\"')
                else:
                    # Starting a new string
                    result.append(char)
                    in_string = True
                    string_start = i
            else:
                result.append(char)
            
            i += 1
        
        repaired = ''.join(result)
        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            pass
    except Exception:
        pass
    
    return None


def repair_truncated_json(content: str) -> Optional[str]:
    """
    Attempt to repair truncated JSON by closing unterminated strings and objects.
    Returns repaired JSON if possible, None otherwise.
    
    Handles common truncation scenarios:
    - Truncation mid-string (unterminated quote)
    - Truncation after colon but before value (e.g., "key":  with no value)
    - Truncation with unbalanced brackets
    - Truncation mid-number or mid-keyword
    """
    # Check if it looks like truncated JSON (starts with { or [ but doesn't end properly)
    stripped = content.strip()
    if not (stripped.startswith('{') or stripped.startswith('[')):
        return None
    
    # Strategy 0: Check if it's already valid JSON
    try:
        json.loads(content)
        return content
    except json.JSONDecodeError:
        pass
    
    # Find the last complete key-value pair position and truncate there
    def find_last_complete_position(text: str) -> int:
        """
        Find the position after the last complete JSON value.
        Returns -1 if no complete value found.
        """
        depth = 0
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
                
            if char == '\\' and in_string:
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
                    rest = text[i+1:].lstrip()
                    if rest and rest[0] in ',}]':
                        last_complete_pos = i
                i += 1
                continue
                
            if in_string:
                i += 1
                continue
                
            # Not in string
            if char in '{[':
                bracket_stack.append(char)
                depth += 1
            elif char == '}':
                if bracket_stack and bracket_stack[-1] == '{':
                    bracket_stack.pop()
                    depth -= 1
                    # Completed an object
                    rest = text[i+1:].lstrip()
                    if not rest or rest[0] in ',}]':
                        last_complete_pos = i
            elif char == ']':
                if bracket_stack and bracket_stack[-1] == '[':
                    bracket_stack.pop()
                    depth -= 1
                    # Completed an array
                    rest = text[i+1:].lstrip()
                    if not rest or rest[0] in ',}]':
                        last_complete_pos = i
            elif char == ',':
                # Comma marks end of a value
                last_complete_pos = i - 1  # Position before comma
            # Check for complete literals (true, false, null) or numbers
            elif char in 'tfn' and depth > 0:
                # Might be true, false, null
                for literal in ['true', 'false', 'null']:
                    if text[i:i+len(literal)] == literal:
                        rest = text[i+len(literal):].lstrip()
                        if rest and rest[0] in ',}]':
                            last_complete_pos = i + len(literal) - 1
                        i += len(literal) - 1
                        break
            elif char in '-0123456789' and depth > 0:
                # Might be a number, find its end
                j = i
                while j < len(text) and text[j] in '-0123456789.eE+':
                    j += 1
                if j > i:
                    rest = text[j:].lstrip()
                    if rest and rest[0] in ',}]':
                        last_complete_pos = j - 1
                    i = j - 1
            
            i += 1
        
        return last_complete_pos
    
    try:
        # Strategy 1: Check if truncation is after a colon (awaiting value)
        # Pattern: "key": followed by nothing or incomplete value
        stripped_end = stripped.rstrip()
        
        # Check for truncation after colon with optional whitespace
        if re.search(r':\s*$', stripped_end):
            # Truncated right after colon - remove the incomplete key-value pair
            # Find the last comma before this colon
            last_colon = stripped_end.rfind(':')
            # Find the key that precedes this colon
            search_area = stripped_end[:last_colon]
            # Find the last complete comma or opening bracket
            last_comma = search_area.rfind(',')
            last_open_brace = search_area.rfind('{')
            last_open_bracket = search_area.rfind('[')
            
            truncate_at = max(last_comma, last_open_brace, last_open_bracket)
            if truncate_at > 0:
                if stripped_end[truncate_at] == ',':
                    # Remove from comma onwards
                    repaired = stripped_end[:truncate_at]
                else:
                    # Keep the opening bracket, remove the incomplete entry
                    repaired = stripped_end[:truncate_at + 1]
                
                # Count remaining open brackets
                open_braces = repaired.count('{') - repaired.count('}')
                open_brackets = repaired.count('[') - repaired.count(']')
                
                # Close them
                repaired += '}' * open_braces + ']' * open_brackets
                
                try:
                    json.loads(repaired)
                    print(f"✅ Repaired truncation after colon")
                    return repaired
                except json.JSONDecodeError:
                    pass
        
        # Strategy 2: Track JSON structure state for more complex repairs
        depth = 0
        in_string = False
        escape_next = False
        string_start_pos = -1
        last_complete_value_pos = -1
        bracket_stack = []  # Track opening brackets: '{' or '['
        
        for i, char in enumerate(content):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
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
                    last_complete_value_pos = i
            elif not in_string:
                if char in '{[':
                    depth += 1
                    bracket_stack.append(char)
                    last_complete_value_pos = i
                elif char in '}]':
                    if bracket_stack:
                        bracket_stack.pop()
                    depth -= 1
                    last_complete_value_pos = i
                elif char in ',:' and depth > 0:
                    # Valid separator, mark as complete position
                    last_complete_value_pos = i
        
        # Strategy 2a: If we're in a string, try to close it with empty content
        if in_string and string_start_pos != -1:
            # Option A: Close the string immediately and complete structure
            repaired = content[:string_start_pos+1] + '"'
            
            # Close all open brackets in reverse order
            temp_stack = bracket_stack.copy()
            while temp_stack:
                bracket = temp_stack.pop()
                if bracket == '{':
                    repaired += '}'
                elif bracket == '[':
                    repaired += ']'
            
            try:
                json.loads(repaired)
                print(f"✅ Repaired by closing unterminated string")
                return repaired
            except json.JSONDecodeError:
                pass
        
        # Strategy 2b: If we have unmatched brackets but not in a string
        if not in_string and depth > 0:
            repaired = content.rstrip()
            
            # Check if ends with incomplete value after colon
            if re.search(r':\s*\S*$', repaired) and not re.search(r':\s*(".*"|true|false|null|\d+\.?\d*|\{.*\}|\[.*\])$', repaired):
                # Ends with incomplete value - add null
                if repaired.rstrip()[-1] == ':':
                    repaired = repaired.rstrip() + 'null'
                else:
                    # Has partial value, try to find last complete point
                    last_complete = find_last_complete_position(content)
                    if last_complete > 0:
                        repaired = content[:last_complete + 1]
            
            # Remove trailing comma if present
            repaired = repaired.rstrip()
            if repaired.endswith(','):
                repaired = repaired[:-1]
            
            # Close brackets in reverse order
            temp_stack = bracket_stack.copy()
            while temp_stack:
                bracket = temp_stack.pop()
                if bracket == '{':
                    repaired += '}'
                elif bracket == '[':
                    repaired += ']'
            try:
                json.loads(repaired)
                print(f"✅ Repaired by closing unmatched brackets")
                return repaired
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Find last complete position and truncate there
        last_complete = find_last_complete_position(content)
        if last_complete > 0:
            repaired = content[:last_complete + 1]
            
            # Remove trailing comma
            repaired = repaired.rstrip()
            if repaired.endswith(','):
                repaired = repaired[:-1]
            
            # Count and close unmatched brackets
            open_braces = repaired.count('{') - repaired.count('}')
            open_brackets = repaired.count('[') - repaired.count(']')
            
            repaired += '}' * max(0, open_braces) + ']' * max(0, open_brackets)
            
            try:
                json.loads(repaired)
                print(f"✅ Repaired by truncating at last complete value")
                return repaired
            except json.JSONDecodeError:
                pass
                
    except Exception as e:
        print(f"⚠️  Error in repair_truncated_json: {e}")
    
    # Strategy 4: Simple bracket counting approach as last resort
    try:
        # Remove any clearly incomplete trailing content
        repaired = content.rstrip()
        
        # Remove trailing incomplete property (after colon with no value)
        repaired = re.sub(r',\s*"[^"]*"\s*:\s*$', '', repaired)
        repaired = re.sub(r':\s*$', ':null', repaired)
        
        # Remove trailing comma
        if repaired.rstrip().endswith(','):
            repaired = repaired.rstrip()[:-1]
        
        # Count brackets
        open_braces = repaired.count('{') - repaired.count('}')
        open_brackets = repaired.count('[') - repaired.count(']')
        
        # Close them
        repaired += '}' * max(0, open_braces) + ']' * max(0, open_brackets)
        
        try:
            json.loads(repaired)
            print(f"✅ Repaired with simple bracket closing")
            return repaired
        except json.JSONDecodeError:
            pass
    except Exception:
        pass
    
    return None


def parse_json_response(content: str) -> tuple[str, bool]:
    """
    Parse JSON from LLM response content.
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
        parsed = json.loads(content)
        # Even if it parses, check for and fix duplicate keys
        # (Python's json keeps last duplicate, but we want clean output)
        repaired_duplicates = repair_duplicate_keys(content)
        if repaired_duplicates and repaired_duplicates != content:
            return repaired_duplicates, True
        return content, True
    except json.JSONDecodeError as e:
        error_str = str(e)
        # Check if it's an unescaped quote issue (common error patterns)
        if "Expecting" in error_str and ("delimiter" in error_str or "property name" in error_str):
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
                    final_repaired = repair_duplicate_keys(repaired_quotes) or repaired_quotes
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


async def fix_json_with_retry(
    backend: Any,
    model_config: Any,
    normalized_messages: List[NormalizedMessage],
    params: GenerationParams,
    invalid_content: str,
    max_retries: int = 1,
) -> Optional[str]:
    """
    Attempt to fix invalid JSON by re-querying the LLM with a correction prompt.
    Returns fixed JSON content if successful, None otherwise.
    """
    # Estimate tokens needed based on truncated content length
    # Rough estimate: ~4 chars per token for English/JSON
    estimated_output_tokens = len(invalid_content) // 3  # Be generous
    
    # Calculate retry max_tokens
    # If original had max_tokens set, use at least 1.5x that
    # Otherwise, estimate based on truncated output + buffer
    original_max = params.max_tokens or 4096
    buffer_tokens = max(2000, estimated_output_tokens // 2)  # At least 2000 or half the estimated output
    retry_max_tokens = max(
        int(original_max * 1.5),  # 1.5x original
        estimated_output_tokens + buffer_tokens,  # Estimated + buffer
        8192  # Minimum retry size
    )
    
    # Cap at reasonable maximum to avoid excessive token usage
    retry_max_tokens = min(retry_max_tokens, 16384)
    
    print(f"📊 Retry token calculation: original={original_max}, estimated_output={estimated_output_tokens}, retry_max={retry_max_tokens}")
    
    # Construct a more specific correction prompt
    # Build repair prompt (system + user) using the provided schema if available
    schema_text = ""
    if params.response_format and isinstance(params.response_format, dict):
        if "json_schema" in params.response_format:
            print(f"🔍 Schema HERE: {params.response_format['json_schema']}")
            try:
                schema_text = json.dumps(params.response_format["json_schema"], indent=2)
            except Exception:
                print(f"🔍 NO SCHEMA FOUND")
                schema_text = str(params.response_format.get("json_schema"))
    if not schema_text:
        schema_text = "(schema not provided)"
    
    system_repair_prompt = """You are a JSON repair engine.

Your task is to output a SINGLE, VALID JSON OBJECT that CONFORMS to the provided JSON Schema.

CRITICAL RULES:
- Output MUST be valid JSON ONLY (no markdown, no code fences, no prose, no explanations).
- Output MUST be a DATA INSTANCE that matches the schema — NOT a schema, NOT an explanation.
- NEVER include schema or metadata keywords in the output, including:
  "$schema", "$id", "$defs", "$ref", "definitions", "properties",
  "required", "anyOf", "oneOf", "allOf", "type", "title", "description"
  (unless the schema explicitly defines these as required DATA fields).
- Do NOT output keys that start with "$".
- Do NOT invent fields that are not allowed by the schema.
- If required fields are missing, populate them with reasonable empty/default values
  that satisfy the schema ("" for required strings, [] for required arrays, {} for required objects),
  unless the schema provides a default.
- If the input contains multiple JSON objects or extra text, select the object
  that best matches the schema and repair ONLY that.

FAILURE RULE:
- If a valid instance cannot be produced, output EXACTLY:
  {"error":"UNABLE_TO_REPAIR"}"""
    
    truncation_note = ""
    if is_json_truncated(invalid_content):
        truncation_note = " (Input appears truncated.)"
    
    correction_text = f"""Schema:
{schema_text}

Input to repair{truncation_note}:
{invalid_content[:800]}"""
    
    repair_system_message = NormalizedMessage(
        role="system",
        content=[TextPart(text=system_repair_prompt)]
    )
    correction_message = NormalizedMessage(
        role="user",
        content=[TextPart(text=correction_text)]
    )
    
    retry_messages = normalized_messages + [repair_system_message, correction_message]
    
    # Create retry params with increased max_tokens to avoid truncation
    retry_params = GenerationParams(
        temperature=max(0.3, params.temperature - 0.2),  # Lower temperature for more deterministic output
        max_tokens=retry_max_tokens,
        stream=params.stream,
        response_format=params.response_format,
    )
    
    for attempt in range(max_retries):
        try:
            if hasattr(backend, "generate_text_chat"):
                if asyncio.iscoroutinefunction(backend.generate_text_chat):
                    result: ChatResult = await backend.generate_text_chat(
                        model_config, retry_messages, retry_params
                    )
                else:
                    result: ChatResult = backend.generate_text_chat(
                        model_config, retry_messages, retry_params
                    )
            else:
                # Legacy backend path
                legacy_messages = []
                for msg in retry_messages:
                    text_content = " ".join(
                        part.text for part in msg.content if isinstance(part, TextPart)
                    )
                    legacy_messages.append({"role": msg.role, "content": text_content})
                
                if hasattr(backend, 'chat_with_temperature'):
                    if asyncio.iscoroutinefunction(backend.chat_with_temperature):
                        output = await backend.chat_with_temperature(legacy_messages, params.temperature)
                    else:
                        output = backend.chat_with_temperature(legacy_messages, params.temperature)
                else:
                    if asyncio.iscoroutinefunction(backend.chat):
                        output = await backend.chat(legacy_messages)
                    else:
                        output = backend.chat(legacy_messages)
                
                usage = None
                if hasattr(backend, 'last_usage'):
                    usage = backend.last_usage
                result = ChatResult(content=output, usage=usage)
            
            fixed_content, is_valid = parse_json_response(result.content)
            if is_valid:
                return fixed_content
            # If extraction found something but validation failed, still try to return it
            # (the validation in parse_json_response should have caught it, but just in case)
            
        except Exception as e:
            print(f"⚠️  Retry attempt {attempt + 1} failed: {e}")
            continue
    
    return None


def create_openai_response(
    content: str,
    model_name: str,
    usage: Optional[Dict] = None,
) -> ChatCompletionResponse:
    """Create an OpenAI-style chat completion response"""
    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    
    if usage is None:
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    return ChatCompletionResponse(
        id=response_id,
        created=created,
        model=model_name,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=Message(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=Usage(**usage),
    )


def openai_error(error_type: str, message: str, status_code: int = 400):
    """Create an OpenAI-style error response"""
    raise HTTPException(
        status_code=status_code,
        detail={
            "error": {
                "type": error_type,
                "message": message,
                "code": None,
            }
        }
    )


# ============================================================================
# Routes
# ============================================================================

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse, dependencies=[Depends(require_app_auth)])
async def chat_completions(req: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    
    Supports:
    - Text-only messages (string content)
    - Multimodal messages (structured content with text + images)
    - Model selection via model field (supports aliases: full, lightweight, vision, cloud)
    """
    try:
        model_service_url = os.getenv("MODEL_SERVICE_URL")
        internal_token = os.getenv("MODEL_SERVICE_TOKEN") or os.getenv("LLM_PROXY_INTERNAL_TOKEN")
        if not model_service_url:
            openai_error("internal_server_error", "MODEL_SERVICE_URL is not set; API is passthrough-only.", 500)
        import httpx
        url = model_service_url.rstrip("/") + "/internal/model/chat"
        headers = {}
        if internal_token:
            headers["X-Internal-Token"] = internal_token
        async with httpx.AsyncClient(timeout=float(os.getenv("MODEL_SERVICE_TIMEOUT", "60"))) as client:
            resp = await client.post(url, json=req.dict(), headers=headers)
            if resp.status_code != 200:
                openai_error("internal_server_error", f"Model service error {resp.status_code}: {resp.text}", 500)
            data = resp.json()
            content = data.get("content")
            usage = data.get("usage")
            model_name = req.model
            return create_openai_response(content=content, model_name=model_name, usage=usage)
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error during chat completion: {e}")
        import traceback
        traceback.print_exc()
        openai_error("internal_server_error", f"Internal error: {str(e)}", 500)


def _require_internal_token(token: Optional[str]):
    expected = os.getenv("LLM_PROXY_INTERNAL_TOKEN")
    if expected and token != expected:
        raise HTTPException(status_code=401, detail={"error": {"type": "unauthorized", "message": "Invalid internal token", "code": "unauthorized"}})


def _parse_created_at(ts: str) -> float:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except Exception:
        try:
            return float(ts)
        except Exception:
            return time.time()


@app.post("/internal/queue/enqueue", response_model=EnqueueResponse, dependencies=[Depends(require_app_auth)])
async def enqueue_job_endpoint(req: EnqueueRequest, request: Request, x_internal_token: Optional[str] = Header(default=None)):
    """
    Enqueue a job for asynchronous LLM processing.
    """
    _require_internal_token(x_internal_token)

    # Basic validation for schema requirement
    if req.request.response_format and req.request.response_format.type == "json_object":
        if req.request.response_format.json_schema is None:
            raise HTTPException(
                status_code=400,
                detail={"error": {"type": "invalid_request_error", "message": "json_schema required when response_format.type=json_object", "code": "missing_schema"}},
            )

    now = time.time()
    created_ts = _parse_created_at(req.created_at)
    if created_ts + req.ttl_seconds < now:
        raise HTTPException(
            status_code=400,
            detail={"error": {"type": "invalid_request_error", "message": "Job already expired", "code": "expired"}},
        )

    conn = get_redis_connection()
    # Dedup check
    if existing_dedup(conn, req.job_id, req.idempotency_key):
        return EnqueueResponse(accepted=True, job_id=req.job_id, deduped=True)

    if not mark_deduped(conn, req.job_id, req.idempotency_key, req.ttl_seconds):
        # Another writer won the race
        return EnqueueResponse(accepted=True, job_id=req.job_id, deduped=True)

    payload = req.dict()
    payload["received_at"] = datetime.utcnow().isoformat()
    payload["queue_name"] = os.getenv("LLM_PROXY_QUEUE_NAME", "llm_proxy_jobs")

    try:
        enqueue_job(payload, req.ttl_seconds, queue_name=payload["queue_name"])
    except Exception as e:
        # Roll back dedupe key on failure to enqueue
        conn.delete(f"llmproxy:dedupe:{req.job_id}:{req.idempotency_key}")
        raise HTTPException(status_code=500, detail={"error": {"type": "internal_server_error", "message": f"Failed to enqueue job: {e}", "code": "enqueue_failed"}})

    return EnqueueResponse(accepted=True, job_id=req.job_id, deduped=False)


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    """Proxy model list from model service"""
    model_service_url = os.getenv("MODEL_SERVICE_URL")
    internal_token = os.getenv("MODEL_SERVICE_TOKEN") or os.getenv("LLM_PROXY_INTERNAL_TOKEN")
    if not model_service_url:
        openai_error("internal_server_error", "MODEL_SERVICE_URL is not set; cannot list models.", 500)
    import httpx
    url = model_service_url.rstrip("/") + "/internal/model/models"
    headers = {}
    if internal_token:
        headers["X-Internal-Token"] = internal_token
    async with httpx.AsyncClient(timeout=float(os.getenv("MODEL_SERVICE_TIMEOUT", "60"))) as client:
        resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            openai_error("internal_server_error", f"Model service error {resp.status_code}: {resp.text}", 500)
        data = resp.json()
        models = []
        for m in data.get("models", []):
            models.append({
                "id": m.get("id"),
                "object": "model",
                "created": 0,
                "owned_by": "jarvis",
            })
        return ModelListResponse(object="list", data=models)


@app.get("/v1/health")
async def health():
    """Health proxy to model service"""
    model_service_url = os.getenv("MODEL_SERVICE_URL")
    internal_token = os.getenv("MODEL_SERVICE_TOKEN") or os.getenv("LLM_PROXY_INTERNAL_TOKEN")
    if not model_service_url:
        return {"status": "degraded", "reason": "MODEL_SERVICE_URL not set"}
    import httpx
    url = model_service_url.rstrip("/") + "/health"
    headers = {}
    if internal_token:
        headers["X-Internal-Token"] = internal_token
    async with httpx.AsyncClient(timeout=float(os.getenv("MODEL_SERVICE_TIMEOUT", "60"))) as client:
        resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            return {"status": "degraded", "reason": f"Model service error {resp.status_code}", "body": resp.text[:200]}
        data = resp.json()
        return {"status": "healthy", "model_service": data}


# Legacy routes removed per PRD:
# - /api/v{version:int}/chat
# - /api/v{version:int}/lightweight/chat
# - /api/v{version:int}/chat/conversation/{conversation_id}/warmup
# - /api/v{version:int}/lightweight/chat/conversation/{conversation_id}/warmup
# - /api/v{version:int}/model-swap
# - /api/v{version:int}/lightweight/model-swap
# - /api/v{version:int}/conversation/{conversation_id}/status
# - /api/v{version:int}/model/reset
# - /api/v{version:int}/lightweight/model/reset
# - /api/v{version:int}/debug-request
#
# All chat interactions now use /v1/chat/completions with model selection.
