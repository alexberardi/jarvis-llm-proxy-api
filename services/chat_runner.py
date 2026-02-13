import asyncio
import base64
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from services.settings_helpers import get_setting

logger = logging.getLogger("uvicorn")

from managers.chat_types import (
    NormalizedMessage,
    TextPart,
    ImagePart,
    GenerationParams,
    ChatResult,
)
from models.api_models import Message, ChatCompletionRequest
from services.json_grammar import schema_to_gbnf


# JSON system message to inject when response_format is json_object
JSON_SYSTEM_MESSAGE = """You must respond with valid JSON only. Do not include any text before or after the JSON.
Do not wrap the JSON in markdown code blocks.
Output compact JSON without unnecessary whitespace.
Ensure all strings are properly escaped and all brackets are balanced."""


def openai_error(error_type: str, message: str, status_code: int = 400):
    """Raise an HTTPException in OpenAI-style shape."""
    logger.error(f"🚨 OpenAI error: type={error_type}, status={status_code}, message={message[:200]}")
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


def parse_data_url(data_url: str) -> tuple[bytes, str]:
    match = re.match(r'^data:([^;]+);base64,(.+)$', data_url)
    if not match:
        raise ValueError("Invalid data URL format. Expected: data:<mime_type>;base64,<data>")
    mime_type = match.group(1)
    b64_data = match.group(2)
    image_bytes = base64.b64decode(b64_data)
    return image_bytes, mime_type


def normalize_message(message: Message) -> NormalizedMessage:
    content_parts: List[Any] = []
    if isinstance(message.content, str):
        content_parts.append(TextPart(text=message.content))
    elif isinstance(message.content, list):
        for part in message.content:
            if part.type == "text" and part.text is not None:
                content_parts.append(TextPart(text=part.text))
            elif part.type == "image_url" and part.image_url is not None:
                url = part.image_url.url
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
    return [normalize_message(msg) for msg in messages]


def has_images(messages: List[NormalizedMessage]) -> bool:
    for msg in messages:
        for part in msg.content:
            if isinstance(part, ImagePart):
                return True
    return False


def inject_json_system_message(messages: List[NormalizedMessage]) -> List[NormalizedMessage]:
    has_system = any(msg.role == "system" for msg in messages)
    if has_system:
        augmented = []
        for msg in messages:
            if msg.role == "system":
                existing_text = ""
                for part in msg.content:
                    if isinstance(part, TextPart):
                        existing_text += part.text + "\n"
                if "JSON" not in existing_text.upper() or "valid json" not in existing_text.lower():
                    augmented_text = existing_text.strip() + "\n\n" + JSON_SYSTEM_MESSAGE
                else:
                    augmented_text = existing_text
                augmented.append(NormalizedMessage(
                    role="system",
                    content=[TextPart(text=augmented_text)]
                ))
            else:
                augmented.append(msg)
        return augmented
    else:
        system_msg = NormalizedMessage(
            role="system",
            content=[TextPart(text=JSON_SYSTEM_MESSAGE)]
        )
        return [system_msg] + messages


def is_json_truncated(content: str) -> bool:
    stripped = content.strip()
    if not stripped:
        return False
    if not (stripped.startswith('{') or stripped.startswith('[')):
        return False
    if stripped.startswith('{') and not stripped.endswith('}'):
        return True
    if stripped.startswith('[') and not stripped.endswith(']'):
        return True
    if re.search(r':\s*$', stripped):
        return True
    open_braces = stripped.count('{') - stripped.count('}')
    open_brackets = stripped.count('[') - stripped.count(']')
    if open_braces > 0 or open_brackets > 0:
        return True
    return False


def repair_duplicate_keys(content: str) -> Optional[str]:
    try:
        def object_pairs_hook(pairs):
            obj = {}
            for k, v in pairs:
                if k in obj:
                    if not isinstance(obj[k], list):
                        obj[k] = [obj[k]]
                    obj[k].append(v)
                else:
                    obj[k] = v
            return obj

        parsed = json.loads(content, object_pairs_hook=object_pairs_hook)
        return json.dumps(parsed, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def repair_unescaped_quotes(content: str) -> Optional[str]:
    try:
        repaired = re.sub(r'(?<!\\)"([^"\\]*(\\.[^"\\]*)*)"', lambda m: m.group(0), content)
        repaired = repaired.replace('\\"', '"').replace('"', '\\"')
        repaired = '"' + repaired + '"'
        repaired = repaired.replace('"\\"', '"').replace('\\""', '"')
        repaired_json = json.loads(f'[{repaired}]')[0]
        return json.dumps(repaired_json)
    except (json.JSONDecodeError, TypeError, ValueError, re.error):
        return None


def extract_json_from_text(content: str) -> Optional[str]:
    try:
        matches = list(re.finditer(r'\{[\s\S]*\}|\[[\s\S]*\]', content))
        for match in matches:
            candidate = match.group()
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue
    except (re.error, TypeError):
        return None
    return None


def repair_truncated_json(content: str) -> Optional[str]:
    try:
        last_brace = max(content.rfind('}'), content.rfind(']'))
        if last_brace != -1:
            truncated = content[:last_brace+1]
            try:
                json.loads(truncated)
                return truncated
            except json.JSONDecodeError:
                pass
        repaired = content
        repaired = re.sub(r',\s*([\}\]])', r'\1', repaired)
        repaired = re.sub(r',\s*$', '', repaired)
        open_braces = repaired.count('{') - repaired.count('}')
        open_brackets = repaired.count('[') - repaired.count(']')
        repaired += '}' * max(0, open_braces) + ']' * max(0, open_brackets)
        json.loads(repaired)
        return repaired
    except (json.JSONDecodeError, TypeError, ValueError, re.error):
        return None


def parse_json_response(content: str) -> tuple[str, bool]:
    try:
        parsed = json.loads(content)
        repaired_duplicates = repair_duplicate_keys(content)
        if repaired_duplicates and repaired_duplicates != content:
            return repaired_duplicates, True
        return content, True
    except json.JSONDecodeError as e:
        error_str = str(e)
        if "Expecting" in error_str and ("delimiter" in error_str or "property name" in error_str):
            repaired = repair_unescaped_quotes(content)
            if repaired:
                try:
                    json.loads(repaired)
                    return repaired, True
                except json.JSONDecodeError:
                    pass
        if "Unterminated string" in error_str:
            repaired = repair_truncated_json(content)
            if repaired:
                try:
                    json.loads(repaired)
                    return repaired, True
                except json.JSONDecodeError:
                    pass
    repaired_duplicates = repair_duplicate_keys(content)
    if repaired_duplicates:
        try:
            json.loads(repaired_duplicates)
            return repaired_duplicates, True
        except json.JSONDecodeError:
            pass
    extracted = extract_json_from_text(content)
    if extracted:
        try:
            final_extracted = repair_duplicate_keys(extracted) or extracted
            json.loads(final_extracted)
            return final_extracted, True
        except json.JSONDecodeError as e:
            repaired_quotes = repair_unescaped_quotes(extracted)
            if repaired_quotes:
                try:
                    final_repaired = repair_duplicate_keys(repaired_quotes) or repaired_quotes
                    json.loads(final_repaired)
                    return final_repaired, True
                except json.JSONDecodeError:
                    pass
            if "Unterminated string" in str(e):
                repaired = repair_truncated_json(extracted)
                if repaired:
                    try:
                        final_repaired = repair_duplicate_keys(repaired) or repaired
                        json.loads(final_repaired)
                        return final_repaired, True
                    except json.JSONDecodeError:
                        pass
    repaired_quotes = repair_unescaped_quotes(content)
    if repaired_quotes:
        try:
            final_repaired = repair_duplicate_keys(repaired_quotes) or repaired_quotes
            json.loads(final_repaired)
            return final_repaired, True
        except json.JSONDecodeError:
            pass
    repaired = repair_truncated_json(content)
    if repaired:
        try:
            json.loads(repaired)
            return repaired, True
        except json.JSONDecodeError:
            pass
    return content, False


def _matches_type(value: Any, schema_type: str) -> bool:
    if schema_type == "null":
        return value is None
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return (isinstance(value, int) and not isinstance(value, bool)) or isinstance(value, float)
    if schema_type == "object":
        return isinstance(value, dict)
    if schema_type == "array":
        return isinstance(value, list)
    return True


def validate_json_schema(value: Any, schema: Dict[str, Any], path: str = "$") -> Optional[str]:
    if not schema:
        return None

    schema_type = schema.get("type")
    if schema_type is not None:
        if isinstance(schema_type, list):
            if not any(_matches_type(value, t) for t in schema_type):
                return f"{path} expected type {schema_type}, got {type(value).__name__}"
        else:
            if not _matches_type(value, schema_type):
                return f"{path} expected type {schema_type}, got {type(value).__name__}"

    if schema_type == "object" or (schema_type is None and isinstance(value, dict)):
        if not isinstance(value, dict):
            return f"{path} expected object"
        required = schema.get("required") or []
        for key in required:
            if key not in value:
                return f"{path}.{key} is required"
        properties = schema.get("properties") or {}
        for key, prop_schema in properties.items():
            if key in value:
                error = validate_json_schema(value[key], prop_schema, f"{path}.{key}")
                if error:
                    return error

    if schema_type == "array" or (schema_type is None and isinstance(value, list)):
        if not isinstance(value, list):
            return f"{path} expected array"
        item_schema = schema.get("items")
        if item_schema:
            for idx, item in enumerate(value):
                error = validate_json_schema(item, item_schema, f"{path}[{idx}]")
                if error:
                    return error

    return None


def summarize_json_schema(schema: Optional[Dict[str, Any]]) -> str:
    if not schema:
        return ""
    properties = schema.get("properties") or {}
    required = schema.get("required") or []
    prop_types = []
    for name, prop_schema in properties.items():
        prop_type = prop_schema.get("type", "any")
        prop_types.append(f"{name}:{prop_type}")
    summary = "Required keys: " + ", ".join(required) if required else "Required keys: (none)"
    if prop_types:
        summary += "\nAllowed keys: " + ", ".join(prop_types)
    return summary


async def fix_json_with_retry(
    backend: Any,
    model_config: Any,
    normalized_messages: List[NormalizedMessage],
    params: GenerationParams,
    invalid_content: str,
    response_schema: Optional[Dict[str, Any]] = None,
    max_retries: int = 1,
) -> Optional[str]:
    estimated_output_tokens = len(invalid_content) // 3
    original_max = params.max_tokens or 4096
    buffer_tokens = max(2000, estimated_output_tokens // 2)
    retry_max_tokens = max(int(original_max * 1.5), estimated_output_tokens + buffer_tokens, 8192)
    retry_max_tokens = min(retry_max_tokens, 16384)

    truncation_note = ""
    if is_json_truncated(invalid_content):
        truncation_note = " The response was truncated. Please provide a COMPLETE response."

    schema_hint = summarize_json_schema(response_schema)
    schema_block = f"\n\nSchema constraints:\n{schema_hint}" if schema_hint else ""
    correction_text = f"""The previous response was not valid JSON or did not match the required schema.{truncation_note}

Please provide ONLY valid JSON with no explanation text. Ensure:
- All brackets and braces are properly closed
- All strings are properly quoted and escaped  
- The response is complete
{schema_block}

Previous (truncated/invalid) response preview:
{invalid_content[:800]}

Return the corrected, complete JSON:"""

    correction_message = NormalizedMessage(
        role="user",
        content=[
            TextPart(text=correction_text)
        ]
    )

    retry_messages = normalized_messages + [correction_message]

    retry_params = GenerationParams(
        temperature=max(0.3, params.temperature - 0.2),
        max_tokens=retry_max_tokens,
        stream=params.stream,
        response_format=params.response_format,
    )

    for attempt in range(max_retries):
        try:
            logger.warning(f"⚠️ JSON invalid, retrying ({attempt + 1}/{max_retries})...")
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
                if response_schema:
                    parsed = json.loads(fixed_content)
                    schema_error = validate_json_schema(parsed, response_schema)
                    if schema_error:
                        logger.warning(f"⚠️ JSON schema validation failed on retry: {schema_error}")
                        continue
                return fixed_content
        except (json.JSONDecodeError, TypeError, ValueError, RuntimeError) as e:
            logger.debug(f"fix_json_with_retry attempt failed: {e}")
            continue
    return None


async def run_chat_completion(
    model_manager: Any,
    req: ChatCompletionRequest,
    allow_images: bool = True,
) -> ChatResult:
    model_config = model_manager.get_model_config(req.model)
    if not model_config:
        openai_error(
            "model_not_found",
            f"Model '{req.model}' does not exist. Available models: {list(model_manager.registry.keys())} and aliases: {list(model_manager.aliases.keys())}",
            404,
        )

    normalized_messages = normalize_messages(req.messages)

    requires_json = False

    has_images_flag = has_images(normalized_messages)
    if has_images_flag and not allow_images:
        openai_error("invalid_request_error", "Images are not allowed for this request")

    if has_images_flag and not model_config.supports_images:
        openai_error(
            "invalid_request_error",
            f"Model '{req.model}' does not support images. Use a vision-capable model instead.",
        )

    backend = model_config.backend_instance
    response_format_dict = None
    if req.response_format:
        response_format_dict = req.response_format.model_dump(exclude_none=True)
        logger.debug(f"🔎 response_format={response_format_dict}")
        if response_format_dict.get("type") == "json_object":
            requires_json = True

    if requires_json:
        normalized_messages = inject_json_system_message(normalized_messages)

    response_schema = None
    if requires_json and response_format_dict:
        response_schema = response_format_dict.get("json_schema")

    grammar = None
    if (
        response_schema
        and model_config.backend_type == "GGUF"
        and hasattr(model_config.backend_instance, "inference_engine")
        and model_config.backend_instance.inference_engine == "llama_cpp"
    ):
        try:
            grammar = schema_to_gbnf(response_schema)
            dump_path = get_setting(
                "debug.dump_gbnf_path", "JARVIS_DUMP_GBNF_PATH", ""
            )
            if dump_path:
                try:
                    with open(dump_path, "w", encoding="utf-8") as handle:
                        handle.write(grammar)
                    logger.debug(f"🧩 GGUF grammar dumped to {dump_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to dump GGUF grammar: {e}")
            logger.info("🧩 GGUF JSON grammar enabled for this request")
        except Exception as e:
            logger.warning(f"⚠️ Failed to build GGUF JSON grammar: {e}")

    # Extract adapter settings if provided and enabled
    adapter_settings_dict = None
    if req.adapter_settings and req.adapter_settings.enabled:
        adapter_settings_dict = req.adapter_settings.model_dump()
        logger.info(f"🧩 Adapter settings: hash={req.adapter_settings.hash}, scale={req.adapter_settings.scale}")

    params = GenerationParams(
        temperature=req.temperature or 0.7,
        max_tokens=req.max_tokens,
        stream=req.stream or False,
        response_format=response_format_dict,
        grammar=grammar,
        adapter_settings=adapter_settings_dict,
    )

    try:
        start_time = time.time()
        if has_images_flag:
            if not hasattr(backend, "generate_vision_chat"):
                openai_error(
                    "internal_server_error",
                    f"Backend '{model_config.backend_type}' does not support vision (missing generate_vision_chat method).",
                    500,
                )
            if asyncio.iscoroutinefunction(backend.generate_vision_chat):
                result: ChatResult = await backend.generate_vision_chat(
                    model_config, normalized_messages, params
                )
            else:
                result: ChatResult = backend.generate_vision_chat(
                    model_config, normalized_messages, params
                )
        else:
            if hasattr(backend, "generate_text_chat"):
                if asyncio.iscoroutinefunction(backend.generate_text_chat):
                    result: ChatResult = await backend.generate_text_chat(
                        model_config, normalized_messages, params
                    )
                else:
                    result: ChatResult = backend.generate_text_chat(
                        model_config, normalized_messages, params
                    )
            else:
                legacy_messages = []
                for msg in normalized_messages:
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

        final_content = result.content

        if requires_json:
            repaired_content, json_valid = parse_json_response(final_content)
            schema_error = None
            if json_valid:
                if response_schema:
                    try:
                        parsed = json.loads(repaired_content)
                        schema_error = validate_json_schema(parsed, response_schema)
                    except Exception as e:
                        schema_error = f"invalid JSON: {e}"
                    if schema_error:
                        logger.warning(f"⚠️ JSON schema validation failed: {schema_error}")
                    else:
                        final_content = repaired_content
                else:
                    final_content = repaired_content

            if not json_valid or schema_error:
                fixed = await fix_json_with_retry(
                    backend,
                    model_config,
                    normalized_messages,
                    params,
                    invalid_content=final_content,
                    response_schema=response_schema,
                    max_retries=1,
                )
                if fixed:
                    final_content = fixed
                else:
                    openai_error(
                        "invalid_response_error",
                        f"Model returned invalid JSON response. {schema_error or ''}".strip(),
                        500,
                    )

        usage = result.usage
        end_time = time.time()
        # store timing? currently unused
        return ChatResult(content=final_content, usage=usage)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Internal error during chat completion: {e}")
        openai_error("internal_server_error", f"Internal error: {str(e)}", 500)

