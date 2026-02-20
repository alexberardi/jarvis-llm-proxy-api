"""Response helper functions.

Utilities for creating OpenAI-compatible responses and error handling.
"""

import time
import uuid
from typing import Dict, List, Optional

from fastapi import HTTPException

from models.api_models import (
    ChatCompletionChoice,
    ChatCompletionResponse,
    Message,
    Usage,
)


def create_openai_response(
    content: str,
    model_name: str,
    usage: Optional[Dict] = None,
    date_keys: Optional[List[str]] = None,
    tool_calls: Optional[List[Dict]] = None,
    finish_reason: Optional[str] = None,
) -> ChatCompletionResponse:
    """Create an OpenAI-style chat completion response."""
    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    if usage is None:
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # Determine finish_reason: native tool_calls override
    if finish_reason is None:
        finish_reason = "tool_calls" if tool_calls else "stop"

    # Build message with optional tool_calls
    message_kwargs: Dict = {"role": "assistant", "content": content}
    if tool_calls:
        from models.api_models import ToolCall, FunctionCall
        message_kwargs["tool_calls"] = [
            ToolCall(
                id=tc.get("id", f"call_{uuid.uuid4().hex[:12]}"),
                function=FunctionCall(
                    name=tc.get("function", {}).get("name", ""),
                    arguments=tc.get("function", {}).get("arguments", "{}"),
                ),
            )
            for tc in tool_calls
        ]

    return ChatCompletionResponse(
        id=response_id,
        created=created,
        model=model_name,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=Message(**message_kwargs),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(**usage),
        date_keys=date_keys,
    )


def openai_error(error_type: str, message: str, status_code: int = 400):
    """Create an OpenAI-style error response.

    Raises HTTPException with the appropriate error format.
    """
    raise HTTPException(
        status_code=status_code,
        detail={
            "error": {
                "type": error_type,
                "message": message,
                "code": None,
            }
        },
    )
