from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Literal, Annotated

from pydantic import BaseModel, Field, ConfigDict


class APIModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class ImageUrl(APIModel):
    url: str
    detail: Optional[str] = None


class TextContentPart(APIModel):
    type: Literal["text"] = "text"
    text: str


class ImageContentPart(APIModel):
    type: Literal["image_url"] = "image_url"
    image_url: ImageUrl


ContentPart = Annotated[Union[TextContentPart, ImageContentPart], Field(discriminator="type")]


class Message(APIModel):
    role: str
    content: Union[str, List[ContentPart]]


class ResponseFormat(APIModel):
    type: str
    json_schema: Optional[Dict[str, Any]] = None


class AdapterSettings(APIModel):
    """Settings for per-request LoRA adapter loading.

    Enables dynamic adapter selection on a per-node basis. The adapter is
    identified by its hash (from the adapter store) and can be tuned with
    optional parameters.

    Attributes:
        hash: Required. Unique identifier for the adapter (e.g., "b2b8ccb4").
              Used to locate the adapter in S3/cache.
        scale: LoRA scaling factor. Higher values = stronger adapter influence.
               Default 1.0 means full adapter strength.
        enabled: Toggle to easily disable adapter without removing the object.
                 Useful for A/B testing or quick fallback to base model.
    """
    hash: str
    scale: float = 1.0
    enabled: bool = True


class ChatCompletionRequest(APIModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = None
    response_format: Optional[ResponseFormat] = None
    # Jarvis extensions (OpenAI-compatible: extra fields are allowed)
    include_date_context: Optional[bool] = None  # If true, extract date keys from input
    adapter_settings: Optional[AdapterSettings] = None  # Per-request LoRA adapter selection


class ChatCompletionChoice(APIModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None


class Usage(APIModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(APIModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage
    # Jarvis extensions (OpenAI-compatible: extra fields are allowed)
    date_keys: Optional[List[str]] = None  # Extracted date keys when include_date_context=true


class ErrorDetail(APIModel):
    type: str
    message: str
    code: Optional[str] = None


class ErrorResponse(APIModel):
    error: ErrorDetail


class ModelInfo(APIModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "jarvis"


class ModelListResponse(APIModel):
    object: str = "list"
    data: List[ModelInfo]
