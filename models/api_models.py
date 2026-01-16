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


class ChatCompletionRequest(APIModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = None
    response_format: Optional[ResponseFormat] = None


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
