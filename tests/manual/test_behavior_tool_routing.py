"""Behavior lane: a real cheap cloud model must route utterances to the right tool.

This is the "does it actually work" signal (not plumbing): gpt-4.1-nano, given
function-calling tools through llm-proxy's REST backend (the native tool-calling
passthrough), must pick the correct tool for each voice utterance.

LIVE / on-demand only — needs an OpenAI key in JARVIS_REST_AUTH_TOKEN; skipped
otherwise. Excluded from normal CI (tests/manual/). Run:

    source ~/.jarvis/secrets/openai.env
    JARVIS_REST_PROVIDER=openai JARVIS_REST_AUTH_TYPE=bearer \
      .venv/bin/python -m pytest tests/manual/test_behavior_tool_routing.py -v

Assert tool *selection*, not prose (temperature=0 to minimize nondeterminism).
"""

from __future__ import annotations

import os

import pytest

from backends.rest_backend import RestClient
from managers.chat_types import GenerationParams, NormalizedMessage, TextPart


requires_key = pytest.mark.skipif(
    not os.getenv("JARVIS_REST_AUTH_TOKEN"),
    reason="JARVIS_REST_AUTH_TOKEN (OpenAI key) not set; behavior lane is live-only",
)

MODEL = os.getenv("JARVIS_REST_MODEL_NAME", "gpt-4.1-nano")
BASE_URL = os.getenv("JARVIS_REST_BASE_URL", "https://api.openai.com")

# Concise system prompt in the spirit of the ChatGPTOpenAI provider: tools are
# delivered natively, so they are NOT embedded here.
SYSTEM_PROMPT = (
    "You are Jarvis, a function-calling voice assistant. "
    "Use the provided functions to act on the user's request: call the single "
    "best-matching function with its required parameters. Do not ask follow-up "
    "questions; infer sensible defaults from the request."
)


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


TOOLS = [
    _tool(
        "set_timer",
        "Start a countdown timer.",
        {"duration_minutes": {"type": "integer", "description": "Length in minutes"}},
        ["duration_minutes"],
    ),
    _tool(
        "get_weather",
        "Get the current weather or forecast.",
        {"location": {"type": "string", "description": "City or place; optional"}},
        [],
    ),
    _tool(
        "control_device",
        "Control a smart-home device (lights, plugs, etc.).",
        {
            "device": {"type": "string", "description": "Device or room, e.g. 'kitchen lights'"},
            "action": {"type": "string", "description": "e.g. on, off, dim, set brightness"},
        },
        ["device", "action"],
    ),
    _tool(
        "play_music",
        "Play music or audio.",
        {"query": {"type": "string", "description": "Artist, genre, song, or playlist"}},
        ["query"],
    ),
    _tool("get_news", "Read the latest news headlines.", {}, []),
    _tool(
        "add_to_list",
        "Add an item to a list (shopping, todo, etc.).",
        {
            "item": {"type": "string", "description": "What to add"},
            "list_name": {"type": "string", "description": "Which list; optional"},
        },
        ["item"],
    ),
]

# (utterance, expected tool) — the behavior corpus.
CORPUS = [
    ("set a 10 minute timer", "set_timer"),
    ("set a timer for 5 minutes", "set_timer"),
    ("what's the weather like today", "get_weather"),
    ("is it going to rain tomorrow", "get_weather"),
    ("turn off the kitchen lights", "control_device"),
    ("dim the bedroom lights to 50 percent", "control_device"),
    ("play some jazz", "play_music"),
    ("what's in the news", "get_news"),
    ("add milk to my shopping list", "add_to_list"),
    ("put eggs on the grocery list", "add_to_list"),
]


def _route(utterance: str) -> tuple[str | None, str]:
    """Return (chosen_tool_name_or_None, raw_content) for an utterance."""
    client = RestClient(base_url=BASE_URL, model_name=MODEL, model_type="live")
    client.provider = "openai"
    client.auth_type = "bearer"
    client.auth_token = os.getenv("JARVIS_REST_AUTH_TOKEN", "")
    client.headers = client._setup_headers()

    messages = [
        NormalizedMessage(role="system", content=[TextPart(text=SYSTEM_PROMPT)]),
        NormalizedMessage(role="user", content=[TextPart(text=utterance)]),
    ]
    params = GenerationParams(
        temperature=0.0, max_tokens=256, tools=TOOLS, tool_choice="auto"
    )
    result = client.generate_text_chat(None, messages, params)
    if result.tool_calls:
        return result.tool_calls[0]["function"]["name"], result.content or ""
    return None, result.content or ""


@requires_key
@pytest.mark.parametrize("utterance,expected", CORPUS, ids=[c[0] for c in CORPUS])
def test_utterance_routes_to_expected_tool(utterance: str, expected: str) -> None:
    chosen, content = _route(utterance)
    assert chosen is not None, f"{utterance!r}: no tool call (said {content!r})"
    assert chosen == expected, f"{utterance!r} routed to {chosen!r}, expected {expected!r}"
