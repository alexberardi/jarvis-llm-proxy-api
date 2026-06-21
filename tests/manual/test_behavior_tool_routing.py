"""Behavior lane: a real cheap cloud model must route utterances to the right
tool AND fill its arguments correctly.

This is the "does it actually work" signal (not plumbing): gpt-4.1-nano, given
function-calling tools through llm-proxy's REST backend (the native tool-calling
passthrough), must (a) pick the correct tool for each voice utterance, (b)
populate the arguments sensibly, and (c) NOT call a tool on small talk.

The corpus and tool schemas live in ``behavior/`` as provider-agnostic YAML so
the same expectations can later (T6) be routed through command-center's real
``ChatGPTOpenAI`` provider, not just this inlined toolset.

LIVE / on-demand only — needs an OpenAI key in JARVIS_REST_AUTH_TOKEN; skipped
otherwise. Excluded from normal CI (tests/manual/). Run:

    source ~/.jarvis/secrets/openai.env
    JARVIS_REST_PROVIDER=openai JARVIS_REST_AUTH_TYPE=bearer \
      .venv/bin/python -m pytest tests/manual/test_behavior_tool_routing.py -v

The model is PINNED to a dated snapshot (gpt-4.1-nano-2025-04-14) for
reproducibility; override with JARVIS_REST_MODEL_NAME. temperature=0 minimizes
nondeterminism. Assertions check tool *selection* and *argument shape*, never
exact prose.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - surfaced by the guard test below
    yaml = None

from backends.rest_backend import RestClient
from managers.chat_types import GenerationParams, NormalizedMessage, TextPart


requires_key = pytest.mark.skipif(
    not os.getenv("JARVIS_REST_AUTH_TOKEN"),
    reason="JARVIS_REST_AUTH_TOKEN (OpenAI key) not set; behavior lane is live-only",
)

# Pinned snapshot for reproducibility (floating "gpt-4.1-nano" can drift mid-run).
MODEL = os.getenv("JARVIS_REST_MODEL_NAME", "gpt-4.1-nano-2025-04-14")
BASE_URL = os.getenv("JARVIS_REST_BASE_URL", "https://api.openai.com")

# Concise system prompt in the spirit of the ChatGPTOpenAI provider: tools are
# delivered natively, so they are NOT embedded here. It mandates a call for any
# actionable request (omitting optional params the user didn't give, never asking
# a follow-up) while still allowing a direct answer for pure small talk — so the
# negatives are a real signal, not forced calls.
SYSTEM_PROMPT = (
    "You are Jarvis, a hands-free voice assistant that acts through function "
    "calls. For any actionable request or question a function can handle, you "
    "MUST immediately call the single best-matching function. Omit optional "
    "parameters the user did not specify — the system fills in defaults such as "
    "the user's location — and NEVER ask a follow-up question for an actionable "
    "request. Only when the request is pure small talk that no function "
    "addresses (a greeting, thanks, or chit-chat) should you answer directly "
    "without calling a function."
)

_BEHAVIOR_DIR = Path(__file__).parent / "behavior"


def _load_yaml(name: str) -> list:
    if yaml is None:
        return []
    path = _BEHAVIOR_DIR / name
    with path.open() as fh:
        return yaml.safe_load(fh) or []


TOOLS = _load_yaml("tools.yaml")
CORPUS = _load_yaml("corpus.yaml")


def _route(utterance: str) -> tuple[str | None, dict, str]:
    """Return (chosen_tool_name_or_None, parsed_arguments, raw_content)."""
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
    if not result.tool_calls:
        return None, {}, result.content or ""

    call = result.tool_calls[0]["function"]
    raw_args = call.get("arguments")
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            args = {}
    else:
        args = raw_args or {}
    return call["name"], args, result.content or ""


def _as_number(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _check_arg(name: str, matcher: dict, args: dict) -> str | None:
    """Return an error message if the arg fails its matcher, else None."""
    if name not in args:
        return f"arg {name!r} missing (got {sorted(args)})"
    value = args[name]
    sval = str(value).strip().lower()

    if "equals" in matcher:
        expected = matcher["equals"]
        num_v, num_e = _as_number(value), _as_number(expected)
        if num_v is not None and num_e is not None:
            if num_v != num_e:
                return f"{name}={value!r} != equals {expected!r}"
        elif sval != str(expected).strip().lower():
            return f"{name}={value!r} != equals {expected!r}"
    if "contains" in matcher:
        if str(matcher["contains"]).strip().lower() not in sval:
            return f"{name}={value!r} does not contain {matcher['contains']!r}"
    if "in" in matcher:
        opts = [str(o).strip().lower() for o in matcher["in"]]
        if sval not in opts:
            return f"{name}={value!r} not in {matcher['in']!r}"
    if "any_of" in matcher:
        opts = [str(o).strip().lower() for o in matcher["any_of"]]
        if not any(o in sval for o in opts):
            return f"{name}={value!r} contains none of {matcher['any_of']!r}"
    return None


@requires_key
def test_corpus_and_tools_loaded() -> None:
    # Guard: in live/nightly context pyyaml is installed; a load failure here is
    # a real problem rather than a silently-empty parametrization.
    assert TOOLS, "tools.yaml failed to load (is pyyaml installed?)"
    assert CORPUS, "corpus.yaml failed to load (is pyyaml installed?)"


@requires_key
@pytest.mark.parametrize("entry", CORPUS, ids=[e["utterance"] for e in CORPUS])
def test_utterance_routing(entry: dict) -> None:
    utterance = entry["utterance"]
    expected_tool = entry.get("tool")
    chosen, args, content = _route(utterance)

    if expected_tool is None:
        assert chosen is None, (
            f"{utterance!r}: expected NO tool call, but routed to {chosen!r} "
            f"args={args!r}"
        )
        return

    assert chosen is not None, f"{utterance!r}: no tool call (said {content!r})"
    assert chosen == expected_tool, (
        f"{utterance!r} routed to {chosen!r}, expected {expected_tool!r}"
    )
    for arg_name, matcher in (entry.get("args") or {}).items():
        err = _check_arg(arg_name, matcher, args)
        assert err is None, f"{utterance!r} -> {chosen}: {err}"
