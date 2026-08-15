"""GGUFClient._apply_no_think_prefill — the in-process thinking-off mechanism.

Two triggers, one behavior: a `/no_think` token in any user turn (Qwen3 prompt
providers) or an explicit per-request ``reasoning_budget=0`` (the backend-agnostic
contract queue jobs use). Either seeds an empty, already-closed ``<think>`` block
so the model skips its reasoning pass entirely.
"""

from backends.gguf_backend import GGUFClient

PREFILL = "<think>\n\n</think>\n\n"


def _user(content):
    return {"role": "user", "content": content}


class TestNoThinkPrefill:
    def test_no_think_token_triggers_prefill(self):
        out = GGUFClient._apply_no_think_prefill([_user("hi /no_think")])
        assert out[-1] == {"role": "assistant", "content": PREFILL}

    def test_reasoning_budget_zero_triggers_prefill_without_token(self):
        out = GGUFClient._apply_no_think_prefill([_user("hi")], reasoning_budget=0)
        assert out[-1] == {"role": "assistant", "content": PREFILL}

    def test_no_trigger_leaves_messages_unchanged(self):
        msgs = [_user("hi")]
        assert GGUFClient._apply_no_think_prefill(msgs) is msgs

    def test_unrestricted_budget_does_not_suppress(self):
        msgs = [_user("hi")]
        assert GGUFClient._apply_no_think_prefill(msgs, reasoning_budget=-1) is msgs

    def test_idempotent_when_caller_already_primed(self):
        msgs = [_user("hi"), {"role": "assistant", "content": PREFILL}]
        out = GGUFClient._apply_no_think_prefill(msgs, reasoning_budget=0)
        assert out == msgs

    def test_empty_messages_no_op(self):
        assert GGUFClient._apply_no_think_prefill([], reasoning_budget=0) == []
