"""Worker callback dispatch (_send_callback) — auth-type handling.

Regression for the 2026-08-07 incident: CC's async-job enqueues set
callback.auth_type="bearer" + a token so the fail-closed /…/callback endpoints
authenticate, but _send_callback dispatched ONLY on cb_type=="internal" — a real
"bearer" callback fell to the unsupported branch and NEVER posted, so every
memory-extraction / deep-research / characterization / adapter callback silently
dropped (nothing persisted; transcripts never drained). Both "internal" (empty
auth_type) and "bearer" must POST; the bearer header is applied when a token is present.
"""
from unittest.mock import patch

from queues.tasks import _send_callback


def _capture_client(captured: dict):
    class _Resp:
        status_code = 200
        text = "ok"

    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = headers or {}
            captured["json"] = json
            return _Resp()

    return _Client


def _send(callback: dict, captured: dict):
    with patch("queues.tasks.httpx.Client", _capture_client(captured)):
        return _send_callback(
            callback=callback,
            job_id="job-1",
            job_type="chat",
            status="succeeded",
            error=None,
            result={"content": "[]"},
            timing={"processing_ms": 5},
            trace_id="trace-1",
            metadata={"type": "memory_extraction"},
        )


class TestSendCallbackAuthType:
    def test_bearer_callback_posts_with_authorization_header(self):
        """auth_type='bearer' must POST (not 'unsupported type') and attach the token."""
        captured: dict = {}
        env = _send(
            {"url": "http://cc/api/v0/memory-extraction/callback", "auth_type": "bearer", "token": "tok123"},
            captured,
        )
        assert captured.get("url") == "http://cc/api/v0/memory-extraction/callback", (
            f"bearer callback never POSTed — {env.get('callback_reason')}"
        )
        assert captured["headers"].get("Authorization") == "Bearer tok123"
        assert env.get("callback_status") == 200

    def test_internal_callback_still_posts(self):
        """Empty auth_type -> cb_type 'internal' -> still posts (no auth header)."""
        captured: dict = {}
        env = _send({"url": "http://cc/callback"}, captured)
        assert captured.get("url") == "http://cc/callback"
        assert "Authorization" not in captured["headers"]
        assert env.get("callback_status") == 200

    def test_truly_unsupported_type_is_rejected(self):
        """A genuinely unknown auth_type still fails closed (no post)."""
        captured: dict = {}
        env = _send({"url": "http://cc/callback", "auth_type": "quantum", "token": "x"}, captured)
        assert captured.get("url") is None, "unknown auth_type must NOT post"
        assert env.get("callback_status") == "error"
