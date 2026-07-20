"""Tests for the REST backend's auth-token resolution.

These cover the DB-backed `rest.auth_token` setting (with the
`JARVIS_REST_AUTH_TOKEN` env var preserved as a fallback):

- DB setting is read first and used to build the bearer header
- env fallback when the DB setting is unset (back-compat / CI behavior lane)
- DB value takes precedence over the env var
- no token (neither DB nor env) -> no auth header injected

Header assembly happens synchronously in ``RestClient.__init__`` ->
``_setup_headers``; no network call is required.
"""

import os
from unittest.mock import patch

from backends.rest_backend import RestClient
from managers.chat_types import GenerationParams, NormalizedMessage, TextPart


class _StubSettings:
    """Minimal settings-service stub: ``.get(key)`` reads from a dict."""

    def __init__(self, values):
        self._values = values

    def get(self, key):
        return self._values.get(key)


def _patch_settings(values):
    """Patch the settings-service singleton ``get_setting`` resolves through."""
    return patch(
        "services.settings_helpers._get_settings_service",
        return_value=_StubSettings(values),
    )


class TestRestClientAuthToken:
    """Tests for RestClient auth-token sourcing (DB setting + env fallback)."""

    def test_rest_client_reads_auth_token_from_db_setting(self):
        """RestClient sources the token from the DB setting and builds the bearer header."""
        values = {"rest.auth_token": "db-key-123", "rest.auth_type": "bearer"}
        with _patch_settings(values), patch.dict(os.environ, {}, clear=True):
            client = RestClient(base_url="https://api.openai.com")

        assert client.auth_token == "db-key-123"
        assert client.headers["Authorization"] == "Bearer db-key-123"

    def test_rest_client_falls_back_to_env_auth_token_when_setting_unset(self):
        """Back-compat: env-only deployments / the CI behavior lane keep working."""
        with _patch_settings({}), patch.dict(
            os.environ,
            {"JARVIS_REST_AUTH_TOKEN": "env-key-456", "JARVIS_REST_AUTH_TYPE": "bearer"},
            clear=True,
        ):
            client = RestClient(base_url="https://api.openai.com")

        assert client.auth_token == "env-key-456"
        assert client.headers["Authorization"] == "Bearer env-key-456"

    def test_rest_client_db_setting_overrides_env_auth_token(self):
        """DB-first precedence matches get_setting (DB beats env)."""
        values = {"rest.auth_token": "db-wins", "rest.auth_type": "bearer"}
        with _patch_settings(values), patch.dict(
            os.environ, {"JARVIS_REST_AUTH_TOKEN": "env-loses"}, clear=True
        ):
            client = RestClient(base_url="https://api.openai.com")

        assert client.auth_token == "db-wins"

    def test_rest_client_no_auth_token_yields_no_auth_header(self):
        """Neither DB nor env set -> no auth header injected (auth_type defaults to 'none')."""
        with _patch_settings({}), patch.dict(os.environ, {}, clear=True):
            client = RestClient(base_url="https://api.openai.com")

        assert client.auth_token == ""
        assert "Authorization" not in client.headers


class TestSyncBridgeLoopStability:
    """The persistent ``httpx.AsyncClient`` must stay bound to ONE event loop.

    Regression: chat_runner offloads sync generations with ``asyncio.to_thread``
    (#47). ``generate_text_chat`` used to ask "is a loop running?" as a proxy
    for "is my caller async, so is my client persistent?". Inside a to_thread
    worker no loop is running, so every model-service call ran on a throwaway
    ``asyncio.run`` loop. The first call bound the connection pool to a loop
    that was then closed, so the NEXT call died during cleanup with
    "RuntimeError: Event loop is closed" — the alternating 200/500/200/500 in
    the CI behavior corpus (2026-07-20).

    These drive ``generate_text_chat`` itself, so they fail on the heuristic.
    """

    def _backend(self):
        with _patch_settings({}):
            return RestClient("http://example.invalid/v1")

    @staticmethod
    def _msg(text="hi"):
        return NormalizedMessage(role="user", content=[TextPart(text=text)])

    def _capturing_backend(self, loops):
        """Backend whose async leaf records the loop it actually ran on."""
        import asyncio as _asyncio

        backend = self._backend()

        async def _fake_chat(dict_messages, temperature):
            loops.append(id(_asyncio.get_running_loop()))
            return "ok"

        backend.chat_with_temperature = _fake_chat
        backend.last_usage = None
        return backend

    def test_sequential_calls_from_to_thread_share_one_loop(self):
        """Ten calls in the exact shape chat_runner uses. The bug alternated,
        so anything fewer than two calls would have passed."""
        import asyncio as _asyncio

        loops: list[int] = []
        backend = self._capturing_backend(loops)
        params = GenerationParams(temperature=0.1)

        async def _drive():
            for _ in range(10):
                await _asyncio.to_thread(
                    backend.generate_text_chat, None, [self._msg()], params
                )

        _asyncio.run(_drive())

        assert len(loops) == 10
        assert len(set(loops)) == 1, f"alternated across loops: {loops}"

    def test_loop_is_not_closed_between_calls(self):
        """The pooled connection is poisoned by the loop CLOSING, so assert the
        loop the work ran on is still alive afterwards."""
        import asyncio as _asyncio

        loops: list[int] = []
        backend = self._capturing_backend(loops)
        params = GenerationParams(temperature=0.1)

        async def _drive():
            await _asyncio.to_thread(
                backend.generate_text_chat, None, [self._msg()], params
            )

        _asyncio.run(_drive())

        bg = backend._bg_loop
        assert bg is not None, "work did not run on the dedicated loop"
        assert not bg.is_closed()
        assert id(bg) == loops[0]

    def test_plain_sync_caller_uses_the_same_loop(self):
        """A sync caller (tests, scripts) must not get a different loop than
        the async path — one client, one loop, always."""
        loops: list[int] = []
        backend = self._capturing_backend(loops)
        params = GenerationParams(temperature=0.1)

        backend.generate_text_chat(None, [self._msg()], params)
        backend.generate_text_chat(None, [self._msg()], params)

        # Identity alone is not enough: CPython reuses the id() of a
        # garbage-collected throwaway loop, so a buggy run can coincidentally
        # report one id. Assert the work landed on the DEDICATED loop, which
        # only happens when the bridge routes there.
        bg = backend._bg_loop
        assert bg is not None, "sync caller bypassed the dedicated loop"
        assert not bg.is_closed()
        assert loops == [id(bg), id(bg)], f"sync caller alternated: {loops}"
