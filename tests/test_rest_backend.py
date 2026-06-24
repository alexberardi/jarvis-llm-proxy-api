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
