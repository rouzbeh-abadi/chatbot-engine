"""The engine's startup contract.

The engine holds the provider credentials and has no notion of end users, so an
unauthenticated engine on a reachable port is an open provider account. That is
the one thing it must refuse to do.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from chatbot_engine.app import InsecureConfiguration, app
from chatbot_engine.settings import get_settings


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    """Set explicitly rather than trusted from this machine's .env."""
    get_settings.cache_clear()
    monkeypatch.setenv("ENGINE_API_KEY", "")
    yield monkeypatch
    get_settings.cache_clear()


def test_production_refuses_to_start_without_a_key(env: pytest.MonkeyPatch) -> None:
    env.setenv("ENGINE_ENV", "production")

    with pytest.raises(InsecureConfiguration, match="ENGINE_API_KEY"):
        with TestClient(app):
            pass


def test_local_still_starts_without_any_configuration(
    env: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    env.setenv("ENGINE_ENV", "local")

    with caplog.at_level("WARNING"):
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200

    assert "ENGINE_API_KEY" in caplog.text
