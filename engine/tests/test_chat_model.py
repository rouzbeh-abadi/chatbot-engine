"""The chat model, and the settings it reads.

Mostly negative tests: a missing key must not stop the engine from starting, or
block the routes that need no model.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from chatbot_engine.api.dependencies import reset_dependency_cache
from chatbot_engine.agent.client import build_chat_model
from chatbot_engine.errors import NotConfiguredError
from chatbot_engine.models.chat import AssistantConfig
from chatbot_engine.rag.splitter import DocumentChunker
from chatbot_engine.settings import Settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    """The repo's own .env would otherwise leak a real key into these tests."""
    monkeypatch.setenv("ENGINE_OPENROUTER_API_KEY", "")
    reset_dependency_cache()

    yield

    reset_dependency_cache()


# --- building the model ------------------------------------------------------


def _config(**overrides: object) -> AssistantConfig:
    base = {"project_id": "support", "name": "Support", "system_prompt": "p"}

    return AssistantConfig(**(base | overrides))


def test_a_missing_key_is_reported_not_raised_at_import() -> None:
    """Name the variable -- this is the error people hit on day one."""
    with pytest.raises(NotConfiguredError, match="ENGINE_OPENROUTER_API_KEY"):
        build_chat_model(_config(), Settings(openrouter_api_key=None))


def test_a_blank_key_counts_as_missing() -> None:
    """`ENGINE_OPENROUTER_API_KEY=` in .env arrives as "", not None."""
    assert Settings(openrouter_api_key="").openrouter_api_key is None

    with pytest.raises(NotConfiguredError):
        build_chat_model(_config(), Settings(openrouter_api_key=""))


def test_the_assistant_model_wins_and_the_engine_default_fills_in() -> None:
    settings = Settings(openrouter_api_key="k")

    assert build_chat_model(_config(model=None), settings).model_name == settings.chat_model
    assert (
        build_chat_model(_config(model="anthropic/claude-sonnet-4.5"), settings).model_name
        == "anthropic/claude-sonnet-4.5"
    )


def test_temperature_is_passed_through_and_omitted_when_unset() -> None:
    settings = Settings(openrouter_api_key="k")

    assert build_chat_model(_config(temperature=0.2), settings).temperature == 0.2
    assert build_chat_model(_config(), settings).temperature is None


def test_the_base_url_is_overridable_per_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local model or a proxy needs no code change."""
    monkeypatch.setenv("ENGINE_OPENROUTER_BASE_URL", "http://localhost:11434/v1")

    model = build_chat_model(_config(), Settings(openrouter_api_key="k"))

    assert model.openai_api_base == "http://localhost:11434/v1"


def test_usage_is_reported_while_streaming() -> None:
    """Without this, a streamed answer carries no token count for `UsageEvent`."""
    model = build_chat_model(_config(), Settings(openrouter_api_key="k"))

    assert model.stream_usage is True


def test_documents_still_work_without_a_provider_key(client: TestClient) -> None:
    """Ingestion needs no model, so a missing key must not block it."""
    response = client.put(
        "/documents",
        data={"project_id": "support", "external_id": "baggage.md"},
        files={"file": ("baggage.md", b"# Baggage\n\nOne bag.\n", "text/markdown")},
    )

    assert response.status_code == 201
    assert response.json()["status"] == "received", "chunked and recorded, not embedded"


# --- the one chunker setting that is easy to get wrong ------------------------


def test_a_zero_overlap_is_honoured() -> None:
    """`if x is None` rather than `or`: 0 is a real value, not "unset"."""
    assert DocumentChunker(chunk_overlap=0)._splitter._chunk_overlap == 0




