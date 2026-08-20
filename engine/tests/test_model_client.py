"""The provider client.

Mostly negative tests: a missing key must not stop the engine from starting, or
block the routes that need no model.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from chatbot_engine.api.deps import (
    get_ingest_pipeline,
    get_model_client,
    reset_dependency_cache,
)
from chatbot_engine.errors import NotConfiguredError
from chatbot_engine.llm.client import build_model_client
from chatbot_engine.rag.splitter import DocumentChunker
from chatbot_engine.settings import Settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    """The repo's own .env would otherwise leak a real key into these tests."""
    monkeypatch.setenv("ENGINE_OPENROUTER_API_KEY", "")
    reset_dependency_cache()

    yield

    reset_dependency_cache()


# --- building the client -----------------------------------------------------


def test_a_missing_key_is_reported_not_raised_at_import() -> None:
    """Name the variable -- this is the error people hit on day one."""
    with pytest.raises(NotConfiguredError, match="ENGINE_OPENROUTER_API_KEY"):
        build_model_client(Settings(openrouter_api_key=None))


def test_a_blank_key_counts_as_missing() -> None:
    """`ENGINE_OPENROUTER_API_KEY=` in .env arrives as "", not None."""
    assert Settings(openrouter_api_key="").openrouter_api_key is None

    with pytest.raises(NotConfiguredError):
        build_model_client(Settings(openrouter_api_key=""))


def test_a_configured_key_builds_a_client_pointed_at_openrouter() -> None:
    client = build_model_client(Settings(openrouter_api_key="sk-or-test"))

    assert str(client.base_url).rstrip("/") == Settings().openrouter_base_url
    assert client.api_key == "sk-or-test"


def test_the_base_url_is_overridable_per_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local model or a proxy needs no code change."""
    monkeypatch.setenv("ENGINE_OPENROUTER_BASE_URL", "http://localhost:11434/v1")

    client = build_model_client(Settings(openrouter_api_key="k"))

    assert str(client.base_url).rstrip("/") == "http://localhost:11434/v1"


def test_the_client_is_async() -> None:
    """A sync client would hold the event loop for a whole generation."""
    client = build_model_client(Settings(openrouter_api_key="k"))

    assert type(client).__name__ == "AsyncOpenAI"


# --- the factory in deps -----------------------------------------------------


def test_the_factory_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """One client, one connection pool, for the process."""
    monkeypatch.setenv("ENGINE_OPENROUTER_API_KEY", "sk-or-test")
    reset_dependency_cache()

    assert get_model_client() is get_model_client()


def test_a_missing_key_does_not_stop_the_engine_starting(
    client: TestClient,
) -> None:
    """The whole reason the client is built on demand, not at import."""
    assert client.get("/health").status_code == 200
    assert client.get("/health/ready").json()["documents"] is True


def test_documents_still_work_without_a_provider_key(client: TestClient) -> None:
    """Ingestion needs no model, so a missing key must not block it."""
    response = client.put(
        "/documents",
        data={"project_id": "support", "external_id": "baggage.md"},
        files={"file": ("baggage.md", b"# Baggage\n\nOne bag.\n", "text/markdown")},
    )

    assert response.status_code == 201


# --- chunking comes from settings too ----------------------------------------


def test_the_chunker_defaults_to_the_configured_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`ENGINE_CHUNK_SIZE` reaches the splitter with nobody passing it down."""
    monkeypatch.setenv("ENGINE_CHUNK_SIZE", "250")
    monkeypatch.setenv("ENGINE_CHUNK_OVERLAP", "25")
    reset_dependency_cache()

    splitter = get_ingest_pipeline()._chunker._splitter

    assert splitter._chunk_size == 250
    assert splitter._chunk_overlap == 25


def test_an_explicit_size_wins_and_the_rest_still_comes_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENGINE_CHUNK_SIZE", "250")
    monkeypatch.setenv("ENGINE_CHUNK_OVERLAP", "25")
    reset_dependency_cache()

    splitter = DocumentChunker(chunk_size=600)._splitter

    assert splitter._chunk_size == 600
    assert splitter._chunk_overlap == 25


def test_a_zero_overlap_is_honoured() -> None:
    """`if x is None` rather than `or`: 0 is a real value, not "unset"."""
    assert DocumentChunker(chunk_overlap=0)._splitter._chunk_overlap == 0


# --- the vector store's location ---------------------------------------------


def test_the_chroma_location_is_configurable(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Compose points this at a volume, so vectors survive a restart."""
    monkeypatch.setenv("ENGINE_CHROMA_DIR", str(tmp_path / "vectors"))
    monkeypatch.setenv("ENGINE_CHROMA_COLLECTION", "other")
    reset_dependency_cache()

    settings = Settings()

    assert settings.chroma_dir == tmp_path / "vectors"
    assert settings.chroma_collection == "other"
