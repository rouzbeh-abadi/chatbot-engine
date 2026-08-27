from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding

from chatbot_engine.api.dependencies import reset_dependency_cache
from chatbot_engine.api.rate_limit import reset_rate_limits
from chatbot_engine.rag import embeddings as embeddings_module
from chatbot_engine.rag import vector_store as vector_store_module


@pytest.fixture(autouse=True)
def offline_vectors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[None]:
    """A real Chroma in a temporary directory, with fake vectors.

    Real store, so the tests exercise the actual write, replace and delete paths.
    Fake embedder, so the suite needs no API key, costs nothing and stays fast --
    `DeterministicFakeEmbedding` hashes the text, so identical text still gives
    identical vectors.
    """
    monkeypatch.setenv("ENGINE_CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("ENGINE_REGISTRY_DB", str(tmp_path / "documents.sqlite3"))
    monkeypatch.setenv("ENGINE_BLOB_DIR", str(tmp_path / "blobs"))
    monkeypatch.setenv("ENGINE_OPENROUTER_API_KEY", "sk-or-fake-for-tests")

    fake = DeterministicFakeEmbedding(size=64)
    # Both modules: `vector_store` imports the name directly, so patching only
    # `embeddings` would leave it holding the real one.
    monkeypatch.setattr(embeddings_module, "get_embeddings", lambda: fake)
    monkeypatch.setattr(vector_store_module, "get_embeddings", lambda: fake)
    reset_dependency_cache()
    reset_rate_limits()

    yield

    reset_dependency_cache()
    reset_rate_limits()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A client with no API key configured, so the engine is open."""
    monkeypatch.delenv("ENGINE_API_KEY", raising=False)
    reset_dependency_cache()

    from chatbot_engine.app import create_app

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    reset_dependency_cache()


@pytest.fixture
def project() -> dict[str, object]:
    return {
        "project_id": "support",
        "name": "Support",
        "system_prompt": "You are a fixture.",
    }
