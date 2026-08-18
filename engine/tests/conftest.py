from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from chatbot_engine.api.deps import reset_dependency_cache


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
