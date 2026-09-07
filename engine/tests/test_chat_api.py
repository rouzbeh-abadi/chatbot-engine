"""The chat contract: what the backend may send, and what it gets back.

Request validation, and the one 501 the engine answers with: no model provider
key. That has to arrive as a status code, which means before the stream starts.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from chatbot_engine.api.dependencies import reset_dependency_cache


def test_a_missing_provider_key_is_501_not_an_empty_200(
    monkeypatch: pytest.MonkeyPatch, project: dict[str, object]
) -> None:
    """The key is checked when the agent is built, which happens before the
    response starts. Checked lazily instead, it would surface inside a 200
    stream where a caller cannot tell it from an answer that never came."""
    monkeypatch.setenv("ENGINE_OPENROUTER_API_KEY", "")
    monkeypatch.delenv("ENGINE_API_KEY", raising=False)
    reset_dependency_cache()

    from chatbot_engine.app import create_app

    with TestClient(create_app()) as client:
        response = client.post("/chat", json={"project": project, "message": "hi"})

    reset_dependency_cache()

    assert response.status_code == 501
    assert "ENGINE_OPENROUTER_API_KEY" in response.json()["detail"]
    assert not response.headers["content-type"].startswith("application/x-ndjson")


def test_chat_rejects_unknown_request_fields(
    client: TestClient, project: dict[str, object]
) -> None:
    """`extra="forbid"` means a typo fails loudly instead of being ignored."""
    response = client.post(
        "/chat",
        json={"project": project, "message": "hi", "sytem_prompt": "oops"},
    )

    assert response.status_code == 422


def test_mcp_servers_must_declare_an_allowlist(
    client: TestClient, project: dict[str, object]
) -> None:
    """Tool descriptions reach the prompt, so an open list is an injection risk."""
    response = client.post(
        "/chat",
        json={
            "project": {
                **project,
                "mcp_servers": [
                    {
                        "name": "support-tools",
                        "url": "http://localhost:8200/mcp",
                        "allowed_tools": [],
                    }
                ],
            },
            "message": "hi",
        },
    )

    assert response.status_code == 422


def test_a_deliberate_engine_error_is_500_not_501(client: TestClient) -> None:
    """`NotConfiguredError` is an `EngineError`. Handler registration must keep
    "missing configuration" (501) distinguishable from "broken" (500), whatever
    the MRO order happens to be."""
    from chatbot_engine.api import dependencies
    from chatbot_engine.errors import EngineError
    from chatbot_engine.services.chat import ChatService

    class Exploding:
        def run(self, request):
            raise EngineError("retriever exploded")

    client.app.dependency_overrides[dependencies.get_chat_service] = lambda: (
        ChatService(agent=Exploding())
    )
    try:
        response = client.post(
            "/chat",
            json={
                "project": {
                    "project_id": "p",
                    "name": "n",
                    "system_prompt": "s",
                },
                "message": "hi",
            },
        )
        assert response.status_code == 500
        assert "retriever exploded" in response.json()["detail"]
    finally:
        client.app.dependency_overrides.clear()
