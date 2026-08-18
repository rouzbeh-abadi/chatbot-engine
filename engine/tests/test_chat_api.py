"""The chat contract: what the backend may send, and what it gets back.

The turn itself is unimplemented, so these assert the boundary -- request
validation, and a 501 that names what to write.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_chat_reports_no_agent_is_registered(
    client: TestClient, project: dict[str, object]
) -> None:
    response = client.post("/chat", json={"project": project, "message": "hi"})

    assert response.status_code == 501
    detail = response.json()["detail"]
    assert "Agent" in detail
    assert "get_agent" in detail, "the 501 should name where to register it"


def test_a_missing_implementation_is_501_not_an_empty_200(
    client: TestClient, project: dict[str, object]
) -> None:
    """The readiness check must happen before the streaming response starts."""
    response = client.post("/chat", json={"project": project, "message": "hi"})

    assert response.status_code == 501
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


def test_chat_rejects_an_empty_message(
    client: TestClient, project: dict[str, object]
) -> None:
    response = client.post("/chat", json={"project": project, "message": ""})

    assert response.status_code == 422


def test_chat_requires_a_project(client: TestClient) -> None:
    """The engine stores no config, so a turn without one is meaningless."""
    assert client.post("/chat", json={"message": "hi"}).status_code == 422


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
    """`NotConfiguredError` inherits from both `EngineError` and
    `NotImplementedError`. Handler registration must keep "unwritten" (501)
    distinguishable from "broken" (500), whatever the MRO order happens to be."""
    from chatbot_engine.api import deps
    from chatbot_engine.errors import EngineError
    from chatbot_engine.services.chat import ChatService

    class Exploding:
        def run(self, request):  # noqa: ANN001, ANN202
            raise EngineError("retriever exploded")

    client.app.dependency_overrides[deps.get_chat_service] = lambda: ChatService(
        agent=Exploding()
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
