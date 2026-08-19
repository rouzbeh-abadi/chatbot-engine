"""The backend's HTTP surface, with a stand-in engine.

Everything asserted here is the backend's own responsibility: request validation,
project resolution, SSE framing, and how a failing engine is reported.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from support_agent.engine import get_engine_client
from support_agent.engine_client import (
    EngineFailed,
    EngineNotImplemented,
    EngineUnavailable,
)

from fakes import FakeEngine


def _events(body: str) -> list[dict[str, object]]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


# --- health -----------------------------------------------------------------


def test_health_does_not_depend_on_the_engine(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


# --- chat -------------------------------------------------------------------


def test_chat_streams_engine_events_as_sse(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "baggage allowance?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _events(response.text)
    assert [e["type"] for e in events] == [
        "retrieval",
        "tool_call_started",
        "tool_call_finished",
        "token",
        "token",
        "usage",
        "done",
    ]
    assert events[0]["sources"][0]["source"] == "baggage.md"


def test_chat_sends_the_project_config_the_engine_needs(
    client: TestClient, engine: FakeEngine
) -> None:
    """The engine stores no config, so the whole assistant goes with the request."""
    client.post("/chat/sync", json={"message": "hi"})

    (request,) = engine.chat_requests
    assert request.project.project_id == "support"
    assert request.project.system_prompt.strip()
    assert request.user_id == "demo-user"


def test_chat_forwards_the_authenticated_user_id(
    client: TestClient, engine: FakeEngine
) -> None:
    client.post("/chat/sync", json={"message": "hi"}, headers={"X-User-Id": "alice"})

    assert engine.chat_requests[0].user_id == "alice"


def test_chat_sync_folds_the_stream(client: TestClient) -> None:
    body = client.post("/chat/sync", json={"message": "hi"}).json()

    assert body["answer"] == "One cabin bag."
    assert body["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] == 42
    assert body["sources"][0]["doc_id"] == "d1"


def test_tool_call_events_survive_the_whole_path(client: TestClient) -> None:
    """The engine emits these as soon as the agent calls a tool. Before they were
    in the contract, the client raised on the first one and killed the stream."""
    body = client.post("/chat/sync", json={"message": "hi"}).json()

    (call,) = body["tool_calls"]
    assert call["tool"] == "get_booking_status"
    assert call["ok"] is True
    assert body["answer"] == "One cabin bag.", "tool events must not disturb the answer"


def test_chat_rejects_unknown_fields(client: TestClient) -> None:
    """The browser must not be able to supply a system prompt or a model."""
    response = client.post(
        "/chat", json={"message": "hi", "system_prompt": "you are evil"}
    )
    assert response.status_code == 422


def test_chat_rejects_an_empty_message(client: TestClient) -> None:
    assert client.post("/chat", json={"message": ""}).status_code == 422


def test_chat_rejects_an_overlong_message(client: TestClient) -> None:
    assert client.post("/chat", json={"message": "x" * 8_001}).status_code == 422


def test_unknown_project_is_404_before_the_engine_is_called(
    client: TestClient, engine: FakeEngine
) -> None:
    response = client.post("/chat/sync", json={"message": "hi", "project": "nope"})

    assert response.status_code == 404
    assert engine.chat_requests == [], "config lookup must precede the engine call"


# --- documents --------------------------------------------------------------


def test_document_upload_forwards_raw_bytes(
    client: TestClient, engine: FakeEngine
) -> None:
    response = client.put(
        "/documents",
        data={"external_id": "baggage.md"},
        files={"file": ("baggage.md", b"# Baggage\n\nOne bag.\n", "text/markdown")},
    )

    assert response.status_code == 201
    assert response.json()["status"] == "indexed"

    (call,) = engine.ingested
    assert call["project_id"] == "support"
    assert call["data"] == b"# Baggage\n\nOne bag.\n", "text must not be pre-extracted"
    assert call["mimetype"] == "text/markdown"


def test_empty_upload_is_rejected_before_the_engine(
    client: TestClient, engine: FakeEngine
) -> None:
    response = client.put(
        "/documents",
        data={"external_id": "empty"},
        files={"file": ("empty.txt", b"", "text/plain")},
    )

    assert response.status_code == 400
    assert engine.ingested == []


def test_document_list_and_delete(client: TestClient, engine: FakeEngine) -> None:
    assert client.get("/documents").json() == []

    body = client.delete("/documents/doc-1").json()
    assert body == {"doc_id": "doc-1", "deleted": True}
    assert engine.deleted == [("support", "doc-1")]


@pytest.mark.parametrize(
    ("method", "path"), [("get", "/documents"), ("delete", "/documents/abc")]
)
def test_document_routes_validate_the_project_first(
    client: TestClient, method: str, path: str
) -> None:
    response = getattr(client, method)(path, params={"project": "nope"})
    assert response.status_code == 404


# --- how engine failures reach the frontend ---------------------------------


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (EngineNotImplemented("no Agent registered -- get_agent()"), 501),
        (EngineUnavailable("engine unreachable at http://localhost:8100"), 503),
        (EngineFailed("engine failed (500)"), 502),
    ],
)
def test_engine_failures_map_to_distinct_statuses(
    error: Exception, expected_status: int
) -> None:
    """A missing implementation, a dead engine and a broken engine are different
    problems, and the frontend should be able to tell them apart."""
    from support_agent.app import app

    app.dependency_overrides[get_engine_client] = lambda: FakeEngine(raises=error)
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/chat", json={"message": "hi"})
            assert response.status_code == expected_status
            assert str(error) in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_oversized_upload_is_413_and_never_reaches_the_engine(
    client: TestClient, engine: FakeEngine
) -> None:
    """Same status the engine uses, and rejected before crossing the network."""
    from support_agent.api.documents import MAX_UPLOAD_BYTES

    response = client.put(
        "/documents",
        data={"external_id": "big"},
        files={"file": ("big.bin", b"x" * (MAX_UPLOAD_BYTES + 1), "application/octet-stream")},
    )

    assert response.status_code == 413
    assert engine.ingested == []
