"""The NDJSON stream contract.

A test-only agent stands in for the real one, so the framing, the media type and
the mid-stream failure behaviour are pinned before any AI logic exists. The fake
agent lives here, not in the package -- it is a fixture, not an implementation.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from chatbot_engine.api import deps
from chatbot_engine.api.streaming import MEDIA_TYPE
from chatbot_engine.models.events import (
    DoneEvent,
    RetrievalEvent,
    SourceRef,
    TokenEvent,
    ToolCallFinishedEvent,
    ToolCallStartedEvent,
    UsageEvent,
)
from chatbot_engine.services.chat import ChatService


class _EchoAgent:
    """Emits one of each event type. No model, no retrieval."""

    async def run(self, request):  # noqa: ANN001, ANN202
        yield RetrievalEvent(
            query=request.message,
            sources=[SourceRef(doc_id="d1", source="baggage.md", score=0.9)],
        )
        yield ToolCallStartedEvent(
            call_id="c1",
            tool="get_booking_status",
            server="support-tools",
            arguments={"booking_reference": "AB12CD"},
        )
        yield ToolCallFinishedEvent(
            call_id="c1", tool="get_booking_status", ok=True, duration_ms=31
        )
        yield TokenEvent(text="One ")
        yield TokenEvent(text="bag.")
        yield UsageEvent(total_tokens=16, model="fake/echo")
        yield DoneEvent()


class _FailingAgent:
    """Fails after the response has already started."""

    async def run(self, request):  # noqa: ANN001, ANN202
        yield TokenEvent(text="partial")
        raise RuntimeError("retriever died")


def _with_agent(client: TestClient, agent: object) -> None:
    client.app.dependency_overrides[deps.get_chat_service] = lambda: ChatService(
        agent=agent
    )


def _lines(body: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in body.splitlines() if line.strip()]


def test_a_turn_streams_ndjson_one_event_per_line(
    client: TestClient, project: dict[str, object]
) -> None:
    _with_agent(client, _EchoAgent())

    response = client.post("/chat", json={"project": project, "message": "baggage?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(MEDIA_TYPE)

    events = _lines(response.text)
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
    assert events[-1]["finish_reason"] == "stop"


def test_the_stream_is_not_buffered(
    client: TestClient, project: dict[str, object]
) -> None:
    """No content-length: the caller must be able to read tokens as they arrive."""
    _with_agent(client, _EchoAgent())

    with client.stream(
        "POST", "/chat", json={"project": project, "message": "hi"}
    ) as response:
        assert "content-length" not in response.headers
        first = next(response.iter_lines())

    assert json.loads(first)["type"] == "retrieval"


def test_a_mid_stream_failure_ends_with_error_then_done(
    client: TestClient, project: dict[str, object]
) -> None:
    """The 200 is already sent by then, so raising would truncate the body with no
    explanation. The stream has to close itself properly instead."""
    _with_agent(client, _FailingAgent())

    response = client.post("/chat", json={"project": project, "message": "hi"})

    assert response.status_code == 200
    events = _lines(response.text)
    assert [e["type"] for e in events] == ["token", "error", "done"]
    assert "retriever died" in str(events[1]["message"])
    assert events[-1]["finish_reason"] == "error"
