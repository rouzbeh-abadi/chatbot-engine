"""The NDJSON stream contract.

A test-only agent stands in for the real one, so the framing, the media type and
the mid-stream failure behaviour are pinned before any AI logic exists. The fake
agent lives here, not in the package -- it is a fixture, not an implementation.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from chatbot_engine.api import dependencies
from chatbot_engine.api.streaming import MEDIA_TYPE, describe
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

    async def run(self, request):
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

    async def run(self, request):
        yield TokenEvent(text="partial")
        raise RuntimeError("retriever died")


def _with_agent(client: TestClient, agent: object) -> None:
    client.app.dependency_overrides[dependencies.get_chat_service] = lambda: (
        ChatService(agent=agent)
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


class _TaskGroupAgent:
    """Fails the way anything built on anyio task groups fails."""

    async def run(self, request):
        yield TokenEvent(text="partial")
        raise ExceptionGroup(
            "unhandled errors in a TaskGroup",
            [ConnectionError("All connection attempts failed")],
        )


def test_an_exception_group_reports_its_real_cause(
    client: TestClient, project: dict[str, object]
) -> None:
    """`str(ExceptionGroup)` alone says only "unhandled errors in a TaskGroup"."""
    _with_agent(client, _TaskGroupAgent())

    response = client.post("/chat", json={"project": project, "message": "hi"})

    error = next(e for e in _lines(response.text) if e["type"] == "error")
    assert "All connection attempts failed" in error["message"]


def test_a_cause_chain_is_included(
    client: TestClient, project: dict[str, object]
) -> None:
    """`raise ... from exc` is how the engine adds context, so keep both parts."""
    assert "outer" in describe(_chained())
    assert "inner" in describe(_chained())


def _chained() -> Exception:
    try:
        try:
            raise ValueError("inner")
        except ValueError as inner:
            raise RuntimeError("outer") from inner
    except RuntimeError as exc:
        return exc


# --- what a failed turn spent -------------------------------------------------


def _reply(tokens_in: int, tokens_out: int):
    from langchain_core.messages import AIMessage

    return AIMessage(
        content="",
        usage_metadata={
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "total_tokens": tokens_in + tokens_out,
        },
    )


async def _ndjson_of(events) -> list[dict[str, object]]:
    from chatbot_engine.api.streaming import to_ndjson

    return [json.loads(line) async for line in to_ndjson(events, model="m/answer")]


def test_a_turn_that_fails_after_spending_says_what_it_spent_before_the_error() -> None:
    """The caller counts the turn from this, instead of estimating it."""
    import asyncio

    from chatbot_engine.agent.client import add_usage, empty_totals

    async def events():
        totals = empty_totals()
        add_usage(totals, _reply(300, 20), utility=True)  # the rewrite
        add_usage(totals, _reply(2_000, 0))  # the answer model, cut off
        yield TokenEvent(text="Thirty")
        raise RuntimeError("provider timeout")

    lines = asyncio.run(_ndjson_of(events()))

    assert [line["type"] for line in lines] == ["token", "usage", "error", "done"]
    usage = lines[1]
    assert (usage["input_tokens"], usage["output_tokens"], usage["model"]) == (
        2_300,
        20,
        "m/answer",
    )
    assert (usage["utility_input_tokens"], usage["utility_output_tokens"]) == (300, 20)


def test_a_turn_that_fails_after_reporting_or_before_spending_adds_no_usage() -> None:
    import asyncio

    from chatbot_engine.agent.client import add_usage, empty_totals

    async def reported():
        totals = empty_totals()
        add_usage(totals, _reply(100, 10))
        yield UsageEvent(input_tokens=100, output_tokens=10, total_tokens=110)
        raise RuntimeError("late")

    async def nothing():
        raise RuntimeError("engine down")
        yield  # pragma: no cover

    assert [line["type"] for line in asyncio.run(_ndjson_of(reported()))] == [
        "usage",
        "error",
        "done",
    ]
    assert [line["type"] for line in asyncio.run(_ndjson_of(nothing()))] == [
        "error",
        "done",
    ]


def test_the_usage_event_names_the_utility_model_when_part_of_the_turn_ran_on_it(
    monkeypatch,
) -> None:
    from chatbot_engine.agent.client import Usage, usage_event
    from chatbot_engine.api.dependencies import reset_dependency_cache

    monkeypatch.setenv("ENGINE_UTILITY_MODEL", "openai/gpt-4.1-mini")
    reset_dependency_cache()
    event = usage_event(
        Usage(
            input_tokens=500,
            output_tokens=50,
            total_tokens=550,
            utility_input_tokens=300,
            utility_output_tokens=20,
            model="m/answer",
        )
    )
    assert (event.utility_model, event.utility_input_tokens) == (
        "openai/gpt-4.1-mini",
        300,
    )
    assert (
        usage_event(
            Usage(input_tokens=5, output_tokens=5, total_tokens=10)
        ).utility_model
        is None
    )
    reset_dependency_cache()
