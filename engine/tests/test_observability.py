"""What the engine reports about itself.

A request id that follows a turn across services, one log line per turn with
the facts an operator asks for first, and metrics a scraper can read.
"""

from __future__ import annotations

import json
import logging

from fastapi.testclient import TestClient
from prometheus_client import generate_latest

from chatbot_engine.api import dependencies
from chatbot_engine.mcp.client import caller_headers
from chatbot_engine.models.events import (
    DoneEvent,
    TokenEvent,
    ToolCallFinishedEvent,
    UsageEvent,
)
from chatbot_engine.observability import (
    REQUEST_ID_HEADER,
    JsonFormatter,
    RequestIdFilter,
    _request_id,
)
from chatbot_engine.services.chat import ChatService


class _Agent:
    """A turn with one tool call and a priced usage event. No model."""

    async def run(self, request):
        yield ToolCallFinishedEvent(call_id="c1", tool="get_booking_status", ok=True)
        yield ToolCallFinishedEvent(call_id="c2", tool="get_flight_status", ok=False)
        yield TokenEvent(text="Delayed.")
        yield UsageEvent(
            input_tokens=80,
            output_tokens=30,
            total_tokens=110,
            cost_usd=0.00008,
            model="openai/gpt-5-mini",
        )
        yield DoneEvent()


def _with_agent(client: TestClient) -> None:
    client.app.dependency_overrides[dependencies.get_chat_service] = lambda: (
        ChatService(agent=_Agent())
    )


# --- the request id --------------------------------------------------------------


def test_every_response_carries_a_request_id(client: TestClient) -> None:
    response = client.get("/health")

    assert len(response.headers[REQUEST_ID_HEADER]) == 32


def test_a_callers_id_is_kept_so_logs_line_up_across_services(
    client: TestClient,
) -> None:
    response = client.get("/health", headers={REQUEST_ID_HEADER: "backend-7f3a"})

    assert response.headers[REQUEST_ID_HEADER] == "backend-7f3a"


def test_a_malformed_id_is_replaced_not_trusted(client: TestClient) -> None:
    """It lands in logs and in headers to other services, so its shape is
    ours to decide."""
    response = client.get("/health", headers={REQUEST_ID_HEADER: "x" * 65})
    assert response.headers[REQUEST_ID_HEADER] != "x" * 65

    response = client.get("/health", headers={REQUEST_ID_HEADER: "bad id\\n"})
    assert " " not in response.headers[REQUEST_ID_HEADER]


def test_the_id_is_forwarded_to_the_tool_server() -> None:
    token = _request_id.set("req-42")
    try:
        headers = caller_headers("u1", "s1")
    finally:
        _request_id.reset(token)

    assert headers == {
        "X-User-Id": "u1",
        "X-Session-Id": "s1",
        REQUEST_ID_HEADER: "req-42",
    }
    assert REQUEST_ID_HEADER not in caller_headers("u1", "s1"), "none outside a request"


# --- logging ---------------------------------------------------------------------


def test_log_lines_carry_the_request_id() -> None:
    record = logging.LogRecord("t", logging.INFO, "", 0, "hello", None, None)

    token = _request_id.set("req-42")
    try:
        RequestIdFilter().filter(record)
    finally:
        _request_id.reset(token)

    assert record.request_id == "req-42"


def test_json_lines_carry_the_turns_fields() -> None:
    record = logging.LogRecord(
        "chatbot_engine.turn", logging.INFO, "", 0, "turn", None, None
    )
    record.request_id = "req-42"
    record.outcome = "stop"
    record.cost_usd = 0.00008

    line = json.loads(JsonFormatter().format(record))

    assert line["request_id"] == "req-42"
    assert line["outcome"] == "stop"
    assert line["cost_usd"] == 0.00008
    assert line["level"] == "INFO"


def test_a_turn_is_logged_once_with_what_an_operator_asks_first(
    client: TestClient, project: dict[str, object], caplog
) -> None:
    _with_agent(client)

    with caplog.at_level(logging.INFO, logger="chatbot_engine.turn"):
        client.post(
            "/chat", json={"project": {**project, "agent": "loop"}, "message": "hi"}
        )

    (record,) = [r for r in caplog.records if r.name == "chatbot_engine.turn"]
    assert record.outcome == "stop"
    assert record.agent == "loop"
    assert record.model == "openai/gpt-5-mini"
    assert (record.input_tokens, record.output_tokens) == (80, 30)
    assert (record.tool_calls_ok, record.tool_calls_failed) == (1, 1)
    assert record.caller == "unauthenticated"
    assert record.seconds >= 0


# --- metrics ----------------------------------------------------------------------


def test_metrics_are_served_and_count_the_turn(
    client: TestClient, project: dict[str, object]
) -> None:
    _with_agent(client)
    client.post("/chat", json={"project": project, "message": "hi"})

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    assert (
        'chatbot_engine_turns_total{agent="loop",caller="unauthenticated",outcome="stop"}'
        in body
    )
    assert (
        'chatbot_engine_tool_calls_total{ok="false",tool="get_flight_status"}' in body
    )
    assert (
        'chatbot_engine_tokens_total{caller="unauthenticated",direction="input",model="openai/gpt-5-mini"}'
        in body
    )


def test_metrics_need_no_api_key(monkeypatch) -> None:
    monkeypatch.setenv("ENGINE_API_KEY", "s3cret")
    dependencies.reset_dependency_cache()
    from chatbot_engine.app import create_app

    with TestClient(create_app()) as keyed:
        assert keyed.get("/metrics").status_code == 200
        assert keyed.get("/agents").status_code == 401, "sanity: the key is enforced"

    dependencies.reset_dependency_cache()


def test_metrics_can_be_switched_off(monkeypatch) -> None:
    monkeypatch.setenv("ENGINE_METRICS_ENABLED", "false")
    dependencies.reset_dependency_cache()
    from chatbot_engine.app import create_app

    with TestClient(create_app()) as quiet:
        assert quiet.get("/metrics").status_code == 404

    dependencies.reset_dependency_cache()
    assert generate_latest()  # the registry itself is untouched
