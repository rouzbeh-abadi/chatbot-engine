"""Installing your own agent, without forking the engine.

The engine ships one agent; the point of the registry is the next one:
an adopter registers a factory under an entry point, installs the package
alongside the engine, and names it in the assistant config.

These tests fake the entry point rather than pip-installing a package, but the
path they exercise is the real one: discovery, construction, selection, and the
error when the name is not installed.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from chatbot_engine.agent import registry
from chatbot_engine.agent.registry import (
    ENTRY_POINT_GROUP,
    UnknownAgentError,
    build_agent,
)


class MyAgent:
    """What an adopter would write: anything with `run`."""

    def __init__(self, tools) -> None:
        self.tools = tools

    def run(self, request):
        async def events():
            yield "ran-my-agent"

        return events()


def build(tools) -> MyAgent:
    """The factory the entry point would point at."""
    return MyAgent(tools)


class FakeEntryPoint:
    name = "my-graph"

    def load(self):
        return build


def _with_plugin():
    """Pretend `my-graph` is installed under the entry-point group."""
    return patch.object(
        registry, "entry_points", lambda group: [FakeEntryPoint()] if group == ENTRY_POINT_GROUP else []
    )


# --- discovery ----------------------------------------------------------------


def test_a_plugin_is_constructed_with_the_tool_provider() -> None:
    """The one dependency an agent cannot make for itself."""
    tools = object()

    with _with_plugin():
        agent = build_agent("my-graph", tools)

    assert isinstance(agent, MyAgent)
    assert agent.tools is tools


def test_a_plugin_may_replace_a_builtin() -> None:
    """Swapping the default should not need the engine's permission."""

    class ReplacesLoop(FakeEntryPoint):
        name = "loop"

    with patch.object(registry, "entry_points", lambda group: [ReplacesLoop()]):
        assert isinstance(build_agent("loop", object()), MyAgent)


# --- the error when it is not installed ---------------------------------------


def test_an_unknown_agent_names_what_is_installed() -> None:
    """The list is the real installed set, so it stays true as plugins change."""
    with pytest.raises(UnknownAgentError) as caught:
        build_agent("nope", object())

    message = str(caught.value)
    assert "nope" in message
    assert "loop" in message and "graph" in message
    assert ENTRY_POINT_GROUP in message


def test_an_unknown_agent_is_a_422_over_http(
    client: TestClient, project: dict[str, object]
) -> None:
    """A caller's mistake, not a broken engine."""
    response = client.post(
        "/chat", json={"project": {**project, "agent": "nope"}, "message": "hi"}
    )

    assert response.status_code == 422
    assert "nope" in response.json()["detail"]


# --- end to end ---------------------------------------------------------------


def test_a_plugin_agent_serves_a_real_request(
    client: TestClient, project: dict[str, object]
) -> None:
    """The whole point: install an agent, name it, and the engine runs it."""

    class Streaming(MyAgent):
        def run(self, request):
            from chatbot_engine.models.events import DoneEvent, TokenEvent

            async def events():
                yield TokenEvent(text="from my agent")
                yield DoneEvent(finish_reason="stop")

            return events()

    with patch.object(
        registry,
        "entry_points",
        lambda group: [type("EP", (), {"name": "my-graph", "load": lambda s: Streaming})()],
    ):
        response = client.post(
            "/chat",
            json={"project": {**project, "agent": "my-graph"}, "message": "hi"},
        )

    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert events[0]["text"] == "from my agent"
    assert events[-1]["type"] == "done"
