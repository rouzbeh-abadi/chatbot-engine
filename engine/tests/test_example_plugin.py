"""The bundled plugin is really installed, and really reachable.

`test_agent_registry.py` proves the mechanism with a fake entry point. This
proves the real one: `examples/langgraph-agent` is a separate package, installed
like any third-party agent, discovered without the engine knowing it exists.

If these fail, the plugin path has broken for everyone, not just this example.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from chatbot_engine.agent.registry import _builtin, build_agent

pytest.importorskip(
    "langgraph_agent",
    reason="the plugin is a workspace member; run `uv sync`",
)


def test_the_engine_ships_exactly_one_agent() -> None:
    """Anything else is a plugin. The engine picks no framework for you."""
    assert sorted(_builtin()) == ["loop"]


def test_the_engine_never_imports_the_plugin() -> None:
    """It is found through its entry point, not wired in by name.

    The prose mentions LangGraph as an example; what must not appear is the
    module, which would mean the engine depends on it after all.
    """
    import chatbot_engine.agent.registry as registry

    assert "langgraph_agent" not in open(registry.__file__).read()


def test_it_builds_and_gets_the_tool_provider() -> None:
    tools = object()

    agent = build_agent("graph", tools)

    assert agent._tools is tools


def test_it_is_offered_over_http(client: TestClient) -> None:
    """What the picker fetches, so it appears in the dropdown."""
    assert "graph" in client.get("/agents").json()
