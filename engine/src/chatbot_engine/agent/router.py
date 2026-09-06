"""Pick the agent a request asked for.

Which agent runs a turn is part of the assistant config, so it is decided per
request rather than at wiring time. One engine can therefore serve the built-in
loop, the graph, and any agent an adopter has installed, at the same time.

This is itself an `Agent`. Everything upstream -- the service, the route, the
event stream -- is unchanged and unaware there is a choice at all.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from chatbot_engine.agent.registry import build_agent
from chatbot_engine.models.chat import ChatRequest
from chatbot_engine.models.events import Event
from chatbot_engine.ports.agent import Agent, ToolProvider
from chatbot_engine.settings import get_settings


class AgentRouter:
    """Builds and runs whichever agent the assistant config names."""

    def __init__(self, tools: ToolProvider) -> None:
        self._tools = tools
        #: Built once per name and reused. An agent holds no per-turn state --
        #: everything about a turn arrives with the request -- so rebuilding one
        #: for every message would be waste, not isolation.
        self._built: dict[str, Agent] = {}

    def run(self, request: ChatRequest) -> AsyncIterator[Event]:
        """Run one turn on the requested agent, or the engine's default."""
        name = request.project.agent or get_settings().agent

        if name not in self._built:
            self._built[name] = build_agent(name, self._tools)

        return self._built[name].run(request)
