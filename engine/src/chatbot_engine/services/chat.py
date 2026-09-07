"""The chat service: the boundary between the HTTP layer and the agent.

Holds no AI logic. It exists so the route depends on one object rather than on
the agent's construction, which keeps `api/chat.py` free of wiring and lets a
test hand the route any agent it likes.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from chatbot_engine.models.chat import ChatRequest
from chatbot_engine.models.events import Event
from chatbot_engine.ports.agent import Agent


class ChatService:
    """Runs a chat turn on the configured `Agent`."""

    def __init__(self, *, agent: Agent) -> None:
        self._agent = agent

    def stream(self, request: ChatRequest) -> AsyncIterator[Event]:
        """The event stream for one turn.

        A plain function returning an iterator, not an async generator, so that
        anything the agent raises while setting up the turn (an unknown agent
        name, a missing provider key) surfaces before the response starts and
        can still become a status code.
        """
        return self._agent.run(request)
