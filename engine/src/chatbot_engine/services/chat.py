"""Chat orchestration boundary.

This class exists so the HTTP layer has something stable to call. It holds no AI
logic: it checks that an `Agent` implementation has been registered and delegates
to it. Everything a chat turn actually does -- retrieval, prompt construction,
the model call, the tool loop -- belongs in the `Agent`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from chatbot_engine.errors import NotConfiguredError
from chatbot_engine.models.chat import ChatRequest
from chatbot_engine.models.events import Event
from chatbot_engine.ports.agent import Agent


class ChatService:
    """Holds only the agent. A `ToolProvider` belongs to the agent itself, which
    is constructed in `api/deps.py` -- routing it through here would be a
    parameter this class never reads."""

    def __init__(self, *, agent: Agent | None = None) -> None:
        self._agent = agent

    @property
    def is_ready(self) -> bool:
        """Whether a chat turn can be served at all."""
        return self._agent is not None

    def stream(self, request: ChatRequest) -> AsyncIterator[Event]:
        """Return the event stream for one turn.

        Deliberately a plain function returning an iterator, not an async
        generator: the readiness check has to raise *before* the response starts,
        or the caller gets a 200 with an empty body instead of a 501.
        """
        if self._agent is None:
            raise NotConfiguredError(
                "no Agent is registered -- implement one under "
                "chatbot_engine/agent/ and return it from "
                "chatbot_engine.api.deps.get_agent()"
            )
        return self._agent.run(request)
