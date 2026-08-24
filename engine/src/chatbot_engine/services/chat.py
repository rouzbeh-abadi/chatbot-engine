"""The chat service: a stable boundary between the HTTP layer and the agent.

Holds no AI logic. It checks that an `Agent` is registered and delegates to it;
retrieval, prompting, the model call, and the tool loop all live in the `Agent`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from chatbot_engine.errors import NotConfiguredError
from chatbot_engine.models.chat import ChatRequest
from chatbot_engine.models.events import Event
from chatbot_engine.ports.agent import Agent


class ChatService:
    """Delegates a chat turn to the registered `Agent`, or reports it is missing.

    Holds only the agent. The agent owns its own `ToolProvider`, so passing one
    through here would be a parameter this class never reads.
    """

    def __init__(self, *, agent: Agent | None = None) -> None:
        self._agent = agent

    @property
    def is_ready(self) -> bool:
        """Whether an agent is registered, so a chat turn can be served."""
        return self._agent is not None

    def stream(self, request: ChatRequest) -> AsyncIterator[Event]:
        """Return the event stream for one turn.

        A plain function returning an iterator, not an async generator: the
        readiness check must raise *before* the response starts, or the caller
        gets a 200 with an empty body instead of a 501.
        """
        if self._agent is None:
            raise NotConfiguredError(
                "no Agent is registered -- implement one under "
                "chatbot_engine/agent/ and return it from "
                "chatbot_engine.api.dependencies.get_agent()"
            )
        return self._agent.run(request)
