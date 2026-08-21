"""The chat agent: one turn, as a stream of events."""

from __future__ import annotations

from collections.abc import AsyncIterator

from chatbot_engine.agent.client import get_completion
from chatbot_engine.agent.retriever import retrieve, to_context, to_source_refs
from chatbot_engine.models.chat import ChatRequest
from chatbot_engine.models.events import (
    DoneEvent,
    Event,
    RetrievalEvent,
    TokenEvent,
)
from chatbot_engine.ports.agent import ToolProvider


class ChatAgent:
    """Answers a turn in one piece.

    Every event the contract allows is legal but optional, so this emits the two
    that matter: the answer, then `done`. Streaming means yielding a `TokenEvent`
    per chunk instead of one at the end -- the shape here does not change.
    """

    def __init__(self, tools: ToolProvider) -> None:
        self._tools = tools

    async def run(self, request: ChatRequest) -> AsyncIterator[Event]:
        """Answer one turn."""
        hits = await retrieve(request)

        # Before the answer, so the UI can show what it was based on while the
        # model is still thinking.
        yield RetrievalEvent(
            query=request.message, sources=to_source_refs(hits)
        )

        answer = await get_completion(request, self._tools, to_context(hits))

        yield TokenEvent(text=answer)
        yield DoneEvent(finish_reason="stop")
