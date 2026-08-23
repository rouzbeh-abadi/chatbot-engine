"""The chat agent: one turn, as a stream of events."""

from __future__ import annotations

from collections.abc import AsyncIterator

from chatbot_engine.agent.client import stream_completion
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
    """Answers a turn as the model writes it.

    Every event the contract allows is legal but optional. This emits what was
    retrieved, the answer in pieces, and `done`. Tool progress and token cost are
    still to come.
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

        async for text in stream_completion(request, self._tools, to_context(hits)):
            yield TokenEvent(text=text)

        yield DoneEvent(finish_reason="stop")
