"""The chat agent: one turn, as a stream of events."""

from __future__ import annotations

from collections.abc import AsyncIterator

from chatbot_engine.agent.client import Usage, stream_completion
from chatbot_engine.agent.retriever import retrieve, to_context, to_source_refs
from chatbot_engine.models.chat import ChatRequest
from chatbot_engine.models.events import (
    DoneEvent,
    Event,
    RetrievalEvent,
    TokenEvent,
    UsageEvent,
)
from chatbot_engine.ports.agent import ToolProvider


class ChatAgent:
    """Run one chat turn through retrieval, model generation, and streaming.

    The agent retrieves relevant context, emits the sources, streams the model's

    answer as token events, and finishes with a done event.
    """

    def __init__(self, tools: ToolProvider) -> None:
        self._tools = tools

    async def run(self, request: ChatRequest) -> AsyncIterator[Event]:
        """Process one chat request and yield events as the answer is produced."""
        hits = await retrieve(request)

        # Before the answer, so the UI can show what it was based on while the model is still thinking.
        yield RetrievalEvent(
            query=request.message, sources=to_source_refs(hits)
        )

        # stream_completion yields answer text as it is generated, tool
        # started/finished events around any tool call, and one Usage value at
        # the end; turn each into the matching event.
        async for item in stream_completion(request, self._tools, to_context(hits)):
            if isinstance(item, Usage):
                yield UsageEvent(
                    input_tokens=item.input_tokens,
                    output_tokens=item.output_tokens,
                    total_tokens=item.total_tokens,
                    cost_usd=item.cost_usd,
                    model=item.model,
                )
            elif isinstance(item, str):
                yield TokenEvent(text=item)
            else:
                # Already an Event (a tool started/finished), pass it through.
                yield item

        yield DoneEvent(finish_reason="stop")
