"""The chat side of the engine: the interfaces future AI logic must satisfy.

Nothing here has behaviour. These exist so `services/chat.py` can be written
against a stable shape, and so the HTTP layer never needs to change when the
real implementation arrives.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Protocol

from chatbot_engine.models.chat import AssistantConfig, ChatRequest
from chatbot_engine.models.events import Event


class Agent(Protocol):
    """One chat turn, as a stream of events.

    An implementation is expected to: retrieve for the question, emit a
    `RetrievalEvent` with its sources, build the prompt from
    `request.project.system_prompt` plus history plus retrieved chunks, call the
    model, emit `TokenEvent`s as it streams, run any tool calls (bounded by
    `request.project.max_tool_iterations`), then emit `UsageEvent` and
    `DoneEvent`.

    A stream rather than a return value, because a UI needs sources and progress
    before the answer is finished.
    """

    def run(self, request: ChatRequest) -> AsyncIterator[Event]: ...


class ToolProvider(Protocol):
    """Tool discovery and invocation, over MCP.

    Implemented by `chatbot_engine.mcp.client.McpToolProvider`. Kept as a port so
    the agent depends on the capability rather than on MCP specifically.
    """

    async def list_tools(self, config: AssistantConfig) -> Sequence[Mapping[str, Any]]:
        """Allowlisted tool schemas, ready to hand to a model."""
        ...

    async def call_tool(
        self,
        *,
        server: str,
        name: str,
        arguments: Mapping[str, Any],
        user_id: str | None = None,
    ) -> str:
        """Invoke one tool and return its result as text."""
        ...
