"""Interfaces (ports) for the chat side of the engine.

A `Protocol` is a structural interface: any class with matching methods satisfies
it, without subclassing. These describe *what* the engine needs, so the rest of
the code depends on the interface, not on a concrete class. The implementations
live in `agent/` and `mcp/` and are wired in at `api/dependencies.py`, so a model
provider, agent framework, or tool protocol can be swapped without touching the
callers.

- `Agent` runs one chat turn and streams back events.
- `ToolProvider` discovers and invokes the application's external tools.
- `Judge` scores a finished evaluation run.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Protocol

from chatbot_engine.models.chat import AssistantConfig, ChatRequest
from chatbot_engine.models.evals import JudgeReport, JudgeRequest
from chatbot_engine.models.events import Event


class Agent(Protocol):
    """Processes one chat turn end to end.

    An implementation does retrieval, builds the prompt, calls the model, runs
    any tools the model requests, and streams the result back as events.
    `ChatAgent` in `agent/chat_agent.py` is the one used today.
    """

    def run(
        self,
        request: ChatRequest,
    ) -> AsyncIterator[Event]:
        """Process one chat request and stream back engine events.

        Args:
            request: The full chat request: assistant config, the user's
                message, history, and caller context.

        Returns:
            An async stream of events (retrieval, tokens, tool activity, usage,
            done) describing the turn as it happens.
        """
        ...


class ToolProvider(Protocol):
    """Define discovery and invocation of external application-owned tools."""

    async def list_tools(
        self,
        config: AssistantConfig,
    ) -> Sequence[Mapping[str, Any]]:
        """Return tools allowed by the assistant configuration.

        Args:
            config: Assistant configuration containing MCP server definitions
                and tool allowlists.

        Returns:
            Tool schemas that may be exposed to the model.
        """
        ...

    async def call_tool(
        self,
        *,
        config: AssistantConfig,
        server: str,
        name: str,
        arguments: Mapping[str, Any],
        user_id: str | None = None,
    ) -> str:
        """Invoke one allowlisted tool on a configured tool server.

        Args:
            config: Assistant configuration used to resolve the MCP server
                connection and tool allowlist.
            server: Name of the configured MCP server exposing the tool.
            name: Name of the tool to invoke.
            arguments: Arguments passed to the tool.
            user_id: Opaque user identifier forwarded for authorization when
                supported by the tool transport.

        Returns:
            Tool result serialized as text.
        """
        ...

class Judge(Protocol):
    """Answers a dataset and grades the answers.

    A callable rather than a class: the implementation is a handful of plain
    functions, and there is no state to hold between runs.
    """

    async def __call__(self, request: JudgeRequest) -> JudgeReport:
        """Return one verdict per case.

        Args:
            request: The rubric, the cases, and the model configuration.
        """
        ...
