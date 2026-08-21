"""Interfaces for the chat side of the AI engine.

These protocols define the capabilities required by the engine without tying the
implementation to a specific model provider, agent framework, or tool protocol.

`Agent` handles one chat turn as a stream of events.
`ToolProvider` exposes external tools to the agent and invokes them when needed.
`Judge` scores a finished evaluation run.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Protocol

from chatbot_engine.models.chat import AssistantConfig, ChatRequest
from chatbot_engine.models.evals import JudgeReport, JudgeRequest
from chatbot_engine.models.events import Event


class Agent(Protocol):
    """Define the contract for processing one chat turn.

    Implementations are responsible for retrieval, prompt construction, model
    execution, optional tool calls, and streaming response events.
    """

    def run(
        self,
        request: ChatRequest,
    ) -> AsyncIterator[Event]:
        """Process one chat request and stream engine events.

        Args:
            request: Complete chat request including assistant configuration,
                user message, history, and caller context.
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
