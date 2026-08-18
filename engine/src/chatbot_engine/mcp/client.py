"""MCP client connectivity.

The engine is an MCP *client*. The tools themselves live in the application
backend and run in its process, because a tool reads the business database and
must execute with the calling user's permissions -- something the engine cannot
evaluate.

    engine  --MCP over streamable HTTP-->  backend MCP server (:8200)

What is prepared here: target resolution, the transport URL, the timeout, the
allowlist gate, and the shape the agent will call. What is deliberately absent:
the session handshake, discovery, and invocation. Those need the `mcp` SDK, which
is declared as the `mcp` extra and intentionally not imported yet.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from chatbot_engine.errors import NotConfiguredError
from chatbot_engine.models.chat import AssistantConfig


class McpToolProvider:
    """Satisfies `ports.agent.ToolProvider`. Connectivity prepared, calls pending."""

    def __init__(self, *, timeout_s: float) -> None:
        self._timeout_s = timeout_s

    async def list_tools(self, config: AssistantConfig) -> Sequence[Mapping[str, Any]]:
        """Discover tools, keep only allowlisted ones, return model-ready schemas.

        Implementation sketch: `resolve_targets(config, timeout_s=self._timeout_s)`
        gives the servers this request may reach; for each one open a
        streamable-HTTP session, call `tools/list`, drop anything
        `target.allows()` rejects, then map the remainder onto your model's
        tool-schema format.
        """
        raise NotConfiguredError(
            "MCP tool discovery is not implemented -- see "
            "chatbot_engine/mcp/client.py and the `mcp` extra"
        )

    async def call_tool(
        self,
        *,
        server: str,
        name: str,
        arguments: Mapping[str, Any],
        user_id: str | None = None,
    ) -> str:
        """Invoke one tool on one server and return its result as text.

        Two rules for the implementation: check `target.allows(name)` again here,
        since the name arrives from model output; and treat whatever comes back as
        untrusted data, never as instructions.
        """
        raise NotConfiguredError(
            "MCP tool invocation is not implemented -- see "
            "chatbot_engine/mcp/client.py and the `mcp` extra"
        )
