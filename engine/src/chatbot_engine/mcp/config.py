"""Resolve which MCP servers to call and which of their tools are allowed.

Configuration only -- no network calls, no tool execution. Two rules matter here:

* MCP servers arrive with each request (in `AssistantConfig`), so the engine
  stores no tool configuration of its own;
* a tool's description is placed into the model's prompt, so only tools the
  caller explicitly allowlisted are ever exposed to the model.
"""

from __future__ import annotations

from dataclasses import dataclass

from chatbot_engine.models.chat import AssistantConfig, McpServerConfig


@dataclass(frozen=True, slots=True)
class McpTarget:
    """One resolved server connection: where to reach it and which tools it allows."""

    name: str
    url: str
    allowed_tools: frozenset[str]
    timeout_s: float

    def allows(self, tool_name: str) -> bool:
        """Whether this tool is allowlisted, so it may be shown to the model.

        Checked after discovery, before a tool's schema reaches the prompt. So a
        server that starts advertising a new tool does not gain access to it.
        """
        return tool_name in self.allowed_tools


def resolve_targets(config: AssistantConfig, *, timeout_s: float) -> list[McpTarget]:
    """Turn the request's MCP server declarations into connection targets."""
    return [_target(server, timeout_s=timeout_s) for server in config.mcp_servers]


def _target(server: McpServerConfig, *, timeout_s: float) -> McpTarget:
    """Build one target from a server config, applying the shared timeout."""
    return McpTarget(
        name=server.name,
        url=server.url,
        allowed_tools=frozenset(server.allowed_tools),
        timeout_s=timeout_s,
    )
