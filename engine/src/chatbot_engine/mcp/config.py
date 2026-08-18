"""Which MCP servers to talk to, and which of their tools are permitted.

Configuration only -- no protocol, no orchestration. The rules here are the ones
worth settling before any code calls a tool:

* servers arrive per request, inside `AssistantConfig`, so the engine holds no
  tool configuration of its own;
* a server's tool *descriptions* end up inside the prompt, so only names the
  caller explicitly allowlisted may ever be exposed to a model.
"""

from __future__ import annotations

from dataclasses import dataclass

from chatbot_engine.models.chat import AssistantConfig, McpServerConfig


@dataclass(frozen=True, slots=True)
class McpTarget:
    """One resolved connection: where to reach a server and what it may offer."""

    name: str
    url: str
    allowed_tools: frozenset[str]
    timeout_s: float

    def allows(self, tool_name: str) -> bool:
        """Whether this tool may be shown to a model.

        Called after discovery, before tool schemas reach the prompt. A server
        that starts advertising a new tool does not silently gain access.
        """
        return tool_name in self.allowed_tools


def resolve_targets(config: AssistantConfig, *, timeout_s: float) -> list[McpTarget]:
    """Turn the request's server declarations into connection targets."""
    return [_target(server, timeout_s=timeout_s) for server in config.mcp_servers]


def _target(server: McpServerConfig, *, timeout_s: float) -> McpTarget:
    return McpTarget(
        name=server.name,
        url=server.url,
        allowed_tools=frozenset(server.allowed_tools),
        timeout_s=timeout_s,
    )
