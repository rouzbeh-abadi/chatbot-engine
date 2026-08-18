"""MCP client connectivity: configuration and transport, not orchestration.

The engine calls tools that live in the application backend. Only the connection
and allowlist layer is prepared here -- discovery and invocation are yours.
"""

from chatbot_engine.mcp.client import McpToolProvider
from chatbot_engine.mcp.config import McpTarget, resolve_targets

__all__ = ["McpTarget", "McpToolProvider", "resolve_targets"]
