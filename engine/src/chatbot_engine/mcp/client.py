"""MCP client: discover and invoke the application's tools over MCP.

This is where the engine actually talks to a tool server. It opens a streamable
HTTP connection to each configured server, lists the tools it offers (keeping
only the allowlisted ones), and invokes a tool when the model asks for one.

The configuration side - which servers, which tools are allowed - lives in
`mcp/config.py`; this module is the network side that acts on it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import (
    create_mcp_http_client,
    streamable_http_client,
)
from mcp.types import TextContent

from chatbot_engine.mcp.config import McpTarget, resolve_targets
from chatbot_engine.models.chat import AssistantConfig


@asynccontextmanager
async def _session(target: McpTarget, *, user_id: str | None = None):
    """Open one MCP session to a target, with the configured timeout.

    The timeout matters: without it a tool server that accepts a connection but
    never answers would hang the whole turn. `user_id`, when present, is
    forwarded as an `X-User-Id` header so the tool server can authorise the call.

    A caller-provided http client is not lifecycle-managed by the transport, so
    it is opened here and closed when the session ends.
    """
    headers = {"X-User-Id": user_id} if user_id else None
    async with (
        create_mcp_http_client(
            headers=headers, timeout=httpx2.Timeout(target.timeout_s)
        ) as http_client,
        streamable_http_client(target.url, http_client=http_client) as (
            read_stream,
            write_stream,
        ),
        ClientSession(read_stream, write_stream) as session,
    ):
        await session.initialize()
        yield session


class McpServerNotFoundError(ValueError):
    """Raised when the requested MCP server is not configured."""


class McpToolNotAllowedError(ValueError):
    """Raised when the requested MCP tool is not allowlisted."""


class McpToolProvider:
    """Discover and invoke application-owned tools through MCP."""

    def __init__(self, *, timeout_s: float) -> None:
        """Initialize the MCP tool provider.

        Args:
            timeout_s: Maximum time allowed for MCP server operations.
        """
        self._timeout_s = timeout_s

    async def list_tools(
        self,
        config: AssistantConfig,
    ) -> Sequence[Mapping[str, Any]]:
        """Discover tools allowed by the assistant configuration.

        Args:
            config: Assistant configuration containing MCP server definitions.

        Returns:
            Allowlisted tool schemas discovered from configured MCP servers.
        """
        tools: list[Mapping[str, Any]] = []

        targets = resolve_targets(
            config,
            timeout_s=self._timeout_s,
        )

        for target in targets:
            async with _session(target) as session:
                result = await session.list_tools()

                for tool in result.tools:
                    if not target.allows(tool.name):
                        continue

                    tools.append(
                        {
                            "server": target.name,
                            "name": tool.name,
                            "description": tool.description or "",
                            "input_schema": tool.input_schema,
                        }
                    )

        return tools

    async def call_tool(
        self,
        *,
        config: AssistantConfig,
        server: str,
        name: str,
        arguments: Mapping[str, Any],
        user_id: str | None = None,
    ) -> str:
        """Invoke one allowlisted tool on a configured MCP server.

        Args:
            config: Assistant configuration containing permitted MCP servers.
            server: Name of the MCP server exposing the tool.
            name: Name of the tool to invoke.
            arguments: Arguments passed to the tool.
            user_id: Opaque user identifier, forwarded to the tool server as an
                `X-User-Id` header so it can authorise the call.

        Returns:
            Tool result serialized as text.

        Raises:
            McpServerNotFoundError: If the requested server is not configured.
            McpToolNotAllowedError: If the tool is not allowlisted.
        """
        target = self._find_target(
            config=config,
            server=server,
        )

        if not target.allows(name):
            raise McpToolNotAllowedError(
                f"Tool {name!r} is not allowed on MCP server {server!r}."
            )

        async with _session(target, user_id=user_id) as session:
            result = await session.call_tool(
                name,
                dict(arguments),
            )

        if result.structured_content is not None:
            return json.dumps(
                result.structured_content,
                ensure_ascii=False,
            )

        text_parts = [
            block.text
            for block in result.content
            if isinstance(block, TextContent)
        ]

        return "\n".join(text_parts)

    def _find_target(
        self,
        *,
        config: AssistantConfig,
        server: str,
    ) -> McpTarget:
        """Find one configured MCP server by name.

        Args:
            config: Assistant configuration containing MCP server definitions.
            server: Name of the MCP server to resolve.

        Returns:
            Matching MCP connection target.

        Raises:
            McpServerNotFoundError: If no configured server matches the name.
        """
        targets = resolve_targets(
            config,
            timeout_s=self._timeout_s,
        )

        for target in targets:
            if target.name == server:
                return target

        raise McpServerNotFoundError(
            f"MCP server {server!r} is not configured."
        )