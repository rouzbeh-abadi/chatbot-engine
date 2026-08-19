"""MCP connectivity: target resolution and the allowlist gate.

The allowlist is the security-relevant part. Tool names and descriptions come from
a server and end up inside the prompt, so a name that was never allowlisted must
be refused -- both when filtering discovery results and again at call time, since
by then the name has passed through model output.
"""

from __future__ import annotations

import pytest

from chatbot_engine.mcp import McpToolProvider, resolve_targets
from chatbot_engine.mcp.client import (
    McpServerNotFoundError,
    McpToolNotAllowedError,
)
from chatbot_engine.models.chat import AssistantConfig, McpServerConfig


def _config(**servers: object) -> AssistantConfig:
    return AssistantConfig(
        project_id="support",
        name="Support",
        system_prompt="s",
        mcp_servers=[
            McpServerConfig(
                name="support-tools",
                url="http://localhost:8200/mcp",
                allowed_tools=["get_booking_status", "get_flight_status"],
            )
        ],
    )


def test_no_servers_means_no_targets() -> None:
    bare = AssistantConfig(project_id="p", name="n", system_prompt="s")

    assert resolve_targets(bare, timeout_s=30.0) == []


def test_targets_carry_the_url_and_the_allowlist() -> None:
    (target,) = resolve_targets(_config(), timeout_s=12.5)

    assert target.name == "support-tools"
    assert target.url == "http://localhost:8200/mcp"
    assert target.timeout_s == 12.5


def test_only_allowlisted_tools_are_permitted() -> None:
    """A server that starts advertising a new tool gains nothing by doing so."""
    (target,) = resolve_targets(_config(), timeout_s=30.0)

    assert target.allows("get_booking_status")
    assert not target.allows("delete_all_bookings")


async def test_calling_a_tool_outside_the_allowlist_is_refused() -> None:
    """The tool name arrives from model output, so it is checked again at call
    time -- not just filtered during discovery. Refused before any network call,
    which is why this test needs no server."""
    provider = McpToolProvider(timeout_s=30.0)

    with pytest.raises(McpToolNotAllowedError, match="delete_all_bookings"):
        await provider.call_tool(
            config=_config(),
            server="support-tools",
            name="delete_all_bookings",
            arguments={},
        )


async def test_calling_an_unconfigured_server_is_refused() -> None:
    """A server the request never declared cannot be reached, whatever the model
    asks for."""
    provider = McpToolProvider(timeout_s=30.0)

    with pytest.raises(McpServerNotFoundError, match="somewhere-else"):
        await provider.call_tool(
            config=_config(),
            server="somewhere-else",
            name="get_booking_status",
            arguments={},
        )
