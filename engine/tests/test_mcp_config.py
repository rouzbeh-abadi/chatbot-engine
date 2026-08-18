"""MCP connectivity configuration -- the part that is implemented.

Discovery and invocation are not written yet; the allowlist gate is, because it is
a security rule rather than AI logic and is easy to get wrong later.
"""

from __future__ import annotations

import pytest

from chatbot_engine.errors import NotConfiguredError
from chatbot_engine.mcp import McpToolProvider, resolve_targets
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


async def test_discovery_and_invocation_are_not_implemented_yet() -> None:
    provider = McpToolProvider(timeout_s=30.0)

    with pytest.raises(NotConfiguredError):
        await provider.list_tools(_config())

    with pytest.raises(NotConfiguredError):
        await provider.call_tool(server="support-tools", name="x", arguments={})
