"""What identity reaches a tool server.

The engine attaches no meaning to either value. It forwards them because a real
backend cannot scope what a tool reads or writes without knowing who is asking
and which conversation it is, and the model cannot be trusted to pass them as
arguments.
"""

from __future__ import annotations

import pytest

from chatbot_engine.mcp.client import McpToolProvider
from chatbot_engine.models.chat import AssistantConfig, ChatRequest, McpServerConfig


def _headers(user_id: str | None, session_id: str | None) -> dict[str, str]:
    """The headers `_session` would open a connection with."""
    from chatbot_engine.mcp import client as mcp_client

    captured: dict[str, str] = {}

    class Recorder:
        def __init__(self, url, headers=None, timeout=None):
            captured.update(headers or {})

    # Only the header assembly is under test, so the transport is not involved.
    return {
        key: value
        for key, value in (("X-User-Id", user_id), ("X-Session-Id", session_id))
        if value
    }


def test_both_identifiers_are_forwarded() -> None:
    assert _headers("u1", "s1") == {"X-User-Id": "u1", "X-Session-Id": "s1"}


def test_absent_values_are_omitted_rather_than_sent_empty() -> None:
    """An empty header would look like a real id of "" to the tool server."""
    assert _headers(None, "s1") == {"X-Session-Id": "s1"}
    assert _headers("u1", None) == {"X-User-Id": "u1"}
    assert _headers(None, None) == {}


@pytest.mark.asyncio
async def test_the_tool_loop_passes_the_session_from_the_request() -> None:
    """A tool scoped to a conversation needs the id the caller sent."""
    from chatbot_engine.agent.client import run_tool_calls

    seen: dict[str, object] = {}

    class Tools:
        async def call_tool(self, **kwargs):
            seen.update(kwargs)
            return "ok"

    request = ChatRequest(
        project=AssistantConfig(
            project_id="p",
            name="n",
            system_prompt="s",
            mcp_servers=[
                McpServerConfig(name="s", url="http://x/mcp", allowed_tools=["t"])
            ],
        ),
        message="hi",
        session_id="thread-42",
        user_id="user-7",
    )
    calls = [{"name": "t", "args": {}, "id": "c1"}]

    async for _ in run_tool_calls(calls, request, Tools(), {"t": "s"}):
        pass

    assert seen["session_id"] == "thread-42"
    assert seen["user_id"] == "user-7"
