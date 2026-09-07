"""What the engine sends to a tool server, and what it makes of the answer.

Three properties. The allowlist is enforced at discovery and again at call
time, because the tool name arrives from model output. A result flagged
`isError` is a failed call, not a successful one with odd text. And the
caller's identifiers reach the server as headers, never as tool arguments.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from mcp.types import CallToolResult, TextContent

from chatbot_engine.agent.client import run_tool_calls
from chatbot_engine.mcp import client as mcp_client
from chatbot_engine.mcp.client import (
    McpServerNotFoundError,
    McpToolNotAllowedError,
    McpToolProvider,
)
from chatbot_engine.models.chat import AssistantConfig, ChatRequest, McpServerConfig
from chatbot_engine.models.events import ToolCallFinishedEvent

CONFIG = AssistantConfig(
    project_id="p",
    name="n",
    system_prompt="s",
    mcp_servers=[
        McpServerConfig(
            name="s", url="http://x/mcp", allowed_tools=["get_booking_status"]
        )
    ],
)


def _serving(*, result: CallToolResult | None = None, tools: list[str] = ()):
    """A tool server session with these tools, answering every call with `result`."""
    opened: list[dict] = []

    class Session:
        async def list_tools(self):
            return SimpleNamespace(
                tools=[
                    SimpleNamespace(name=name, description="", input_schema={})
                    for name in tools
                ]
            )

        async def call_tool(self, name, arguments):
            return result

    @asynccontextmanager
    async def session(target, **kwargs):
        opened.append(kwargs)
        yield Session()

    return patch.object(mcp_client, "_session", session), opened


def _result(text: str, *, is_error: bool) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=text)], isError=is_error
    )


# --- the allowlist ------------------------------------------------------------


async def test_discovery_drops_tools_outside_the_allowlist() -> None:
    """A server that starts advertising a new tool gains nothing by doing so:
    the name and description never reach the prompt."""
    serving, _ = _serving(tools=["get_booking_status", "delete_all_bookings"])

    with serving:
        tools = await McpToolProvider(timeout_s=1).list_tools(CONFIG)

    assert [tool["name"] for tool in tools] == ["get_booking_status"]


async def test_a_call_outside_the_allowlist_is_refused_before_any_network() -> None:
    """Checked again at call time, since by then the name came from the model."""
    with pytest.raises(McpToolNotAllowedError, match="delete_all_bookings"):
        await McpToolProvider(timeout_s=1).call_tool(
            config=CONFIG, server="s", name="delete_all_bookings", arguments={}
        )


async def test_a_server_the_request_never_declared_is_refused() -> None:
    with pytest.raises(McpServerNotFoundError, match="elsewhere"):
        await McpToolProvider(timeout_s=1).call_tool(
            config=CONFIG, server="elsewhere", name="get_booking_status", arguments={}
        )


# --- the error flag -----------------------------------------------------------


async def test_a_result_flagged_as_an_error_is_reported_as_a_failed_call() -> None:
    """The bug this guards: `ok=true` with the error text as the result."""
    serving, _ = _serving(result=_result("database unavailable", is_error=True))
    request = ChatRequest(project=CONFIG, message="hi")
    calls = [{"name": "get_booking_status", "args": {}, "id": "c1"}]

    with serving:
        finished = [
            event
            async for event in run_tool_calls(
                calls,
                request,
                McpToolProvider(timeout_s=1),
                {"get_booking_status": "s"},
            )
            if isinstance(event, ToolCallFinishedEvent)
        ]

    (event,) = finished
    assert event.ok is False
    assert "database unavailable" in (event.error or "")


# --- caller context -----------------------------------------------------------


async def test_the_callers_identifiers_reach_the_server_as_headers() -> None:
    """A tool that scopes what it reads needs them, and must not get them from
    an argument the model could have written."""
    serving, opened = _serving(result=_result("ok", is_error=False))
    request = ChatRequest(
        project=CONFIG, message="hi", session_id="thread-42", user_id="user-7"
    )
    calls = [{"name": "get_booking_status", "args": {}, "id": "c1"}]

    with serving:
        async for _ in run_tool_calls(
            calls, request, McpToolProvider(timeout_s=1), {"get_booking_status": "s"}
        ):
            pass

    assert opened == [{"user_id": "user-7", "session_id": "thread-42"}]
