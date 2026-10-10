"""What the engine sends to a tool server, and what it makes of the answer.

Three properties. The allowlist is enforced at discovery and again at call
time, because the tool name arrives from model output. A result flagged
`isError` is a failed call, not a successful one with odd text. And the
caller's identifiers reach the server as headers, never as tool arguments.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch

import httpx2
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


async def test_a_server_that_is_down_is_left_out_and_the_others_still_serve() -> None:
    """One product's server failing must not take the whole assistant down."""
    config = CONFIG.model_copy(
        update={
            "mcp_servers": [
                McpServerConfig(
                    name="down", url="http://down/mcp", allowed_tools=["book"]
                ),
                McpServerConfig(
                    name="up", url="http://up/mcp", allowed_tools=["get_booking_status"]
                ),
            ]
        }
    )

    class Session:
        async def list_tools(self):
            return SimpleNamespace(
                tools=[
                    SimpleNamespace(
                        name="get_booking_status", description="", input_schema={}
                    )
                ]
            )

    @asynccontextmanager
    async def session(target, **kwargs):
        if target.name == "down":
            raise ConnectionError("All connection attempts failed")
        yield Session()

    with patch.object(mcp_client, "_session", session):
        tools = await McpToolProvider(timeout_s=1).list_tools(config)

    assert [(t["server"], t["name"]) for t in tools] == [("up", "get_booking_status")]


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


# --- discovery is cached ---------------------------------------------------------


async def test_a_servers_tool_list_is_reused_while_fresh() -> None:
    """Every turn needs the list, and a graph agent asks at every step; asking
    the server each time opens a connection to learn nothing new."""
    serving, opened = _serving(tools=["get_booking_status"])
    provider = McpToolProvider(timeout_s=1, tools_ttl_s=60)

    with serving:
        first = await provider.list_tools(CONFIG)
        second = await provider.list_tools(CONFIG)

    assert first == second
    assert len(opened) == 1, "one connection for two discoveries"


async def test_the_cache_is_keyed_by_the_allowlist() -> None:
    """Two assistants on one server with different allowlists must not share
    a filtered list: the second would see tools it never allowed."""
    serving, opened = _serving(tools=["get_booking_status", "get_flight_status"])
    provider = McpToolProvider(timeout_s=1, tools_ttl_s=60)
    wider = CONFIG.model_copy(
        update={
            "mcp_servers": [
                McpServerConfig(
                    name="s",
                    url="http://x/mcp",
                    allowed_tools=["get_booking_status", "get_flight_status"],
                )
            ]
        }
    )

    with serving:
        narrow = await provider.list_tools(CONFIG)
        wide = await provider.list_tools(wider)

    assert [t["name"] for t in narrow] == ["get_booking_status"]
    assert [t["name"] for t in wide] == ["get_booking_status", "get_flight_status"]
    assert len(opened) == 2


async def test_a_zero_ttl_asks_every_time() -> None:
    serving, opened = _serving(tools=["get_booking_status"])
    provider = McpToolProvider(timeout_s=1, tools_ttl_s=0)

    with serving:
        await provider.list_tools(CONFIG)
        await provider.list_tools(CONFIG)

    assert len(opened) == 2


async def test_an_expired_list_is_fetched_again() -> None:
    serving, opened = _serving(tools=["get_booking_status"])
    provider = McpToolProvider(timeout_s=1, tools_ttl_s=60)

    with serving:
        await provider.list_tools(CONFIG)
        # Age the entry past its TTL rather than sleeping.
        key = next(iter(provider._discovered))
        stamp, tools = provider._discovered[key]
        provider._discovered[key] = (stamp - 61, tools)
        await provider.list_tools(CONFIG)

    assert len(opened) == 2


# --- headers for one server ------------------------------------------------------


def test_a_servers_own_headers_win_for_that_server_only() -> None:
    """One server may get its own identity; the others keep the request's."""
    from chatbot_engine.mcp.client import headers_for
    from chatbot_engine.mcp.config import resolve_targets

    config = AssistantConfig(
        project_id="p",
        name="n",
        system_prompt="s",
        mcp_servers=[
            McpServerConfig(
                name="site",
                url="http://site/mcp",
                allowed_tools=["t"],
                headers={"x-user-id": "signed-token", "Authorization": "Bearer abc"},
            ),
            McpServerConfig(name="other", url="http://other/mcp", allowed_tools=["t"]),
        ],
    )
    site, other = resolve_targets(config, timeout_s=1)

    to_site = headers_for(site, "visitor:1", "conv-1")
    to_other = headers_for(other, "visitor:1", "conv-1")

    assert to_site["x-user-id"] == "signed-token"
    assert "X-User-Id" not in to_site, "replaced whatever the case, not sent twice"
    assert to_site["Authorization"] == "Bearer abc"
    assert to_site["X-Session-Id"] == "conv-1"
    assert to_other["X-User-Id"] == "visitor:1"
    assert "Authorization" not in to_other


def test_header_values_never_show_in_a_repr() -> None:
    server = McpServerConfig(
        name="site",
        url="http://site/mcp",
        allowed_tools=["t"],
        headers={"X-User-Id": "secret-token"},
    )
    from chatbot_engine.mcp.config import resolve_targets

    config = AssistantConfig(
        project_id="p", name="n", system_prompt="s", mcp_servers=[server]
    )
    assert "secret-token" not in repr(server)
    assert "secret-token" not in repr(config)
    assert "secret-token" not in repr(resolve_targets(config, timeout_s=1)[0])


@pytest.mark.parametrize(
    "headers",
    [
        {"Bad Name": "SECRETVALUE"},
        {"X-Request-Id": "SECRETVALUE"},
        {"X-User-Id": "SECRET\nVALUE"},
        {"X-User-Id": "SECRETVALUE" * 500},
        {f"H{i}": "SECRETVALUE" for i in range(11)},
    ],
)
def test_unsafe_headers_are_refused_without_echoing_the_value(
    headers: dict[str, str],
) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as caught:
        McpServerConfig(
            name="site", url="http://site/mcp", allowed_tools=["t"], headers=headers
        )
    messages = " ".join(str(e["msg"]) for e in caught.value.errors())
    assert "SECRET" not in messages


async def test_a_server_with_its_own_headers_is_not_cached() -> None:
    """Its list may depend on the credential, and the cache would hold it."""
    config = AssistantConfig(
        project_id="p",
        name="n",
        system_prompt="s",
        mcp_servers=[
            McpServerConfig(
                name="s",
                url="http://x/mcp",
                allowed_tools=["get_booking_status"],
                headers={"X-User-Id": "token"},
            )
        ],
    )
    serving, opened = _serving(tools=["get_booking_status"])
    provider = McpToolProvider(timeout_s=1, tools_ttl_s=60)

    with serving:
        await provider.list_tools(config)
        await provider.list_tools(config)

    assert len(opened) == 2


# --- what the logs say ----------------------------------------------------------

#: The application's own tool server: its address carries the assistant's token.
TOKENED = "https://chatfrom.example/api/mcp/a1b2/tok_SECRET123"


async def test_a_server_that_cannot_be_reached_is_logged_by_name_not_by_address(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = CONFIG.model_copy(
        update={
            "mcp_servers": [
                McpServerConfig(name="app", url=TOKENED, allowed_tools=["book"])
            ]
        }
    )

    @asynccontextmanager
    async def session(target, **kwargs):
        raise ConnectionError(f"Client error '401 Unauthorized' for url '{target.url}'")
        yield  # pragma: no cover

    with patch.object(mcp_client, "_session", session), caplog.at_level("WARNING"):
        await McpToolProvider(timeout_s=1).list_tools(config)

    assert "'app'" in caplog.text
    assert "https://chatfrom.example/…" in caplog.text
    assert "tok_SECRET123" not in caplog.text


async def test_a_failed_discovery_names_the_servers_and_not_their_addresses(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from chatbot_engine.agent.client import discover_tools

    class Broken:
        async def list_tools(self, config):
            raise RuntimeError(f"cannot reach {TOKENED}")

    config = CONFIG.model_copy(
        update={
            "mcp_servers": [
                McpServerConfig(name="app", url=TOKENED, allowed_tools=["book"])
            ]
        }
    )

    with caplog.at_level("WARNING"):
        assert await discover_tools(Broken(), config) == []

    assert "could not discover tools from app:" in caplog.text
    assert "tok_SECRET123" not in caplog.text
    assert "/api/mcp" not in caplog.text


def test_an_address_in_a_log_line_keeps_its_host_only() -> None:
    from chatbot_engine.agent.client import without_paths

    assert without_paths(f"failed on {TOKENED}.") == (
        "failed on https://chatfrom.example/…"
    )
    assert without_paths("http://user:pw@host:8200/mcp?key=1") == "http://host:8200/…"
    assert without_paths("http://down") == "http://down"
    assert without_paths("no address here") == "no address here"


# --- MCP-3: an answer is bounded in bytes, gzip included ------------------------


def _capped(response: Callable[[httpx2.Request], httpx2.Response], limit: int):
    from chatbot_engine.mcp.capped import CappedTransport

    return httpx2.AsyncClient(
        transport=CappedTransport(limit, inner=httpx2.MockTransport(response))
    )


async def test_a_compressed_answer_inside_the_limit_reads_as_usual() -> None:
    import gzip

    asked: list[str] = []

    def answer(request: httpx2.Request) -> httpx2.Response:
        asked.append(request.headers["accept-encoding"])
        return httpx2.Response(
            200,
            headers={"content-encoding": "gzip"},
            content=gzip.compress(b'{"ok": true}'),
        )

    async with _capped(answer, 1024) as client:
        response = await client.get("https://mcp.example/mcp")

    assert response.json() == {"ok": True}
    assert asked == ["gzip, deflate"]


async def test_a_small_body_that_inflates_past_the_limit_is_not_read_further() -> None:
    import gzip

    from chatbot_engine.mcp.capped import McpResponseRefusedError

    bomb = gzip.compress(b"a" * (8 * 1024 * 1024))
    assert len(bomb) < 64 * 1024

    def answer(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, headers={"content-encoding": "gzip"}, content=bomb)

    async with _capped(answer, 64 * 1024) as client:
        with pytest.raises(McpResponseRefusedError, match="more than 65,536 bytes"):
            await client.get("https://mcp.example/mcp")


async def test_a_plain_answer_past_the_limit_is_refused() -> None:
    from chatbot_engine.mcp.capped import McpResponseRefusedError

    def answer(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=b"a" * 100_000)

    async with _capped(answer, 64 * 1024) as client:
        with pytest.raises(McpResponseRefusedError):
            await client.get("https://mcp.example/mcp")


async def test_an_encoding_that_was_not_asked_for_is_refused() -> None:
    from chatbot_engine.mcp.capped import McpResponseRefusedError

    def answer(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, headers={"content-encoding": "br"}, content=b"x")

    async with _capped(answer, 1024) as client:
        with pytest.raises(McpResponseRefusedError, match="'br'"):
            await client.get("https://mcp.example/mcp")


# --- MCP-2: a redirect is followed only within the server's own origin --------


async def test_a_redirect_within_the_server_is_followed():
    """`/mcp` to `/mcp/` is how many servers answer the address without its
    slash; it stays on the server's origin, so it is followed. A redirect to
    another origin is refused (engine/tests/review/test_mcp*.py, MCP-2)."""
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Server(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, status: int, body: bytes = b"", **headers: str) -> None:
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name.replace("_", "-"), value)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(405)

        def do_DELETE(self):
            self._send(200)

        def do_POST(self):
            n = int(self.headers.get("content-length") or 0)
            message = _json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/mcp":
                return self._send(307, location="/mcp/")
            if "id" not in message:
                return self._send(202)
            results = {
                "initialize": {
                    "protocolVersion": message.get("params", {}).get("protocolVersion"),
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "slash", "version": "1"},
                },
                "tools/list": {
                    "tools": [
                        {
                            "name": "lookup",
                            "description": "d",
                            "inputSchema": {"type": "object"},
                        }
                    ]
                },
            }
            body = _json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "result": results[message["method"]],
                }
            )
            self._send(200, body.encode(), content_type="application/json")

    server = HTTPServer(("127.0.0.1", 0), Server)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        config = AssistantConfig(
            project_id="p",
            name="p",
            system_prompt=".",
            mcp_servers=[
                McpServerConfig(
                    name="s",
                    url=f"http://127.0.0.1:{server.server_address[1]}/mcp",
                    allowed_tools=["lookup"],
                )
            ],
        )
        tools = await McpToolProvider(timeout_s=5).list_tools(config)
    finally:
        server.shutdown()

    assert [t["name"] for t in tools] == ["lookup"]
