"""Who the MCP tool server answers: the Host it knows, and the token it was given.

The tools take their owner from the `X-User-Id` a call carries, so the server
is only as safe as the set of callers that can reach it. These drive the app as
`main` serves it, with no database: an `initialize` touches none.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from support_agent import assistant, mcp_tools
from support_agent.settings import Settings, get_settings

INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "1"},
    },
}


def _initialize(settings: Settings, base_url: str, **headers: str) -> int:
    """The status the server answers an MCP `initialize` with."""
    app = mcp_tools.http_app(settings)
    with TestClient(app, base_url=base_url) as client:
        response = client.post(
            "/mcp",
            content=json.dumps(INITIALIZE),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                **headers,
            },
        )
    return response.status_code


def _settings(**overrides: object) -> Settings:
    """Settings from explicit values only, ignoring any .env on this machine."""
    return Settings.model_construct(
        **{"mcp_tools_url": None, "mcp_tools_token": None, **overrides}
    )


def test_it_listens_on_loopback_unless_told_otherwise() -> None:
    assert mcp_tools.HOST == "127.0.0.1"


def test_it_answers_the_host_the_engine_is_told_to_dial() -> None:
    """Under Compose it listens on 0.0.0.0 and the engine dials `mcp-tools`."""
    settings = _settings(mcp_tools_url="http://mcp-tools:8200/mcp")

    assert _initialize(settings, "http://mcp-tools:8200") == 200
    assert _initialize(settings, "http://localhost:8200") == 200


def test_it_refuses_a_name_rebound_to_its_address() -> None:
    """The rebinding guard stays on whatever address the server listens on."""
    settings = _settings(mcp_tools_url="http://mcp-tools:8200/mcp")

    assert _initialize(settings, "http://rebind.attacker.example") == 421


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({}, 401),
        ({"Authorization": "Bearer wrong"}, 401),
        ({"Authorization": "s3cret-tools-token"}, 401),
        ({"Authorization": "Bearer s3cret-tools-token"}, 200),
    ],
    ids=["none", "wrong", "not bearer", "right"],
)
def test_with_a_token_set_it_takes_only_that_token(
    headers: dict[str, str], expected: int
) -> None:
    settings = _settings(mcp_tools_token="s3cret-tools-token")

    assert _initialize(settings, "http://localhost:8200", **headers) == expected


def test_the_backend_has_the_engine_send_the_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """To the tool server alone: it goes in that server's entry, which the
    engine sends to no other server and never logs."""
    monkeypatch.setenv("BACKEND_MCP_TOOLS_TOKEN", "s3cret-tools-token")
    get_settings.cache_clear()
    assistant.load_project.cache_clear()
    try:
        (server,) = assistant.load_project("support").mcp_servers
        assert server.headers == {"Authorization": "Bearer s3cret-tools-token"}
        assert "s3cret" not in repr(server)
    finally:
        get_settings.cache_clear()
        assistant.load_project.cache_clear()
