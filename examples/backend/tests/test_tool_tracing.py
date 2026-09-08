"""A tool call can be found in this server's log by the engine's request id."""

from __future__ import annotations

import logging
from types import SimpleNamespace

from support_agent.mcp_tools import _trace


def test_a_tool_call_is_logged_with_the_request_id_the_engine_sent(caplog) -> None:
    ctx = SimpleNamespace(headers={"x-request-id": "live-trace-2", "x-user-id": "u1"})

    with caplog.at_level(logging.INFO, logger="support_agent.mcp_tools"):
        _trace(ctx, "get_booking_status")

    assert "tool get_booking_status [live-trace-2]" in caplog.text
    assert "u1" not in caplog.text, "only the request id; the rest is the customer's"


def test_a_call_with_no_request_id_is_still_logged(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="support_agent.mcp_tools"):
        _trace(SimpleNamespace(headers=None), "remember")

    assert "tool remember [-]" in caplog.text
