"""A hand-off tells someone before it promises the visitor anyone.

The step used to stream "a person will follow up" and then call its tool,
whatever came of the call, and call nothing at all when the tool was not
offered. Now the tool goes first; the message is said only when the call
worked, and otherwise the visitor hears the assistant's `unavailable_message`
and the turn ends there, as a failed tool step ends it.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from test_agent_parity import ScriptedModel, _no_retrieval

from chatbot_engine.agent.client import DEFAULT_UNAVAILABLE_MESSAGE
from chatbot_engine.models.chat import (
    AssistantConfig,
    ChatRequest,
    McpServerConfig,
    Message,
)
from chatbot_engine.models.events import (
    DoneEvent,
    TokenEvent,
    ToolCallFinishedEvent,
    ToolCallStartedEvent,
)

pytest.importorskip("langgraph_agent.workflow")
from langgraph_agent.workflow import WorkflowAgent

PROMISE = "A colleague will email you shortly."

HANDOFF = {
    "start": "handoff",
    "nodes": [
        {
            "id": "handoff",
            "type": "handoff",
            "message": PROMISE,
            "tool": "hand_off_to_human",
        },
        {"id": "after", "type": "reply", "text": " (handed off: {{vars.handed_off}})"},
    ],
    "edges": [{"from": "handoff", "to": "after"}],
}


class HandoffTools:
    """Offers the hand-off tool unless told not to; records every call."""

    def __init__(self, *, offered: bool = True, fails: bool = False) -> None:
        self.offered = offered
        self.fails = fails
        self.calls: list[dict] = []

    async def list_tools(self, config):
        if not self.offered:
            return []
        return [
            {
                "server": "s",
                "name": "hand_off_to_human",
                "description": "d",
                "input_schema": {},
            }
        ]

    async def call_tool(self, *, name, arguments, **kwargs):
        self.calls.append(dict(arguments))
        if self.fails:
            raise RuntimeError("helpdesk down")
        return "ticket 42"


def _request(spec: dict, history: list[Message] | None = None) -> ChatRequest:
    return ChatRequest(
        project=AssistantConfig(
            project_id="support",
            name="S",
            system_prompt="You are helpful.",
            mcp_servers=[
                McpServerConfig(
                    name="s", url="http://s", allowed_tools=["hand_off_to_human"]
                )
            ],
            workflow=spec,
        ),
        message="I want a person",
        history=history or [],
    )


async def _run(spec: dict, tools: HandoffTools, history=None) -> list:
    agent = WorkflowAgent(tools=tools)
    with (
        patch(
            "langgraph_agent.workflow.build_chat_model",
            return_value=ScriptedModel(rounds=[], seen=[]),
        ),
        patch("langgraph_agent.workflow.retrieve_with_usage", new=_no_retrieval),
    ):
        return [event async for event in agent.run(_request(spec, history))]


def _text(events: list) -> str:
    return "".join(e.text for e in events if isinstance(e, TokenEvent))


async def test_the_promise_comes_after_the_call_that_worked():
    tools = HandoffTools()

    events = await _run(HANDOFF, tools)

    assert len(tools.calls) == 1
    finished = next(
        i for i, e in enumerate(events) if isinstance(e, ToolCallFinishedEvent)
    )
    first_token = next(i for i, e in enumerate(events) if isinstance(e, TokenEvent))
    assert finished < first_token, "nothing is promised before the call returns"
    assert events[finished].ok
    assert _text(events) == f"{PROMISE} (handed off: true)"
    assert isinstance(events[-1], DoneEvent) and events[-1].finish_reason == "stop"


async def test_a_failed_call_promises_nothing_and_ends_the_turn():
    tools = HandoffTools(fails=True)

    events = await _run(HANDOFF, tools)

    finished = [e for e in events if isinstance(e, ToolCallFinishedEvent)]
    assert [e.ok for e in finished] == [False]
    assert _text(events) == DEFAULT_UNAVAILABLE_MESSAGE, "the steps after it do not run"
    assert PROMISE not in _text(events)
    assert isinstance(events[-1], DoneEvent) and events[-1].finish_reason == "stop"


async def test_a_tool_that_is_not_offered_is_reported_and_promises_nothing():
    """It used to be skipped in silence, the promise already made."""
    tools = HandoffTools(offered=False)

    events = await _run(HANDOFF, tools)

    assert tools.calls == []
    started = [e for e in events if isinstance(e, ToolCallStartedEvent)]
    finished = [e for e in events if isinstance(e, ToolCallFinishedEvent)]
    assert [e.tool for e in started] == ["hand_off_to_human"]
    assert finished[0].ok is False and "unavailable" in (finished[0].error or "")
    assert _text(events) == DEFAULT_UNAVAILABLE_MESSAGE


async def test_a_hand_off_without_a_tool_says_its_message():
    spec = {
        "start": "handoff",
        "nodes": [{"id": "handoff", "type": "handoff", "message": PROMISE}],
    }
    tools = HandoffTools()

    events = await _run(spec, tools)

    assert tools.calls == []
    assert _text(events) == PROMISE


async def test_the_transcript_carries_the_whole_conversation():
    """The query rewrite reads only the last few turns; the person who takes
    over reads all of them."""
    history = [
        Message(role="user" if i % 2 == 0 else "assistant", content=f"turn {i}")
        for i in range(10)
    ]
    tools = HandoffTools()

    await _run(HANDOFF, tools, history=history)

    transcript = tools.calls[0]["transcript"]
    assert transcript.startswith("user: turn 0\n")
    assert transcript.endswith("user: I want a person")
