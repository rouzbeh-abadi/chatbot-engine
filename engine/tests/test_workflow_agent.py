"""The workflow agent runs the graph an assistant describes."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessageChunk
from test_agent_parity import (
    FakeTools,
    ScriptedModel,
    _no_retrieval,
    _rounds_with_a_tool_call,
)

from chatbot_engine.models.chat import AssistantConfig, ChatRequest, McpServerConfig
from chatbot_engine.models.events import (
    DoneEvent,
    TokenEvent,
    ToolCallFinishedEvent,
    ToolCallStartedEvent,
    UsageEvent,
)

pytest.importorskip("langgraph_agent.workflow")
from langgraph_agent.workflow import WorkflowAgent


def _request(workflow: dict | None) -> ChatRequest:
    return ChatRequest(
        project=AssistantConfig(
            project_id="support",
            name="S",
            system_prompt="You are helpful.",
            model="openai/gpt-5-mini",
            mcp_servers=[
                McpServerConfig(
                    name="tools",
                    url="http://tools",
                    allowed_tools=["get_booking_status"],
                )
            ],
            workflow=workflow,
        ),
        message="is my flight delayed?",
        user_id="u1",
        session_id="s1",
    )


async def _run(
    workflow: dict | None,
    model: ScriptedModel,
    tools: FakeTools | None = None,
    request: ChatRequest | None = None,
) -> list:
    agent = WorkflowAgent(tools=tools or FakeTools())
    with (
        patch("langgraph_agent.workflow.build_chat_model", return_value=model),
        patch("langgraph_agent.workflow.retrieve_with_usage", new=_no_retrieval),
    ):
        return [event async for event in agent.run(request or _request(workflow))]


def _text(events: list) -> str:
    return "".join(e.text for e in events if isinstance(e, TokenEvent))


async def test_without_a_workflow_it_retrieves_and_answers():
    events = await _run(
        None, ScriptedModel(rounds=[[AIMessageChunk(content="On time.")]], seen=[])
    )
    assert _text(events) == "On time."
    assert isinstance(events[-1], DoneEvent) and isinstance(events[-2], UsageEvent)


async def test_a_cut_reply_ends_the_turn_with_length():
    cut = AIMessageChunk(content="", response_metadata={"finish_reason": "length"})
    events = await _run(
        None, ScriptedModel(rounds=[[AIMessageChunk(content="On"), cut]], seen=[])
    )
    assert isinstance(events[-1], DoneEvent)
    assert events[-1].finish_reason == "length"


async def test_a_tool_step_reports_timing_and_failure_like_a_model_call():
    """The Tool Call step goes through the engine's runner: a started and a
    finished event with a real duration, and a failure that fills the variable
    with nothing rather than ending the turn."""
    spec = {
        "start": "lookup",
        "nodes": [
            {
                "id": "lookup",
                "type": "tool",
                "tool": "get_booking_status",
                "arguments": {"ref": "{{message}}"},
                "var": "status",
            },
            {"id": "say", "type": "reply", "text": "Status: {{vars.status}}"},
        ],
        "edges": [{"from": "lookup", "to": "say"}],
    }
    events = await _run(spec, ScriptedModel(rounds=[], seen=[]))
    started = [e for e in events if isinstance(e, ToolCallStartedEvent)]
    finished = [e for e in events if isinstance(e, ToolCallFinishedEvent)]
    assert [e.tool for e in started] == ["get_booking_status"]
    assert (
        finished[0].ok
        and finished[0].duration_ms >= 0
        and finished[0].call_id == "wf-lookup"
    )
    assert _text(events).startswith("Status: ")

    class BrokenTools(FakeTools):
        async def call_tool(self, *, name, **kwargs):
            raise RuntimeError("tool server down")

    events = await _run(spec, ScriptedModel(rounds=[], seen=[]), tools=BrokenTools())
    finished = [e for e in events if isinstance(e, ToolCallFinishedEvent)]
    assert finished[0].ok is False and "tool server down" in (finished[0].error or "")
    assert _text(events) == "Status: ", "a failed call leaves the variable empty"
    assert isinstance(events[-1], DoneEvent)


async def test_a_model_step_that_runs_out_of_tool_rounds_ends_with_tool_limit():
    """The Chat Model step counts like the loop agent: two tool rounds, three
    model calls, then `tool_limit` with the usage of every call."""
    asks = _rounds_with_a_tool_call()[0]
    request = _request(None)
    request.project = request.project.model_copy(update={"max_tool_iterations": 2})

    events = await _run(
        None, ScriptedModel(rounds=[asks, asks, asks], seen=[]), request=request
    )

    assert isinstance(events[-1], DoneEvent)
    assert events[-1].finish_reason == "tool_limit"
    assert [e.tool for e in events if isinstance(e, ToolCallFinishedEvent)] == [
        "get_booking_status",
        "get_booking_status",
    ]
    usage = next(e for e in events if isinstance(e, UsageEvent))
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (
        90,
        30,
        120,
    )


async def test_a_reply_node_speaks_a_template_and_ends():
    spec = {
        "start": "hello",
        "nodes": [
            {
                "id": "hello",
                "type": "reply",
                "text": "Hi {{user_id}}, you asked: {{message}}",
            }
        ],
    }
    events = await _run(spec, ScriptedModel(rounds=[], seen=[]))
    assert _text(events) == "Hi u1, you asked: is my flight delayed?"
    assert isinstance(events[-1], DoneEvent)


async def test_a_condition_routes_to_the_branch_the_model_names():
    spec = {
        "start": "kind",
        "nodes": [
            {
                "id": "kind",
                "type": "condition",
                "question": "Order or other?",
                "branches": {"order": "lookup", "other": "shrug"},
            },
            {
                "id": "lookup",
                "type": "tool",
                "tool": "get_booking_status",
                "arguments": {"reference": "{{message}}"},
                "var": "booking",
            },
            {"id": "tell", "type": "reply", "text": "Status: {{vars.booking}}"},
            {"id": "shrug", "type": "reply", "text": "No idea."},
        ],
        "edges": [{"from": "lookup", "to": "tell"}],
    }
    model = ScriptedModel(rounds=[[AIMessageChunk(content="order")]], seen=[])
    events = await _run(spec, model)
    assert _text(events) == 'Status: {"status": "delayed"}'
    assert any(isinstance(e, ToolCallFinishedEvent) and e.ok for e in events)


async def test_a_turn_without_a_model_step_still_names_the_model_it_is_priced_at():
    """A condition-only workflow on an assistant using the engine's default
    model reports that default, not null, so the caller can price the turn."""
    from chatbot_engine.settings import get_settings

    spec = {
        "start": "kind",
        "nodes": [
            {
                "id": "kind",
                "type": "condition",
                "question": "?",
                "branches": {"a": "ya", "b": "yb"},
            },
            {"id": "ya", "type": "reply", "text": "A"},
            {"id": "yb", "type": "reply", "text": "B"},
        ],
    }
    request = _request(spec)
    request.project = request.project.model_copy(update={"model": None})
    chunk = AIMessageChunk(
        content="a",
        usage_metadata={"input_tokens": 7, "output_tokens": 1, "total_tokens": 8},
    )

    events = await _run(spec, ScriptedModel(rounds=[[chunk]], seen=[]), request=request)

    usage = next(e for e in events if isinstance(e, UsageEvent))
    assert usage.model == get_settings().chat_model
    assert usage.total_tokens == 8


async def test_a_condition_falls_back_to_the_first_branch():
    spec = {
        "start": "kind",
        "nodes": [
            {
                "id": "kind",
                "type": "condition",
                "question": "?",
                "branches": {"a": "ya", "b": "yb"},
            },
            {"id": "ya", "type": "reply", "text": "A"},
            {"id": "yb", "type": "reply", "text": "B"},
        ],
    }
    events = await _run(
        spec, ScriptedModel(rounds=[[AIMessageChunk(content="mumble")]], seen=[])
    )
    assert _text(events) == "A"


async def test_a_model_step_can_store_its_reply_for_a_later_step():
    spec = {
        "start": "draft",
        "nodes": [
            {"id": "draft", "type": "model", "var": "draft", "tools": False},
            {"id": "say", "type": "reply", "text": "Draft was: {{vars.draft}}"},
        ],
        "edges": [{"from": "draft", "to": "say"}],
    }
    events = await _run(
        spec, ScriptedModel(rounds=[[AIMessageChunk(content="secret")]], seen=[])
    )
    assert _text(events) == "Draft was: secret"


async def test_a_handoff_marks_the_turn_and_speaks():
    spec = {
        "start": "h",
        "nodes": [
            {
                "id": "h",
                "type": "handoff",
                "message": "A person will reply to {{user_id}}.",
            }
        ],
    }
    events = await _run(spec, ScriptedModel(rounds=[], seen=[]))
    assert _text(events) == "A person will reply to u1."


async def test_a_handoff_step_passes_the_reason_and_the_transcript_to_its_tool():
    """The hand-off tool gets what a ticket or an email needs, through the
    same runner as every other tool call, so the call shows in the log."""

    class RecordingTools(FakeTools):
        def __init__(self) -> None:
            super().__init__()
            self.arguments: list[dict] = []

        async def list_tools(self, config):
            return [
                *await super().list_tools(config),
                {
                    "server": "s",
                    "name": "hand_off_to_human",
                    "description": "d",
                    "input_schema": {"type": "object", "properties": {}},
                },
            ]

        async def call_tool(self, *, name, arguments, **kwargs):
            self.calls.append(name)
            self.arguments.append(dict(arguments))
            return "A person has been notified."

    spec = {
        "start": "h",
        "nodes": [
            {
                "id": "h",
                "type": "handoff",
                "message": "A person will reply.",
                "tool": "hand_off_to_human",
                "reason": "The visitor asked for a person about {{message}}",
            }
        ],
    }
    tools = RecordingTools()

    events = await _run(spec, ScriptedModel(rounds=[], seen=[]), tools=tools)

    assert tools.calls == ["hand_off_to_human"]
    assert tools.arguments == [
        {
            "reason": "The visitor asked for a person about is my flight delayed?",
            "transcript": "user: is my flight delayed?",
        }
    ]
    started = [e for e in events if isinstance(e, ToolCallStartedEvent)]
    finished = [e for e in events if isinstance(e, ToolCallFinishedEvent)]
    assert [e.tool for e in started] == ["hand_off_to_human"]
    assert finished[0].ok and finished[0].call_id == "wf-h"
    assert isinstance(events[-1], DoneEvent)
