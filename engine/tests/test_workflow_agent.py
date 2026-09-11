"""The workflow agent runs the graph an assistant describes."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessageChunk
from test_agent_parity import FakeTools, ScriptedModel, _no_retrieval

from chatbot_engine.models.chat import AssistantConfig, ChatRequest, McpServerConfig
from chatbot_engine.models.events import (
    DoneEvent,
    TokenEvent,
    ToolCallFinishedEvent,
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


async def _run(workflow: dict | None, model: ScriptedModel) -> list:
    agent = WorkflowAgent(tools=FakeTools())
    with (
        patch("langgraph_agent.workflow.build_chat_model", return_value=model),
        patch("langgraph_agent.workflow.retrieve_with_usage", new=_no_retrieval),
    ):
        return [event async for event in agent.run(_request(workflow))]


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
