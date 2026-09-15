"""The ask step: a workflow turn that pauses on a question and resumes with the answer.

This is LangGraph's human-in-the-loop: `interrupt()` inside the step, a
checkpointer keeping the graph between two requests, `Command(resume=...)`
continuing it. The tests run the real graph with an in-memory checkpointer
and the scripted model, and resume with a fresh agent, as a second request
would.
"""

from __future__ import annotations

import json
import time
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from pydantic import ValidationError
from test_agent_parity import ScriptedModel, _no_retrieval, _usage_chunk

from chatbot_engine.models.chat import (
    AssistantConfig,
    ChatRequest,
    McpServerConfig,
    Message,
    ResumeInput,
)
from chatbot_engine.models.events import (
    DoneEvent,
    ErrorEvent,
    InputRequiredEvent,
    TokenEvent,
    UsageEvent,
)
from chatbot_engine.models.workflow import WorkflowSpec

pytest.importorskip("langgraph_agent.workflow")
from langgraph_agent.pauses import Pause, Pauses
from langgraph_agent.workflow import WorkflowAgent


class Tools:
    """Two tools: slots returns a JSON list, book records what it was given."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def list_tools(self, config):
        return [
            {"server": "s", "name": n, "description": "d", "input_schema": {}}
            for n in ("list_slots", "book")
        ]

    async def call_tool(self, *, name, arguments, **kwargs):
        self.calls.append((name, dict(arguments)))
        if name == "list_slots":
            return json.dumps(
                [
                    {"value": "2026-09-16T10:00", "label": "Wed 10:00"},
                    {"value": "2026-09-16T14:00", "label": "Wed 14:00"},
                ]
            )
        return "booked"


CALLBACK = {
    "start": "phone",
    "nodes": [
        {
            "id": "phone",
            "type": "ask",
            "prompt": "What number should we call?",
            "input": "phone",
            "var": "phone",
        },
        {"id": "slots", "type": "tool", "tool": "list_slots", "var": "slots"},
        {
            "id": "when",
            "type": "ask",
            "prompt": "When suits you?",
            "input": "choice",
            "options_from": "slots",
            "var": "slot",
        },
        {
            "id": "book",
            "type": "tool",
            "tool": "book",
            "arguments": {
                "phone": "{{vars.phone}}",
                "slot": "{{vars.slot}}",
                "why": "{{message}}",
            },
            "var": "ticket",
        },
        {"id": "done", "type": "reply", "text": "Booked for {{vars.slot_label}}."},
    ],
    "edges": [
        {"from": "phone", "to": "slots"},
        {"from": "slots", "to": "when"},
        {"from": "when", "to": "book"},
        {"from": "book", "to": "done"},
    ],
}


def _request(
    workflow: dict,
    message: str = "please call me",
    resume: dict | None = None,
    project_id: str = "support",
    session_id: str = "s1",
    history: list[Message] | None = None,
) -> ChatRequest:
    return ChatRequest(
        project=AssistantConfig(
            project_id=project_id,
            name="S",
            system_prompt="You are helpful.",
            model="openai/gpt-5-mini",
            mcp_servers=[
                McpServerConfig(
                    name="s", url="http://s", allowed_tools=["list_slots", "book"]
                )
            ],
            workflow=workflow,
        ),
        message=message,
        session_id=session_id,
        user_id="u1",
        history=history or [],
        resume=ResumeInput(**resume) if resume else None,
    )


async def _run(
    request: ChatRequest,
    pauses: Pauses,
    tools: Tools,
    model: ScriptedModel | None = None,
) -> list:
    # A new agent every time: a resume is a separate request, and nothing may
    # survive between the two except what the checkpointer kept.
    agent = WorkflowAgent(tools=tools, pauses=pauses)
    with (
        patch(
            "langgraph_agent.workflow.build_chat_model",
            return_value=model or ScriptedModel(rounds=[], seen=[]),
        ),
        patch("langgraph_agent.workflow.retrieve_with_usage", new=_no_retrieval),
    ):
        return [event async for event in agent.run(request)]


def _asked(events: list) -> InputRequiredEvent:
    return next(e for e in events if isinstance(e, InputRequiredEvent))


def _text(events: list) -> str:
    return "".join(e.text for e in events if isinstance(e, TokenEvent))


async def test_a_turn_pauses_on_a_question_and_resumes_where_it_stopped():
    pauses, tools = Pauses.memory(), Tools()

    first = await _run(_request(CALLBACK), pauses, tools)
    asked = _asked(first)
    assert _text(first) == "What number should we call?"
    assert asked.input == "phone" and asked.node == "phone"
    assert asked.thread_id.startswith("support:")
    assert (
        isinstance(first[-1], DoneEvent) and first[-1].finish_reason == "input_required"
    )
    assert tools.calls == [], "nothing after the question ran"

    second = await _run(
        _request(
            CALLBACK,
            message="+44 20 7946 0958",
            resume={"thread_id": asked.thread_id, "value": "+44 (20) 7946-0958"},
        ),
        pauses,
        tools,
    )
    choice = _asked(second)
    assert choice.thread_id == asked.thread_id
    assert [o.label for o in choice.options] == ["Wed 10:00", "Wed 14:00"]
    assert _text(second) == "When suits you?"

    third = await _run(
        _request(
            CALLBACK,
            message="Wed 14:00",
            resume={"thread_id": asked.thread_id, "value": "2026-09-16T14:00"},
        ),
        pauses,
        tools,
    )
    assert _text(third) == "Booked for Wed 14:00."
    assert third[-1].finish_reason == "stop"
    # The answers, tidied, and the message that started the turn, not the last answer.
    assert tools.calls[-1] == (
        "book",
        {"phone": "+442079460958", "slot": "2026-09-16T14:00", "why": "please call me"},
    )
    assert await pauses.find(asked.thread_id) is None, "a finished turn keeps nothing"


async def test_a_refused_answer_asks_again_with_the_reason():
    pauses, tools = Pauses.memory(), Tools()
    asked = _asked(await _run(_request(CALLBACK), pauses, tools))

    again = await _run(
        _request(
            CALLBACK,
            message="call me",
            resume={"thread_id": asked.thread_id, "value": "call me"},
        ),
        pauses,
        tools,
    )

    retry = _asked(again)
    assert retry.node == "phone" and "phone number" in (retry.error or "")
    assert _text(again) == "", "the question is not said twice"
    assert tools.calls == []

    later = await _run(
        _request(
            CALLBACK,
            message="+4915112345678",
            resume={"thread_id": asked.thread_id, "value": "+4915112345678"},
        ),
        pauses,
        tools,
    )
    assert _asked(later).node == "when"


async def test_a_choice_outside_the_options_is_refused():
    pauses, tools = Pauses.memory(), Tools()
    thread = _asked(await _run(_request(CALLBACK), pauses, tools)).thread_id
    await _run(
        _request(
            CALLBACK, "+4915112345678", {"thread_id": thread, "value": "+4915112345678"}
        ),
        pauses,
        tools,
    )

    events = await _run(
        _request(CALLBACK, "Friday", {"thread_id": thread, "value": "Friday"}),
        pauses,
        tools,
    )

    assert _asked(events).error == "Please choose one of the options."
    assert not any(name == "book" for name, _ in tools.calls)


PROJECT = {
    "start": "url",
    "nodes": [
        {
            "id": "url",
            "type": "ask",
            "prompt": "What is the website?",
            "input": "url",
            "optional": True,
            "skip_label": "No website",
            "var": "url",
        },
        {
            "id": "create",
            "type": "tool",
            "tool": "book",
            "arguments": {"name": "Project X", "url": "{{vars.url}}"},
            "var": "created",
        },
    ],
    "edges": [{"from": "url", "to": "create"}],
}


async def test_skipping_an_optional_question_leaves_the_argument_out():
    pauses, tools = Pauses.memory(), Tools()
    asked = _asked(await _run(_request(PROJECT, "create Project X"), pauses, tools))
    assert asked.optional and asked.skip_label == "No website"

    await _run(
        _request(
            PROJECT, "No website", {"thread_id": asked.thread_id, "skipped": True}
        ),
        pauses,
        tools,
    )

    assert tools.calls == [("book", {"name": "Project X"})]


async def test_a_web_address_gets_a_scheme():
    pauses, tools = Pauses.memory(), Tools()
    thread = _asked(await _run(_request(PROJECT), pauses, tools)).thread_id

    await _run(
        _request(PROJECT, "x.com", {"thread_id": thread, "value": "x.com"}),
        pauses,
        tools,
    )

    assert tools.calls == [("book", {"name": "Project X", "url": "https://x.com"})]


async def test_a_resume_from_another_conversation_or_project_is_refused():
    pauses, tools = Pauses.memory(), Tools()
    thread = _asked(await _run(_request(PROJECT), pauses, tools)).thread_id
    answer = {"thread_id": thread, "value": "x.com"}

    for other in (
        _request(PROJECT, "x.com", answer, session_id="someone-else"),
        _request(PROJECT, "x.com", answer, project_id="another"),
        _request(PROJECT, "x.com", {"thread_id": "support:unknown", "value": "x"}),
    ):
        events = await _run(other, pauses, tools)
        assert isinstance(events[0], ErrorEvent) and events[0].code == "resume_expired"
        assert events[-1].finish_reason == "error"
    assert tools.calls == []


async def test_a_workflow_edited_while_waiting_is_not_resumed():
    pauses, tools = Pauses.memory(), Tools()
    thread = _asked(await _run(_request(PROJECT), pauses, tools)).thread_id
    edited = {
        **PROJECT,
        "nodes": [{**PROJECT["nodes"][0], "prompt": "Website?"}, PROJECT["nodes"][1]],
    }

    events = await _run(
        _request(edited, "x.com", {"thread_id": thread, "value": "x.com"}),
        pauses,
        tools,
    )

    assert isinstance(events[0], ErrorEvent) and events[0].code == "resume_changed"


async def test_an_expired_pause_is_forgotten():
    pauses, tools = Pauses.memory(ttl_s=60), Tools()
    thread = _asked(await _run(_request(PROJECT), pauses, tools)).thread_id

    with patch("langgraph_agent.pauses.time.time", return_value=time.time() + 120):
        assert await pauses.find(thread) is None


async def test_usage_before_the_pause_is_reported_once():
    """The condition's tokens are reported when the turn pauses, not again at the end."""
    spec = {
        "start": "kind",
        "nodes": [
            {
                "id": "kind",
                "type": "condition",
                "question": "Call?",
                "branches": {"call": "url", "other": "url"},
            },
            *PROJECT["nodes"],
        ],
        "edges": PROJECT["edges"],
    }
    pauses, tools = Pauses.memory(), Tools()
    condition = ScriptedModel(
        rounds=[[AIMessageChunk(content="call"), _usage_chunk(100, 1)]], seen=[]
    )
    first = await _run(_request(spec), pauses, tools, condition)
    paused_usage = next(e for e in first if isinstance(e, UsageEvent))
    assert paused_usage.input_tokens == 100

    thread = _asked(first).thread_id
    last = await _run(
        _request(spec, "x.com", {"thread_id": thread, "value": "x.com"}), pauses, tools
    )

    assert next(e for e in last if isinstance(e, UsageEvent)).input_tokens == 0


async def test_a_choice_with_no_options_is_skipped_rather_than_shown_empty():
    spec = {
        "start": "when",
        "nodes": [
            {
                "id": "when",
                "type": "ask",
                "prompt": "When?",
                "input": "choice",
                "options_from": "slots",
                "var": "slot",
            },
            {"id": "say", "type": "reply", "text": "slot=[{{vars.slot}}]"},
        ],
        "edges": [{"from": "when", "to": "say"}],
    }
    events = await _run(_request(spec), Pauses.memory(), Tools())

    assert _text(events) == "slot=[]"
    assert not any(isinstance(e, InputRequiredEvent) for e in events)


async def test_a_workflow_without_questions_never_opens_the_checkpointer():
    class Unusable(Pauses):
        async def saver(self):
            raise AssertionError("opened")

    spec = {"start": "say", "nodes": [{"id": "say", "type": "reply", "text": "hi"}]}
    events = await _run(_request(spec), Unusable.memory(), Tools())

    assert _text(events) == "hi"


async def test_a_model_step_after_a_tool_step_reads_the_call_with_its_result():
    spec = {
        "start": "slots",
        "nodes": [
            {"id": "slots", "type": "tool", "tool": "list_slots", "var": "slots"},
            {"id": "answer", "type": "model", "tools": False},
        ],
        "edges": [{"from": "slots", "to": "answer"}],
    }
    model = ScriptedModel(rounds=[[AIMessageChunk(content="ok")]], seen=[])
    await _run(_request(spec), Pauses.memory(), Tools(), model)

    seen = model.seen[0]
    call = next(i for i, m in enumerate(seen) if isinstance(m, ToolMessage))
    assert (
        isinstance(seen[call - 1], AIMessage)
        and seen[call - 1].tool_calls[0]["id"] == "wf-slots"
    )


async def test_a_model_step_after_a_pause_reads_only_what_followed():
    """The conversation the resumed request sends already holds what was said before."""
    spec = {
        "start": "draft",
        "nodes": [
            {"id": "draft", "type": "model", "var": "draft", "tools": False},
            {
                "id": "url",
                "type": "ask",
                "prompt": "Website?",
                "input": "text",
                "var": "site",
            },
            {"id": "answer", "type": "model", "tools": False},
        ],
        "edges": [{"from": "draft", "to": "url"}, {"from": "url", "to": "answer"}],
    }
    pauses, tools = Pauses.memory(), Tools()
    drafter = ScriptedModel(rounds=[[AIMessageChunk(content="DRAFTED")]], seen=[])
    thread = _asked(await _run(_request(spec), pauses, tools, drafter)).thread_id

    answerer = ScriptedModel(rounds=[[AIMessageChunk(content="done")]], seen=[])
    await _run(
        _request(
            spec,
            "my site",
            {"thread_id": thread, "value": "my site"},
            history=[
                Message(role="user", content="please call me"),
                Message(role="assistant", content="Website?"),
            ],
        ),
        pauses,
        tools,
        answerer,
    )

    assert "DRAFTED" not in " ".join(str(m.content) for m in answerer.seen[0])


async def test_the_sqlite_store_keeps_a_pause_across_restarts(tmp_path):
    path = tmp_path / "checkpoints.sqlite3"
    first = Pauses.sqlite(path, ttl_s=3600)
    tools = Tools()
    thread = _asked(await _run(_request(PROJECT), first, tools)).thread_id

    # A new store on the same file: an engine that restarted in between.
    second = Pauses.sqlite(path, ttl_s=3600)
    pause = await second.find(thread)
    assert isinstance(pause, Pause) and pause.session_id == "s1"
    await _run(
        _request(PROJECT, "x.com", {"thread_id": thread, "value": "x.com"}),
        second,
        tools,
    )

    assert tools.calls == [("book", {"name": "Project X", "url": "https://x.com"})]
    assert await second.find(thread) is None


@pytest.mark.parametrize(
    "node, message",
    [
        ({"input": "choice"}, "no options"),
        ({"input": "text", "options": ["a"]}, "has options"),
        ({"input": "choice", "options": ["a"], "options_from": "x"}, "both"),
        ({"input": "date"}, "input"),
    ],
)
def test_malformed_questions_are_refused(node, message):
    ask = {"id": "q", "type": "ask", "prompt": "?", "var": "v", **node}
    with pytest.raises(ValidationError, match=message):
        WorkflowSpec.model_validate({"start": "q", "nodes": [ask]})
