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
from typing import Any
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
from chatbot_engine.models.workflow import AskNode, WorkflowSpec

pytest.importorskip("langgraph_agent.workflow")
from langgraph_agent.pauses import Pause, Pauses
from langgraph_agent.workflow import WorkflowAgent, check_answer, parse_verdict


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


CALLBACK: dict[str, Any] = {
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


def _verdict(outcome: str, value: str = "", reply: str = "") -> list:
    """One scripted reading of a reply, as the utility model returns it."""
    return [
        AIMessageChunk(
            content=json.dumps({"outcome": outcome, "value": value, "reply": reply})
        )
    ]


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

    # Read as an answer, but not a phone number: asked again, saying why.
    again = await _run(
        _request(
            CALLBACK,
            message="call me",
            resume={"thread_id": asked.thread_id, "value": "call me"},
        ),
        pauses,
        tools,
        ScriptedModel(rounds=[_verdict("answered", "call me")], seen=[]),
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
        ScriptedModel(rounds=[_verdict("answered", "Friday")], seen=[]),
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

    answerer = ScriptedModel(
        rounds=[_verdict("answered", "my site"), [AIMessageChunk(content="done")]],
        seen=[],
    )
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

    assert "DRAFTED" not in " ".join(str(m.content) for m in answerer.seen[-1])


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


# --- reading the reply -------------------------------------------------------

NAMED: dict[str, Any] = {
    "start": "name",
    "nodes": [
        {
            "id": "name",
            "type": "ask",
            "prompt": "What should the project be called?",
            "input": "text",
            "var": "name",
        },
        {
            "id": "create",
            "type": "tool",
            "tool": "book",
            "arguments": {"name": "{{vars.name}}"},
            "var": "created",
        },
        {"id": "done", "type": "reply", "text": "Created {{vars.name}}."},
    ],
    "edges": [{"from": "name", "to": "create"}, {"from": "create", "to": "done"}],
}


async def _reply(spec: dict, text: str, *rounds: list, pauses=None, tools=None):
    """Ask, then reply with `text`, the model scripted with `rounds`."""
    pauses, tools = pauses or Pauses.memory(), tools or Tools()
    thread = _asked(await _run(_request(spec, "new project"), pauses, tools)).thread_id
    model = ScriptedModel(rounds=list(rounds), seen=[])
    events = await _run(
        _request(spec, text, {"thread_id": thread, "value": text}),
        pauses,
        tools,
        model,
    )
    return events, tools, pauses, thread, model


async def test_an_answer_in_other_words_keeps_only_the_value():
    events, tools, *_ = await _reply(
        NAMED, "call it Apollo please", _verdict("answered", "Apollo")
    )

    assert tools.calls == [("book", {"name": "Apollo"})]
    assert _text(events) == "Created Apollo."


async def test_a_visitor_who_declines_is_not_taken_as_answering():
    """The case that started this: 'no, I don't want a new project' became the project's name."""
    events, tools, pauses, thread, _ = await _reply(
        NAMED,
        "no, I don't want a new project",
        _verdict("declined", reply="No problem, I won't create one."),
    )

    assert tools.calls == [], "nothing was created"
    assert _text(events) == "No problem, I won't create one."
    assert events[-1].finish_reason == "stop"
    assert await pauses.find(thread) is None, "the question no longer waits"


async def test_a_decline_goes_where_the_step_says_or_says_its_own_words():
    spec = {
        **NAMED,
        "nodes": [
            {**NAMED["nodes"][0], "on_decline": "stop"},
            *NAMED["nodes"][1:],
            {"id": "stop", "type": "reply", "text": "Okay, nothing created."},
        ],
    }
    events, tools, *_ = await _reply(spec, "never mind", _verdict("declined"))
    assert _text(events) == "Okay, nothing created." and tools.calls == []

    worded = {
        **NAMED,
        "nodes": [{**NAMED["nodes"][0], "decline_reply": "Fine."}, *NAMED["nodes"][1:]],
    }
    events, *_ = await _reply(worded, "no", _verdict("declined", reply="ignored"))
    assert _text(events) == "Fine."


async def test_a_question_back_is_answered_and_the_question_asked_again_then_left():
    pauses, tools = Pauses.memory(), Tools()
    events, _, _, thread, model = await _reply(
        NAMED,
        "what is a project?",
        _verdict("other"),
        [AIMessageChunk(content="A project groups your tasks.")],
        pauses=pauses,
        tools=tools,
    )

    again = _asked(events)
    assert again.node == "name" and again.thread_id == thread and not again.error
    assert (
        _text(events)
        == "A project groups your tasks.\n\nWhat should the project be called?"
    )
    # The reply was written knowing what was asked, and not to ask it itself.
    note = " ".join(str(m.content) for m in model.seen[-1])
    assert (
        "What should the project be called?" in note
        and "Do not ask your question again" in note
    )

    # Still no answer, with one retry used: replied to once more, and the question is left.
    last = await _run(
        _request(
            NAMED,
            "how much does it cost?",
            {"thread_id": thread, "value": "how much does it cost?"},
        ),
        pauses,
        tools,
        ScriptedModel(
            rounds=[_verdict("other"), [AIMessageChunk(content="It is free.")]], seen=[]
        ),
    )
    assert _text(last) == "It is free."
    assert not any(isinstance(e, InputRequiredEvent) for e in last)
    assert tools.calls == [] and await pauses.find(thread) is None


async def test_replies_that_still_do_not_answer_go_where_the_step_says():
    spec = {
        **NAMED,
        "nodes": [
            {**NAMED["nodes"][0], "retries": 0, "on_other": "help"},
            *NAMED["nodes"][1:],
            {"id": "help", "type": "reply", "text": "Let me pass you to the team."},
        ],
    }
    events, tools, *_ = await _reply(spec, "is it raining?", _verdict("other"))

    assert _text(events) == "Let me pass you to the team." and tools.calls == []


async def test_a_valid_reply_of_a_checked_kind_needs_no_reading():
    """A phone number that passes the check is kept without a model call."""
    pauses, tools = Pauses.memory(), Tools()
    thread = _asked(await _run(_request(CALLBACK), pauses, tools)).thread_id
    model = ScriptedModel(rounds=[], seen=[])
    events = await _run(
        _request(
            CALLBACK, "+4915112345678", {"thread_id": thread, "value": "+4915112345678"}
        ),
        pauses,
        tools,
        model,
    )

    assert _asked(events).node == "when" and model.seen == []


async def test_a_typed_choice_is_read_as_the_option_it_means():
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
        _request(
            CALLBACK,
            "the afternoon one",
            {"thread_id": thread, "value": "the afternoon one"},
        ),
        pauses,
        tools,
        ScriptedModel(rounds=[_verdict("answered", "Wed 14:00")], seen=[]),
    )

    assert _text(events) == "Booked for Wed 14:00."
    assert tools.calls[-1][1]["slot"] == "2026-09-16T14:00"


async def test_with_understanding_off_a_reply_is_the_answer_as_it_is():
    spec = {
        **NAMED,
        "nodes": [{**NAMED["nodes"][0], "understand": False}, *NAMED["nodes"][1:]],
    }
    _events, tools, *_, model = await _reply(spec, "no, I don't want a new project")

    assert tools.calls == [("book", {"name": "no, I don't want a new project"})]
    assert model.seen == []


async def test_an_unreadable_reading_takes_the_reply_as_it_stands():
    _events, tools, *_ = await _reply(
        NAMED, "Apollo", [AIMessageChunk(content="sure!")]
    )

    assert tools.calls == [("book", {"name": "Apollo"})]


@pytest.mark.parametrize(
    "extra, message",
    [
        ({"on_decline": "nowhere"}, "on_decline of 'q' names an unknown node"),
        ({"on_other": "nowhere"}, "on_other of 'q' names an unknown node"),
        ({"on_decline": "q"}, "on_decline of 'q' names the step itself"),
        ({"on_other": "q"}, "on_other of 'q' names the step itself"),
        ({"retries": 4}, "retries"),
    ],
)
def test_where_a_question_goes_must_exist(extra, message):
    ask = {"id": "q", "type": "ask", "prompt": "?", "var": "v", **extra}
    with pytest.raises(ValidationError, match=message):
        WorkflowSpec.model_validate({"start": "q", "nodes": [ask]})


def test_a_node_reached_only_through_a_question_is_reachable():
    spec = WorkflowSpec.model_validate(
        {
            "start": "q",
            "nodes": [
                {
                    "id": "q",
                    "type": "ask",
                    "prompt": "?",
                    "var": "v",
                    "on_decline": "bye",
                    "on_other": "help",
                },
                {"id": "bye", "type": "reply", "text": "Bye."},
                {"id": "help", "type": "reply", "text": "Help."},
            ],
        }
    )
    assert spec.start == "q"


async def test_a_typed_no_for_a_no_option_is_that_option_and_needs_no_reading():
    spec = {
        **NAMED,
        "nodes": [
            {**NAMED["nodes"][0], "input": "choice", "options": ["Yes", "No"]},
            *NAMED["nodes"][1:],
        ],
    }
    # No scripted rounds: a reading would fail on an empty script.
    _events, tools, *_ = await _reply(spec, "no")
    assert tools.calls == [("book", {"name": "No"})]


async def test_an_answer_of_the_wrong_kind_is_asked_again_however_often():
    """A typo in a phone number is not a refusal: `retries` is for replies that do not answer at all."""
    spec = {
        **CALLBACK,
        "nodes": [{**CALLBACK["nodes"][0], "retries": 0}, *CALLBACK["nodes"][1:]],
    }
    pauses, tools = Pauses.memory(), Tools()
    thread = _asked(await _run(_request(spec), pauses, tools)).thread_id
    for text in ("+44 20", "+44 207"):
        events = await _run(
            _request(spec, text, {"thread_id": thread, "value": text}),
            pauses,
            tools,
            ScriptedModel(rounds=[_verdict("answered", text)], seen=[]),
        )
        assert "phone number" in (_asked(events).error or "")
    assert await pauses.find(thread) is not None, "the question still waits"
    assert tools.calls == []


async def test_a_reading_that_cannot_be_had_asks_again_and_keeps_the_question():
    class Broken(ScriptedModel):
        async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
            raise RuntimeError("blocked by the key's guardrail")
            yield  # pragma: no cover

    pauses, tools = Pauses.memory(), Tools()
    thread = _asked(await _run(_request(NAMED, "new project"), pauses, tools)).thread_id
    events = await _run(
        _request(NAMED, "no thanks", {"thread_id": thread, "value": "no thanks"}),
        pauses,
        tools,
        Broken(rounds=[], seen=[]),
    )
    assert "did not catch" in (_asked(events).error or "")
    assert await pauses.find(thread) is not None, "the pause is kept"
    assert tools.calls == [], "the reply was not taken as the name"


async def test_the_reading_of_a_refused_answer_is_reported_with_the_pause():
    pauses, tools = Pauses.memory(), Tools()
    thread = _asked(await _run(_request(CALLBACK), pauses, tools)).thread_id
    events = await _run(
        _request(
            CALLBACK,
            "call me on my mobile",
            {"thread_id": thread, "value": "call me on my mobile"},
        ),
        pauses,
        tools,
        ScriptedModel(
            rounds=[[*_verdict("answered", "mobile"), _usage_chunk(500, 20)]], seen=[]
        ),
    )
    assert "phone number" in (_asked(events).error or "")
    spent = next(e for e in events if isinstance(e, UsageEvent))
    assert (spent.input_tokens, spent.output_tokens) == (500, 20)
    later = await _run(
        _request(
            CALLBACK, "+4915112345678", {"thread_id": thread, "value": "+4915112345678"}
        ),
        pauses,
        tools,
    )
    assert next(e for e in later if isinstance(e, UsageEvent)).input_tokens == 0


async def test_a_text_reply_over_the_limit_is_asked_again_before_any_reading():
    events, tools, _pauses, _thread, model = await _reply(NAMED, "x" * 1_200)
    assert "1,000 characters" in (_asked(events).error or "")
    assert tools.calls == [] and model.seen == []


async def test_a_question_says_whether_typed_words_are_read():
    asked = _asked(await _run(_request(NAMED, "new project"), Pauses.memory(), Tools()))
    assert asked.understand is True
    off = {
        **NAMED,
        "nodes": [{**NAMED["nodes"][0], "understand": False}, *NAMED["nodes"][1:]],
    }
    assert (
        _asked(
            await _run(_request(off, "new project"), Pauses.memory(), Tools())
        ).understand
        is False
    )


def test_a_web_address_needs_a_host_with_a_domain():
    node = AskNode(
        id="site", type="ask", prompt="Your website?", input="url", var="site"
    )
    for word in ("No.", "nope.", "N.O.", "no"):
        assert check_answer(node, word, [])[1], word
    assert check_answer(node, "example.com.", []) == ("https://example.com", None)
    # Names in other scripts are addresses too; a bare machine name is not.
    assert check_answer(node, "münchen.de", []) == ("https://münchen.de", None)
    assert check_answer(node, "example.xn--p1ai", []) == (
        "https://example.xn--p1ai",
        None,
    )
    assert check_answer(node, "localhost", [])[1]
    assert check_answer(node, "https://shop.example/page?x=1", []) == (
        "https://shop.example/page?x=1",
        None,
    )


def test_a_reading_is_found_among_words_and_fences():
    fenced = 'Sure:\n```json\n{"outcome": "declined", "reply": "Fine."}\n```\nDone }'
    assert parse_verdict(fenced, "raw").outcome == "declined"
    assert (
        parse_verdict('{"outcome": "answered", "value": "Apollo"} }', "raw").value
        == "Apollo"
    )
    assert parse_verdict("nothing here", "raw") == ("answered", "raw", "")
    # An outcome in capitals, and a brace before the JSON, are read too.
    assert parse_verdict('{"outcome": "Declined"}', "raw").outcome == "declined"
    assert parse_verdict('Reading {x}: {"outcome": "other"}', "raw").outcome == "other"


OPTIONAL: dict[str, Any] = {
    **NAMED,
    "nodes": [
        {**NAMED["nodes"][0], "optional": True, "skip_label": "Not now"},
        *NAMED["nodes"][1:],
    ],
}


async def test_the_skip_typed_by_its_name_skips_the_question_without_a_reading():
    _events, tools, *_ = await _reply(OPTIONAL, "not now!")
    # An empty answer, as pressing the skip gives: the tool step leaves an empty argument out.
    assert tools.calls == [("book", {})]


async def test_nothing_to_give_said_in_other_words_skips_an_optional_question():
    _events, tools, _pauses, _thread, model = await _reply(
        OPTIONAL, "I have nothing to add", _verdict("answered", "")
    )
    assert tools.calls == [("book", {})]
    # The reading was told the question may be skipped, on a line of its own.
    assert (
        'It asks for a short free answer.\nIt may be skipped, by saying "Not now".'
        in (model.seen[0][1].content)
    )


async def test_a_reading_that_cannot_be_had_twice_takes_the_reply_as_it_stands():
    class Broken(ScriptedModel):
        async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
            raise RuntimeError("blocked by the key's guardrail")
            yield  # pragma: no cover

    pauses, tools = Pauses.memory(), Tools()
    thread = _asked(await _run(_request(NAMED, "new project"), pauses, tools)).thread_id
    first = await _run(
        _request(NAMED, "Apollo", {"thread_id": thread, "value": "Apollo"}),
        pauses,
        tools,
        Broken(rounds=[], seen=[]),
    )
    assert "did not catch" in (_asked(first).error or "") and tools.calls == []
    second = await _run(
        _request(NAMED, "Apollo", {"thread_id": thread, "value": "Apollo"}),
        pauses,
        tools,
        Broken(rounds=[], seen=[]),
    )
    assert tools.calls == [("book", {"name": "Apollo"})]
    assert _text(second) == "Created Apollo."


async def test_a_question_asked_twice_in_one_turn_starts_its_count_afresh():
    """name -> echo -> name: a question back on the second visit is answered and asked again, not left."""
    loop = {
        "start": "name",
        "nodes": [
            {**NAMED["nodes"][0], "retries": 1},
            {"id": "echo", "type": "reply", "text": "Once more."},
        ],
        "edges": [{"from": "name", "to": "echo"}, {"from": "echo", "to": "name"}],
        "max_steps": 12,
    }
    pauses, tools = Pauses.memory(), Tools()
    thread = _asked(await _run(_request(loop, "new project"), pauses, tools)).thread_id
    resume = lambda text, *rounds: _run(  # noqa: E731
        _request(loop, text, {"thread_id": thread, "value": text}),
        pauses,
        tools,
        ScriptedModel(rounds=list(rounds), seen=[]),
    )
    await resume("what?", _verdict("other"), [AIMessageChunk(content="A label.")])
    await resume("Apollo", _verdict("answered", "Apollo"))
    events = await resume(
        "why?", _verdict("other"), [AIMessageChunk(content="Just a label.")]
    )
    assert any(isinstance(e, InputRequiredEvent) for e in events), (
        "asked again, not left"
    )
