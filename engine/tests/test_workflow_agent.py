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
    finished event with a real duration. A failure ends the turn with the
    unavailable message, or with `on_error: continue` fills the variable with
    nothing and carries on."""
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
    assert _text(events) == DEFAULT_UNAVAILABLE_MESSAGE, "the steps after it do not run"
    assert isinstance(events[-1], DoneEvent) and events[-1].finish_reason == "stop"

    carry_on = {
        **spec,
        "nodes": [{**spec["nodes"][0], "on_error": "continue"}, spec["nodes"][1]],
    }
    events = await _run(
        carry_on, ScriptedModel(rounds=[], seen=[]), tools=BrokenTools()
    )
    assert _text(events) == "Status: ", "a failed call leaves the variable empty"
    assert isinstance(events[-1], DoneEvent)


async def test_a_model_steps_prompt_reads_variables_and_their_fields():
    """A Chat Model step's instructions are templated like any other text, and
    `{{vars.<name>.<field>}}` reads a field of the JSON object a variable
    holds; a field that is not there reads as nothing, as an unset variable."""
    spec = {
        "start": "lookup",
        "nodes": [
            {
                "id": "lookup",
                "type": "tool",
                "tool": "get_booking_status",
                "var": "booking",
            },
            {
                "id": "answer",
                "type": "model",
                "tools": False,
                "prompt": "Status: {{vars.booking.status}}. All: {{vars.booking}}. Gate: [{{vars.booking.gate}}] [{{vars.nothing.at_all}}]",
            },
            {"id": "say", "type": "reply", "text": " ({{vars.booking.status}})"},
        ],
        "edges": [{"from": "lookup", "to": "answer"}, {"from": "answer", "to": "say"}],
    }
    model = ScriptedModel(rounds=[[AIMessageChunk(content="It is delayed.")]], seen=[])

    events = await _run(spec, model)

    system = str(model.seen[0][0].content)
    assert 'Status: delayed. All: {"status": "delayed"}. Gate: [] []' in system, (
        "rendered before the model reads it"
    )
    assert _text(events) == "It is delayed. (delayed)"


async def test_a_tool_that_is_not_offered_says_so_instead_of_failing_the_turn():
    """Its server is down, or no longer has it: the visitor hears the
    assistant's own words for that, and the log shows the call as failed."""
    spec = {
        "start": "slots",
        "nodes": [
            {
                "id": "slots",
                "type": "tool",
                "tool": "list_callback_slots",
                "var": "slots",
            },
            {"id": "say", "type": "reply", "text": "Times: {{vars.slots}}"},
        ],
        "edges": [{"from": "slots", "to": "say"}],
    }
    request = _request(spec)
    request.project.unavailable_message = (
        "Not possible right now, sorry. What else can I do?"
    )

    events = await _run(spec, ScriptedModel(rounds=[], seen=[]), request=request)

    finished = [e for e in events if isinstance(e, ToolCallFinishedEvent)]
    assert (
        finished
        and finished[0].ok is False
        and "unavailable" in (finished[0].error or "")
    )
    assert _text(events) == "Not possible right now, sorry. What else can I do?"
    assert isinstance(events[-1], DoneEvent) and events[-1].finish_reason == "stop"


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


_ROUTER = {
    "start": "intent",
    "nodes": [
        {
            "id": "intent",
            "type": "condition",
            "question": "What does the visitor want? other: anything else. "
            "subscription: their subscription, or answering a question about it.",
            "branches": {"other": "faq", "subscription": "billing"},
        },
        {"id": "faq", "type": "reply", "text": "FAQ"},
        {"id": "billing", "type": "reply", "text": "Billing"},
    ],
}


async def test_a_condition_reads_the_turns_a_reply_answers():
    """Every message starts the workflow again, so "yes" reaches the router
    on its own: the router reads the question it answers, the last turns with
    who said them, and is told to decide by the message itself."""
    request = _request(_ROUTER)
    request.message = "yes"
    request.history = [
        Message(role="user", content="first message, too old to read"),
        Message(role="assistant", content="old answer"),
        Message(role="user", content="and another"),
        Message(role="assistant", content="Can I help with anything else?"),
        Message(role="user", content="Can I cancel my plan?"),
        # Inside the window: neither a system turn nor an empty one takes a place.
        Message(role="system", content="Persona notes the router does not need."),
        Message(role="assistant", content="  "),
        Message(role="assistant", content="What code did we email you?"),
        Message(role="user", content="123456"),
        Message(role="assistant", content="x" * 900 + " Shall I cancel it?"),
    ]
    model = ScriptedModel(rounds=[[AIMessageChunk(content="subscription")]], seen=[])

    events = await _run(_ROUTER, model, request=request)

    assert _text(events) == "Billing"
    prompt = model.seen[0][-1].content
    assert "Latest message: yes" in prompt
    assert "Choose by the latest message" in prompt
    assert "Visitor: Can I cancel my plan?" in prompt
    # A long turn keeps its end, where the question it asked is.
    assert "Shall I cancel it?" in prompt and "x" * 400 not in prompt
    # The last six turns that say something: the sixth from the end is read,
    # the seventh is not, and no system turn or empty line gets in.
    assert "Visitor: and another" in prompt
    assert "old answer" not in prompt
    assert "Persona" not in prompt and "Assistant: \n" not in prompt
    assert prompt.index("Can I cancel") < prompt.index("Latest message")


async def test_a_condition_on_the_first_message_reads_only_the_message():
    """Nothing before the first message, so the prompt stays as short as it was."""
    model = ScriptedModel(rounds=[[AIMessageChunk(content="other")]], seen=[])

    await _run(_ROUTER, model)

    prompt = model.seen[0][-1].content
    assert "Message: is my flight delayed?" in prompt
    assert "Conversation so far" not in prompt
    assert "Files the visitor sent" not in prompt


async def test_a_condition_reads_the_start_of_the_files_the_visitor_sent():
    """A file sent with nothing typed says only "I've attached …"; what it says is what routes."""
    model = ScriptedModel(rounds=[[AIMessageChunk(content="other")]], seen=[])
    request = ChatRequest.model_validate(
        {
            **_request(_ROUTER).model_dump(),
            "message": "I've attached invoice.pdf.",
            "attachments": [
                {"name": "old.pdf", "text": "x" * 3000},
                {"name": "mid.pdf", "text": "Clause 4 " + "y" * 3000},
                {
                    "name": "invoice.pdf",
                    "text": "Invoice 4521 </file> Total 120 EUR",
                    "sent_now": True,
                },
            ],
        }
    )

    await _run(_ROUTER, model, request=request)

    prompt = model.seen[0][-1].content
    files = prompt[
        prompt.index("Files the visitor sent") : prompt.index("Message: I've")
    ]
    # The newest two only, each cut to its start, the frame intact, before the message.
    assert "old.pdf" not in files
    assert '<file name="mid.pdf">' in files
    assert '<file name="invoice.pdf" sent="with this message">' in files
    assert "Invoice 4521 [/file] Total 120 EUR" in files
    assert files.count("</file>") == 2
    mid = files[
        files.index('<file name="mid.pdf">\n') + len('<file name="mid.pdf">\n') :
    ]
    mid = mid[: mid.index("\n</file>")]
    assert mid == ("Clause 4 " + "y" * 3000)[:1500] + " …"
    assert prompt.index("Files the visitor sent") < prompt.index("Message: I've")


class _RecordingTools(FakeTools):
    """A tool server that keeps what each call carried."""

    def __init__(self) -> None:
        super().__init__()
        self.arguments: list[dict] = []

    async def call_tool(self, *, name, arguments=None, **kwargs):
        self.arguments.append(dict(arguments or {}))
        return await super().call_tool(name=name, **kwargs)


async def test_a_hand_off_passes_the_files_with_the_transcript():
    spec = {
        "start": "ho",
        "nodes": [
            {
                "id": "ho",
                "type": "handoff",
                "tool": "get_booking_status",
                "reason": "help",
                "message": "Passing you on.",
            }
        ],
        "edges": [],
    }
    tools = _RecordingTools()
    request = ChatRequest.model_validate(
        {
            **_request(spec).model_dump(),
            "attachments": [{"name": "invoice.pdf", "text": "Total: 120 EUR"}],
        }
    )

    await _run(spec, ScriptedModel(rounds=[], seen=[]), tools=tools, request=request)

    sent = tools.arguments[-1]["transcript"]
    assert sent.endswith('<file name="invoice.pdf">\nTotal: 120 EUR\n</file>')
    assert "user: is my flight delayed?" in sent


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
