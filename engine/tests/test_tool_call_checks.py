"""A tool call is checked before it runs, and a turn ends in words.

Three ways a turn used to go wrong, for the loop and graph agents alike and
for a workflow's Chat Model step:

- arguments that are not a JSON object landed in `invalid_tool_calls`, which
  nothing read, and the turn ended with no text: now the model is told what
  was wrong, in that call's result, and may call again;
- a call cut off by `max_output_tokens` was mended by the partial-JSON
  parser and run with arguments the model never finished: now a reply that
  ended on `length` runs none of its calls;
- a workflow step that ran out of tool rounds left calls with no result in
  the turn's messages, which a later model step sent back (a 400 from
  OpenAI-style APIs): now every call has a result.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from test_agent_parity import (
    AGENTS,
    FakeTools,
    ScriptedModel,
    _no_retrieval,
    _request,
    _run,
    _usage_chunk,
)

from chatbot_engine.agent.client import (
    CUT_CALL_RESULT,
    INVALID_ARGUMENTS_ERROR,
    TOOL_LIMIT_RESULT,
)
from chatbot_engine.models.chat import AssistantConfig, ChatRequest
from chatbot_engine.models.events import (
    DoneEvent,
    TokenEvent,
    ToolCallFinishedEvent,
    ToolCallStartedEvent,
    UsageEvent,
)

pytest.importorskip("langgraph_agent.workflow")
from langgraph_agent.workflow import WorkflowAgent


def _call(args: str, call_id: str = "c1") -> AIMessageChunk:
    return AIMessageChunk(
        content="",
        tool_call_chunks=[
            {"name": "get_booking_status", "args": args, "id": call_id, "index": 0}
        ],
    )


def _text(events: list) -> str:
    return "".join(e.text for e in events if isinstance(e, TokenEvent))


# --- unreadable arguments ---------------------------------------------------


@pytest.mark.parametrize("which", AGENTS)
async def test_a_call_with_unreadable_arguments_is_answered_with_what_was_wrong(
    which: str,
) -> None:
    rounds = [
        [_call("not json"), _usage_chunk(10, 5)],
        [_call('{"ref": "AB12"}', "c2"), _usage_chunk(10, 5)],
        [AIMessageChunk(content="Delayed."), _usage_chunk(10, 5)],
    ]
    tools = FakeTools()
    model = ScriptedModel(rounds=rounds, seen=[])

    events = await _run(which, rounds, tools, model=model)

    assert tools.calls == ["get_booking_status"], "the malformed call is not made"
    finished = [e for e in events if isinstance(e, ToolCallFinishedEvent)]
    assert [(e.call_id, e.ok, e.error) for e in finished] == [
        ("c1", False, INVALID_ARGUMENTS_ERROR),
        ("c2", True, None),
    ]
    # The model reads why, in the result of the call it made, and calls again.
    told = model.seen[1][-1]
    assert isinstance(told, ToolMessage) and told.tool_call_id == "c1"
    assert "not a JSON object" in str(told.content)
    assert _text(events) == "Delayed."
    assert isinstance(events[-1], DoneEvent) and events[-1].finish_reason == "stop"


async def test_both_agents_report_an_unreadable_call_the_same_way() -> None:
    def rounds() -> list:
        return [
            [_call("[1, 2]"), _usage_chunk(10, 5)],
            [AIMessageChunk(content="Sorry."), _usage_chunk(10, 5)],
        ]

    loop = await _run("loop", rounds(), FakeTools())
    graph = await _run("graph", rounds(), FakeTools())

    assert [type(e).__name__ for e in loop] == [type(e).__name__ for e in graph]
    assert _text(loop) == _text(graph) == "Sorry."


# --- a reply cut at max_output_tokens ---------------------------------------


@pytest.mark.parametrize("which", AGENTS)
async def test_a_reply_cut_while_calling_a_tool_runs_none_of_its_calls(
    which: str,
) -> None:
    """The parser mends `{"ref": "A` into `{"ref": "A"}`; run, it would look up
    a booking the model never named."""
    cut = _call('{"ref": "A')
    cut.response_metadata = {"finish_reason": "length"}
    rounds = [[AIMessageChunk(content="Let me check"), cut, _usage_chunk(10, 5)]]
    tools = FakeTools()

    events = await _run(which, rounds, tools)

    assert tools.calls == []
    assert not any(isinstance(e, ToolCallStartedEvent) for e in events)
    assert _text(events) == "Let me check"
    assert isinstance(events[-1], DoneEvent) and events[-1].finish_reason == "length"
    assert sum(isinstance(e, UsageEvent) for e in events) == 1


# --- a model that streams nothing -------------------------------------------


class Silent(ScriptedModel):
    """A model whose stream ends without a single chunk. LangChain's own
    `astream` would raise on that; a client that does not is stood in for."""

    async def astream(self, input, config=None, **kwargs):
        self.seen.append(list(input))
        return
        yield  # pragma: no cover


async def test_both_agents_end_a_turn_whose_model_said_nothing_the_same_way() -> None:
    """The loop agent always answered an empty stream with an empty answer;
    the graph agent asserted it never happened."""
    loop = await _run("loop", [], FakeTools(), model=Silent(rounds=[], seen=[]))
    graph = await _run("graph", [], FakeTools(), model=Silent(rounds=[], seen=[]))

    assert [type(e).__name__ for e in loop] == [type(e).__name__ for e in graph]
    assert [type(e).__name__ for e in graph] == [
        "RetrievalEvent",
        "UsageEvent",
        "DoneEvent",
    ]
    assert isinstance(graph[-1], DoneEvent) and graph[-1].finish_reason == "stop"


# --- a workflow's Chat Model step -------------------------------------------


def _workflow_request(spec: dict, **project: object) -> ChatRequest:
    base = _request()
    return ChatRequest(
        project=AssistantConfig(
            **base.project.model_dump(exclude={"workflow"}) | project, workflow=spec
        ),
        message=base.message,
    )


async def _run_workflow(request: ChatRequest, model: ScriptedModel, tools=None):
    agent = WorkflowAgent(tools=tools or FakeTools())
    with (
        patch("langgraph_agent.workflow.build_chat_model", return_value=model),
        patch("langgraph_agent.workflow.retrieve_with_usage", new=_no_retrieval),
    ):
        return [event async for event in agent.run(request)]


def _unanswered_calls(messages: list) -> list[str]:
    """The ids of the calls in `messages` that no tool result answers."""
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    asked = [
        call["id"]
        for m in messages
        if isinstance(m, AIMessage)
        for call in [*m.tool_calls, *m.invalid_tool_calls]
    ]
    return [call_id for call_id in asked if call_id not in answered]


#: A step that drafts into a variable, with the tools, and one that answers.
TWO_STEPS = {
    "start": "draft",
    "nodes": [
        {"id": "draft", "type": "model", "var": "draft"},
        {"id": "answer", "type": "model", "tools": False},
    ],
    "edges": [{"from": "draft", "to": "answer"}],
}


async def test_a_step_that_runs_out_of_rounds_leaves_no_call_without_a_result():
    """The rounds run out, and even the call made with tools off asks for one
    more: none of the three is left unanswered for the next step to send."""
    model = ScriptedModel(
        rounds=[
            [_call("{}", "c1")],
            [_call("{}", "c2")],
            [_call("{}", "c3")],  # with tools off, a model that asks anyway
            [AIMessageChunk(content="Here is what I found.")],
        ],
        seen=[],
    )
    request = _workflow_request(TWO_STEPS, max_tool_iterations=1)

    events = await _run_workflow(request, model)

    answer_step = model.seen[-1]
    assert _unanswered_calls(answer_step) == []
    results = {
        m.tool_call_id: m.content for m in answer_step if isinstance(m, ToolMessage)
    }
    assert results["c2"] == TOOL_LIMIT_RESULT
    assert results["c3"] == TOOL_LIMIT_RESULT
    assert _text(events) == "Here is what I found."
    assert isinstance(events[-1], DoneEvent) and events[-1].finish_reason == "stop"


async def test_a_step_cut_while_calling_a_tool_runs_nothing_and_closes_the_call():
    cut = _call('{"ref": "A', "c1")
    cut.response_metadata = {"finish_reason": "length"}
    model = ScriptedModel(
        rounds=[[cut], [AIMessageChunk(content="Answered.")]], seen=[]
    )
    tools = FakeTools()

    events = await _run_workflow(_workflow_request(TWO_STEPS), model, tools)

    assert tools.calls == []
    answer_step = model.seen[-1]
    assert _unanswered_calls(answer_step) == []
    assert any(
        isinstance(m, ToolMessage) and m.content == CUT_CALL_RESULT for m in answer_step
    )
    assert _text(events) == "Answered."


async def test_a_step_answers_a_call_with_unreadable_arguments():
    spec = {"start": "answer", "nodes": [{"id": "answer", "type": "model"}]}
    model = ScriptedModel(
        rounds=[[_call("not json")], [AIMessageChunk(content="Delayed.")]], seen=[]
    )
    tools = FakeTools()

    events = await _run_workflow(_workflow_request(spec), model, tools)

    assert tools.calls == []
    finished = [e for e in events if isinstance(e, ToolCallFinishedEvent)]
    assert [(e.ok, e.error) for e in finished] == [(False, INVALID_ARGUMENTS_ERROR)]
    assert _unanswered_calls(model.seen[-1]) == []
    assert _text(events) == "Delayed."
