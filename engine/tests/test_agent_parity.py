"""The engine's agent and a plugin agent must be interchangeable.

`loop` ships with the engine; `graph` is installed as a plugin from
`examples/langgraph-agent`. That they are indistinguishable here is what makes
the plugin contract real: an agent written outside the engine is not a
second-class citizen.


The loop agent and the graph agent run the same turn by different means. That
is only useful if a caller cannot tell them apart: same events, in the same
order, with the same token counts and the same tool calls. Anything else and
switching `agent` in the project config quietly changes the product.

Each test runs one scripted conversation through both and compares.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from chatbot_engine.agent.chat_agent import ChatAgent
from langgraph_agent.agent import LangGraphAgent
from chatbot_engine.models.chat import AssistantConfig, ChatRequest
from chatbot_engine.models.events import (
    DoneEvent,
    TokenEvent,
    ToolCallFinishedEvent,
    ToolCallStartedEvent,
    UsageEvent,
)

AGENTS = ["loop", "graph"]


class ScriptedModel(BaseChatModel):
    rounds: list
    model_name: str = "openai/gpt-5-mini"

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, *args, **kwargs):  # pragma: no cover - unused
        raise NotImplementedError

    def bind_tools(self, tools, **kwargs):
        return self

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        for chunk in self.rounds.pop(0):
            yield ChatGenerationChunk(message=chunk)


class FakeTools:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def list_tools(self, config):
        return [
            {
                "server": "s",
                "name": "get_booking_status",
                "description": "d",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]

    async def call_tool(self, *, name, **kwargs):
        self.calls.append(name)
        return '{"status": "delayed"}'


async def _no_retrieval(_request):
    return []


def _request() -> ChatRequest:
    return ChatRequest(
        project=AssistantConfig(
            project_id="support",
            name="S",
            system_prompt="p",
            model="openai/gpt-5-mini",
        ),
        message="is my flight delayed?",
    )


def _usage_chunk(prompt: int, completion: int) -> AIMessageChunk:
    return AIMessageChunk(
        content="",
        usage_metadata={
            "input_tokens": prompt,
            "output_tokens": completion,
            "total_tokens": prompt + completion,
        },
    )


def _rounds_with_a_tool_call() -> list:
    return [
        [
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": "get_booking_status", "args": "{}", "id": "c1", "index": 0}
                ],
            ),
            _usage_chunk(30, 10),
        ],
        [AIMessageChunk(content="Delayed."), _usage_chunk(50, 20)],
    ]


async def _run(which: str, rounds: list, tools: FakeTools) -> list:
    """Drive one agent through a scripted conversation and collect its events."""
    model = ScriptedModel(rounds=rounds)

    if which == "loop":
        agent = ChatAgent(tools=tools)
        targets = [
            patch("chatbot_engine.agent.client.build_chat_model", return_value=model),
            patch("chatbot_engine.agent.chat_agent.retrieve", new=_no_retrieval),
        ]
    else:
        agent = LangGraphAgent(tools=tools)
        targets = [
            patch(
                "langgraph_agent.agent.build_chat_model",
                return_value=model,
            ),
            patch("langgraph_agent.agent.retrieve", new=_no_retrieval),
        ]

    with targets[0], targets[1]:
        return [event async for event in agent.run(_request())]


# --- each agent on its own -----------------------------------------------------


@pytest.mark.parametrize("which", AGENTS)
async def test_a_turn_with_a_tool_call_produces_the_expected_events(
    which: str,
) -> None:
    tools = FakeTools()

    events = await _run(which, _rounds_with_a_tool_call(), tools)
    types = [type(e).__name__ for e in events]

    assert tools.calls == ["get_booking_status"]
    assert types[0] == "RetrievalEvent"
    assert "ToolCallStartedEvent" in types
    assert "ToolCallFinishedEvent" in types
    assert types.index("UsageEvent") < types.index("DoneEvent")
    assert isinstance(events[-1], DoneEvent)


@pytest.mark.parametrize("which", AGENTS)
async def test_usage_is_summed_across_tool_rounds(which: str) -> None:
    """Both agents must count every model call, not just the last."""
    events = await _run(which, _rounds_with_a_tool_call(), FakeTools())

    usage = next(e for e in events if isinstance(e, UsageEvent))

    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (80, 30, 110)
    assert usage.cost_usd == pytest.approx(0.00008)
    assert usage.model == "openai/gpt-5-mini"


@pytest.mark.parametrize("which", AGENTS)
async def test_the_answer_text_streams_as_tokens(which: str) -> None:
    rounds = [[AIMessageChunk(content="Hi."), _usage_chunk(5, 2)]]

    events = await _run(which, rounds, FakeTools())

    assert "".join(e.text for e in events if isinstance(e, TokenEvent)) == "Hi."


@pytest.mark.parametrize("which", AGENTS)
async def test_a_failing_tool_does_not_end_the_turn(which: str) -> None:
    """The model is told the tool failed and still gets to answer."""

    class BrokenTools(FakeTools):
        async def call_tool(self, *, name, **kwargs):
            self.calls.append(name)
            raise RuntimeError("tool server down")

    events = await _run(which, _rounds_with_a_tool_call(), BrokenTools())

    finished = next(e for e in events if isinstance(e, ToolCallFinishedEvent))
    assert finished.ok is False
    assert isinstance(events[-1], DoneEvent)


# --- the two against each other ------------------------------------------------


async def test_both_agents_emit_the_same_event_sequence() -> None:
    """The whole point: swapping `agent` must not change what the caller sees."""
    loop = await _run("loop", _rounds_with_a_tool_call(), FakeTools())
    graph = await _run("graph", _rounds_with_a_tool_call(), FakeTools())

    assert [type(e).__name__ for e in loop] == [type(e).__name__ for e in graph]


async def test_both_agents_report_the_same_usage_and_answer() -> None:
    loop = await _run("loop", _rounds_with_a_tool_call(), FakeTools())
    graph = await _run("graph", _rounds_with_a_tool_call(), FakeTools())

    def summary(events: list) -> tuple:
        usage = next(e for e in events if isinstance(e, UsageEvent))
        text = "".join(e.text for e in events if isinstance(e, TokenEvent))
        started = [e.tool for e in events if isinstance(e, ToolCallStartedEvent)]
        return text, started, usage.total_tokens, usage.cost_usd

    assert summary(loop) == summary(graph)
