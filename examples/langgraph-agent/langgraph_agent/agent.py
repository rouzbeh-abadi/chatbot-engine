"""A LangGraph agent for the chatbot engine, provided as a plugin.

`ChatAgent` runs retrieval, the model, and the tool loop as straight-line code.
This runs the identical turn as a LangGraph state machine: three nodes and one
conditional edge. Same inputs, same events, same answer. The difference is that
the control flow is data you can inspect and extend rather than a `for` loop.

    retrieve -> model -> (tool calls?) -> tools -> model -> ... -> END

This lives outside the engine on purpose. The engine defines what an agent is
and how one is found; picking LangGraph is an application decision, so it is
made here. An adopter who wants a different framework writes their own package
the same way and changes nothing in the engine.

A graph earns its place once a turn stops being a straight line: approvals in
the middle, branching on the question, resuming a half-finished turn from a
checkpointer. For a single question and answer the engine's own loop is
simpler, which is why it remains the default.

Events are pushed onto a queue by the nodes rather than reconstructed from
LangGraph's stream. The nodes know exactly what happened; a stream of graph
updates has to be interpreted, and the interpretation changes between LangGraph
versions. The queue keeps this agent's output identical to `ChatAgent`'s.

The nodes do not reimplement the engine. Prompt assembly, tool discovery, the
model stream with its retry, the tool runner, usage arithmetic and pricing are
all the engine's helpers from `chatbot_engine.agent.client`; this module owns
only the graph's shape.
"""

from __future__ import annotations

import asyncio
import operator
from collections.abc import AsyncIterator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessageChunk, BaseMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from chatbot_engine.agent.client import (
    FinishReason,
    add_totals,
    build_chat_model,
    discover_tools,
    finish_reason_of,
    price_usage,
    prompt_messages,
    run_tool_calls,
    stream_reply,
    unavailable_note,
    usage_event,
    usage_of,
)
from chatbot_engine.agent.retriever import (
    retrieve_with_usage,
    to_context,
    to_source_refs,
)
from chatbot_engine.models.chat import ChatRequest
from chatbot_engine.models.events import (
    DoneEvent,
    Event,
    RetrievalEvent,
    TokenEvent,
)
from chatbot_engine.ports.agent import ToolProvider
from chatbot_engine.settings import get_settings
from chatbot_engine.tracing import run_config
from langgraph_agent.runner import run_graph


class _State(TypedDict, total=False):
    """What flows between nodes.

    `messages` is appended to by every node; `usage` is summed. Nothing else is
    carried, because everything else about the turn arrives with the request.
    """

    messages: Annotated[list[BaseMessage], lambda a, b: [*a, *b]]
    usage: Annotated[dict[str, int], add_totals]
    #: Why the last reply ended; `length` when `max_output_tokens` cut it.
    finish_reason: FinishReason
    #: Tool rounds run so far, against `max_tool_iterations`.
    rounds: Annotated[int, operator.add]
    context: str
    #: Taken from the model actually built, so pricing matches the loop agent's.
    model_name: str


class LangGraphAgent:
    """Run one chat turn as a LangGraph state machine.

    Satisfies the same `Agent` port as `ChatAgent`, so the two are
    interchangeable and can be compared against each other in one engine.
    """

    def __init__(self, tools: ToolProvider) -> None:
        self._tools = tools

    async def run(self, request: ChatRequest) -> AsyncIterator[Event]:
        """Process one chat request and yield events as the answer is produced.

        The graph runs as a task while this drains the queue its nodes write to,
        so events reach the caller as they happen rather than at the end.
        """
        events: asyncio.Queue[Any] = asyncio.Queue()
        graph = self._build(request, events)
        config = {
            # The tracer and the ids; the nodes' model calls inherit them.
            **run_config(request, name="graph"),
            # Retrieval, a model step and a tool step per allowed round, the
            # final model step and finish make 2N + 3; one more so the limit
            # is never what stops a legal turn.
            "recursion_limit": 2 * request.project.max_tool_iterations + 4,
        }
        async for event in run_graph(
            graph,
            {"messages": [], "usage": {}, "context": "", "rounds": 0},
            config,
            events,
        ):
            yield event

    def _build(self, request: ChatRequest, events: asyncio.Queue[Any]) -> Any:
        discovered: list[dict[str, Any]] | None = None

        async def tools_of() -> list[dict[str, Any]]:
            """The assistant's tools, discovered once per turn and reused by
            every model and tool step."""
            nonlocal discovered
            if discovered is None:
                discovered = await discover_tools(self._tools, request.project)
            return discovered

        async def retrieve_node(state: _State) -> _State:
            hits, spent = await retrieve_with_usage(request)
            # Before the answer, so the UI can show what it was based on while
            # the model is still thinking.
            await events.put(
                RetrievalEvent(query=request.message, sources=to_source_refs(hits))
            )
            context = to_context(hits)
            # Tools that could not be reached are named in the prompt, as the
            # loop agent does, so both agents send the same system prompt.
            note = unavailable_note(request.project, await tools_of())
            return {
                # The system prompt leads, exactly as in the engine's loop agent.
                "messages": prompt_messages(request, context, extra_system=note),
                "context": context,
                # What retrieval's own model calls cost, so the turn's usage
                # is the whole turn's, as it is for the loop agent.
                "usage": spent,
            }

        async def model_node(state: _State) -> _State:
            tools = await tools_of()
            model = build_chat_model(request.project)
            bound = model.bind_tools(tools) if tools else model

            reply: AIMessageChunk | None = None
            async for chunk in stream_reply(
                lambda: bound.astream(state["messages"]),
                retries=get_settings().provider_max_retries,
            ):
                if chunk.text:
                    await events.put(TokenEvent(text=chunk.text))
                reply = chunk if reply is None else reply + chunk

            assert reply is not None
            return {
                "messages": [reply],
                "usage": usage_of(reply),
                "model_name": model.model_name,
                "finish_reason": finish_reason_of(reply),
            }

        async def tools_node(state: _State) -> _State:
            """Run whatever the model asked for, over MCP.

            Delegates to the same `run_tool_calls` the loop agent uses, rather
            than reimplementing it: two agents that ran tools differently, or
            reported them differently, would be a bug waiting to happen. It
            yields the started and finished events and the `ToolMessage` that
            answers each call.
            """
            tools = await tools_of()
            server_for = {tool["name"]: tool["server"] for tool in tools}
            calls = getattr(state["messages"][-1], "tool_calls", []) or []

            results: list[BaseMessage] = []
            async for item in run_tool_calls(calls, request, self._tools, server_for):
                if isinstance(item, ToolMessage):
                    results.append(item)
                else:
                    await events.put(item)

            return {"messages": results, "rounds": 1}

        async def finish_node(state: _State) -> _State:
            await events.put(
                usage_event(
                    price_usage(state.get("usage", {}), state.get("model_name"))
                )
            )
            # Arriving here with a tool request still open means the rounds
            # ran out: the same `tool_limit` the loop agent reports.
            last = state["messages"][-1] if state.get("messages") else None
            limited = bool(getattr(last, "tool_calls", None))
            await events.put(
                DoneEvent(
                    finish_reason="tool_limit"
                    if limited
                    else state.get("finish_reason", "stop")
                )
            )
            return {}

        def next_step(state: _State) -> str:
            """The one branch in the turn: did the model ask for a tool, and may it still?"""
            last = state["messages"][-1]
            wants_tools = bool(getattr(last, "tool_calls", None))
            allowed = state.get("rounds", 0) < request.project.max_tool_iterations
            return "tools" if wants_tools and allowed else "finish"

        graph = StateGraph(_State)  # ty: ignore[invalid-argument-type]  a TypedDict with reducers is what LangGraph documents
        graph.add_node("retrieve", retrieve_node)
        graph.add_node("model", model_node)
        graph.add_node("tools", tools_node)
        graph.add_node("finish", finish_node)

        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "model")
        graph.add_conditional_edges(
            "model", next_step, {"tools": "tools", "finish": "finish"}
        )
        graph.add_edge("tools", "model")
        graph.add_edge("finish", END)

        return graph.compile()


def build(tools: ToolProvider) -> LangGraphAgent:
    """The factory the entry point names. The engine calls this once."""
    return LangGraphAgent(tools=tools)
