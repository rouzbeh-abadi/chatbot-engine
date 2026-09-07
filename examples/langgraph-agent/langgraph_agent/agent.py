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
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Annotated, Any, TypedDict

from chatbot_engine.agent.client import (
    _usage,
    build_chat_model,
    run_tool_calls,
    to_messages,
)
from chatbot_engine.agent.retriever import retrieve, to_context, to_source_refs
from chatbot_engine.errors import EngineError
from chatbot_engine.models.chat import ChatRequest
from chatbot_engine.models.events import (
    DoneEvent,
    Event,
    RetrievalEvent,
    TokenEvent,
    UsageEvent,
)
from chatbot_engine.ports.agent import ToolProvider
from langchain_core.messages import AIMessageChunk, BaseMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

#: Marks the end of the event stream, so `run` knows the graph has finished.
_DONE = object()


def _merge_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    """Add token counts as the turn goes round.

    A turn with tool calls is several model calls, and reporting only the last
    one would understate the total.
    """
    return {key: left.get(key, 0) + right.get(key, 0) for key in {*left, *right}}


class _State(TypedDict, total=False):
    """What flows between nodes.

    `messages` is appended to by every node; `usage` is summed. Nothing else is
    carried, because everything else about the turn arrives with the request.
    """

    messages: Annotated[list[BaseMessage], lambda a, b: [*a, *b]]
    usage: Annotated[dict[str, int], _merge_usage]
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

        async def drive() -> None:
            try:
                await graph.ainvoke(
                    {"messages": [], "usage": {}, "context": ""},
                    # One more than the tool rounds allowed: each round is a
                    # model step and a tool step, plus the retrieval step.
                    {"recursion_limit": 2 * request.project.max_tool_iterations + 3},
                )
            finally:
                await events.put(_DONE)

        task = asyncio.create_task(drive())
        try:
            while True:
                item = await events.get()
                if item is _DONE:
                    break
                yield item
            # Surfaces anything the graph raised, now that the stream is drained.
            await task
        finally:
            if not task.done():
                task.cancel()

    def _build(self, request: ChatRequest, events: asyncio.Queue[Any]) -> Any:
        async def retrieve_node(state: _State) -> _State:
            hits = await retrieve(request)
            # Before the answer, so the UI can show what it was based on while
            # the model is still thinking.
            await events.put(
                RetrievalEvent(query=request.message, sources=to_source_refs(hits))
            )
            context = to_context(hits)
            return {"messages": to_messages(request, context), "context": context}

        async def model_node(state: _State) -> _State:
            tools = await self._discover(request)
            model = build_chat_model(request.project)
            bound = model.bind_tools([dict(t) for t in tools]) if tools else model

            reply: AIMessageChunk | None = None
            async for chunk in bound.astream(state["messages"]):
                if chunk.text:
                    await events.put(TokenEvent(text=chunk.text))
                reply = chunk if reply is None else reply + chunk

            assert reply is not None
            return {
                "messages": [reply],
                "usage": _usage_of(reply),
                "model_name": model.model_name,
            }

        async def tools_node(state: _State) -> _State:
            """Run whatever the model asked for, over MCP.

            Delegates to the same `run_tool_calls` the loop agent uses, rather
            than reimplementing it: two agents that ran tools differently, or
            reported them differently, would be a bug waiting to happen. It
            yields the started and finished events and the `ToolMessage` that
            answers each call.
            """
            tools = await self._discover(request)
            server_for = {tool["name"]: tool["server"] for tool in tools}
            calls = getattr(state["messages"][-1], "tool_calls", []) or []

            results: list[BaseMessage] = []
            async for item in run_tool_calls(calls, request, self._tools, server_for):
                if isinstance(item, ToolMessage):
                    results.append(item)
                else:
                    await events.put(item)

            return {"messages": results}

        async def finish_node(state: _State) -> _State:
            await events.put(
                _usage_event(state.get("usage", {}), state.get("model_name"))
            )
            await events.put(DoneEvent(finish_reason="stop"))
            return {}

        def next_step(state: _State) -> str:
            """The one branch in the turn: did the model ask for a tool?"""
            last = state["messages"][-1]
            return "tools" if getattr(last, "tool_calls", None) else "finish"

        graph = StateGraph(_State)
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

    async def _discover(self, request: ChatRequest) -> list[dict[str, Any]]:
        """The tools this assistant allows, named in the error if unreachable."""
        try:
            return list(await self._tools.list_tools(request.project))
        except Exception as exc:
            urls = ", ".join(server.url for server in request.project.mcp_servers)
            raise EngineError(f"could not discover tools from {urls}") from exc


def _usage_of(reply: AIMessageChunk) -> dict[str, int]:
    """Token counts off one model reply, or nothing when it reported none."""
    usage = getattr(reply, "usage_metadata", None) or {}
    return {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


def _usage_event(totals: dict[str, int], model_name: str | None) -> UsageEvent:
    """The turn's total spend, priced by the same helper the loop agent uses.

    Shared on purpose: two agents that disagree about what a turn cost would be
    worse than one agent.
    """
    usage = _usage(
        {
            "input_tokens": totals.get("input_tokens", 0),
            "output_tokens": totals.get("output_tokens", 0),
            "total_tokens": totals.get("total_tokens", 0),
        },
        model_name,
    )
    return UsageEvent(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        cost_usd=usage.cost_usd,
        model=usage.model,
    )


def build(tools: ToolProvider) -> LangGraphAgent:
    """The factory the entry point names. The engine calls this once."""
    return LangGraphAgent(tools=tools)
