"""The `workflow` agent: a LangGraph built at run time from an assistant's workflow.

The assistant's `workflow` describes the turn as nodes and edges from a fixed
library (see the engine's models/workflow.py). This module turns that into a
StateGraph for each request, runs it, and streams the same events the other
agents do. An assistant without a workflow gets the same graph the `graph`
agent runs: retrieve, model, end.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Hashable
from typing import Annotated, Any, TypedDict, cast

from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph

from chatbot_engine.agent.client import build_chat_model, run_tool_calls, to_messages
from chatbot_engine.agent.retriever import (
    retrieve_with_usage,
    to_context,
    to_source_refs,
    utility_config,
)
from chatbot_engine.errors import EngineError
from chatbot_engine.models.chat import ChatRequest
from chatbot_engine.models.events import (
    DoneEvent,
    Event,
    RetrievalEvent,
    TokenEvent,
    ToolCallFinishedEvent,
    ToolCallStartedEvent,
)
from chatbot_engine.models.workflow import (
    ConditionNode,
    HandoffNode,
    ModelNode,
    ReplyNode,
    RetrieveNode,
    ToolNode,
    WorkflowSpec,
)
from chatbot_engine.ports.agent import ToolProvider
from chatbot_engine.tracing import run_config
from langgraph_agent.agent import _DONE, _merge_usage, _usage_event, _usage_of

DEFAULT_WORKFLOW = WorkflowSpec.model_validate(
    {
        "start": "retrieve",
        "nodes": [
            {"id": "retrieve", "type": "retrieve"},
            {"id": "answer", "type": "model"},
        ],
        "edges": [{"from": "retrieve", "to": "answer"}],
    }
)


def _merge_vars(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    return {**left, **right}


class _State(TypedDict, total=False):
    #: This turn's model replies and tool results, in order.
    messages: Annotated[list[BaseMessage], lambda a, b: [*a, *b]]
    usage: Annotated[dict[str, int], _merge_usage]
    vars: Annotated[dict[str, str], _merge_vars]
    context: str
    model_name: str
    #: Set by a condition node; read by its routing function.
    route: str


class WorkflowAgent:
    """Run one chat turn as the assistant's workflow."""

    def __init__(self, tools: ToolProvider) -> None:
        self._tools = tools

    async def run(self, request: ChatRequest) -> AsyncIterator[Event]:
        spec = request.project.workflow or DEFAULT_WORKFLOW
        events: asyncio.Queue[Any] = asyncio.Queue()
        graph = self._build(spec, request, events)

        async def drive() -> None:
            try:
                await graph.ainvoke(
                    {"messages": [], "usage": {}, "vars": {}, "context": ""},
                    {
                        **run_config(request, name="workflow"),
                        "recursion_limit": spec.max_steps + 2,
                    },
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
            await task
        finally:
            if not task.done():
                task.cancel()

    # --- nodes --------------------------------------------------------------

    def _build(
        self, spec: WorkflowSpec, request: ChatRequest, events: asyncio.Queue[Any]
    ) -> Any:
        project = request.project
        discovered: list[dict[str, Any]] | None = None

        async def tools_of() -> list[dict[str, Any]]:
            nonlocal discovered
            if discovered is None:
                try:
                    discovered = [
                        dict(t) for t in await self._tools.list_tools(project)
                    ]
                except Exception as exc:
                    urls = ", ".join(s.url for s in project.mcp_servers)
                    raise EngineError(f"could not discover tools from {urls}") from exc
            return discovered

        def render(template: str, state: _State) -> str:
            values = {
                "message": request.message,
                "user_id": request.user_id or "",
                "session_id": request.session_id or "",
            }
            vars_ = state.get("vars", {})

            def sub(m: re.Match[str]) -> str:
                key = m.group(1).strip()
                if key.startswith("vars."):
                    return vars_.get(key[5:], "")
                return values.get(key, "")

            return re.sub(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_.]*)\s*\}\}", sub, template)

        def prompt_messages(state: _State, extra: str = "") -> list[BaseMessage]:
            system = project.system_prompt + (f"\n\n{extra}" if extra else "")
            return [
                SystemMessage(content=system),
                *to_messages(request, state.get("context", "")),
                *state.get("messages", []),
            ]

        async def retrieve(_: _State) -> _State:
            hits, spent = await retrieve_with_usage(request)
            await events.put(
                RetrievalEvent(query=request.message, sources=to_source_refs(hits))
            )
            return {"context": to_context(hits), "usage": dict(spent)}

        def model_node(node: ModelNode):
            async def step(state: _State) -> _State:
                tools = await tools_of() if node.tools else []
                server_for = {t["name"]: t["server"] for t in tools}
                model = build_chat_model(project)
                bound = model.bind_tools(tools) if tools else model
                messages = prompt_messages(state, node.prompt)
                new: list[BaseMessage] = []
                usage: dict[str, int] = {}
                text = ""

                for _ in range(project.max_tool_iterations):
                    reply: AIMessageChunk | None = None
                    async for chunk in bound.astream(messages + new):
                        if chunk.text and node.var is None:
                            await events.put(TokenEvent(text=chunk.text))
                        reply = cast(
                            AIMessageChunk, chunk if reply is None else reply + chunk
                        )
                    if reply is None:
                        break
                    usage = _merge_usage(usage, _usage_of(reply))
                    new.append(reply)
                    text = reply.text or ""
                    if not reply.tool_calls:
                        break
                    async for item in run_tool_calls(
                        reply.tool_calls, request, self._tools, server_for
                    ):
                        if isinstance(item, ToolMessage):
                            new.append(item)
                        else:
                            await events.put(item)
                else:
                    raise EngineError(
                        f"the model was still calling tools after {project.max_tool_iterations} rounds"
                    )

                out: _State = {
                    "messages": new,
                    "usage": usage,
                    "model_name": model.model_name,
                }
                if node.var is not None:
                    out["vars"] = {node.var: text}
                return out

            return step

        def condition_node(node: ConditionNode):
            labels = list(node.branches)

            async def step(state: _State) -> _State:
                model = build_chat_model(utility_config(project))
                prompt = [
                    SystemMessage(
                        content="Answer with exactly one of these labels and nothing else: "
                        + ", ".join(labels)
                    ),
                    HumanMessage(
                        content=f"{node.question}\n\nMessage: {request.message}\n\n"
                        f"Context:\n{state.get('context', '')[:4000]}"
                    ),
                ]
                # Streamed like every other call, so a streaming-only model
                # (the test double included) is enough.
                reply: AIMessageChunk | None = None
                async for chunk in model.astream(
                    prompt, config=run_config(request, name=f"condition:{node.id}")
                ):
                    reply = chunk if reply is None else reply + chunk
                if reply is None:
                    reply = AIMessageChunk(content="")
                answer = (reply.text or "").strip().lower()
                # An exact label first, then a label as a whole word; a bare
                # substring would let "b" match "mumble".
                picked = next(
                    (label for label in labels if label.lower() == answer),
                    next(
                        (
                            label
                            for label in labels
                            if re.search(rf"\b{re.escape(label.lower())}\b", answer)
                        ),
                        labels[0],
                    ),
                )
                return {
                    "route": picked,
                    "vars": {f"condition_{node.id}": picked},
                    "usage": _usage_of(reply),
                }

            return step

        def tool_node(node: ToolNode):
            async def step(state: _State) -> _State:
                tools = await tools_of()
                server = next(
                    (t["server"] for t in tools if t["name"] == node.tool), None
                )
                if server is None:
                    raise EngineError(
                        f"workflow node {node.id!r} names a tool the assistant does not allow: {node.tool!r}"
                    )
                arguments = {k: render(v, state) for k, v in node.arguments.items()}
                call_id = f"wf-{node.id}"
                await events.put(
                    ToolCallStartedEvent(
                        call_id=call_id,
                        tool=node.tool,
                        server=server,
                        arguments=arguments,
                    )
                )
                loop = asyncio.get_running_loop()
                started = loop.time()
                try:
                    result = await self._tools.call_tool(
                        config=project,
                        server=server,
                        name=node.tool,
                        arguments=arguments,
                        user_id=request.user_id,
                        session_id=request.session_id,
                    )
                    ok, error = True, None
                except Exception as exc:
                    result, ok, error = "", False, str(exc)
                await events.put(
                    ToolCallFinishedEvent(
                        call_id=call_id,
                        tool=node.tool,
                        ok=ok,
                        duration_ms=int((loop.time() - started) * 1000),
                        error=error,
                    )
                )
                return {
                    "vars": {node.var: result},
                    "messages": [
                        ToolMessage(
                            content=result or (error or ""), tool_call_id=call_id
                        )
                    ],
                }

            return step

        def reply_node(node: ReplyNode):
            async def step(state: _State) -> _State:
                text = render(node.text, state)
                await events.put(TokenEvent(text=text))
                return {"messages": [AIMessageChunk(content=text)]}

            return step

        def handoff_node(node: HandoffNode):
            async def step(state: _State) -> _State:
                text = render(node.message, state)
                await events.put(TokenEvent(text=text))
                out: _State = {
                    "vars": {"handed_off": "true"},
                    "messages": [AIMessageChunk(content=text)],
                }
                if node.tool:
                    tools = await tools_of()
                    server = next(
                        (t["server"] for t in tools if t["name"] == node.tool), None
                    )
                    if server:
                        call_id = f"wf-{node.id}"
                        transcript = (
                            "\n".join(f"{t.role}: {t.content}" for t in request.history)
                            + f"\nuser: {request.message}"
                        )
                        await events.put(
                            ToolCallStartedEvent(
                                call_id=call_id,
                                tool=node.tool,
                                server=server,
                                arguments={"transcript": transcript},
                            )
                        )
                        try:
                            await self._tools.call_tool(
                                config=project,
                                server=server,
                                name=node.tool,
                                arguments={"transcript": transcript},
                                user_id=request.user_id,
                                session_id=request.session_id,
                            )
                            await events.put(
                                ToolCallFinishedEvent(
                                    call_id=call_id,
                                    tool=node.tool,
                                    ok=True,
                                    duration_ms=0,
                                )
                            )
                        except Exception as exc:
                            await events.put(
                                ToolCallFinishedEvent(
                                    call_id=call_id,
                                    tool=node.tool,
                                    ok=False,
                                    duration_ms=0,
                                    error=str(exc),
                                )
                            )
                return out

            return step

        async def end_step(_state):
            """An `end` node does nothing; the edge to finish does the work."""
            return {}

        async def finish(state: _State) -> _State:
            await events.put(
                _usage_event(
                    state.get("usage", {}), state.get("model_name") or project.model
                )
            )
            await events.put(DoneEvent(finish_reason="stop"))
            return {}

        graph = StateGraph(_State)  # ty: ignore[invalid-argument-type]
        for node in spec.nodes:
            if isinstance(node, RetrieveNode):
                graph.add_node(node.id, retrieve)  # ty: ignore[invalid-argument-type]
            elif isinstance(node, ModelNode):
                graph.add_node(node.id, model_node(node))
            elif isinstance(node, ConditionNode):
                graph.add_node(node.id, condition_node(node))
            elif isinstance(node, ToolNode):
                graph.add_node(node.id, tool_node(node))
            elif isinstance(node, ReplyNode):
                graph.add_node(node.id, reply_node(node))
            elif isinstance(node, HandoffNode):
                graph.add_node(node.id, handoff_node(node))
            else:
                graph.add_node(node.id, end_step)  # ty: ignore[invalid-argument-type]
        graph.add_node("__finish__", finish)

        graph.add_edge(START, spec.start)
        for node in spec.nodes:
            if isinstance(node, ConditionNode):
                routes: dict[Hashable, str] = dict(node.branches.items())
                graph.add_conditional_edges(node.id, lambda s: s["route"], routes)
            else:
                graph.add_edge(node.id, spec.next_of(node.id) or "__finish__")
        graph.add_edge("__finish__", END)
        return graph.compile()


def build(tools: ToolProvider) -> WorkflowAgent:
    """The factory the entry point names. The engine calls this once."""
    return WorkflowAgent(tools=tools)
