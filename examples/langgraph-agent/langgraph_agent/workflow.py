"""The `workflow` agent: a LangGraph built at run time from an assistant's workflow.

The assistant's `workflow` describes the turn as nodes and edges from a fixed
library (see the engine's models/workflow.py). This module turns that into a
StateGraph for each request, runs it, and streams the same events the other
agents do. An assistant without a workflow gets the same graph the `graph`
agent runs: retrieve, model, end.

Like `agent.py`, the steps call the engine's helpers for everything that is
not the graph's shape: prompt assembly, tool discovery, the model stream with
its retry, the tool runner, usage arithmetic and pricing.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections.abc import AsyncIterator, Hashable
from typing import Annotated, Any, TypedDict

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

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
    transcript,
    usage_event,
    usage_of,
)
from chatbot_engine.agent.retriever import (
    retrieve_with_usage,
    to_context,
    to_source_refs,
    utility_config,
)
from chatbot_engine.errors import EngineError
from chatbot_engine.models.chat import ChatRequest
from chatbot_engine.models.events import (
    AskOption,
    DoneEvent,
    ErrorEvent,
    Event,
    InputRequiredEvent,
    RetrievalEvent,
    TokenEvent,
    ToolCallFinishedEvent,
)
from chatbot_engine.models.workflow import (
    AskNode,
    ConditionNode,
    HandoffNode,
    ModelNode,
    ReplyNode,
    RetrieveNode,
    ToolNode,
    WorkflowSpec,
)
from chatbot_engine.ports.agent import ToolProvider
from chatbot_engine.settings import get_settings
from chatbot_engine.tracing import run_config
from langgraph_agent.pauses import Pause, Pauses, spec_hash
from langgraph_agent.runner import run_graph

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
    usage: Annotated[dict[str, int], add_totals]
    #: Why the last spoken reply ended; `length` when `max_output_tokens` cut it.
    finish_reason: FinishReason
    vars: Annotated[dict[str, str], _merge_vars]
    context: str
    model_name: str
    #: Set by a condition node; read by its routing function.
    route: str
    #: The message that started the turn. A resumed request carries the
    #: answer as its message, so `{{message}}` reads this instead.
    message: str
    #: How many of `messages` came before the last pause. The conversation a
    #: resumed request sends already holds what was said then, so model steps
    #: read only what followed.
    since: int
    #: The usage already reported when the turn paused, so the resumed part
    #: reports only its own.
    reported: Annotated[dict[str, int], add_totals]


#: How an answer of each kind is checked and tidied, or why it is refused.
_PHONE = re.compile(r"^\+?[0-9]{6,15}$")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def check_answer(
    node: AskNode, value: str, options: list[AskOption]
) -> tuple[str, str | None]:
    """The answer as stored, or an error to ask again with."""
    value = value.strip()
    if not value:
        return "", "Please answer the question" + (
            f", or choose {node.skip_label}." if node.optional else "."
        )
    if node.input == "choice":
        if not any(o.value == value for o in options):
            return "", "Please choose one of the options."
        return value, None
    if node.input == "phone":
        digits = re.sub(r"[\s().-]", "", value)
        if not _PHONE.fullmatch(digits):
            return (
                "",
                "That does not look like a phone number. Please include the country code, such as +44 20 7946 0958.",
            )
        return digits, None
    if node.input == "email":
        if not _EMAIL.fullmatch(value) or len(value) > 254:
            return "", "That does not look like an email address."
        return value, None
    if node.input == "url":
        url = value if re.match(r"^https?://", value, re.I) else f"https://{value}"
        if " " in url or "." not in url.split("://", 1)[1] or len(url) > 2048:
            return "", "That does not look like a web address, such as example.com."
        return url, None
    if len(value) > 1000:
        return "", "Please keep the answer under 1,000 characters."
    return value, None


def options_of(node: AskNode, vars_: dict[str, str]) -> list[AskOption]:
    """A choice's options: as written, or parsed from the variable that holds them.

    A variable that is not a JSON list (a failed tool step, say) gives no
    options, and the question is then skipped rather than shown empty.
    """
    if node.options:
        return [AskOption(value=o, label=o) for o in node.options]
    if node.options_from is None:
        return []
    try:
        raw = json.loads(vars_.get(node.options_from, ""))
    except ValueError:
        return []
    if isinstance(raw, dict):
        # A tool may wrap the list: {"slots": [...]} or {"options": [...]}.
        raw = next((v for v in raw.values() if isinstance(v, list)), [])
    options: list[AskOption] = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, str) and item:
            options.append(AskOption(value=item[:200], label=item[:120]))
        elif isinstance(item, dict) and item.get("value") is not None:
            value = str(item["value"])[:200]
            options.append(
                AskOption(value=value, label=str(item.get("label") or value)[:120])
            )
    return options[:20]


class WorkflowAgent:
    """Run one chat turn as the assistant's workflow."""

    def __init__(self, tools: ToolProvider, pauses: Pauses | None = None) -> None:
        self._tools = tools
        # Opened on the first turn that can pause, not at import.
        self._pauses = pauses

    @property
    def pauses(self) -> Pauses:
        if self._pauses is None:
            self._pauses = Pauses.from_settings()
        return self._pauses

    async def run(self, request: ChatRequest) -> AsyncIterator[Event]:
        spec = request.project.workflow or DEFAULT_WORKFLOW
        events: asyncio.Queue[Any] = asyncio.Queue()
        can_pause = any(isinstance(n, AskNode) for n in spec.nodes)
        saver = await self.pauses.saver() if can_pause else None
        graph = self._build(spec, request, events, saver)
        config: dict[str, Any] = {
            **run_config(request, name="workflow"),
            "recursion_limit": spec.max_steps + 2,
        }
        project_id = request.project.project_id
        session_id = request.session_id or ""

        start: Any = {
            "messages": [],
            "usage": {},
            "vars": {},
            "context": "",
            "message": request.message,
            "since": 0,
        }
        thread_id = f"{project_id}:{uuid.uuid4().hex}"
        if request.resume is not None:
            pause = (
                await self.pauses.find(request.resume.thread_id) if can_pause else None
            )
            problem = None
            if (
                pause is None
                or pause.project_id != project_id
                or pause.session_id != session_id
            ):
                problem = (
                    "resume_expired",
                    "That question is no longer waiting for an answer. Please ask again.",
                )
            elif pause.spec_hash != spec_hash(spec):
                problem = (
                    "resume_changed",
                    "The chatbot's workflow changed while the question waited. Please ask again.",
                )
            if problem is not None:
                yield ErrorEvent(code=problem[0], message=problem[1])
                yield DoneEvent(finish_reason="error")
                return
            thread_id = request.resume.thread_id
            start = Command(
                resume={
                    "value": request.resume.value or "",
                    "skipped": request.resume.skipped,
                }
            )
        if saver is not None:
            config["configurable"] = {
                **config.get("configurable", {}),
                "thread_id": thread_id,
            }

        outcome: dict[str, Any] = {}
        try:
            async for event in run_graph(graph, start, config, events, outcome):
                yield event
        except Exception:
            if saver is not None:
                await self.pauses.forget(thread_id)
            raise

        asked = (outcome.get("values") or {}).get("__interrupt__") or []
        if saver is None:
            return
        if not asked:
            # Finished: nothing to resume, so nothing is kept.
            await self.pauses.forget(thread_id)
            return

        question = asked[0].value
        await self.pauses.record(
            Pause(thread_id, project_id, session_id, spec_hash(spec), time.time())
        )
        snapshot = await graph.aget_state(config)
        values = snapshot.values
        if not question.get("error"):
            yield TokenEvent(text=question["prompt"])
        yield InputRequiredEvent(thread_id=thread_id, **question)
        # A refused answer asks again straight away, having spent nothing new.
        spent = (
            {}
            if question.get("error")
            else {
                k: v - values.get("reported", {}).get(k, 0)
                for k, v in values.get("usage", {}).items()
            }
        )
        model_name = (
            values.get("model_name")
            or request.project.model
            or get_settings().chat_model
        )
        yield usage_event(price_usage(spent, model_name))
        yield DoneEvent(finish_reason="input_required")

    # --- nodes --------------------------------------------------------------

    def _build(
        self,
        spec: WorkflowSpec,
        request: ChatRequest,
        events: asyncio.Queue[Any],
        saver: Any = None,
    ) -> Any:
        project = request.project
        discovered: list[dict[str, Any]] | None = None

        async def tools_of() -> list[dict[str, Any]]:
            """The assistant's tools, discovered once per turn."""
            nonlocal discovered
            if discovered is None:
                discovered = await discover_tools(self._tools, project)
            return discovered

        def render(template: str, state: _State) -> str:
            values = {
                "message": state.get("message") or request.message,
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

        async def call_one(
            tool: str,
            arguments: dict[str, str],
            call_id: str,
            server_for: dict[str, str],
        ) -> tuple[str, ToolMessage]:
            """One tool call through the engine's runner: the same started and
            finished events, timing and failure handling as the model's own
            calls. Returns the result text (empty on failure) and the message."""
            ok, message = False, ToolMessage(content="", tool_call_id=call_id)
            async for item in run_tool_calls(
                [{"name": tool, "args": arguments, "id": call_id}],
                request,
                self._tools,
                server_for,
            ):
                if isinstance(item, ToolMessage):
                    message = item  # arrives after the finished event
                else:
                    if isinstance(item, ToolCallFinishedEvent):
                        ok = item.ok
                    await events.put(item)
            return (str(message.content) if ok else ""), message

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
                messages = prompt_messages(
                    request,
                    state.get("context", ""),
                    extra_system=node.prompt,
                    prior=state.get("messages", [])[state.get("since", 0) :],
                )
                new: list[BaseMessage] = []
                usage: dict[str, int] = {}
                text = ""
                finish: FinishReason = "stop"
                retries = get_settings().provider_max_retries

                # One model call more than the tool rounds allowed, as the
                # loop agent counts: the last round's results reach the model,
                # and only a request for another round ends as `tool_limit`.
                limit = project.max_tool_iterations
                for rounds_done in range(limit + 1):
                    reply: AIMessageChunk | None = None
                    async for chunk in stream_reply(
                        lambda: bound.astream(messages + new), retries=retries
                    ):
                        if chunk.text and node.var is None:
                            await events.put(TokenEvent(text=chunk.text))
                        reply = chunk if reply is None else reply + chunk
                    if reply is None:
                        break
                    usage = add_totals(usage, usage_of(reply))
                    new.append(reply)
                    text = reply.text or ""
                    finish = finish_reason_of(reply)
                    if not reply.tool_calls:
                        break
                    if rounds_done == limit:
                        finish = "tool_limit"
                        break
                    async for item in run_tool_calls(
                        reply.tool_calls, request, self._tools, server_for
                    ):
                        if isinstance(item, ToolMessage):
                            new.append(item)
                        else:
                            await events.put(item)

                out: _State = {
                    "messages": new,
                    "usage": usage,
                    "model_name": model.model_name,
                }
                if node.var is not None:
                    out["vars"] = {node.var: text}
                else:
                    out["finish_reason"] = finish
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
                async for chunk in stream_reply(
                    lambda: model.astream(
                        prompt, config=run_config(request, name=f"condition:{node.id}")
                    ),
                    retries=get_settings().provider_max_retries,
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
                # No `model_name`: the condition runs on the utility model, and
                # the turn is priced at the answer model's rate (see
                # `price_usage`), so it must not overwrite the answer's name.
                return {
                    "route": picked,
                    "vars": {f"condition_{node.id}": picked},
                    "usage": usage_of(reply),
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
                # An argument that renders empty (a skipped question, say) is
                # left out, so the tool sees it as not given.
                arguments = {
                    k: rendered
                    for k, v in node.arguments.items()
                    if (rendered := render(v, state)) != ""
                }
                call_id = f"wf-{node.id}"
                result, message = await call_one(
                    node.tool, arguments, call_id, {node.tool: server}
                )
                # The call as well as its result, so a model step later in the
                # turn reads a tool result that answers a call, as providers require.
                call = AIMessage(
                    content="",
                    tool_calls=[{"name": node.tool, "args": arguments, "id": call_id}],
                )
                return {"vars": {node.var: result}, "messages": [call, message]}

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
                        # The reason and the conversation so far, so a ticket
                        # tool or a hand-off email has both.
                        await call_one(
                            node.tool,
                            {
                                "reason": render(node.reason, state),
                                "transcript": transcript(request, include_message=True),
                            },
                            f"wf-{node.id}",
                            {node.tool: server},
                        )
                return out

            return step

        def ask_node(node: AskNode):
            async def step(state: _State) -> _State:
                # LangGraph runs this step again from the top when the answer
                # arrives, and `interrupt` then returns that answer instead of
                # pausing; asking again after a refused answer pauses once more.
                options = options_of(node, state.get("vars", {}))
                label = node.var + "_label"
                if node.input == "choice" and not options:
                    return {"vars": {node.var: "", label: ""}}
                question: dict[str, Any] = {
                    "node": node.id,
                    "prompt": render(node.prompt, state),
                    "input": node.input,
                    "options": [o.model_dump() for o in options],
                    "optional": node.optional,
                    "skip_label": node.skip_label if node.optional else None,
                    "placeholder": node.placeholder or None,
                    "error": None,
                }
                while True:
                    answer = interrupt(question)
                    if answer.get("skipped") and node.optional:
                        value, chosen = "", ""
                        break
                    value, error = check_answer(
                        node, str(answer.get("value") or ""), options
                    )
                    if error is None:
                        chosen = next(
                            (o.label for o in options if o.value == value), value
                        )
                        break
                    question = {**question, "error": error}
                return {
                    "vars": {node.var: value, label: chosen},
                    "since": len(state.get("messages", [])),
                    "reported": {
                        k: v - state.get("reported", {}).get(k, 0)
                        for k, v in state.get("usage", {}).items()
                    },
                }

            return step

        async def end_step(_state):
            """An `end` node does nothing; the edge to finish does the work."""
            return {}

        async def finish(state: _State) -> _State:
            # A turn with no model step (condition and reply nodes only) still
            # names the model the turn is priced at, resolved as
            # `build_chat_model` does, so the caller can price it.
            model_name = (
                state.get("model_name") or project.model or get_settings().chat_model
            )
            # After a pause, only what was spent since: the rest was reported then.
            reported = state.get("reported", {})
            spent = {
                k: v - reported.get(k, 0) for k, v in state.get("usage", {}).items()
            }
            await events.put(usage_event(price_usage(spent, model_name)))
            await events.put(
                DoneEvent(finish_reason=state.get("finish_reason", "stop"))
            )
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
            elif isinstance(node, AskNode):
                graph.add_node(node.id, ask_node(node))
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
        return graph.compile(checkpointer=saver)


def build(tools: ToolProvider) -> WorkflowAgent:
    """The factory the entry point names. The engine calls this once."""
    return WorkflowAgent(tools=tools)
