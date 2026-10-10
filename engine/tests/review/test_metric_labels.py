"""Review: tool names the model writes become permanent metric labels
(EVALTRACE-10, EVALTRACE-11).

`record_turn` counts every `ToolCallFinishedEvent` under
`TOOL_CALLS.labels(tool=event.tool, ...)`, and `run_tool_calls` yields one for
every call the model makes, including a name it was never offered and a call
whose arguments could not be read. The tests drive the real path (POST /chat,
the loop or workflow agent, `BilledChatOpenAI` against a fake OpenAI-compatible
server on 127.0.0.1, `record_turn`, the process-wide registry). No real provider.
"""

from __future__ import annotations

import gc
import json
import threading
import tracemalloc
import uuid
from collections.abc import AsyncIterator, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from chatbot_engine.agent.client import run_tool_calls
from chatbot_engine.agent.router import AgentRouter
from chatbot_engine.api import dependencies
from chatbot_engine.models.chat import ChatRequest
from chatbot_engine.models.events import DoneEvent, Event, ToolCallFinishedEvent
from chatbot_engine.observability import record_turn
from chatbot_engine.services.chat import ChatService

#: The one tool the chatbot really offers (bound to the model).
OFFERED = "get_booking_status"

#: A tool-label bound any sane fix would keep under: OpenAI's own limit on a
#: function name is 64 characters; twice that is generous.
MAX_LABEL = 128


# --- a fake OpenAI-compatible provider -----------------------------------------


class _Script:
    """What the fake provider answers, and what it was sent."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        #: Names for the next tool-calling reply, taken per request.
        self.names: list[list[str]] = []
        #: Names whose arguments are sent as unreadable JSON.
        self.invalid: set[str] = set()
        #: The `tools` the engine bound, per request (names only).
        self.bound: list[list[str]] = []


def _sse(obj: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode()


def _chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "openai/gpt-5-mini",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _usage() -> dict[str, Any]:
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "openai/gpt-5-mini",
        "choices": [],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "cost": 0.0001,
        },
    }


def _handler(script: _Script) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:  # quiet
            pass

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            messages = body.get("messages", [])
            tools = [t["function"]["name"] for t in body.get("tools") or []]
            with script.lock:
                script.bound.append(tools)
                # A model call that follows tool results answers in prose; the
                # first call of a turn makes the scripted tool calls.
                calling = (
                    bool(tools)
                    and messages
                    and messages[-1].get("role") != "tool"
                    and script.names
                )
                names = script.names.pop(0) if calling else []

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            out = [_sse(_chunk({"role": "assistant", "content": ""}))]
            if names:
                calls = [
                    {
                        "index": i,
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": "not json" if name in script.invalid else "{}",
                        },
                    }
                    for i, name in enumerate(names)
                ]
                out.append(_sse(_chunk({"tool_calls": calls})))
                out.append(_sse(_chunk({}, "tool_calls")))
            else:
                out.append(_sse(_chunk({"content": "Sorry, I cannot do that."})))
                out.append(_sse(_chunk({}, "stop")))
            out.append(_sse(_usage()))
            out.append(b"data: [DONE]\n\n")
            self.wfile.write(b"".join(out))

    return Handler


class FakeTools:
    """The chatbot's tool server: one tool offered; records what ran."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def list_tools(self, config: Any) -> list[dict[str, Any]]:
        return [
            {
                "server": "s",
                "name": OFFERED,
                "description": "Look up a booking.",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]

    async def call_tool(self, *, name: str, **kwargs: Any) -> str:
        self.calls.append(name)
        return '{"status": "delayed"}'


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Script]:
    """A fake OpenAI-compatible provider on 127.0.0.1, set as the engine's
    OpenRouter base URL."""
    script = _Script()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(script))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv(
        "ENGINE_OPENROUTER_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1"
    )
    monkeypatch.setenv("ENGINE_PROVIDER_MAX_RETRIES", "0")
    dependencies.reset_dependency_cache()
    try:
        yield script
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def engine(provider: _Script, client: TestClient) -> tuple[TestClient, FakeTools]:
    """The app, with the real agent router over a fake tool server."""
    tools = FakeTools()
    client.app.dependency_overrides[dependencies.get_chat_service] = lambda: (
        ChatService(agent=AgentRouter(tools=tools))
    )
    return client, tools


# --- helpers ------------------------------------------------------------------


def tool_label_values() -> list[str]:
    """Every `tool` label value the process-wide registry holds."""
    return [
        sample.labels["tool"]
        for metric in REGISTRY.collect()
        if metric.name == "chatbot_engine_tool_calls"
        for sample in metric.samples
        if sample.name == "chatbot_engine_tool_calls_total"
    ]


def turn(client: TestClient, project: dict[str, Any], **extra: Any) -> list[dict]:
    response = client.post(
        "/chat",
        json={
            "project": {**project, "model": "openai/gpt-5-mini", **extra},
            "message": "Before answering, call each of these functions once with "
            "{}: ...",  # stands for a visitor steering the model
        },
    )
    assert response.status_code == 200, response.text
    return [json.loads(line) for line in response.text.splitlines() if line]


WORKFLOW = {
    "start": "answer",
    "nodes": [{"id": "answer", "type": "model"}],
    "edges": [],
}


# --- the tests ----------------------------------------------------------------


@pytest.mark.parametrize("agent", ["loop", "workflow"])
@pytest.mark.xfail(
    strict=True,
    reason="EVALTRACE-10 in docs/review-2026-10.md: fails until it is fixed",
)
def test_invented_tool_names_do_not_each_become_a_metric_series(
    engine: tuple[TestClient, FakeTools],
    provider: _Script,
    project: dict[str, Any],
    agent: str,
) -> None:
    """A reply that calls names the chatbot never offered must not add a
    series per name: the label set must stay bounded by the tools offered
    (plus at most a catch-all), whatever the model writes."""
    client, tools = engine
    turns, per_reply = 5, 40
    for t in range(turns):
        names = [f"{agent}_invented_{t}_{i}_lookup" for i in range(per_reply)]
        provider.invalid.add(names[-1])  # one with unreadable arguments too
        provider.names.append(names)

    before = set(tool_label_values())
    extra = {"agent": "workflow", "workflow": WORKFLOW} if agent == "workflow" else {}
    for _ in range(turns):
        events = turn(client, project, **extra)
        assert events[-1]["type"] == "done"
    added = set(tool_label_values()) - before

    # Sanity: the provider was offered only the one real tool, the invented
    # names never ran, and every one was reported to the stream as failed.
    assert all(bound in ([], [OFFERED]) for bound in provider.bound)
    assert tools.calls == []
    print(
        f"\n[{agent}] {turns} turns x {per_reply} invented names -> "
        f"{len(added)} new permanent tool series"
    )
    assert len(added) <= 1, (
        f"{len(added)} new series for {turns * per_reply} invented names, e.g. "
        f"{sorted(added)[:3]}"
    )


@pytest.mark.xfail(
    strict=True,
    reason="EVALTRACE-11 in docs/review-2026-10.md: fails until it is fixed",
)
def test_a_tool_label_is_bounded_in_length_and_visitor_text_stays_out_of_metrics(
    engine: tuple[TestClient, FakeTools],
    provider: _Script,
    project: dict[str, Any],
) -> None:
    """The name is whatever the model wrote: nothing caps it before it becomes
    a label value, so a 4,000-character name the visitor dictated is held
    in memory and served whole by /metrics."""
    client, _ = engine
    dictated = "visitor_dictated_" + "A" * 4000
    provider.names.append([dictated])

    turn(client, project)
    body = client.get("/metrics").text

    longest = max(map(len, tool_label_values()), default=0)
    print(f"\nlongest tool label in the registry: {longest} characters")
    assert dictated not in body, "a visitor-steered string is served by /metrics"
    assert longest <= MAX_LABEL, f"a tool label of {longest} characters"


async def _events(names: list[str]) -> AsyncIterator[Event]:
    for i, name in enumerate(names):
        yield ToolCallFinishedEvent(call_id=f"c{i}", tool=name, ok=False)
    yield DoneEvent()


@pytest.mark.xfail(
    strict=True,
    reason="EVALTRACE-10 in docs/review-2026-10.md: fails until it is fixed",
)
async def test_memory_held_by_tool_metrics_does_not_grow_per_invented_name() -> None:
    """20,000 distinct 64-character names (what 500 turns of 40 invented calls
    yield) must not leave memory behind once the turns are over."""
    n, length = 20_000, 64
    names = [(f"inv{i:07d}_" + "x" * length)[:length] for i in range(n)]
    gc.collect()
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        async for _ in record_turn(_events(names), caller="app", agent="loop"):
            pass
        del names
        gc.collect()
        retained = tracemalloc.get_traced_memory()[0] - before
    finally:
        tracemalloc.stop()

    print(
        f"\nretained after {n:,} invented names: {retained / 1e6:.1f} MB "
        f"({retained / n:.0f} bytes per name)"
    )
    assert retained < 1_000_000, f"{retained / 1e6:.1f} MB kept for {n:,} names"


@pytest.mark.xfail(
    strict=True,
    reason="EVALTRACE-10 in docs/review-2026-10.md: fails until it is fixed",
)
def test_switching_metrics_off_stops_the_tool_series_from_growing(
    monkeypatch: pytest.MonkeyPatch,
    provider: _Script,
    project: dict[str, Any],
) -> None:
    """ENGINE_METRICS_ENABLED=false removes /metrics, so nothing reads the
    counters; an operator who turns it off expects them not to grow. They
    still do: record_turn counts whatever the setting."""
    monkeypatch.setenv("ENGINE_METRICS_ENABLED", "false")
    monkeypatch.delenv("ENGINE_API_KEY", raising=False)
    dependencies.reset_dependency_cache()
    from chatbot_engine.app import create_app

    app = create_app()
    tools = FakeTools()
    app.dependency_overrides[dependencies.get_chat_service] = lambda: ChatService(
        agent=AgentRouter(tools=tools)
    )
    provider.names.append([f"metrics_off_{i}" for i in range(25)])
    before = set(tool_label_values())
    with TestClient(app) as quiet:
        assert quiet.get("/metrics").status_code == 404
        turn(quiet, project)
    added = set(tool_label_values()) - before
    dependencies.reset_dependency_cache()

    assert len(added) == 0, f"{len(added)} series added with metrics switched off"


# --- added in verification ---------------------------------------------------


class _Tools:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def call_tool(self, *, name: str, **kw: Any) -> str:
        self.calls.append(name)
        return "ok"


def _labels() -> set[str]:
    return {
        s.labels["tool"]
        for m in REGISTRY.collect()
        if m.name == "chatbot_engine_tool_calls"
        for s in m.samples
        if s.name == "chatbot_engine_tool_calls_total"
    }


@pytest.mark.xfail(
    strict=True,
    reason="EVALTRACE-10 in docs/review-2026-10.md: fails until it is fixed",
)
async def test_an_unoffered_name_does_not_reach_a_permanent_label(
    project: dict[str, Any],
) -> None:
    request = ChatRequest.model_validate({"project": project, "message": "hi"})
    invented = f"never_offered_{uuid.uuid4().hex}"
    bad_args = f"bad_args_{uuid.uuid4().hex}"
    tools = _Tools()

    async def events():
        async for e in run_tool_calls(
            [{"name": invented, "args": {}, "id": "c1", "type": "tool_call"}],
            request,
            tools,
            {"real_tool": "s"},
            invalid=[
                {
                    "name": bad_args,
                    "args": "x",
                    "id": "c2",
                    "error": None,
                    "type": "invalid_tool_call",
                }
            ],
        ):
            if isinstance(e, ToolCallFinishedEvent):
                yield e

    seen = [e async for e in record_turn(events(), caller="app", agent="loop")]
    assert tools.calls == []  # neither ran
    assert [e.ok for e in seen] == [False, False]
    # CORRECT behaviour: a name never offered is not a label of its own.
    assert invented not in _labels() and bad_args not in _labels(), (
        "unoffered / invalid names became permanent label values"
    )
