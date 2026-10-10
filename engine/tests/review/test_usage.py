"""Review: the usage the engine reports, on the paths where a call is cut,
retried or reports nothing (TURN-8, INGEST-18, INGEST-19).

ChatFrom bills owners on this usage and takes a non-null `cost_usd` as exact
(`turnCost` in ChatFrom's `libs/pricing.ts`), so a partial figure presented as
exact undercharges silently. The real `BilledChatOpenAI` and `OpenAIEmbeddings`
talk to a fake OpenAI-compatible server on 127.0.0.1 that streams SSE the way
OpenRouter does, and can end a stream without its usage chunk, cut a
connection mid-stream, answer late, or refuse a later batch. No real provider.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_openai import OpenAIEmbeddings

from chatbot_engine.api.dependencies import reset_dependency_cache
from chatbot_engine.rag import embeddings as embeddings_module
from chatbot_engine.rag import vector_store as vector_store_module

Step = Callable[["_Handler", dict[str, Any]], None]


# --- a fake OpenAI-compatible provider ----------------------------------------


class _Provider:
    """The script of replies, one per request in order, and what was asked."""

    def __init__(self) -> None:
        self.steps: list[Step] = []
        self.seen: list[tuple[str, dict[str, Any]]] = []
        #: What this provider charged for each call, USD: what OpenRouter's
        #: bill would say, whether or not the engine heard about it.
        self.charged: list[float] = []
        self._lock = threading.Lock()

    def next(self, path: str, body: dict[str, Any]) -> Step | None:
        with self._lock:
            self.seen.append((path, body))
            return self.steps.pop(0) if self.steps else None

    def chat_calls(self) -> list[dict[str, Any]]:
        return [b for p, b in self.seen if p.endswith("/chat/completions")]


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _Server

    def log_message(self, *args: object) -> None:  # quiet
        pass

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        step = self.server.provider.next(self.path, body)
        if step is None:
            self.send_error(500, "no scripted reply")
            return
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            step(self, body)

    # helpers for steps
    def start_stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def chunk(self, raw: bytes) -> None:
        self.wfile.write(f"{len(raw):x}\r\n".encode() + raw + b"\r\n")
        self.wfile.flush()

    def event(self, payload: dict[str, Any] | str) -> None:
        data = payload if isinstance(payload, str) else json.dumps(payload)
        self.chunk(f"data: {data}\n\n".encode())

    def end_stream(self) -> None:
        self.event("[DONE]")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def cut(self) -> None:
        """The connection drops mid-body: no [DONE], no last chunk."""
        self.wfile.flush()
        self.close_connection = True
        self.connection.shutdown(socket.SHUT_RDWR)

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        self.wfile.flush()


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    provider: _Provider


def _usage(prompt: int, completion: int, cost: float | None) -> dict[str, Any]:
    usage: dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }
    if cost is not None:
        usage["cost"] = cost
    return usage


def _chunk(model: str, delta: dict[str, Any] | None, **extra: Any) -> dict[str, Any]:
    choices = (
        [{"index": 0, "delta": delta, "finish_reason": extra.pop("finish", None)}]
        if delta is not None
        else []
    )
    return {
        "id": "gen-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": choices,
        **extra,
    }


def reply_json(provider: _Provider, text: str, usage: dict[str, Any]) -> Step:
    """A whole (non-streamed) reply, billed as `usage` says."""
    provider.charged.append(usage.get("cost", 0.0))

    def step(h: _Handler, body: dict[str, Any]) -> None:
        h.send_json(
            200,
            {
                "id": "gen-1",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage,
            },
        )

    return step


def stream(
    provider: _Provider,
    deltas: list[dict[str, Any]],
    *,
    usage: dict[str, Any] | None,
    charged: float,
    cut: bool = False,
) -> Step:
    """A streamed reply: `deltas`, then the usage chunk when given, then
    [DONE], or a dropped connection when `cut`. `charged` is what the provider
    bills for it, sent or not."""
    provider.charged.append(charged)

    def step(h: _Handler, body: dict[str, Any]) -> None:
        model = body["model"]
        h.start_stream()
        for i, delta in enumerate(deltas):
            last = i == len(deltas) - 1
            h.event(_chunk(model, delta, finish="stop" if last and not cut else None))
        if usage is not None:
            h.event(_chunk(model, None, usage=usage))
        if cut:
            h.cut()
            return
        h.end_stream()

    return step


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Provider]:
    """A fake OpenRouter on 127.0.0.1, wired in as the engine's base URL.

    ChatFrom's settings: no ENGINE_PRICING (deploy/docker-compose.yml says
    so on purpose), so a cost is either OpenRouter's bill or null.
    """
    server = _Server(("127.0.0.1", 0), _Handler)
    server.provider = _Provider()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv(
        "ENGINE_OPENROUTER_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1"
    )
    monkeypatch.delenv("ENGINE_PRICING", raising=False)
    monkeypatch.delenv("ENGINE_UTILITY_MODEL", raising=False)
    monkeypatch.delenv("ENGINE_API_KEY", raising=False)
    reset_dependency_cache()
    try:
        yield server.provider
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def api(provider: _Provider) -> Iterator[TestClient]:
    from chatbot_engine.app import create_app

    with TestClient(create_app()) as test_client:
        yield test_client
    reset_dependency_cache()


def _turn(api: TestClient, **request: Any) -> list[dict[str, Any]]:
    body = {
        "project": {
            "project_id": "billing",
            "name": "Billing",
            "system_prompt": "You answer questions.",
            "model": "openai/gpt-5-mini",
        },
        "message": "and for orders from the EU?",
        **request,
    }
    with api.stream("POST", "/chat", json=body) as response:
        assert response.status_code == 200, response.read()
        return [json.loads(line) for line in response.iter_lines() if line]


def _one(events: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    found = [e for e in events if e["type"] == kind]
    assert len(found) == 1, events
    return found[0]


HISTORY = [
    {"role": "user", "content": "How long do refunds take?"},
    {"role": "assistant", "content": "Refunds take 5 working days."},
]


# --- (a) a call that reported no usage makes the others' bill "exact" ----------


@pytest.mark.xfail(
    strict=True, reason="TURN-8 in docs/review-2026-10.md: fails until it is fixed"
)
def test_a_reply_without_its_usage_chunk_does_not_leave_a_partial_bill_presented_as_exact(
    api: TestClient, provider: _Provider
) -> None:
    """price_usage: "The cost is what the provider billed when every model call
    in the turn reported it ... Otherwise it is priced from the table ... null
    when the table does not list the model." add_usage returns before it
    counts `calls` for a reply with no usage_metadata, so a turn of two calls
    where the answer's stream ended without its usage chunk reports the
    rewrite's bill alone, as the turn's exact cost.
    """
    provider.steps = [
        # The query rewrite (history present): a whole reply, billed.
        reply_json(provider, "EU refund time", _usage(120, 6, 0.0000400)),
        # The answer: streamed, ends with [DONE], but no usage chunk came.
        stream(
            provider,
            [
                {"role": "assistant", "content": "EU refunds take "},
                {"content": "14 days."},
            ],
            usage=None,
            charged=0.0031,
        ),
    ]

    events = _turn(api, history=HISTORY)

    calls = provider.chat_calls()
    assert len(calls) == 2
    assert calls[1]["stream"] is True
    assert calls[1].get("stream_options") == {"include_usage": True}
    assert _one(events, "done")["finish_reason"] == "stop"
    usage = _one(events, "usage")
    # The provider charged both calls; the engine heard about only the first.
    # With no price table (ChatFrom's setting) the honest report is null, so
    # the app prices the turn itself rather than taking a figure as billed.
    assert usage["cost_usd"] is None or usage["cost_usd"] >= sum(provider.charged), (
        f"cost_usd={usage['cost_usd']} reported as OpenRouter's exact bill, "
        f"provider charged {sum(provider.charged)}; usage={usage}"
    )


@pytest.mark.xfail(
    strict=True, reason="TURN-8 in docs/review-2026-10.md: fails until it is fixed"
)
def test_a_one_call_turn_without_its_usage_chunk_is_not_reported_as_zero_tokens(
    api: TestClient, provider: _Provider
) -> None:
    """The commonest turn shape (a first message: no rewrite, no rerank) whose
    stream ended without its usage chunk: the visitor got the whole answer,
    and the usage event says 0 tokens and no cost, with nothing to tell that
    from a turn that cost nothing. ChatFrom prices that as 0 (priceSplitTurn
    of 0 tokens is 0, not null, so finalise's estimate does not run) and
    answerCredits(0, 0) is 0 credits.
    """
    provider.steps = [
        stream(
            provider,
            [{"role": "assistant", "content": "Refunds take "}, {"content": "5 days."}],
            usage=None,
            charged=0.0031,
        ),
    ]

    events = _turn(api, message="How long do refunds take?")

    assert len(provider.chat_calls()) == 1
    assert "".join(e["text"] for e in events if e["type"] == "token") == (
        "Refunds take 5 days."
    )
    usage = _one(events, "usage")
    assert usage["total_tokens"] > 0 or usage["cost_usd"], (
        f"an answered turn reported as free: {usage}"
    )


# --- (b) a retried attempt's spend is dropped ----------------------------------


@pytest.mark.xfail(
    strict=True, reason="TURN-8 in docs/review-2026-10.md: fails until it is fixed"
)
def test_a_retried_attempt_that_had_sent_its_bill_is_counted(
    api: TestClient, provider: _Provider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """stream_reply retries a stream that breaks before its first text: a
    reasoning model that was thinking, or a round that was asking for a tool.
    The first attempt was generated and billed; here it had even delivered
    OpenRouter's usage chunk, with its `cost`, before the connection dropped.
    The engine throws that reply away and reports the second attempt's bill
    as the turn's exact cost.
    """
    monkeypatch.setenv("ENGINE_PROVIDER_MAX_RETRIES", "1")
    reset_dependency_cache()
    provider.steps = [
        stream(
            provider,
            [
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning": "Thinking about refunds",
                }
            ],
            usage=_usage(900, 700, 0.0016),
            charged=0.0016,
            cut=True,
        ),
        stream(
            provider,
            [{"role": "assistant", "content": "Refunds take 5 days."}],
            usage=_usage(900, 750, 0.0017),
            charged=0.0017,
        ),
    ]

    events = _turn(api, message="How long do refunds take?")

    assert len(provider.chat_calls()) == 2  # retried
    assert "".join(e["text"] for e in events if e["type"] == "token") == (
        "Refunds take 5 days."
    )
    usage = _one(events, "usage")
    assert usage["cost_usd"] is None or usage["cost_usd"] >= sum(provider.charged), (
        f"cost_usd={usage['cost_usd']} reported as exact; the provider billed "
        f"{provider.charged} and sent both bills; usage={usage}"
    )


@pytest.mark.xfail(
    strict=True, reason="TURN-8 in docs/review-2026-10.md: fails until it is fixed"
)
def test_a_retried_attempt_that_broke_while_reasoning_is_not_hidden_behind_an_exact_bill(
    api: TestClient, provider: _Provider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The common shape of (b): a reasoning model (gpt-5-mini, the engine's
    default) streams reasoning but no text, and the connection breaks before
    the usage chunk. The provider generated and billed those tokens; the
    engine retries and reports only the second attempt, as exact.
    """
    monkeypatch.setenv("ENGINE_PROVIDER_MAX_RETRIES", "1")
    reset_dependency_cache()
    provider.steps = [
        stream(
            provider,
            [{"role": "assistant", "content": "", "reasoning": "step "}] * 5,
            usage=None,
            charged=0.0012,  # 900 prompt + ~500 reasoning tokens at list price
            cut=True,
        ),
        stream(
            provider,
            [{"role": "assistant", "content": "Refunds take 5 days."}],
            usage=_usage(900, 750, 0.0017),
            charged=0.0017,
        ),
    ]

    events = _turn(api, message="How long do refunds take?")

    assert len(provider.chat_calls()) == 2  # retried
    usage = _one(events, "usage")
    assert usage["cost_usd"] is None or usage["cost_usd"] >= sum(provider.charged), (
        f"cost_usd={usage['cost_usd']} reported as exact although an attempt "
        f"the provider billed was dropped unreported; usage={usage}"
    )


# --- (c) a vision call that times out is billed and reported nowhere -----------


@pytest.mark.xfail(
    strict=True, reason="INGEST-18 in docs/review-2026-10.md: fails until it is fixed"
)
def test_an_image_read_that_timed_out_after_the_provider_started_says_what_it_spent(
    api: TestClient, provider: _Provider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """agent/vision.py: no retry, "a call that is billed must also be
    answered"; api/extract.py: "the caller hears what it cost even for an
    image it said nothing about". A read that passes READ_TIMEOUT_S (45 s,
    here the provider timeout stands in, at 0.5 s) is a 502 with no usage,
    though the provider went on to finish and bill it. ChatFrom records
    nothing for a 502 (chat-files-store.ts keepChatFile), and the image can be
    sent again.
    """
    monkeypatch.setenv("ENGINE_PROVIDER_TIMEOUT_S", "0.5")
    reset_dependency_cache()
    provider.charged.append(0.0064)  # 4,000 output tokens of gpt-4.1-mini

    def slow(h: _Handler, body: dict[str, Any]) -> None:
        time.sleep(1.5)
        h.send_json(
            200,
            {
                "id": "gen-1",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Text: ..."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": _usage(1200, 4000, 0.0064),
            },
        )

    provider.steps = [slow]

    response = api.post(
        "/extract",
        files={"file": ("receipt.png", b"\x89PNG\r\n\x1a\n" + b"0" * 64, "image/png")},
    )

    assert len(provider.chat_calls()) == 1
    body = response.json()
    assert "usage" in body and body["usage"] is not None, (
        f"status {response.status_code}, body {body}: a vision call the provider "
        "went on to bill is reported with no usage at all"
    )


@pytest.mark.xfail(
    strict=True, reason="TURN-8 in docs/review-2026-10.md: fails until it is fixed"
)
def test_an_image_read_whose_reply_carried_no_usage_is_not_reported_as_free(
    api: TestClient, provider: _Provider
) -> None:
    """The app's "worth a look" question: is an image read with no usage free?
    describe_image prices `usage_of(reply)`, and a reply with no usage gives
    zeros and cost null, which keepChatFile records as 0 credits
    (answerCredits(null, 0) is 0). The read happened and was billed."""
    provider.steps = [
        lambda h, body: h.send_json(
            200,
            {
                "id": "gen-1",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Text: TOTAL 12.50",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )
    ]

    response = api.post(
        "/extract",
        files={"file": ("receipt.png", b"\x89PNG\r\n\x1a\n" + b"0" * 64, "image/png")},
    )

    assert response.status_code == 200, response.text
    usage = response.json()["usage"]
    assert usage["input_tokens"] + usage["output_tokens"] > 0 or usage["cost_usd"], (
        f"a vision read that ran is reported as free: {usage}"
    )


# --- (d) a failed index after a billed batch bills nothing ---------------------


@pytest.mark.xfail(
    strict=True, reason="INGEST-19 in docs/review-2026-10.md: fails until it is fixed"
)
def test_an_index_that_fails_after_an_embedded_batch_says_what_was_embedded(
    api: TestClient, provider: _Provider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The app bills embedding from the record: `embeddedTokens` counts
    chunk_count only when status is "indexed" (chatFrom libs/ledger.ts). The
    embedder sends 1,000 chunks a request; when a later request fails, the
    earlier ones were billed, the record is `failed` with chunk_count 0 and
    the response is a 502, so nothing says what was spent. Uploading the same
    bytes again embeds the first batch again, and is again unreported.
    """

    # The real OpenAI embedder, over the fake provider (no tiktoken download:
    # batching by count is the same, 1,000 texts a request).
    def real_embedder(*_: Any, **__: Any) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(
            model="openai/text-embedding-3-small",
            api_key="sk-or-fake-for-tests",  # type: ignore[arg-type]
            base_url=os.environ["ENGINE_OPENROUTER_BASE_URL"],
            max_retries=0,
            check_embedding_ctx_length=False,
        )

    monkeypatch.setattr(embeddings_module, "get_embeddings", real_embedder)
    monkeypatch.setattr(vector_store_module, "get_embeddings", real_embedder)
    vector_store_module.reset_vector_store()

    def embedded(h: _Handler, body: dict[str, Any]) -> None:
        inputs = body["input"]
        tokens = 25 * len(inputs)
        h.send_json(
            200,
            {
                "object": "list",
                "model": body["model"],
                "data": [
                    {
                        "object": "embedding",
                        "index": i,
                        "embedding": [0.1 * (i % 7)] * 8,
                    }
                    for i in range(len(inputs))
                ],
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
            },
        )

    def refused(h: _Handler, body: dict[str, Any]) -> None:
        h.send_json(
            400,
            {"error": {"message": "input rejected", "type": "invalid_request_error"}},
        )

    text = "\n\n".join(
        f"Paragraph {i}: the warranty covers parts and labour for item {i}."
        for i in range(1500)
    ).encode()

    def upload() -> Any:
        return api.put(
            "/documents",
            data={
                "project_id": "billing",
                "external_id": "manual.txt",
                "chunk_size": "100",
                "chunk_overlap": "0",
            },
            files={"file": ("manual.txt", text, "text/plain")},
        )

    provider.steps = [embedded, refused]
    first = upload()
    batches = [b for p, b in provider.seen if p.endswith("/embeddings")]
    assert len(batches) == 2 and len(batches[0]["input"]) == 1000
    assert first.status_code == 502

    provider.steps = [embedded, refused]
    again = upload()
    batches = [b for p, b in provider.seen if p.endswith("/embeddings")]
    assert again.status_code == 502
    # Re-uploading the same bytes after a failure embeds again: by design, a
    # failed record is not `unchanged`. Two batches of 1,000 paid, so far.
    assert len(batches) == 4

    record = next(
        d
        for d in api.get("/documents", params={"project_id": "billing"}).json()
        if d["external_id"] == "manual.txt"
    )
    assert record["status"] == "failed"
    assert record["chunk_count"] > 0, (
        f"record {record}: two uploads each paid to embed 1,000 chunks, and the "
        "only figure the app bills from says 0"
    )


# --- added in verification ---------------------------------------------------


@pytest.mark.xfail(
    strict=True, reason="TURN-8 in docs/review-2026-10.md: fails until it is fixed"
)
def test_an_answer_stream_broken_after_its_first_text_does_not_report_the_rewrite_as_exact(
    api: TestClient, provider: _Provider
) -> None:
    """Same root cause as TURN-8, commoner trigger: a follow-up turn
    whose answer stream breaks after its first text (not retried). The
    meter's usage event carries only the rewrite, billed, so cost_usd is
    the rewrite alone, presented as exact; the app takes it as 'billed' and
    finalise's estimate does not run."""
    provider.steps = [
        reply_json(provider, "EU refund time", _usage(120, 6, 0.00004)),
        stream(
            provider,
            [{"role": "assistant", "content": "EU refunds take "}],
            usage=None,
            charged=0.0025,
            cut=True,
        ),
    ]
    events = _turn(api, history=HISTORY)
    assert len(provider.chat_calls()) == 2
    assert any(e["type"] == "error" for e in events), events
    usage = _one(events, "usage")
    assert usage["cost_usd"] is None or usage["cost_usd"] >= sum(provider.charged), (
        f"broken answer dropped, rewrite reported as the exact bill: {usage}"
    )


def test_an_attempt_refused_before_streaming_costs_nothing(
    api: TestClient, provider: _Provider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: an attempt the provider refused outright (503, nothing
    generated) is retried and the retry's bill is the whole bill. This
    should pass: only attempts that broke mid-stream carry hidden spend."""
    monkeypatch.setenv("ENGINE_PROVIDER_MAX_RETRIES", "1")
    reset_dependency_cache()

    def refused(h: _Handler, body: dict[str, Any]) -> None:
        h.send_json(503, {"error": {"message": "overloaded"}})

    provider.steps = [
        refused,
        stream(
            provider,
            [{"role": "assistant", "content": "Refunds take 5 days."}],
            usage=_usage(900, 750, 0.0017),
            charged=0.0017,
        ),
    ]
    events = _turn(api, message="How long do refunds take?")
    assert len(provider.chat_calls()) == 2
    assert _one(events, "usage")["cost_usd"] == pytest.approx(0.0017)
