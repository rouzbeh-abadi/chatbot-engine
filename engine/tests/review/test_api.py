"""Review repros for the API surface: auth, body limits, validation, errors.

Each test asserts the behaviour the engine should have; a failing test is a
confirmed finding. Run by path:

    uv run --no-sync pytest engine/tests/review/test_api.py -q -p no:cacheprovider
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from chatbot_engine.api import dependencies
from chatbot_engine.api.documents import MAX_UPLOAD_BYTES
from chatbot_engine.models.events import DoneEvent, TokenEvent
from chatbot_engine.services.chat import ChatService

KEY = "s3cret-review-key"
MB = 1024 * 1024


# --- helpers -----------------------------------------------------------------


def _keyed_app(monkeypatch: pytest.MonkeyPatch, *, env: str = "production"):
    monkeypatch.setenv("ENGINE_ENV", env)
    monkeypatch.setenv("ENGINE_API_KEY", KEY)
    monkeypatch.delenv("ENGINE_API_KEYS", raising=False)
    dependencies.reset_dependency_cache()
    from chatbot_engine.app import create_app

    return create_app()


def _multipart_chunks(
    file_bytes: int, *, fields: dict[str, str], mimetype: str = "text/plain"
) -> Iterator[bytes]:
    """A multipart body with one file part of `file_bytes`, generated lazily."""
    boundary = b"reviewboundary"
    for name, value in fields.items():
        yield (
            b"--" + boundary + b"\r\n"
            b'Content-Disposition: form-data; name="'
            + name.encode()
            + b'"\r\n\r\n'
            + value.encode()
            + b"\r\n"
        )
    yield (
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="file"; filename="big.txt"\r\n'
        b"Content-Type: " + mimetype.encode() + b"\r\n\r\n"
    )
    sent = 0
    while sent < file_bytes:
        step = min(MB, file_bytes - sent)
        yield b"a" * step
        sent += step
    yield b"\r\n--" + boundary + b"--\r\n"


async def _asgi(
    app,
    method: str,
    path: str,
    headers: list[tuple[bytes, bytes]],
    chunks: Iterator[bytes],
) -> tuple[int, int, bytes]:
    """Drive the app as uvicorn would (spec 2.3, body in chunks, no length).

    Returns the status, how many body bytes the app pulled before it
    answered, and the response body.
    """
    consumed = 0
    finished = False
    status = 0
    body = b""
    started = asyncio.Event()

    async def receive():
        nonlocal consumed, finished
        if not finished:
            try:
                chunk = next(chunks)
            except StopIteration:
                finished = True
                return {"type": "http.request", "body": b"", "more_body": False}
            consumed += len(chunk)
            return {"type": "http.request", "body": chunk, "more_body": True}
        await started.wait()
        await asyncio.sleep(3600)
        return {"type": "http.disconnect"}

    async def send(message):
        nonlocal status, body
        if message["type"] == "http.response.start":
            status = message["status"]
            started.set()
        elif message["type"] == "http.response.body":
            body += message.get("body", b"")

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": headers,
        "client": ("10.0.0.9", 40000),
        "server": ("engine", 8100),
    }
    await app(scope, receive, send)
    return status, consumed, body


# --- API-1: uploads are read in full before auth and before their cap ---------


@pytest.mark.parametrize("path,method", [("/documents", "PUT"), ("/extract", "POST")])
@pytest.mark.xfail(
    strict=True, reason="API-1 in docs/review-2026-10.md: fails until it is fixed"
)
def test_an_unauthenticated_upload_is_refused_before_its_body_is_read(
    monkeypatch: pytest.MonkeyPatch, path: str, method: str
) -> None:
    """A caller without the key should not be able to make the engine take in
    (and spool to disk) an arbitrarily large body before it is told 401."""
    app = _keyed_app(monkeypatch)
    size = 40 * MB  # over the 25 MB upload cap as well
    chunks = _multipart_chunks(size, fields={"project_id": "p", "external_id": "x"})
    headers = [(b"content-type", b"multipart/form-data; boundary=reviewboundary")]

    status, consumed, _ = asyncio.run(_asgi(app, method, path, headers, chunks))

    assert status == 401
    assert consumed <= 64 * 1024, (
        f"the engine read {consumed / MB:.1f} MB of an unauthenticated body "
        "before refusing it"
    )


@pytest.mark.xfail(
    strict=True, reason="API-1 in docs/review-2026-10.md: fails until it is fixed"
)
def test_an_authenticated_upload_over_the_cap_is_refused_as_it_grows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The docs: uploads 'keep their own limit of 25 MB'. Today the route only
    measures the file after Starlette spooled all of it and `file.read()`
    pulled all of it into memory."""
    from starlette.datastructures import UploadFile

    app = _keyed_app(monkeypatch)
    biggest_read = 0
    original = UploadFile.read

    async def recording_read(self, size: int = -1) -> bytes:
        nonlocal biggest_read
        data = await original(self, size)
        biggest_read = max(biggest_read, len(data))
        return data

    monkeypatch.setattr(UploadFile, "read", recording_read)
    size = 40 * MB
    chunks = _multipart_chunks(size, fields={"project_id": "p", "external_id": "x"})
    headers = [
        (b"content-type", b"multipart/form-data; boundary=reviewboundary"),
        (b"x-api-key", KEY.encode()),
    ]

    status, consumed, _ = asyncio.run(_asgi(app, "PUT", "/documents", headers, chunks))

    assert status == 413
    assert biggest_read <= MAX_UPLOAD_BYTES + 1, (
        f"the route held {biggest_read / MB:.1f} MB in memory before its check"
    )
    assert consumed <= MAX_UPLOAD_BYTES + MB, (
        f"the engine received {consumed / MB:.1f} MB before refusing"
    )


# --- API-2: evaluation requests are unbounded --------------------------------


class _CountingJudge:
    def __init__(self) -> None:
        self.cases = 0

    async def __call__(self, request):
        from chatbot_engine.models.evals import JudgeReport

        self.cases = len(request.cases)
        return JudgeReport()


@pytest.mark.xfail(
    strict=True, reason="API-2 in docs/review-2026-10.md: fails until it is fixed"
)
def test_one_judge_request_cannot_carry_an_unbounded_run(
    client: TestClient, project: dict[str, object]
) -> None:
    """`limit_eval` charges one token per request, so the size of a run is the
    only bound on what one request spends; the schema sets none."""
    judge = _CountingJudge()
    client.app.dependency_overrides[dependencies.get_judge] = lambda: judge
    cases = [
        {"id": f"c{i}", "category": "c", "question": "q?", "expected": "e"}
        for i in range(20_000)
    ]

    response = client.post(
        "/judge", json={"project": project, "judge_prompt": "grade", "cases": cases}
    )

    assert response.status_code == 422, (
        f"a single request with {judge.cases} cases was accepted; at the "
        "default limit of 20 runs an hour that is 400,000 answered turns"
    )


@pytest.mark.xfail(
    strict=True, reason="API-2 in docs/review-2026-10.md: fails until it is fixed"
)
def test_one_rag_eval_request_cannot_carry_an_unbounded_run(
    client: TestClient, project: dict[str, object]
) -> None:
    seen: list[int] = []

    async def evaluator(request):
        from chatbot_engine.models.evals import RagReport

        seen.append(len(request.cases))
        return RagReport.model_construct()

    client.app.dependency_overrides[dependencies.get_rag_evaluator] = lambda: evaluator
    cases = [
        {"id": f"c{i}", "category": "c", "question": "q?", "reference": "r"}
        for i in range(20_000)
    ]

    response = client.post("/eval/rag", json={"project": project, "cases": cases})

    assert response.status_code == 422, f"accepted a run of {seen} cases"


# --- API-3: a 422 echoes the secrets it was sent ----------------------------


@pytest.mark.xfail(
    strict=True, reason="API-5 in docs/review-2026-10.md: fails until it is fixed"
)
def test_a_validation_error_does_not_echo_a_servers_header_values(
    client: TestClient, project: dict[str, object]
) -> None:
    """models/chat.py: 'The messages name the header, never its value: a
    value is usually a secret, and a validation error travels back in the
    response body.' FastAPI's 422 carries `input`, the whole headers dict."""
    secret = "Bearer sk-live-SECRET-TOKEN-1234"
    server = {
        "name": "crm",
        "url": "https://crm.example.com/mcp",
        "allowed_tools": ["lookup"],
        "headers": {"Authorization": secret, "X-Bad": "line\nbreak"},
    }
    response = client.post(
        "/chat",
        json={"project": project | {"mcp_servers": [server]}, "message": "hi"},
    )

    assert response.status_code == 422
    assert secret not in response.text, response.text


@pytest.mark.xfail(
    strict=True, reason="API-5 in docs/review-2026-10.md: fails until it is fixed"
)
def test_a_validation_error_does_not_echo_the_provider_key(
    client: TestClient, project: dict[str, object]
) -> None:
    """A project missing a required field: the 422's `input` is the whole
    project, its `provider_api_key` and tracing `secret_key` included."""
    body = {
        "project": {
            "project_id": "p",
            "system_prompt": "x",
            "provider_api_key": "sk-or-OWNER-KEY-5678",
            "tracing": {"public_key": "pk", "secret_key": "sk-lf-TRACE-SECRET"},
        },
        "message": "hi",
    }
    response = client.post("/chat", json=body)

    assert response.status_code == 422
    assert "sk-or-OWNER-KEY-5678" not in response.text
    assert "sk-lf-TRACE-SECRET" not in response.text


# --- API-4: unknown form fields and query parameters are accepted -------------


@pytest.mark.xfail(
    strict=True, reason="API-6 in docs/review-2026-10.md: fails until it is fixed"
)
def test_an_unknown_upload_field_is_refused(client: TestClient) -> None:
    """docs/backend-integration.md: 'Every model uses extra="forbid". A
    misspelled field is a 422 that names it.' A misspelled form field on the
    upload is silently dropped and the document cut with the defaults."""
    response = client.put(
        "/documents",
        data={"project_id": "support", "external_id": "faq", "chunk_szie": "200"},
        files={"file": ("faq.md", b"# Returns\n\nThirty days.\n", "text/markdown")},
    )

    assert response.status_code == 422, (
        f"{response.status_code}: chunk_size={response.json().get('chunk_size')}"
    )


@pytest.mark.xfail(
    strict=True, reason="API-6 in docs/review-2026-10.md: fails until it is fixed"
)
def test_an_unknown_extract_field_is_refused(client: TestClient) -> None:
    response = client.post(
        "/extract",
        data={"provider_key": "sk-or-misspelled"},
        files={"file": ("a.txt", b"hello there", "text/plain")},
    )

    assert response.status_code == 422, response.status_code


# --- holds: a refused path cannot forge a log line (urlsplit drops CR/LF) ----


def test_a_refused_path_cannot_write_a_second_log_line(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    app = _keyed_app(monkeypatch)
    forged = "INFO chatbot_engine.turn [-] turn stop in 0.10s: 0 tokens"
    with TestClient(app) as client, caplog.at_level(logging.WARNING):
        response = client.delete(
            "/documents/x%0A" + forged.replace(" ", "%20") + "?project_id=p"
        )

    assert response.status_code == 401
    rejected = [r.getMessage() for r in caplog.records if "rejected" in r.getMessage()]
    assert rejected, "the refusal is logged"
    assert all("\n" not in line for line in rejected), rejected


# --- checks that hold (kept as regression tests) -----------------------------


class _SlowAgent:
    """Writes a token every 10 ms for up to 5 s, and says when it was closed."""

    def __init__(self) -> None:
        self.tokens = 0
        self.closed = False

    async def run(self, request):
        try:
            for _ in range(500):
                self.tokens += 1
                yield TokenEvent(text="x")
                await asyncio.sleep(0.01)
            yield DoneEvent()
        finally:
            self.closed = True


def test_a_reader_that_goes_away_stops_the_turn(
    monkeypatch: pytest.MonkeyPatch, project: dict[str, object]
) -> None:
    """The app cancels the engine stream on Stop and bills an estimate,
    trusting that the engine drops the model call: through both middlewares
    (RequestId is a BaseHTTPMiddleware) the disconnect must reach the agent."""
    app = _keyed_app(monkeypatch, env="local")
    agent = _SlowAgent()
    app.dependency_overrides[dependencies.get_chat_service] = lambda: ChatService(
        agent=agent
    )
    payload = json.dumps({"project": project, "message": "hi"}).encode()

    async def run() -> None:
        delivered = False
        gone = asyncio.Event()

        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": payload, "more_body": False}
            await gone.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                gone.set()  # the reader leaves after the first line

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/chat",
            "raw_path": b"/chat",
            "root_path": "",
            "query_string": b"",
            "headers": [
                (b"content-type", b"application/json"),
                (b"x-api-key", KEY.encode()),
            ],
            "client": ("10.0.0.9", 40000),
            "server": ("engine", 8100),
        }
        await asyncio.wait_for(app(scope, receive, send), timeout=10)

    asyncio.run(run())

    assert agent.closed
    assert agent.tokens < 50, f"the turn went on for {agent.tokens} tokens"


def test_production_refuses_unknown_json_fields_at_every_level(
    client: TestClient, project: dict[str, object]
) -> None:
    nested = [
        {"project": project, "message": "hi", "extra": 1},
        {"project": project | {"extra": 1}, "message": "hi"},
        {
            "project": project,
            "message": "hi",
            "history": [{"role": "user", "content": "a", "x": 1}],
        },
        {"project": project, "message": "hi", "resume": {"thread_id": "t", "x": 1}},
        {
            "project": project,
            "message": "hi",
            "attachments": [{"name": "a", "text": "b", "x": 1}],
        },
        {
            "project": project
            | {
                "mcp_servers": [
                    {"name": "s", "url": "https://x", "allowed_tools": ["t"], "x": 1}
                ]
            },
            "message": "hi",
        },
        {
            "project": project
            | {"tracing": {"public_key": "p", "secret_key": "s", "x": 1}},
            "message": "hi",
        },
        {
            "project": project
            | {
                "workflow": {
                    "nodes": [{"id": "a", "type": "end", "x": 1}],
                    "start": "a",
                }
            },
            "message": "hi",
        },
    ]
    for body in nested:
        response = client.post("/chat", json=body)
        assert response.status_code == 422, body
        assert "extra_forbidden" in response.text, response.text


def test_an_empty_key_in_production_does_not_open_the_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chatbot_engine.app import InsecureConfiguration, create_app

    monkeypatch.setenv("ENGINE_ENV", "production")
    monkeypatch.setenv("ENGINE_API_KEY", "")
    monkeypatch.delenv("ENGINE_API_KEYS", raising=False)
    dependencies.reset_dependency_cache()

    with pytest.raises(InsecureConfiguration), TestClient(create_app()):
        pass


def test_a_forwarded_for_header_does_not_choose_the_rate_limit_bucket(
    monkeypatch: pytest.MonkeyPatch, project: dict[str, object]
) -> None:
    monkeypatch.setenv("ENGINE_CHAT_RATE_LIMIT_PER_MINUTE", "1")
    app = _keyed_app(monkeypatch)

    class _Quick:
        async def run(self, request):
            yield DoneEvent()

    app.dependency_overrides[dependencies.get_chat_service] = lambda: ChatService(
        agent=_Quick()
    )
    with TestClient(app) as client:
        first = client.post(
            "/chat",
            json={"project": project, "message": "hi"},
            headers={"X-API-Key": KEY, "X-Forwarded-For": "1.1.1.1"},
        )
        second = client.post(
            "/chat",
            json={"project": project, "message": "hi"},
            headers={"X-API-Key": KEY, "X-Forwarded-For": "2.2.2.2"},
        )

    assert first.status_code == 200
    assert second.status_code == 429


# --- API-2b: an evaluation goes on after its caller has gone -----------------


@pytest.mark.xfail(
    strict=True, reason="API-3 in docs/review-2026-10.md: fails until it is fixed"
)
def test_an_evaluation_stops_when_its_caller_goes_away(
    monkeypatch: pytest.MonkeyPatch, project: dict[str, object]
) -> None:
    """The app gives /judge 180 s for a batch of 5 cases, each allowed the
    120 s turn deadline. When it gives up, the engine should stop answering
    (and spending the owner's key) rather than finish a run nobody reads."""
    app = _keyed_app(monkeypatch, env="local")
    answered: list[int] = []
    finished = asyncio.Event()

    async def judge(request):
        from chatbot_engine.models.evals import JudgeReport

        try:
            for index, _ in enumerate(request.cases):
                await asyncio.sleep(0.05)  # one case's model calls
                answered.append(index)
        finally:
            finished.set()
        return JudgeReport()

    app.dependency_overrides[dependencies.get_judge] = lambda: judge
    cases = [
        {"id": f"c{i}", "category": "c", "question": "q?", "expected": "e"}
        for i in range(40)
    ]
    payload = json.dumps(
        {"project": project, "judge_prompt": "grade", "cases": cases}
    ).encode()

    async def run() -> None:
        delivered = False

        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": payload, "more_body": False}
            await asyncio.sleep(0.12)  # the caller's timeout passes
            return {"type": "http.disconnect"}

        async def send(message):
            pass

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/judge",
            "raw_path": b"/judge",
            "root_path": "",
            "query_string": b"",
            "headers": [
                (b"content-type", b"application/json"),
                (b"x-api-key", KEY.encode()),
            ],
            "client": ("10.0.0.9", 40000),
            "server": ("engine", 8100),
        }
        await asyncio.wait_for(app(scope, receive, send), timeout=10)

    asyncio.run(run())

    assert len(answered) < 10, (
        f"{len(answered)} of 40 cases were answered after the caller left at "
        "about the third"
    )


# --- API-6: a lone surrogate in the history breaks every later turn ------------


def _fake_provider(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Answer OpenRouter calls in-process: no network, no key."""
    import httpx

    calls: list[dict] = []

    async def send(self, request, **kwargs):
        body = json.loads(request.content or b"{}")
        calls.append(body)
        usage = {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
        if body.get("stream"):
            chunk = {
                "id": "c",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "ok"},
                        "finish_reason": None,
                    }
                ],
            }
            last = chunk | {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            tail = chunk | {"choices": [], "usage": usage}
            sse = "".join(f"data: {json.dumps(c)}\n\n" for c in (chunk, last, tail))
            sse += "data: [DONE]\n\n"
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=sse.encode(),
                request=request,
            )
        return httpx.Response(
            200,
            json={
                "id": "c",
                "object": "chat.completion",
                "created": 0,
                "model": body.get("model", "m"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "query"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage,
            },
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    monkeypatch.setenv("ENGINE_OPENROUTER_BASE_URL", "http://provider.invalid/api/v1")
    dependencies.reset_dependency_cache()
    return calls


def _events(response) -> list[dict]:
    return [json.loads(line) for line in response.text.splitlines() if line]


def test_a_turn_is_answered_with_the_fake_provider(
    monkeypatch: pytest.MonkeyPatch, project: dict[str, object]
) -> None:
    """Control: the same turn without the cut emoji is answered."""
    calls = _fake_provider(monkeypatch)
    from chatbot_engine.app import create_app

    with TestClient(create_app()) as client:
        response = client.post(
            "/chat",
            json={
                "project": project,
                "message": "and the second one?",
                "history": [
                    {"role": "user", "content": "x" * 50 + "\U0001f600"},
                    {"role": "assistant", "content": "Sure."},
                ],
            },
        )

    kinds = [e["type"] for e in _events(response)]
    assert calls, "the fake provider was reached"
    assert "error" not in kinds, _events(response)


@pytest.mark.xfail(
    strict=True, reason="API-4 in docs/review-2026-10.md: fails until it is fixed"
)
def test_a_lone_surrogate_in_the_history_does_not_break_the_turn(
    monkeypatch: pytest.MonkeyPatch, project: dict[str, object]
) -> None:
    """ChatFrom cuts each history message at 16,000 UTF-16 units, which can
    split an emoji; JSON.stringify sends the half as `\\ud83d`. The engine
    accepts it (no 422), and the provider client then cannot encode it, so
    the turn, and every later turn carrying that message, fails."""
    _fake_provider(monkeypatch)
    from chatbot_engine.app import create_app

    cut = "x" * 50 + "\ud83d"
    body = json.dumps(
        {
            "project": project,
            "message": "and the second one?",
            "history": [
                {"role": "user", "content": cut},
                {"role": "assistant", "content": "Sure."},
            ],
        }
    )
    with TestClient(create_app(), raise_server_exceptions=False) as client:
        response = client.post(
            "/chat", content=body, headers={"content-type": "application/json"}
        )

    assert response.status_code in (200, 422), (
        f"{response.status_code}: {response.text[:200]}"
    )
    if response.status_code == 422:
        return  # refused at the boundary with a reason: acceptable
    events = _events(response)
    errors = [e for e in events if e["type"] == "error"]
    assert not errors, errors
