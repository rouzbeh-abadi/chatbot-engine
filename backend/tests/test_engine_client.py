"""The HTTP client itself, against a mock transport.

This is where the new failure modes live: the engine is remote now, so an
unreachable host, a 501, and a malformed stream all have to behave predictably.
"""

from __future__ import annotations

import httpx
import pytest

from support_agent.engine_client import (
    EngineClient,
    EngineFailed,
    EngineNotImplemented,
    EngineRejected,
    EngineUnavailable,
)
from support_agent.engine_client.models import (
    AssistantConfig,
    EngineChatRequest,
    TokenEvent,
)

NDJSON = "application/x-ndjson"


def _request() -> EngineChatRequest:
    return EngineChatRequest(
        project=AssistantConfig(project_id="support", name="S", system_prompt="s"),
        message="hi",
    )


def _client(handler) -> EngineClient:
    """A real EngineClient whose socket is replaced by a mock transport."""
    return EngineClient(
        base_url="http://engine.test",
        api_key="k",
        transport=httpx.MockTransport(handler),
    )


async def test_chat_parses_the_ndjson_stream_into_events() -> None:
    body = (
        '{"type":"retrieval","sources":[]}\n'
        '{"type":"token","text":"One "}\n'
        '{"type":"token","text":"bag."}\n'
        "\n"  # blank lines are skipped
        '{"type":"usage","total_tokens":7}\n'
        '{"type":"done","finish_reason":"stop"}\n'
    )
    engine = _client(lambda r: httpx.Response(200, text=body, headers={"content-type": NDJSON}))

    events = [e async for e in await engine.start_chat(_request())]

    assert [e.type for e in events] == ["retrieval", "token", "token", "usage", "done"]
    assert "".join(e.text for e in events if isinstance(e, TokenEvent)) == "One bag."


async def test_chat_sends_the_api_key_and_the_payload() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["key"] = request.headers.get("x-api-key")
        seen["url"] = str(request.url)
        seen["body"] = request.read().decode()
        return httpx.Response(200, text='{"type":"done"}\n')

    engine = _client(handler)
    [e async for e in await engine.start_chat(_request())]

    assert seen["key"] == "k"
    assert seen["url"] == "http://engine.test/chat"
    assert '"message":"hi"' in str(seen["body"])


async def test_a_501_raises_before_the_stream_is_consumed() -> None:
    """The route must be able to return 501 instead of an empty 200, so the
    failure has to happen while awaiting `start_chat`, not on first iteration."""
    engine = _client(
        lambda r: httpx.Response(501, json={"detail": "no Agent -- get_agent()"})
    )

    with pytest.raises(EngineNotImplemented, match="get_agent"):
        await engine.start_chat(_request())


async def test_a_connection_failure_is_reported_as_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    engine = _client(handler)

    with pytest.raises(EngineUnavailable, match="unreachable"):
        await engine.start_chat(_request())


async def test_a_4xx_is_reported_as_rejected() -> None:
    engine = _client(lambda r: httpx.Response(422, json={"detail": "bad field"}))

    with pytest.raises(EngineRejected, match="bad field"):
        await engine.start_chat(_request())


async def test_a_5xx_is_reported_as_failed() -> None:
    engine = _client(lambda r: httpx.Response(500, json={"detail": "boom"}))

    with pytest.raises(EngineFailed, match="boom"):
        await engine.start_chat(_request())


async def test_an_unreadable_event_line_fails_loudly() -> None:
    """Silently dropping a bad line would hide a contract mismatch."""
    engine = _client(lambda r: httpx.Response(200, text='{"type":"nonsense"}\n'))

    with pytest.raises(EngineFailed, match="unreadable event"):
        [e async for e in await engine.start_chat(_request())]


async def test_ingest_document_posts_multipart_and_returns_a_record() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PUT"
        assert str(request.url) == "http://engine.test/documents"
        assert b"# Baggage" in request.read()
        return httpx.Response(
            201,
            json={
                "doc_id": "d1",
                "external_id": "baggage.md",
                "project_id": "support",
                "filename": "baggage.md",
                "mimetype": "text/markdown",
                "size_bytes": 11,
                "content_hash": "abc",
                "status": "indexed",
                "chunk_count": 4,
            },
        )

    engine = _client(handler)
    record = await engine.ingest_document(
        project_id="support",
        external_id="baggage.md",
        filename="baggage.md",
        mimetype="text/markdown",
        data=b"# Baggage\n\n",
    )

    assert record.status == "indexed"
    assert record.chunk_count == 4


async def test_list_and_delete_pass_the_project_id() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["project_id"] == "support"
        if request.method == "DELETE":
            return httpx.Response(200, json={"doc_id": "d1", "deleted": True})
        return httpx.Response(200, json=[])

    engine = _client(handler)

    assert await engine.list_documents(project_id="support") == []
    assert (await engine.delete_document(project_id="support", doc_id="d1")).deleted


async def test_abandoning_a_stream_early_closes_the_connection() -> None:
    """A browser that navigates away mid-answer must not leak a connection.

    Starlette closes the response generator on disconnect, which unwinds through
    `_events`' `finally`. If that ever stops holding, the leak is invisible until
    the process runs out of sockets.
    """
    body = "".join(f'{{"type":"token","text":"{i} "}}\n' for i in range(50))
    engine = _client(lambda r: httpx.Response(200, text=body))

    # Reach the response object the generator owns, so closure is observable.
    seen: list[httpx.Response] = []
    original = httpx.AsyncClient.send

    async def spy(self, request, **kwargs):  # noqa: ANN001, ANN202
        response = await original(self, request, **kwargs)
        seen.append(response)
        return response

    httpx.AsyncClient.send = spy  # type: ignore[method-assign]
    try:
        stream = await engine.start_chat(_request())
        async for _ in stream:
            break  # walk away after the first event
        await stream.aclose()
    finally:
        httpx.AsyncClient.send = original  # type: ignore[method-assign]

    assert seen, "the request should have gone through the mock transport"
    assert seen[0].is_closed, "abandoning the stream must close the response"
