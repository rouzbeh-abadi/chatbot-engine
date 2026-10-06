"""A streamed call is retried in one place, while nothing has reached the caller.

`stream_reply` yields the text as it arrives, then the whole reply. It retries
a call that fails before its first token, on a failure a retry can fix, and
sums each attempt's reply afresh, so a retried stream never repeats a tool
call. After a token it passes the failure on.
"""

from __future__ import annotations

import httpx
import openai
import pytest
from langchain_core.messages import AIMessageChunk

from chatbot_engine.agent.client import is_transient, stream_reply

_REQUEST = httpx.Request("POST", "https://x")


def _rate_limited() -> openai.RateLimitError:
    response = httpx.Response(429, request=_REQUEST)
    return openai.RateLimitError("slow down", response=response, body=None)


def _bad_request() -> openai.BadRequestError:
    response = httpx.Response(400, request=_REQUEST)
    return openai.BadRequestError("nope", response=response, body=None)


def _error_in_the_stream() -> openai.APIError:
    """What the client library raises for an error the provider sends inside a
    stream it had already started: no status, only a message."""
    return openai.APIError("Provider returned error", request=_REQUEST, body=None)


class FlakyChain:
    """Fails the first `failures` calls, after streaming what `before` says."""

    def __init__(
        self,
        failures: int,
        *,
        exc: Exception,
        fail_after: int = 0,
        before: list[AIMessageChunk] | None = None,
        then: list[AIMessageChunk] | None = None,
    ) -> None:
        self.failures = failures
        self.exc = exc
        self.before = (
            before
            if before is not None
            else [AIMessageChunk(content=f"partial{i} ") for i in range(fail_after)]
        )
        self.then = (
            then
            if then is not None
            else [AIMessageChunk(content="hello"), AIMessageChunk(content=" world")]
        )
        self.calls = 0

    async def astream(self, _inputs, config=None):
        self.calls += 1
        if self.calls <= self.failures:
            for chunk in self.before:
                yield chunk
            raise self.exc
        for chunk in self.then:
            yield chunk


async def _collect(chain, retries: int | None) -> tuple[str, AIMessageChunk | None]:
    text, reply = [], None
    async for item in stream_reply(
        lambda: chain.astream([]), retries=retries, backoff_s=0.0
    ):
        if isinstance(item, str):
            text.append(item)
        else:
            assert reply is None, "the whole reply comes once"
            reply = item
    return "".join(text), reply


async def test_a_stream_that_breaks_before_its_first_token_is_retried():
    chain = FlakyChain(failures=2, exc=_rate_limited())
    text, reply = await _collect(chain, retries=3)
    assert text == "hello world"
    assert reply is not None and reply.text == "hello world"
    assert chain.calls == 3


async def test_a_stream_that_breaks_after_a_token_is_not_retried():
    chain = FlakyChain(failures=1, exc=_rate_limited(), fail_after=1)
    with pytest.raises(openai.RateLimitError):
        await _collect(chain, retries=3)
    assert chain.calls == 1


async def test_retries_are_bounded():
    chain = FlakyChain(failures=5, exc=_rate_limited())
    with pytest.raises(openai.RateLimitError):
        await _collect(chain, retries=2)
    assert chain.calls == 3


async def test_a_permanent_failure_is_not_retried():
    chain = FlakyChain(failures=1, exc=_bad_request())
    with pytest.raises(openai.BadRequestError):
        await _collect(chain, retries=3)
    assert chain.calls == 1


async def test_a_retried_stream_does_not_repeat_a_tool_call():
    """A stream that sent a tool call's chunks and then broke is retried, as
    no text had reached the caller; the reply is summed afresh, so the call
    is asked for once, not twice with its arguments run together."""
    call = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "name": "get_booking_status",
                "args": '{"ref": "AB"}',
                "id": "c1",
                "index": 0,
            }
        ],
    )
    chain = FlakyChain(failures=1, exc=_rate_limited(), before=[call], then=[call])

    text, reply = await _collect(chain, retries=3)

    assert text == ""
    assert chain.calls == 2
    assert reply is not None
    assert reply.tool_calls == [
        {
            "name": "get_booking_status",
            "args": {"ref": "AB"},
            "id": "c1",
            "type": "tool_call",
        }
    ]


@pytest.mark.parametrize(
    "exc",
    [
        _error_in_the_stream(),
        httpx.RemoteProtocolError("peer closed connection without sending body"),
        httpx.ReadError("connection reset"),
    ],
    ids=["an error inside the stream", "a dropped stream", "a reset"],
)
async def test_a_stream_error_before_the_first_token_is_retried(exc):
    """The client library raises these while the stream is read: the
    provider's own error event, and httpx's errors, unwrapped."""
    chain = FlakyChain(failures=1, exc=exc)
    text, _ = await _collect(chain, retries=1)
    assert text == "hello world"
    assert chain.calls == 2


async def test_the_retries_default_to_the_setting(monkeypatch):
    from chatbot_engine.settings import get_settings

    monkeypatch.setenv("ENGINE_PROVIDER_MAX_RETRIES", "1")
    get_settings.cache_clear()
    try:
        chain = FlakyChain(failures=5, exc=_rate_limited())
        with pytest.raises(openai.RateLimitError):
            await _collect(chain, retries=None)
        assert chain.calls == 2
    finally:
        get_settings.cache_clear()


async def test_a_model_that_says_nothing_yields_no_reply():
    chain = FlakyChain(failures=0, exc=_rate_limited(), then=[])
    assert await _collect(chain, retries=0) == ("", None)


def test_which_failures_count_as_transient():
    assert is_transient(_rate_limited())
    assert is_transient(_error_in_the_stream())
    assert is_transient(httpx.ReadTimeout("slow"))
    assert is_transient(openai.APIConnectionError(request=_REQUEST))
    assert not is_transient(_bad_request())
    assert not is_transient(
        openai.APIResponseValidationError(
            response=httpx.Response(200, request=_REQUEST), body=None
        )
    )
    assert not is_transient(ValueError("not a provider error"))
