"""A stream that breaks before its first token is retried; after it, it is not."""

from __future__ import annotations

import httpx
import openai
import pytest
from langchain_core.messages import AIMessageChunk

from chatbot_engine.agent.client import is_transient, stream_round


def _rate_limited() -> openai.RateLimitError:
    response = httpx.Response(429, request=httpx.Request("POST", "https://x"))
    return openai.RateLimitError("slow down", response=response, body=None)


def _bad_request() -> openai.BadRequestError:
    response = httpx.Response(400, request=httpx.Request("POST", "https://x"))
    return openai.BadRequestError("nope", response=response, body=None)


class FlakyChain:
    """Fails the first `failures` calls, in the way `fail_after` chunks say."""

    def __init__(self, failures: int, *, exc: Exception, fail_after: int = 0) -> None:
        self.failures = failures
        self.exc = exc
        self.fail_after = fail_after
        self.calls = 0

    async def astream(self, _inputs):
        self.calls += 1
        if self.calls <= self.failures:
            for i in range(self.fail_after):
                yield AIMessageChunk(content=f"partial{i} ")
            raise self.exc
        for word in ("hello", " world"):
            yield AIMessageChunk(content=word)


async def _collect(chain, retries: int) -> str:
    return "".join(
        [c.text async for c in stream_round(chain, [], retries=retries, backoff_s=0.0)]
    )


async def test_a_stream_that_breaks_before_its_first_token_is_retried():
    chain = FlakyChain(failures=2, exc=_rate_limited())
    assert await _collect(chain, retries=3) == "hello world"
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


def test_which_failures_count_as_transient():
    assert is_transient(_rate_limited())
    assert not is_transient(_bad_request())
    assert not is_transient(ValueError("not a provider error"))
