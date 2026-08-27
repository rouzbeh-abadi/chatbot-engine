"""The limits on the routes that spend provider credits.

The bucket arithmetic is unit-tested directly; driving it through the app would
mean either sleeping or spending a full allowance to prove one boundary. What is
asserted through the app is the wiring -- which routes are metered and, just as
importantly, which are not.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from chatbot_engine.api.dependencies import reset_dependency_cache
from chatbot_engine.api.rate_limit import RateLimiter, reset_rate_limits


def _spend(limiter: RateLimiter, caller: str, times: int) -> None:
    for _ in range(times):
        limiter.check(caller)


def test_a_caller_may_spend_the_whole_allowance() -> None:
    _spend(RateLimiter(name="chat", capacity=3, window_s=60.0), "web", 3)


def test_the_next_call_is_refused_with_how_long_to_wait() -> None:
    limiter = RateLimiter(name="chat", capacity=3, window_s=60.0)
    _spend(limiter, "web", 3)

    with pytest.raises(Exception) as caught:
        limiter.check("web")

    error = caught.value
    assert getattr(error, "status_code", None) == 429
    # A caller that honours Retry-After must not be refused again on arrival.
    assert int(error.headers["Retry-After"]) >= 1


def test_a_capacity_of_zero_turns_the_limit_off() -> None:
    _spend(RateLimiter(name="chat", capacity=0, window_s=60.0), "web", 100)


def test_the_bucket_refills_over_time() -> None:
    """Continuous refill, so a caller is not locked out for a whole window."""
    limiter = RateLimiter(name="chat", capacity=60, window_s=60.0)
    _spend(limiter, "web", 60)

    # Rewind this caller's clock by two seconds: one token per second, so two
    # are back. Reaching into the bucket beats sleeping in a test.
    limiter._buckets["web"].updated -= 2.0

    _spend(limiter, "web", 2)
    with pytest.raises(Exception):
        limiter.check("web")


# --- the wiring --------------------------------------------------------------


@pytest.fixture
def metered_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """An engine whose ingest limit is two, so a test can reach it."""
    monkeypatch.setenv("ENGINE_INGEST_RATE_LIMIT_PER_MINUTE", "2")
    reset_dependency_cache()
    reset_rate_limits()

    from chatbot_engine.app import create_app

    return TestClient(create_app())


def _upload(client: TestClient, external_id: str) -> int:
    return client.put(
        "/documents",
        data={"project_id": "support", "external_id": external_id},
        files={"file": ("probe.md", b"# Probe\n\nOne bag.\n", "text/markdown")},
    ).status_code


def test_uploading_is_metered(metered_client: TestClient) -> None:
    """Indexing embeds every chunk, which is billed per token."""
    assert _upload(metered_client, "one") == 201
    assert _upload(metered_client, "two") == 201

    assert _upload(metered_client, "three") == 429


def test_listing_and_deleting_are_not_metered(metered_client: TestClient) -> None:
    """Neither calls a provider, so neither may be throttled with the upload."""
    for _ in range(10):
        assert (
            metered_client.get("/documents", params={"project_id": "support"}).status_code
            == 200
        )
        assert (
            metered_client.delete(
                "/documents/missing", params={"project_id": "support"}
            ).status_code
            == 200
        )
