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
    limiter.store._buckets["web"].updated -= 2.0

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


# --- shared buckets ------------------------------------------------------------
#
# The property a Redis store exists for: two replicas charging the same caller
# see one allowance, not one each. `fakeredis` runs the real Lua script, so the
# arithmetic under test is the one production runs.


def _shared_redis():
    import fakeredis

    return fakeredis.FakeRedis()


def _replica(redis, capacity: int = 3) -> RateLimiter:
    from chatbot_engine.api.rate_limit import RedisBuckets

    store = RedisBuckets("redis://unused", name="chat", client=redis)
    return RateLimiter(name="chat", capacity=capacity, window_s=60.0, store=store)


def test_two_replicas_share_one_allowance() -> None:
    """The bug an in-memory store has: each replica grants the full capacity."""
    redis = _shared_redis()
    first, second = _replica(redis), _replica(redis)

    first.check("web")
    second.check("web")
    first.check("web")

    with pytest.raises(Exception) as caught:
        second.check("web")
    assert getattr(caught.value, "status_code", None) == 429
    assert int(caught.value.headers["Retry-After"]) >= 1


def test_callers_are_still_separate_in_redis() -> None:
    redis = _shared_redis()
    limiter = _replica(redis, capacity=1)

    limiter.check("web")
    limiter.check("batch")

    with pytest.raises(Exception):
        limiter.check("web")


def test_the_redis_store_is_selected_by_the_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one switch between per-replica and shared limits."""
    from chatbot_engine.api.rate_limit import MemoryBuckets, RedisBuckets, _store_for
    from chatbot_engine.settings import Settings

    monkeypatch.setattr("redis.Redis.from_url", lambda url: _shared_redis())

    assert isinstance(_store_for("chat", Settings(redis_url=None)), MemoryBuckets)
    assert isinstance(
        _store_for("chat", Settings(redis_url="redis://cache:6379/0")), RedisBuckets
    )
