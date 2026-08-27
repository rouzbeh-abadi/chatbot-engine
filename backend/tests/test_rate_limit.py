"""The limits on the routes that cost money.

The bucket arithmetic is unit-tested directly, because driving it through the
app would mean either sleeping or spending 30 requests to prove one boundary.
The wiring -- that the routes are actually metered, and that callers get their
own allowance -- is asserted through the app, since that is where it can break.
"""

from __future__ import annotations

import pytest
from fakes import FakeEngine
from fastapi.testclient import TestClient

from support_agent.api.rate_limit import RateLimiter


def _spend(limiter: RateLimiter, caller: str, times: int) -> None:
    for _ in range(times):
        limiter.check(caller)


def test_a_caller_may_spend_the_whole_allowance() -> None:
    limiter = RateLimiter(name="chat", capacity=3, window_s=60.0)

    _spend(limiter, "ip:1.2.3.4", 3)


def test_the_next_call_is_refused_with_how_long_to_wait() -> None:
    limiter = RateLimiter(name="chat", capacity=3, window_s=60.0)
    _spend(limiter, "ip:1.2.3.4", 3)

    with pytest.raises(Exception) as caught:
        limiter.check("ip:1.2.3.4")

    error = caught.value
    assert getattr(error, "status_code", None) == 429
    # A client that honours Retry-After must not be refused again on arrival.
    assert int(error.headers["Retry-After"]) >= 1


def test_callers_do_not_spend_each_others_allowance() -> None:
    limiter = RateLimiter(name="chat", capacity=1, window_s=60.0)
    limiter.check("ip:1.2.3.4")

    limiter.check("ip:5.6.7.8")


def test_a_capacity_of_zero_turns_the_limit_off() -> None:
    limiter = RateLimiter(name="chat", capacity=0, window_s=60.0)

    _spend(limiter, "ip:1.2.3.4", 100)


def test_the_bucket_refills_over_time() -> None:
    """Refill is continuous, so a caller is not locked out for a whole window."""
    limiter = RateLimiter(name="chat", capacity=60, window_s=60.0)
    _spend(limiter, "ip:1.2.3.4", 60)

    # Rewind this caller's clock by two seconds: one token per second, so two
    # are back. Reaching into the bucket beats sleeping in a test.
    limiter._buckets["ip:1.2.3.4"].updated -= 2.0

    _spend(limiter, "ip:1.2.3.4", 2)
    with pytest.raises(Exception):
        limiter.check("ip:1.2.3.4")


# --- the wiring --------------------------------------------------------------


def test_chat_is_metered(limited_client: TestClient, engine: FakeEngine) -> None:
    """Two chat turns allowed, the third refused -- and it never reaches the engine."""
    for _ in range(2):
        assert limited_client.post("/chat/sync", json={"message": "hi"}).status_code == 200

    response = limited_client.post("/chat/sync", json={"message": "hi"})

    assert response.status_code == 429
    assert len(engine.chat_requests) == 2


def test_the_eval_route_is_metered(limited_client: TestClient) -> None:
    for _ in range(2):
        assert (
            limited_client.post("/admin/eval/system-prompt?only=greeting").status_code
            == 200
        )

    assert (
        limited_client.post("/admin/eval/system-prompt?only=greeting").status_code
        == 429
    )


def test_listing_is_not_metered(limited_client: TestClient) -> None:
    """Only the routes that call a model are limited; browsing the cases is free."""
    for _ in range(10):
        assert (
            limited_client.get("/admin/eval/system-prompt/cases").status_code == 200
        )
