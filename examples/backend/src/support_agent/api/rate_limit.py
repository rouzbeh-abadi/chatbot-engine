"""Rate limiting for the routes that cost money.

Every chat turn and every evaluation case is a model call billed to whoever owns
the provider key. Without a limit, one script -- or one enthusiastic crawler --
turns an open endpoint into someone else's bill, which is the failure mode that
actually happens to small deployments.

Where the buckets live is a deployment choice: in this process's memory by
default, which is exact for one replica and multiplies by the replica count for
several; or in Redis, set by `BACKEND_REDIS_URL`, where every replica charges
the same bucket. The arithmetic is identical in both, and mirrors the engine's.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Protocol

from fastapi import Depends, HTTPException, Request, status

from support_agent.api.identity import ANONYMOUS_USER_ID, UserIdDep
from support_agent.settings import Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover - only for an annotation
    import redis

SettingsDep = Annotated[Settings, Depends(get_settings)]

#: Buckets are kept per caller, so an unbounded key space is a memory leak. Well
#: above any real concurrent-caller count for a deployment this size; reaching it
#: means a spray of forged addresses, which is exactly when to start forgetting.
MAX_TRACKED_CALLERS = 10_000


@dataclass
class _Bucket:
    """One caller's allowance, refilling continuously rather than in steps.

    A token bucket rather than a fixed window: a window lets a caller spend the
    whole allowance in its last second and the whole next allowance in the
    following one, which is the burst the limit exists to prevent.
    """

    tokens: float
    updated: float

    def take(self, *, capacity: float, per_second: float, now: float) -> float:
        """Spend one token. Returns the seconds to wait, or 0.0 when allowed."""
        self.tokens = min(capacity, self.tokens + (now - self.updated) * per_second)
        self.updated = now

        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return 0.0

        return (1.0 - self.tokens) / per_second


class BucketStore(Protocol):
    """Where one named limit keeps its buckets."""

    def take(
        self, caller: str, *, capacity: float, per_second: float, now: float
    ) -> float:
        """Spend one of `caller`'s tokens. Returns the seconds to wait, or 0.0."""
        ...


@dataclass
class MemoryBuckets:
    """Buckets in this process. Exact for one replica, per-replica for several."""

    _buckets: dict[str, _Bucket] = field(default_factory=dict)

    def take(
        self, caller: str, *, capacity: float, per_second: float, now: float
    ) -> float:
        bucket = self._buckets.get(caller)
        if bucket is None:
            self._prune(now=now, capacity=capacity, per_second=per_second)
            bucket = self._buckets.setdefault(
                caller, _Bucket(tokens=capacity, updated=now)
            )

        return bucket.take(capacity=capacity, per_second=per_second, now=now)

    def _prune(self, *, now: float, capacity: float, per_second: float) -> None:
        """Forget callers whose allowance has fully refilled; they cost nothing.

        Only when the table is large: pruning on every miss would walk the whole
        dict per new caller, which is the cost this is meant to avoid.
        """
        if len(self._buckets) < MAX_TRACKED_CALLERS:
            return

        self._buckets = {
            caller: bucket
            for caller, bucket in self._buckets.items()
            if bucket.tokens + (now - bucket.updated) * per_second < capacity
        }


#: The same arithmetic as `_Bucket.take`, run inside Redis. One script call is
#: atomic, so two replicas charging the same caller at once cannot both read
#: the old balance. The key expires once a full refill has elapsed.
_TAKE = """
local capacity = tonumber(ARGV[1])
local per_second = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

local state = redis.call('HMGET', KEYS[1], 'tokens', 'updated')
local tokens = tonumber(state[1])
local updated = tonumber(state[2])
if tokens == nil then
  tokens = capacity
  updated = now
end

tokens = math.min(capacity, tokens + (now - updated) * per_second)
local wait = 0.0
if tokens >= 1.0 then
  tokens = tokens - 1.0
else
  wait = (1.0 - tokens) / per_second
end

redis.call('HSET', KEYS[1], 'tokens', tokens, 'updated', now)
redis.call('PEXPIRE', KEYS[1], math.ceil(capacity / per_second * 1000))
return tostring(wait)
"""


class RedisBuckets:
    """Buckets in Redis, shared by every replica that points at the same URL."""

    def __init__(
        self, url: str, *, name: str, client: redis.Redis | None = None
    ) -> None:
        import redis as redis_client

        self._redis = client if client is not None else redis_client.Redis.from_url(url)
        self._take = self._redis.register_script(_TAKE)
        self._prefix = f"support-agent:ratelimit:{name}:"

    def take(
        self, caller: str, *, capacity: float, per_second: float, now: float
    ) -> float:
        wait = self._take(
            keys=[self._prefix + caller], args=[capacity, per_second, now]
        )
        return float(wait)


@dataclass
class RateLimiter:
    """A named allowance of `capacity` calls per `window_s`, per caller."""

    name: str
    capacity: int
    window_s: float
    store: BucketStore = field(default_factory=MemoryBuckets)

    @property
    def enabled(self) -> bool:
        """A capacity of zero turns the limit off, for local runs and tests."""
        return self.capacity > 0

    def check(self, caller: str) -> None:
        """Charge one call to `caller`, or raise 429 with how long to wait."""
        if not self.enabled:
            return

        wait_s = self.store.take(
            caller,
            capacity=float(self.capacity),
            per_second=self.capacity / self.window_s,
            # Wall-clock, not monotonic: with Redis the clock has to mean the
            # same thing on every replica.
            now=time.time(),
        )
        if wait_s == 0.0:
            return

        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"rate limit exceeded: at most {self.capacity} "
                f"{self.name} calls per {round(self.window_s)}s"
            ),
            # Whole seconds, rounded up: the header has no fractional form, and
            # rounding down would refuse a client that obeyed it exactly.
            headers={"Retry-After": str(max(1, math.ceil(wait_s)))},
        )


def caller_of(request: Request, user_id: str | None = None) -> str:
    """Who to charge: the authenticated user if there is one, else the address.

    `request.client.host` is the peer address, which behind a proxy is the proxy
    -- one bucket for everyone. Uvicorn rewrites it from `X-Forwarded-For` when
    started with `--proxy-headers --forwarded-allow-ips`, which is how the
    container runs it. That flag is doing security work: the header is trivially
    forged, so it must only be believed from a proxy you run.

    Anonymous callers all share the `anonymous` id, which would be one bucket for
    the whole internet -- so they are charged by address instead.
    """
    if user_id and user_id != ANONYMOUS_USER_ID:
        return f"user:{user_id}"

    client = request.client
    return f"ip:{client.host}" if client else "ip:unknown"


#: The live buckets, one table per named limit. Process-wide by design -- see
#: the module docstring -- and rebuilt when the configured capacity changes, so
#: a settings override in a test takes effect on the next request.
_LIMITERS: dict[str, RateLimiter] = {}


def _store_for(name: str, settings: Settings) -> BucketStore:
    if settings.redis_url:
        return RedisBuckets(settings.redis_url, name=name)
    return MemoryBuckets()


def _limiter(
    name: str, capacity: int, window_s: float, settings: Settings
) -> RateLimiter:
    """The named limiter, built on first use and kept until its capacity changes."""
    existing = _LIMITERS.get(name)
    if existing is None or existing.capacity != capacity:
        existing = RateLimiter(
            name=name,
            capacity=capacity,
            window_s=window_s,
            store=_store_for(name, settings),
        )
        _LIMITERS[name] = existing
    return existing


async def limit_chat(
    request: Request, user_id: UserIdDep, settings: SettingsDep
) -> None:
    """Charge a chat turn to whoever asked for it."""
    _limiter("chat", settings.chat_rate_limit_per_minute, 60.0, settings).check(
        caller_of(request, user_id)
    )


async def limit_eval(request: Request, settings: SettingsDep) -> None:
    """Charge an evaluation run to the operator's address.

    No user id here: `/admin` is authenticated by a shared key, so every operator
    looks the same. The address is the only thing distinguishing them.
    """
    _limiter("evaluation", settings.eval_rate_limit_per_hour, 3600.0, settings).check(
        caller_of(request)
    )


def reset_rate_limits() -> None:
    """Forget every bucket. For tests, and after a configuration change."""
    _LIMITERS.clear()
