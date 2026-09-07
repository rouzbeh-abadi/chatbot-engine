"""Rate limiting for the routes that spend money.

Every chat turn, every judged case, every RAGAS metric and every document
ingested is a billed provider call. The engine is where that spending actually
happens, so the limit belongs here, not only in whatever calls it. A caller's
own limiter protects the caller from itself; it is not a control the engine can
rely on, because the engine cannot verify it exists.

Counted per authenticated caller name, which is what the named keys in
`ENGINE_API_KEYS` are for: one runaway client can be throttled, and found,
without touching the others.

The allowance is a token bucket, refilling continuously. Where the buckets live
is a deployment choice: in this process's memory by default, which is exact for
one engine and multiplies by the replica count for several; or in Redis, set by
`ENGINE_REDIS_URL`, where every replica charges the same bucket. The arithmetic
is identical in both. The Redis form runs it as one script so two replicas
cannot interleave a read and a write on the same caller.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Protocol

from fastapi import Depends, HTTPException, status

from chatbot_engine.api.auth import CallerDep
from chatbot_engine.settings import Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover - only for the annotation below
    import redis

SettingsDep = Annotated[Settings, Depends(get_settings)]

#: Buckets are kept per caller, so an unbounded key space would be a memory
#: leak. Callers are named keys here, so this is only reachable if someone
#: configures thousands of them: the cap is a backstop, not a design point.
MAX_TRACKED_CALLERS = 10_000


class BucketStore(Protocol):
    """Where one named limit keeps its buckets."""

    def take(
        self, caller: str, *, capacity: float, per_second: float, now: float
    ) -> float:
        """Spend one of `caller`'s tokens. Returns the seconds to wait, or 0.0."""
        ...


@dataclass
class _Bucket:
    """One caller's allowance, refilling continuously rather than in steps.

    A token bucket rather than a fixed window: a window lets a caller spend a
    whole allowance in its last second and the next one in the following second,
    which is the burst the limit exists to prevent.
    """

    tokens: float
    updated: float

    def take(self, *, capacity: float, per_second: float, now: float) -> float:
        self.tokens = min(capacity, self.tokens + (now - self.updated) * per_second)
        self.updated = now

        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return 0.0

        return (1.0 - self.tokens) / per_second


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
        """Forget callers whose allowance has fully refilled; they cost nothing."""
        if len(self._buckets) < MAX_TRACKED_CALLERS:
            return

        self._buckets = {
            caller: bucket
            for caller, bucket in self._buckets.items()
            if bucket.tokens + (now - bucket.updated) * per_second < capacity
        }


#: The same arithmetic as `_Bucket.take`, run inside Redis. One script call is
#: atomic, so two replicas charging the same caller at once cannot both read
#: the old balance. The key expires once a full refill has elapsed, since a
#: full bucket is indistinguishable from a missing one.
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
        # Imported here: `redis` is an optional extra, needed only when the
        # setting that selects this store is present.
        import redis as redis_client

        self._redis = client if client is not None else redis_client.Redis.from_url(url)
        self._take = self._redis.register_script(_TAKE)
        self._prefix = f"chatbot-engine:ratelimit:{name}:"

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
            # rounding down would refuse a caller that obeyed it exactly.
            headers={"Retry-After": str(max(1, math.ceil(wait_s)))},
        )


#: The live limiters, one per named limit. Rebuilt when the configured
#: capacity changes, so a settings override in a test takes effect at once.
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


async def limit_chat(caller: CallerDep, settings: SettingsDep) -> None:
    """One answered turn: a model call, plus an embedding for the retrieval."""
    _limiter("chat", settings.chat_rate_limit_per_minute, 60.0, settings).check(
        caller.name
    )


async def limit_eval(caller: CallerDep, settings: SettingsDep) -> None:
    """The expensive one: a full run is several model calls per case."""
    _limiter("evaluation", settings.eval_rate_limit_per_hour, 3600.0, settings).check(
        caller.name
    )


async def limit_ingest(caller: CallerDep, settings: SettingsDep) -> None:
    """Indexing a document embeds every chunk of it, which is billed per token."""
    _limiter("ingest", settings.ingest_rate_limit_per_minute, 60.0, settings).check(
        caller.name
    )


def reset_rate_limits() -> None:
    """Forget every limiter. For tests, and after a configuration change."""
    _LIMITERS.clear()
