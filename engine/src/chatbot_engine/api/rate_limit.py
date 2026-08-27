"""Rate limiting for the routes that spend money.

Every chat turn, every judged case, every RAGAS metric and every document
ingested is a billed provider call. The engine is where that spending actually
happens, so the limit belongs here -- not only in whatever calls it. A caller's
own limiter protects the caller from itself; it is not a control the engine can
rely on, because the engine cannot verify it exists.

Counted per authenticated caller name, which is what the named keys in
`ENGINE_API_KEYS` are for: one runaway client can be throttled, and found,
without touching the others.

Scope, stated plainly: the buckets live in this process's memory. Behind two
replicas the effective limit is doubled, and a restart forgets everything. That
is a real limitation and it is still worth having -- these stop runaway loops
and accidental retries, which is the failure that actually empties a provider
account. For an exact global limit, back `_Bucket` with Redis; nothing above
`RateLimiter.check` changes.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Annotated

from fastapi import Depends, HTTPException, status

from chatbot_engine.api.auth import CallerDep
from chatbot_engine.settings import Settings, get_settings

SettingsDep = Annotated[Settings, Depends(get_settings)]

#: Buckets are kept per caller, so an unbounded key space would be a memory
#: leak. Callers are named keys here, so this is only reachable if someone
#: configures thousands of them -- the cap is a backstop, not a design point.
MAX_TRACKED_CALLERS = 10_000


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
        """Spend one token. Returns the seconds to wait, or 0.0 when allowed."""
        self.tokens = min(capacity, self.tokens + (now - self.updated) * per_second)
        self.updated = now

        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return 0.0

        return (1.0 - self.tokens) / per_second


@dataclass
class RateLimiter:
    """A named allowance of `capacity` calls per `window_s`, per caller."""

    name: str
    capacity: int
    window_s: float
    _buckets: dict[str, _Bucket] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        """A capacity of zero turns the limit off, for local runs and tests."""
        return self.capacity > 0

    def check(self, caller: str) -> None:
        """Charge one call to `caller`, or raise 429 with how long to wait."""
        if not self.enabled:
            return

        now = time.monotonic()
        per_second = self.capacity / self.window_s

        bucket = self._buckets.get(caller)
        if bucket is None:
            self._prune(now=now, capacity=self.capacity, per_second=per_second)
            bucket = self._buckets.setdefault(
                caller, _Bucket(tokens=float(self.capacity), updated=now)
            )

        wait_s = bucket.take(capacity=self.capacity, per_second=per_second, now=now)
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

    def _prune(self, *, now: float, capacity: float, per_second: float) -> None:
        """Forget callers whose allowance has fully refilled -- they cost nothing."""
        if len(self._buckets) < MAX_TRACKED_CALLERS:
            return

        self._buckets = {
            caller: bucket
            for caller, bucket in self._buckets.items()
            if bucket.tokens + (now - bucket.updated) * per_second < capacity
        }


#: The live buckets, one table per named limit. Rebuilt when the configured
#: capacity changes, so a settings override in a test takes effect at once.
_LIMITERS: dict[str, RateLimiter] = {}


def _limiter(name: str, capacity: int, window_s: float) -> RateLimiter:
    """The named limiter, built on first use and kept until its capacity changes."""
    existing = _LIMITERS.get(name)
    if existing is None or existing.capacity != capacity:
        existing = RateLimiter(name=name, capacity=capacity, window_s=window_s)
        _LIMITERS[name] = existing
    return existing


async def limit_chat(caller: CallerDep, settings: SettingsDep) -> None:
    """One answered turn: a model call, plus an embedding for the retrieval."""
    _limiter("chat", settings.chat_rate_limit_per_minute, 60.0).check(caller.name)


async def limit_eval(caller: CallerDep, settings: SettingsDep) -> None:
    """The expensive one: a full run is several model calls per case."""
    _limiter("evaluation", settings.eval_rate_limit_per_hour, 3600.0).check(
        caller.name
    )


async def limit_ingest(caller: CallerDep, settings: SettingsDep) -> None:
    """Indexing a document embeds every chunk of it, which is billed per token."""
    _limiter("ingest", settings.ingest_rate_limit_per_minute, 60.0).check(caller.name)


def reset_rate_limits() -> None:
    """Forget every bucket. For tests, and after a configuration change."""
    _LIMITERS.clear()
