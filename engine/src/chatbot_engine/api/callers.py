"""Work a caller waits for, stopped when the caller goes away.

Starlette cancels a streaming answer when its reader leaves, but not a plain
JSON route: an evaluation went on answering and grading every case on the
caller's key after the caller had timed out, and a retry paid for all of it
again (docs/review-2026-10.md, API-3).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable

from fastapi import HTTPException, Request


async def while_waited_for[T](request: Request, work: Awaitable[T]) -> T:
    """`work`'s result, or, once the caller has gone, `work` cancelled and a
    499 nobody reads."""
    task = asyncio.ensure_future(work)
    watcher = asyncio.ensure_future(_gone(request))
    try:
        done, _ = await asyncio.wait(
            {task, watcher}, return_when=asyncio.FIRST_COMPLETED
        )
        if task in done:
            return task.result()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        raise HTTPException(status_code=499, detail="the caller went away")
    finally:
        for pending in (task, watcher):
            if not pending.done():
                pending.cancel()


async def _gone(request: Request) -> None:
    """Return once the caller has disconnected, and never otherwise.

    Listens as a streaming answer does: the body has been read, so what the
    server hands on next is the disconnect. A server that cannot say so is
    taken to keep the caller.
    """
    try:
        while (await request.receive())["type"] != "http.disconnect":
            pass
    except Exception:
        await asyncio.Event().wait()
