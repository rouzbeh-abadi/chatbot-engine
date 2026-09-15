"""Where a workflow turn waits between asking a question and getting the answer.

LangGraph pauses a graph with `interrupt()` and resumes it with
`Command(resume=...)`, but only with a checkpointer that kept the state in
between. This module owns that checkpointer and a small record of each paused
turn: which project and conversation it belongs to, which workflow it ran,
and when it paused. The record is what makes a resume safe:

- a `thread_id` from one project or conversation cannot resume another's turn;
- a workflow edited while a turn waited is not resumed into a different graph;
- a turn nobody answered is forgotten after `pause_ttl_s`, state and all.

Only a workflow that contains an ask step is compiled with the checkpointer,
so turns that never pause write nothing to disk, and a finished turn's state
is deleted as soon as it ends.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from chatbot_engine.models.workflow import WorkflowSpec
from chatbot_engine.settings import get_settings


@dataclass(frozen=True)
class Pause:
    """A paused turn, as recorded when it paused."""

    thread_id: str
    project_id: str
    session_id: str
    spec_hash: str
    paused_at: float


def spec_hash(spec: WorkflowSpec) -> str:
    """A fingerprint of the workflow, so a resume can tell it was edited."""
    return hashlib.sha256(spec.model_dump_json(by_alias=True).encode()).hexdigest()


class Pauses:
    """The checkpointer plus the record of paused turns.

    `memory()` for tests and a single process; `sqlite(path)` for an engine
    that restarts, opened on first use inside the running event loop.
    """

    def __init__(self, open_saver: Any, ttl_s: int) -> None:
        self._open_saver = open_saver
        self._ttl_s = ttl_s
        self._saver: BaseCheckpointSaver | None = None
        self._records: dict[str, Pause] = {}
        self._conn: Any = None
        self._lock = asyncio.Lock()

    @classmethod
    def memory(cls, ttl_s: int = 86_400) -> Pauses:
        async def open_saver(_: Pauses) -> BaseCheckpointSaver:
            return InMemorySaver()

        return cls(open_saver, ttl_s)

    @classmethod
    def sqlite(cls, path: Path, ttl_s: int) -> Pauses:
        async def open_saver(self: Pauses) -> BaseCheckpointSaver:
            import aiosqlite
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = await aiosqlite.connect(str(path))
            await self._conn.execute(
                "CREATE TABLE IF NOT EXISTS workflow_pauses ("
                " thread_id TEXT PRIMARY KEY, project_id TEXT NOT NULL,"
                " session_id TEXT NOT NULL, spec_hash TEXT NOT NULL,"
                " paused_at REAL NOT NULL)"
            )
            await self._conn.commit()
            saver = AsyncSqliteSaver(self._conn)
            await saver.setup()
            return saver

        return cls(open_saver, ttl_s)

    @classmethod
    def from_settings(cls) -> Pauses:
        settings = get_settings()
        return cls.sqlite(settings.checkpoint_db, settings.pause_ttl_s)

    async def saver(self) -> BaseCheckpointSaver:
        async with self._lock:
            if self._saver is None:
                self._saver = await self._open_saver(self)
            return self._saver

    async def record(self, pause: Pause) -> None:
        """Remember a turn that just paused, and forget the ones left too long."""
        await self.saver()
        if self._conn is None:
            self._records[pause.thread_id] = pause
        else:
            await self._conn.execute(
                "INSERT OR REPLACE INTO workflow_pauses VALUES (?, ?, ?, ?, ?)",
                (
                    pause.thread_id,
                    pause.project_id,
                    pause.session_id,
                    pause.spec_hash,
                    pause.paused_at,
                ),
            )
            await self._conn.commit()
        await self._prune()

    async def find(self, thread_id: str) -> Pause | None:
        """The paused turn under `thread_id`, if it is still waiting."""
        await self.saver()
        if self._conn is None:
            pause = self._records.get(thread_id)
        else:
            async with self._conn.execute(
                "SELECT thread_id, project_id, session_id, spec_hash, paused_at"
                " FROM workflow_pauses WHERE thread_id = ?",
                (thread_id,),
            ) as cursor:
                row = await cursor.fetchone()
            pause = Pause(*row) if row else None
        if pause is not None and time.time() - pause.paused_at > self._ttl_s:
            await self.forget(thread_id)
            return None
        return pause

    async def forget(self, thread_id: str) -> None:
        """Delete a turn's saved state and its record: it finished, or expired."""
        saver = await self.saver()
        await saver.adelete_thread(thread_id)
        if self._conn is None:
            self._records.pop(thread_id, None)
        else:
            await self._conn.execute(
                "DELETE FROM workflow_pauses WHERE thread_id = ?", (thread_id,)
            )
            await self._conn.commit()

    async def _prune(self) -> None:
        cutoff = time.time() - self._ttl_s
        if self._conn is None:
            stale = [t for t, p in self._records.items() if p.paused_at < cutoff]
        else:
            async with self._conn.execute(
                "SELECT thread_id FROM workflow_pauses WHERE paused_at < ?", (cutoff,)
            ) as cursor:
                stale = [row[0] for row in await cursor.fetchall()]
        for thread_id in stale:
            await self.forget(thread_id)
