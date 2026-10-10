"""Where a workflow turn waits between asking a question and getting the answer.

LangGraph pauses a graph with `interrupt()` and resumes it with
`Command(resume=...)`, but only with a checkpointer that kept the state in
between. This module owns that checkpointer and a small record of each paused
turn: which project and conversation it belongs to, which workflow it ran,
and when it paused. The record is what makes a resume safe:

- a `thread_id` from one project or conversation cannot resume another's turn;
- a workflow edited while a turn waited is not resumed into a different graph;
- a turn nobody answered is forgotten after `pause_ttl_s`, state and all;
- a pause is resumed once: the request that resumes it claims its record
  first, so a second request with the same answer (a double click, a retry)
  finds nothing waiting.

Only a workflow that contains an ask step is compiled with the checkpointer,
so turns that never pause write nothing to disk, and a finished turn's state
is deleted as soon as it ends.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from chatbot_engine.models.workflow import WorkflowSpec
from chatbot_engine.settings import get_settings

logger = logging.getLogger(__name__)


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


#: How often, at most, paused turns past their time are forgotten while the
#: store is used (seconds).
_PRUNE_EVERY_S = 60.0

#: How long saved state that no paused turn names is kept after it was last
#: written (seconds): far longer than any turn runs, so a turn still running,
#: in this process or another sharing the file, is never caught.
_ORPHAN_AFTER_S = 3600.0

#: 100-nanosecond steps from the start of the Gregorian calendar to 1970.
_GREGORIAN_TO_UNIX = 0x01B21DD213814000


def _written_at(checkpoint_id: object) -> float | None:
    """When a checkpoint was written, read from its id: LangGraph's are UUIDv6,
    which carry the time. None for an id of any other kind."""
    try:
        uid = uuid.UUID(str(checkpoint_id))
    except ValueError:
        return None
    if uid.version != 6:
        return None
    ticks = (uid.time_low << 28) | (uid.time_mid << 12) | (uid.time_hi_version & 0x0FFF)
    return (ticks - _GREGORIAN_TO_UNIX) / 1e7


#: `PRAGMA auto_vacuum`'s value for INCREMENTAL.
_INCREMENTAL = 2
#: How large the write-ahead log is left after a checkpoint.
_WAL_BYTES = 4 * 1024 * 1024


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
        self._pruned_at = 0.0

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
            # So the room a forgotten turn took goes back to the volume, which
            # every tenant's index shares: pages freed as they are deleted
            # (`incremental_vacuum` in `forget`), and a write-ahead log cut
            # back after it is checkpointed (docs/review-2026-10.md,
            # WORKFLOW-4). A file made before this is rebuilt once to it.
            async with self._conn.execute("PRAGMA auto_vacuum") as cursor:
                row = await cursor.fetchone()
            if row is None or row[0] != _INCREMENTAL:
                await self._conn.execute("PRAGMA auto_vacuum = INCREMENTAL")
                try:
                    # Rewrites the file, so it needs as much room again; with
                    # too little, turns still pause, only the room is not
                    # given back yet.
                    await self._conn.execute("VACUUM")
                except sqlite3.OperationalError as exc:
                    logger.warning("checkpoint file not rebuilt: %s", exc)
            await self._conn.execute(f"PRAGMA journal_size_limit = {_WAL_BYTES}")
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
            saver = self._saver
        # Every use of the store, a turn that never pauses included, forgets
        # what is past its time, so a question nobody answered does not wait
        # for another turn to pause before it goes (WORKFLOW-5).
        await self._prune_if_due()
        return saver

    async def forget_where(self, project_id: str, session_id: str | None = None) -> int:
        """Forget the paused turns of a project, or of one of its sessions,
        with their saved state; how many."""
        await self.saver()
        if self._conn is None:
            threads = [
                thread_id
                for thread_id, pause in self._records.items()
                if pause.project_id == project_id
                and session_id in (None, pause.session_id)
            ]
        else:
            query = "SELECT thread_id FROM workflow_pauses WHERE project_id = ?"
            args: tuple[str, ...] = (project_id,)
            if session_id is not None:
                query += " AND session_id = ?"
                args = (project_id, session_id)
            async with self._conn.execute(query, args) as cursor:
                threads = [row[0] for row in await cursor.fetchall()]
        for thread_id in threads:
            await self.forget(thread_id)
        return len(threads)

    async def _prune_if_due(self) -> None:
        if time.time() - self._pruned_at >= _PRUNE_EVERY_S:
            await self._prune()

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

    async def claim(
        self, thread_id: str, project_id: str, session_id: str
    ) -> Pause | None:
        """Take the paused turn under `thread_id` for one resume, or None.

        The record is removed as it is read, in one step, so of two requests
        resuming the same turn only one gets it; the other finds nothing
        waiting, as for a turn that expired. A pause from another project or
        conversation is neither returned nor removed. The turn's saved state
        stays for the resume to read: a turn that pauses again is recorded
        again, and one that ends forgets its state.
        """
        await self.saver()
        if self._conn is None:
            # No await between the look and the removal: atomic in one loop.
            pause = self._records.get(thread_id)
            if (
                pause is None
                or pause.project_id != project_id
                or pause.session_id != session_id
            ):
                return None
            del self._records[thread_id]
        else:
            # One statement, so two engines on one file cannot both delete it.
            async with self._conn.execute(
                "DELETE FROM workflow_pauses"
                " WHERE thread_id = ? AND project_id = ? AND session_id = ?"
                " RETURNING thread_id, project_id, session_id, spec_hash, paused_at",
                (thread_id, project_id, session_id),
            ) as cursor:
                row = await cursor.fetchone()
            await self._conn.commit()
            if row is None:
                return None
            pause = Pause(*row)
        if time.time() - pause.paused_at > self._ttl_s:
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
            # Give the freed pages back to the volume, and cut the log back.
            # Each read to its end: a pragma left part-read holds the
            # connection, and one step of this one frees one page. Skipped
            # while another writer holds the file; the next forget does it.
            with contextlib.suppress(sqlite3.OperationalError):
                for pragma in ("incremental_vacuum", "wal_checkpoint(TRUNCATE)"):
                    async with self._conn.execute(f"PRAGMA {pragma}") as cursor:
                        await cursor.fetchall()

    async def _prune(self) -> None:
        # Noted first: forgetting uses the store, which would prune again.
        self._pruned_at = time.time()
        cutoff = time.time() - self._ttl_s
        if self._conn is None:
            stale = [t for t, p in self._records.items() if p.paused_at < cutoff]
        else:
            async with self._conn.execute(
                "SELECT thread_id FROM workflow_pauses WHERE paused_at < ?", (cutoff,)
            ) as cursor:
                stale = [row[0] for row in await cursor.fetchall()]
            stale += await self._orphans()
        for thread_id in stale:
            await self.forget(thread_id)

    async def _orphans(self) -> list[str]:
        """Saved state that no paused turn names and no turn has written to
        for `_ORPHAN_AFTER_S`: a turn the engine was stopped in the middle of,
        or one whose forget failed on a full disk. Nothing will resume it, and
        it can hold a visitor's answers (WORKFLOW-5)."""
        async with self._conn.execute(
            "SELECT thread_id, MAX(checkpoint_id) FROM checkpoints"
            " WHERE thread_id NOT IN (SELECT thread_id FROM workflow_pauses)"
            " GROUP BY thread_id"
        ) as cursor:
            rows = await cursor.fetchall()
        cutoff = time.time() - _ORPHAN_AFTER_S
        return [
            thread_id
            for thread_id, newest in rows
            if (written := _written_at(newest)) is not None and written < cutoff
        ]
