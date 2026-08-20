"""The registry, on disk.

The vectors persist, so this has to as well. When it did not, an engine restart
left Chroma holding chunks that `GET /documents` no longer listed -- documents the
engine would answer from while reporting that it had none, and that nothing could
delete because their ids were gone.

Plain `sqlite3` on a thread rather than an async driver: one small table, written
once per upload, so a new dependency would buy nothing.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from chatbot_engine.models.documents import DocumentRecord, IngestStatus
from chatbot_engine.ports.documents import DocumentRegistry

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    project_id   TEXT    NOT NULL,
    doc_id       TEXT    NOT NULL,
    external_id  TEXT    NOT NULL,
    filename     TEXT    NOT NULL,
    mimetype     TEXT    NOT NULL,
    size_bytes   INTEGER NOT NULL,
    content_hash TEXT    NOT NULL,
    status       TEXT    NOT NULL,
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    error        TEXT,
    created_at   TEXT,
    updated_at   TEXT,
    PRIMARY KEY (project_id, doc_id)
);
"""

_COLUMNS = (
    "project_id, doc_id, external_id, filename, mimetype, size_bytes, "
    "content_hash, status, chunk_count, error, created_at, updated_at"
)


def _to_row(record: DocumentRecord) -> tuple[object, ...]:
    return (
        record.project_id,
        record.doc_id,
        record.external_id,
        record.filename,
        record.mimetype,
        record.size_bytes,
        record.content_hash,
        record.status.value,
        record.chunk_count,
        record.error,
        record.created_at.isoformat() if record.created_at else None,
        record.updated_at.isoformat() if record.updated_at else None,
    )


def _to_record(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        project_id=row["project_id"],
        doc_id=row["doc_id"],
        external_id=row["external_id"],
        filename=row["filename"],
        mimetype=row["mimetype"],
        size_bytes=row["size_bytes"],
        content_hash=row["content_hash"],
        status=IngestStatus(row["status"]),
        chunk_count=row["chunk_count"],
        error=row["error"],
        created_at=(
            datetime.fromisoformat(row["created_at"]) if row["created_at"] else None
        ),
        updated_at=(
            datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None
        ),
    )


class SqliteDocumentRegistry(DocumentRegistry):
    """Document metadata in a SQLite file, so it outlives the process."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as connection:
            connection.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row

        return connection

    async def upsert(self, record: DocumentRecord) -> DocumentRecord:
        """Create or replace one record, keyed by project and document id."""

        def write() -> None:
            with self._connect() as connection:
                placeholders = ", ".join("?" * 12)
                connection.execute(
                    f"INSERT OR REPLACE INTO documents ({_COLUMNS}) "
                    f"VALUES ({placeholders})",
                    _to_row(record),
                )

        await asyncio.to_thread(write)

        return record

    async def get(self, *, project_id: str, doc_id: str) -> DocumentRecord | None:
        def read() -> DocumentRecord | None:
            with self._connect() as connection:
                row = connection.execute(
                    f"SELECT {_COLUMNS} FROM documents "
                    "WHERE project_id = ? AND doc_id = ?",
                    (project_id, doc_id),
                ).fetchone()

            return _to_record(row) if row else None

        return await asyncio.to_thread(read)

    async def list(self, *, project_id: str) -> Sequence[DocumentRecord]:
        def read() -> list[DocumentRecord]:
            with self._connect() as connection:
                rows = connection.execute(
                    f"SELECT {_COLUMNS} FROM documents WHERE project_id = ? "
                    "ORDER BY external_id",
                    (project_id,),
                ).fetchall()

            return [_to_record(row) for row in rows]

        return await asyncio.to_thread(read)

    async def set_status(self, *, doc_id: str, status: IngestStatus) -> None:
        def write() -> None:
            with self._connect() as connection:
                connection.execute(
                    "UPDATE documents SET status = ? WHERE doc_id = ?",
                    (status.value, doc_id),
                )

        await asyncio.to_thread(write)

    async def delete(self, *, project_id: str, doc_id: str) -> bool:
        def write() -> bool:
            with self._connect() as connection:
                cursor = connection.execute(
                    "DELETE FROM documents WHERE project_id = ? AND doc_id = ?",
                    (project_id, doc_id),
                )

            return cursor.rowcount > 0

        return await asyncio.to_thread(write)
