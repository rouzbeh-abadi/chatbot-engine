"""The registry in Postgres, for an engine that runs as several replicas.

The SQLite registry is a file, and a file belongs to one writer. Several
replicas that all ingest need one registry they all read and write, and a
database is that. Same table shape as the SQLite one, same contract.

Plain `psycopg` with a connection per operation: the table is written once
per upload and read once per listing, so a pool would be machinery for a load
this registry never sees. The `postgres` extra provides the driver.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from chatbot_engine.models.documents import DocumentRecord, IngestStatus
from chatbot_engine.ports.documents import DocumentRegistry

TABLE = "engine_documents"

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
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
    created_at   TIMESTAMPTZ,
    updated_at   TIMESTAMPTZ,
    PRIMARY KEY (project_id, doc_id)
)
"""

_COLUMNS = (
    "project_id, doc_id, external_id, filename, mimetype, size_bytes, "
    "content_hash, status, chunk_count, error, created_at, updated_at"
)

_UPSERT = f"""
INSERT INTO {TABLE} ({_COLUMNS})
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (project_id, doc_id) DO UPDATE SET
    external_id = EXCLUDED.external_id,
    filename = EXCLUDED.filename,
    mimetype = EXCLUDED.mimetype,
    size_bytes = EXCLUDED.size_bytes,
    content_hash = EXCLUDED.content_hash,
    status = EXCLUDED.status,
    chunk_count = EXCLUDED.chunk_count,
    error = EXCLUDED.error,
    created_at = EXCLUDED.created_at,
    updated_at = EXCLUDED.updated_at
"""


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
        record.created_at,
        record.updated_at,
    )


def _to_record(row: tuple) -> DocumentRecord:
    (
        project_id, doc_id, external_id, filename, mimetype, size_bytes,
        content_hash, status, chunk_count, error, created_at, updated_at,
    ) = row  # fmt: skip
    return DocumentRecord(
        project_id=project_id,
        doc_id=doc_id,
        external_id=external_id,
        filename=filename,
        mimetype=mimetype,
        size_bytes=size_bytes,
        content_hash=content_hash,
        status=IngestStatus(status),
        chunk_count=chunk_count,
        error=error,
        created_at=_aware(created_at),
        updated_at=_aware(updated_at),
    )


def _aware(value: datetime | None) -> datetime | None:
    """Postgres returns timestamptz in the session zone; keep them comparable
    to the UTC values the pipeline writes."""
    if value is None:
        return None
    from datetime import UTC

    return value.astimezone(UTC)


class PostgresDocumentRegistry(DocumentRegistry):
    """Document metadata in a Postgres table shared by every replica."""

    def __init__(self, url: str) -> None:
        self._url = url
        self._ready = False

    async def _connection(self):
        import psycopg

        connection = await psycopg.AsyncConnection.connect(self._url)
        if not self._ready:
            async with connection.cursor() as cursor:
                await cursor.execute(_SCHEMA)
            await connection.commit()
            self._ready = True
        return connection

    async def upsert(self, record: DocumentRecord) -> DocumentRecord:
        async with await self._connection() as connection:
            await connection.execute(_UPSERT, _to_row(record))
            await connection.commit()
        return record

    async def get(self, *, project_id: str, doc_id: str) -> DocumentRecord | None:
        async with await self._connection() as connection:
            cursor = await connection.execute(
                f"SELECT {_COLUMNS} FROM {TABLE} WHERE project_id = %s AND doc_id = %s",
                (project_id, doc_id),
            )
            row = await cursor.fetchone()
        return _to_record(row) if row else None

    async def list(self, *, project_id: str) -> Sequence[DocumentRecord]:
        async with await self._connection() as connection:
            cursor = await connection.execute(
                f"SELECT {_COLUMNS} FROM {TABLE} WHERE project_id = %s "
                "ORDER BY external_id",
                (project_id,),
            )
            rows = await cursor.fetchall()
        return [_to_record(row) for row in rows]

    async def delete(self, *, project_id: str, doc_id: str) -> bool:
        async with await self._connection() as connection:
            cursor = await connection.execute(
                f"DELETE FROM {TABLE} WHERE project_id = %s AND doc_id = %s",
                (project_id, doc_id),
            )
            await connection.commit()
            return (cursor.rowcount or 0) > 0
