"""The registry has to be as durable as the vectors.

Chroma writes to disk. When the registry did not, restarting the engine left
chunks that `GET /documents` no longer listed: the engine would answer from
documents it reported having none of, and nothing could delete them because their
ids were gone. A second registry on the same file stands in for a restart.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from chatbot_engine.documents.sqlite_registry import SqliteDocumentRegistry
from chatbot_engine.models.documents import DocumentRecord, IngestStatus
from chatbot_engine.rag.vector_store import count_chunks

LONG = ("Cabin baggage is one bag up to eight kilograms. " * 40).encode()


def _record(**overrides: object) -> DocumentRecord:
    base = {
        "doc_id": "doc-1",
        "external_id": "baggage.md",
        "project_id": "support",
        "filename": "baggage.md",
        "mimetype": "text/markdown",
        "size_bytes": 42,
        "content_hash": "abc123",
        "status": IngestStatus.INDEXED,
        "chunk_count": 3,
        "created_at": datetime(2026, 8, 20, 9, 0, tzinfo=UTC),
        "updated_at": datetime(2026, 8, 20, 9, 0, tzinfo=UTC),
    }

    return DocumentRecord(**(base | overrides))


@pytest.fixture
def registry(tmp_path: Path) -> SqliteDocumentRegistry:
    return SqliteDocumentRegistry(tmp_path / "documents.sqlite3")


# --- the contract ------------------------------------------------------------


async def test_a_record_round_trips_intact(registry: SqliteDocumentRegistry) -> None:
    """Including the timestamps, which go through the file as ISO strings."""
    stored = _record()
    await registry.upsert(stored)

    assert await registry.get(project_id="support", doc_id="doc-1") == stored


async def test_upsert_replaces_rather_than_duplicates(
    registry: SqliteDocumentRegistry,
) -> None:
    await registry.upsert(_record(chunk_count=3))
    await registry.upsert(_record(chunk_count=9))

    listed = await registry.list(project_id="support")

    assert [record.chunk_count for record in listed] == [9]


# --- surviving a restart -----------------------------------------------------


def test_documents_still_listed_after_the_engine_restarts(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bug this file exists for: vectors on disk, metadata gone."""
    from chatbot_engine.api.dependencies import reset_dependency_cache

    uploaded = client.put(
        "/documents",
        data={"project_id": "support", "external_id": "baggage.md"},
        files={"file": ("baggage.md", LONG, "text/markdown")},
    ).json()
    assert uploaded["status"] == "indexed"

    # Drop every cached singleton: a fresh registry and a fresh Chroma client,
    # both reopening the same files.
    reset_dependency_cache()

    listed = client.get("/documents", params={"project_id": "support"}).json()

    assert [record["doc_id"] for record in listed] == [uploaded["doc_id"]]
    assert count_chunks() == uploaded["chunk_count"]


async def test_chunking_settings_survive_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "documents.sqlite3"
    await SqliteDocumentRegistry(path).upsert(
        _record(chunking_strategy="page", chunk_size=1200, chunk_overlap=100)
    )

    restored = await SqliteDocumentRegistry(path).get(
        project_id="support", doc_id="doc-1"
    )

    assert restored is not None
    assert (
        restored.chunking_strategy,
        restored.chunk_size,
        restored.chunk_overlap,
    ) == ("page", 1200, 100)


async def test_a_registry_from_before_chunking_columns_is_upgraded_in_place(
    tmp_path: Path,
) -> None:
    """An existing file keeps its rows; the new columns arrive empty."""
    import sqlite3

    path = tmp_path / "documents.sqlite3"
    with sqlite3.connect(path) as old:
        old.executescript(
            """
            CREATE TABLE documents (
                project_id TEXT NOT NULL, doc_id TEXT NOT NULL, external_id TEXT NOT NULL,
                filename TEXT NOT NULL, mimetype TEXT NOT NULL, size_bytes INTEGER NOT NULL,
                content_hash TEXT NOT NULL, status TEXT NOT NULL, chunk_count INTEGER NOT NULL DEFAULT 0,
                error TEXT, created_at TEXT, updated_at TEXT, PRIMARY KEY (project_id, doc_id)
            );
            INSERT INTO documents VALUES ('support', 'doc-1', 'baggage.md', 'baggage.md', 'text/markdown', 42, 'abc123', 'indexed', 3, NULL, NULL, NULL);
            """
        )

    registry = SqliteDocumentRegistry(path)
    kept = await registry.get(project_id="support", doc_id="doc-1")

    assert kept is not None and kept.chunk_count == 3
    assert kept.chunking_strategy is None
    await registry.upsert(
        kept.model_copy(
            update={
                "chunking_strategy": "size",
                "chunk_size": 1000,
                "chunk_overlap": 200,
            }
        )
    )
    assert (await registry.get(project_id="support", doc_id="doc-1")).chunk_size == 1000
