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
