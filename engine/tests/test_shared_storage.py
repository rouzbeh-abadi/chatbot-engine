"""The registry and the original uploads on shared services.

The registry contract is run against both backends. Postgres needs a server:
`ENGINE_TEST_POSTGRES_URL`, or the demo stack's database, and the tests skip
without one. The bucket is served by moto in-process, so those always run.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest

from chatbot_engine.documents.blobs import DocumentBlobs
from chatbot_engine.documents.sqlite_registry import SqliteDocumentRegistry
from chatbot_engine.models.documents import DocumentRecord, IngestStatus

POSTGRES_URL = os.environ.get(
    "ENGINE_TEST_POSTGRES_URL",
    "postgresql://support_agent:support_agent@localhost:5432/support_agent",
)


def _record(**over: object) -> DocumentRecord:
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
    return DocumentRecord(**(base | over))


async def _postgres_reachable() -> bool:
    try:
        import psycopg

        async with await psycopg.AsyncConnection.connect(
            POSTGRES_URL, connect_timeout=2
        ):
            return True
    except Exception:
        return False


@pytest.fixture(params=["sqlite", "postgres"])
async def registry(request, tmp_path):
    if request.param == "sqlite":
        yield SqliteDocumentRegistry(tmp_path / "documents.sqlite3")
        return

    if not await _postgres_reachable():
        pytest.skip(f"no Postgres at {POSTGRES_URL}")
    from chatbot_engine.documents.postgres_registry import (
        TABLE,
        PostgresDocumentRegistry,
    )

    store = PostgresDocumentRegistry(POSTGRES_URL)
    yield store
    # Leave the shared database as it was found.
    import psycopg

    async with await psycopg.AsyncConnection.connect(POSTGRES_URL) as connection:
        await connection.execute(
            f"DELETE FROM {TABLE} WHERE project_id IN ('support', 'sales')"
        )
        await connection.commit()


# --- the registry contract, on both backends ----------------------------------


async def test_a_record_round_trips_intact(registry) -> None:
    """Including the timestamps, which each backend stores its own way."""
    await registry.upsert(_record())

    assert await registry.get(project_id="support", doc_id="doc-1") == _record()


async def test_upsert_replaces_rather_than_duplicates(registry) -> None:
    await registry.upsert(_record(chunk_count=3))
    await registry.upsert(_record(chunk_count=9))

    assert [r.chunk_count for r in await registry.list(project_id="support")] == [9]


async def test_listing_is_scoped_and_ordered(registry) -> None:
    await registry.upsert(_record(doc_id="b", external_id="refunds.md"))
    await registry.upsert(_record(doc_id="a", external_id="baggage.md"))
    await registry.upsert(_record(doc_id="c", project_id="sales"))

    listed = await registry.list(project_id="support")

    assert [r.external_id for r in listed] == ["baggage.md", "refunds.md"]


async def test_delete_reports_whether_it_existed(registry) -> None:
    await registry.upsert(_record())

    assert await registry.delete(project_id="support", doc_id="doc-1") is True
    assert await registry.delete(project_id="support", doc_id="doc-1") is False
    assert await registry.get(project_id="support", doc_id="doc-1") is None


# --- the bucket ---------------------------------------------------------------


@pytest.fixture
def bucket(monkeypatch):
    from moto import mock_aws

    from chatbot_engine.documents.s3_storage import S3BlobStore

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    with mock_aws():
        store = S3BlobStore(bucket="uploads", prefix="engine", region="us-east-1")
        store.ensure_bucket()
        yield store


async def test_uploads_round_trip_through_the_bucket(bucket) -> None:
    blobs = DocumentBlobs(store=bucket)

    uri = await blobs.write(doc_id="abc123", data=b"hello", mimetype="text/plain")

    assert uri == "s3://uploads/engine/abc123"
    assert await blobs.read(doc_id="abc123") == b"hello"


async def test_the_uri_is_recomputed_from_the_id_alone(bucket) -> None:
    """Nothing stores the URI: a replica that never wrote the document can
    still read and delete it by id."""
    writer = DocumentBlobs(store=bucket)
    await writer.write(doc_id="abc123", data=b"hello", mimetype="text/plain")

    reader = DocumentBlobs(store=bucket)
    assert await reader.read(doc_id="abc123") == b"hello"

    await reader.delete(doc_id="abc123")
    with pytest.raises(Exception):  # noqa: B017 - boto raises its own ClientError subclass
        await reader.read(doc_id="abc123")


async def test_ensure_bucket_creates_a_missing_one_and_tolerates_an_existing_one(
    bucket,
) -> None:
    bucket.ensure_bucket()
    bucket.ensure_bucket()

    assert bucket._client.head_bucket(Bucket="uploads")


def test_the_bucket_store_is_selected_by_the_setting(monkeypatch) -> None:
    from chatbot_engine.api.dependencies import get_blob_store, reset_dependency_cache
    from chatbot_engine.documents import s3_storage

    created: dict = {}

    class Fake(s3_storage.S3BlobStore):
        def __init__(self, **kw):
            created.update(kw)

        def ensure_bucket(self):
            created["ensured"] = True

        def uri_for(self, key):
            return f"s3://{created['bucket']}/{key}"

    monkeypatch.setattr(s3_storage, "S3BlobStore", Fake)
    monkeypatch.setenv("ENGINE_BLOB_S3_BUCKET", "uploads")
    monkeypatch.setenv("ENGINE_BLOB_S3_ENDPOINT_URL", "http://minio:9000")
    reset_dependency_cache()

    blobs = get_blob_store()

    assert created["bucket"] == "uploads"
    assert created["endpoint_url"] == "http://minio:9000"
    assert created["ensured"] is True
    assert blobs._uri("x") == "s3://uploads/x"
    reset_dependency_cache()
