"""One document's uploads and re-indexes take turns, and each decides what to
keep from the record and original as they are once its turn has come.

Read before its turn, they could be an earlier version's: a failure would
then write that version's record back over the one that answers, a re-index
would rebuild it, and a delete that came while it waited would be undone.
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from chatbot_engine.documents.blobs import DocumentBlobs
from chatbot_engine.documents.sqlite_registry import SqliteDocumentRegistry
from chatbot_engine.errors import DocumentDeletedError, OriginalChangedError
from chatbot_engine.rag.pipeline import DocumentIngestPipeline, doc_id_for
from chatbot_engine.rag.splitter import DocumentChunker
from chatbot_engine.rag.vector_store import ChromaChunkStore, stored_chunks
from chatbot_engine.services.documents import DocumentService
from chatbot_engine.settings import get_settings

V1 = b"# Refunds\n\nVERSION ONE: refunds within fourteen days.\n"
V2 = b"# Refunds\n\nVERSION TWO: refunds within thirty days.\n"
V3 = b"# Refunds\n\nVERSION THREE: refunds within sixty days.\n"
DOC = doc_id_for("support", "refunds.md")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _Held:
    """Embedding that holds VERSION TWO until released, and fails VERSION
    THREE, as a provider outage would."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        real = DeterministicFakeEmbedding.embed_documents

        def embed(fake, texts):
            if any("VERSION TWO" in t for t in texts):
                self.started.set()
                self.release.wait(30)
            if any("VERSION THREE" in t for t in texts):
                raise RuntimeError("provider down")
            return real(fake, texts)

        monkeypatch.setattr(DeterministicFakeEmbedding, "embed_documents", embed)

    async def under_way(self) -> None:
        while not self.started.is_set():
            await asyncio.sleep(0.01)


def _wired(tmp_path: Path):
    registry = SqliteDocumentRegistry(get_settings().registry_db)
    blobs = DocumentBlobs(root=tmp_path / "blobs")
    pipeline = DocumentIngestPipeline(
        registry=registry,
        chunker=DocumentChunker(chunk_size=200, chunk_overlap=0),
        vectors=ChromaChunkStore(),
        blobs=blobs,
    )
    return pipeline, registry, blobs


def _put(pipeline: DocumentIngestPipeline, data: bytes):
    return pipeline.ingest(
        project_id="support",
        external_id="refunds.md",
        filename="refunds.md",
        mimetype="text/markdown",
        data=data,
    )


def _texts() -> list[str]:
    return stored_chunks(where={"doc_id": DOC})["documents"]


async def test_an_update_that_fails_in_its_turn_keeps_the_version_before_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v2 is embedding when v3 is sent; v2 lands, v3 fails. The record goes on
    describing v2, which answers, not v1, which answered when v3 was sent."""
    pipeline, registry, blobs = _wired(tmp_path)
    await _put(pipeline, V1)
    held = _Held(monkeypatch)

    v2 = asyncio.create_task(_put(pipeline, V2))
    await held.under_way()
    v3 = asyncio.create_task(_put(pipeline, V3))
    await asyncio.sleep(0.2)
    held.release.set()
    assert (await v2).status == "indexed"
    with pytest.raises(RuntimeError, match="provider down"):
        await v3

    record = await registry.get(project_id="support", doc_id=DOC)
    assert record is not None and record.content_hash == _sha(V2)
    assert record.status == "indexed" and "provider down" in (record.error or "")
    assert all("VERSION TWO" in t for t in _texts())
    assert await blobs.read(doc_id=DOC) == V2


async def test_a_delete_also_cancels_an_upload_waiting_its_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The delete answers while v2 embeds and v3 waits: neither brings the
    document back."""
    pipeline, registry, blobs = _wired(tmp_path)
    service = DocumentService(
        pipeline=pipeline, registry=registry, vectors=ChromaChunkStore(), blobs=blobs
    )
    await _put(pipeline, V1)
    held = _Held(monkeypatch)

    v2 = asyncio.create_task(_put(pipeline, V2))
    await held.under_way()
    v3 = asyncio.create_task(_put(pipeline, V3))
    await asyncio.sleep(0.2)
    await service.delete(project_id="support", doc_id=DOC)
    held.release.set()
    for upload in (v2, v3):
        with pytest.raises(DocumentDeletedError):
            await upload

    assert await registry.get(project_id="support", doc_id=DOC) is None
    assert _texts() == []
    assert not (tmp_path / "blobs" / DOC).exists()


async def test_a_reindex_waits_for_an_upload_and_rebuilds_what_it_left(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asked for while v2 is embedding, a re-index rebuilds v2, not the v1
    that was indexed when it was asked for (INGEST-3)."""
    pipeline, _, blobs = _wired(tmp_path)
    await _put(pipeline, V1)
    held = _Held(monkeypatch)

    v2 = asyncio.create_task(_put(pipeline, V2))
    await held.under_way()
    rebuilt = asyncio.create_task(pipeline.reindex(project_id="support", doc_id=DOC))
    await asyncio.sleep(0.2)
    held.release.set()
    await v2
    record = await rebuilt

    assert record.content_hash == _sha(V2)
    assert all("VERSION TWO" in t for t in _texts())
    assert await blobs.read(doc_id=DOC) == V2


async def test_uploading_again_restores_an_original_that_is_not_the_records(
    tmp_path: Path,
) -> None:
    """What the 409 of a re-index tells its caller to do works: the same
    bytes, sent again, answer `unchanged` and store the original."""
    pipeline, _, blobs = _wired(tmp_path)
    await _put(pipeline, V1)
    await blobs.write(doc_id=DOC, data=V2, mimetype="text/markdown")

    with pytest.raises(OriginalChangedError):
        await pipeline.reindex(project_id="support", doc_id=DOC)
    assert (await _put(pipeline, V1)).status == "unchanged"

    assert await blobs.read(doc_id=DOC) == V1
    assert (
        await pipeline.reindex(project_id="support", doc_id=DOC)
    ).status == "indexed"


class _SlowRead:
    """A blob store whose read waits, so a delete can come during it."""

    def __init__(self, blobs: DocumentBlobs, monkeypatch: pytest.MonkeyPatch) -> None:
        self.entered = asyncio.Event()
        self.go = asyncio.Event()
        real = blobs.read

        async def read(*, doc_id: str) -> bytes:
            self.entered.set()
            await self.go.wait()
            return await real(doc_id=doc_id)

        monkeypatch.setattr(blobs, "read", read)


@pytest.mark.parametrize("how", ["delete", "purge"])
async def test_an_unchanged_upload_does_not_store_an_original_a_delete_took(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    """The same bytes again, with the stored original missing, while the
    document (or its chatbot) is deleted: no file is left that no record
    names, and the caller is not told the document is there."""
    pipeline, registry, blobs = _wired(tmp_path)
    service = DocumentService(
        pipeline=pipeline, registry=registry, vectors=ChromaChunkStore(), blobs=blobs
    )
    await _put(pipeline, V1)
    await blobs.delete(doc_id=DOC)
    slow = _SlowRead(blobs, monkeypatch)

    again = asyncio.create_task(_put(pipeline, V1))
    await slow.entered.wait()
    if how == "delete":
        await service.delete(project_id="support", doc_id=DOC)
    else:
        await service.purge(project_id="support")
    slow.go.set()
    with pytest.raises(DocumentDeletedError):
        await again

    assert await registry.get(project_id="support", doc_id=DOC) is None
    assert not (tmp_path / "blobs" / DOC).exists()


async def test_sending_the_same_bytes_again_stores_a_missing_original_in_a_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bucket says a missing object its own way; it is missing all the
    same, and the upload stores it, as for a file on disk."""
    from moto import mock_aws

    from chatbot_engine.documents.s3_storage import S3BlobStore

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    with mock_aws():
        store = S3BlobStore(bucket="uploads", prefix="engine", region="us-east-1")
        store.ensure_bucket()
        blobs = DocumentBlobs(store=store)
        pipeline = DocumentIngestPipeline(
            registry=SqliteDocumentRegistry(get_settings().registry_db),
            chunker=DocumentChunker(chunk_size=200, chunk_overlap=0),
            vectors=ChromaChunkStore(),
            blobs=blobs,
        )
        await _put(pipeline, V1)
        await blobs.delete(doc_id=DOC)
        with pytest.raises(OriginalChangedError, match="no original"):
            await pipeline.reindex(project_id="support", doc_id=DOC)

        assert (await _put(pipeline, V1)).status == "unchanged"
        assert await blobs.read(doc_id=DOC) == V1


async def test_a_failure_once_the_new_version_has_landed_records_it_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v2's chunks all land, then removing v1's fails: v2 is what answers, so
    the record must not say v1 stays, and sending v2 again indexes it again
    rather than answering `unchanged`."""
    from chatbot_engine.rag import vector_store

    pipeline, registry, _ = _wired(tmp_path)
    await _put(pipeline, V1)

    def broken(name: str, removed: int) -> None:
        raise RuntimeError("the collection's count could not be written")

    with monkeypatch.context() as patched:
        patched.setattr(vector_store, "_count_deleted", broken)
        with pytest.raises(RuntimeError):
            await _put(pipeline, V2)

    record = await registry.get(project_id="support", doc_id=DOC)
    assert record is not None
    assert record.status == "failed" and record.content_hash == _sha(V2)
    assert (await _put(pipeline, V2)).status == "indexed"
    assert all("VERSION TWO" in t for t in _texts())


async def test_a_slice_embedded_but_not_written_is_counted_as_embedded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The provider bills a slice once it is embedded, written or not."""
    from chatbot_engine.rag import vector_store

    pipeline, registry, _ = _wired(tmp_path)

    def full(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(vector_store, "_write", full)
    with pytest.raises(OSError):
        await _put(pipeline, V1)

    record = await registry.get(project_id="support", doc_id=DOC)
    assert record is not None and record.status == "failed"
    assert record.chunk_count == 1


async def test_the_version_that_answers_sent_again_clears_an_update_that_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, registry, _ = _wired(tmp_path)
    await _put(pipeline, V1)
    _Held(monkeypatch)
    with pytest.raises(RuntimeError):
        await _put(pipeline, V3)
    failed = await registry.get(project_id="support", doc_id=DOC)
    assert failed is not None and failed.error is not None

    again = await _put(pipeline, V1)

    assert again.status == "unchanged" and again.error is None
    stored = await registry.get(project_id="support", doc_id=DOC)
    assert stored is not None and stored.error is None
