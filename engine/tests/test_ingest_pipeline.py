"""The pipeline directly, for what HTTP cannot see.

The chunks are dropped until a vector store exists, so the metadata on them --
the basis of every later citation -- is only observable from in here.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from test_attachments import _pdf_pages

from chatbot_engine.documents.models import ExtractedDocument
from chatbot_engine.documents.sqlite_registry import SqliteDocumentRegistry
from chatbot_engine.errors import DocumentRejectedError
from chatbot_engine.rag import pipeline as pipeline_module
from chatbot_engine.rag.pipeline import DocumentIngestPipeline, doc_id_for
from chatbot_engine.rag.splitter import DocumentChunker
from chatbot_engine.rag.vector_store import (
    ChromaChunkStore,
    count_chunks,
    open_vector_store,
)
from chatbot_engine.settings import get_settings

TEXT = ("Cabin baggage is one bag up to 8 kg. " * 12).encode()


class RecordingChunker(DocumentChunker):
    """The real splitter, keeping the chunks it produced for inspection."""

    def __init__(self) -> None:
        super().__init__(chunk_size=80, chunk_overlap=10)
        self.produced: list[Document] = []

    def chunk(
        self, extracted: ExtractedDocument, metadata: dict[str, object]
    ) -> list[Document]:
        self.produced = super().chunk(extracted, metadata)

        return self.produced


def _pipeline() -> tuple[
    DocumentIngestPipeline, RecordingChunker, SqliteDocumentRegistry
]:
    chunker = RecordingChunker()
    registry = SqliteDocumentRegistry(get_settings().registry_db)

    return (
        DocumentIngestPipeline(registry=registry, chunker=chunker),
        chunker,
        registry,
    )


async def _ingest(pipeline: DocumentIngestPipeline, data: bytes = TEXT):
    return await pipeline.ingest(
        project_id="support",
        external_id="policies/baggage.md",
        filename="baggage.md",
        mimetype="text/markdown",
        data=data,
    )


# --- the identifier ----------------------------------------------------------


def test_doc_id_separates_the_two_fields() -> None:
    """Without a separator, ("a", "bc") and ("ab", "c") would be one document."""
    assert doc_id_for("a", "bc") != doc_id_for("ab", "c")


def test_doc_id_is_scoped_by_project() -> None:
    assert doc_id_for("support", "baggage.md") != doc_id_for("sales", "baggage.md")


# --- what the chunks carry ---------------------------------------------------


async def test_every_chunk_knows_which_document_it_came_from() -> None:
    pipeline, chunker, _ = _pipeline()

    record = await _ingest(pipeline)

    assert chunker.produced, "the splitter should have been handed something"
    for chunk in chunker.produced:
        assert chunk.metadata["doc_id"] == record.doc_id
        assert chunk.metadata["project_id"] == "support"
        assert chunk.metadata["source"] == "policies/baggage.md"
        assert chunk.metadata["filename"] == "baggage.md"


# --- doing no work twice -----------------------------------------------------


async def test_identical_bytes_skip_the_splitter_entirely() -> None:
    """The point of the hash check: embedding is what costs money."""
    pipeline, chunker, _ = _pipeline()

    await _ingest(pipeline)
    first_pass = list(chunker.produced)
    chunker.produced = []

    second = await _ingest(pipeline)

    assert second.status == "unchanged"
    assert chunker.produced == [], "the splitter ran again on unchanged bytes"
    assert first_pass, "sanity: the first pass did split something"


async def test_unchanged_is_reported_without_being_stored() -> None:
    """`unchanged` describes the call; the record keeps the status it earned."""
    pipeline, _, registry = _pipeline()

    stored = await _ingest(pipeline)
    await _ingest(pipeline)

    listed: Sequence = await registry.list(project_id="support")
    assert [record.status for record in listed] == [stored.status]


async def test_a_changed_document_keeps_its_original_created_at() -> None:
    pipeline, _, _ = _pipeline()

    first = await _ingest(pipeline)
    second = await _ingest(pipeline, data=TEXT + b" Flexible fares allow two.")

    assert second.created_at == first.created_at
    assert second.updated_at is not None
    assert first.updated_at is not None
    assert second.updated_at >= first.updated_at


# --- chunking is part of what "unchanged" means -------------------------------


async def test_the_record_says_how_the_document_was_cut() -> None:
    """After defaults, so a caller can compare with its own settings."""
    pipeline, _, _ = _pipeline()

    record = await _ingest(pipeline)

    assert (record.chunking_strategy, record.chunk_size, record.chunk_overlap) == (
        "size",
        80,
        10,
    )


async def test_identical_bytes_with_new_chunking_are_cut_again() -> None:
    """Same file, new settings: skipping it would keep the old boundaries."""
    pipeline, _, _ = _pipeline()
    await _ingest(pipeline)

    again = await pipeline.ingest(
        project_id="support",
        external_id="policies/baggage.md",
        filename="baggage.md",
        mimetype="text/markdown",
        data=TEXT,
        chunk_size=200,
        chunk_overlap=20,
    )

    assert again.status != "unchanged"
    assert (again.chunk_size, again.chunk_overlap) == (200, 20)


async def test_a_record_from_before_chunking_was_recorded_stays_current() -> None:
    """An engine upgrade alone must not re-embed a knowledge base."""
    pipeline, chunker, registry = _pipeline()
    stored = await _ingest(pipeline)
    await registry.upsert(
        stored.model_copy(
            update={
                "chunking_strategy": None,
                "chunk_size": None,
                "chunk_overlap": None,
            }
        )
    )
    chunker.produced = []

    second = await _ingest(pipeline)

    assert second.status == "unchanged"
    assert chunker.produced == []


async def test_an_overlap_as_long_as_the_chunk_is_refused() -> None:
    from chatbot_engine.rag.splitter import ChunkingError

    pipeline, _, _ = _pipeline()
    with pytest.raises(ChunkingError, match="chunk_overlap"):
        await pipeline.ingest(
            project_id="support",
            external_id="policies/baggage.md",
            filename="baggage.md",
            mimetype="text/markdown",
            data=TEXT,
            chunk_size=300,
            chunk_overlap=300,
        )


# --- the embedding model is part of what "unchanged" means --------------------


def _embedding_pipeline() -> tuple[DocumentIngestPipeline, SqliteDocumentRegistry]:
    """A pipeline that embeds, into the test's own Chroma (see conftest)."""
    registry = SqliteDocumentRegistry(get_settings().registry_db)
    pipeline = DocumentIngestPipeline(
        registry=registry,
        chunker=DocumentChunker(chunk_size=80, chunk_overlap=10),
        vectors=ChromaChunkStore(),
    )
    return pipeline, registry


async def _ingest_with(
    pipeline: DocumentIngestPipeline, *, data: bytes = TEXT, model: str | None = None
):
    return await pipeline.ingest(
        project_id="support",
        external_id="policies/baggage.md",
        filename="baggage.md",
        mimetype="text/markdown",
        data=data,
        embedding_model=model,
    )


def _outage(self, texts):
    raise RuntimeError("provider down")


async def test_the_record_says_which_model_made_its_vectors() -> None:
    pipeline, _ = _embedding_pipeline()

    record = await _ingest_with(pipeline)

    assert record.status == "indexed"
    assert record.embedding_model == get_settings().embedding_model


async def test_a_document_that_was_not_embedded_names_no_model() -> None:
    """No vectors, so no model made them."""
    pipeline, _, _ = _pipeline()

    assert (await _ingest(pipeline)).embedding_model is None


async def test_identical_bytes_with_another_model_are_embedded_again() -> None:
    pipeline, _ = _embedding_pipeline()
    await _ingest_with(pipeline, model="openai/text-embedding-3-small")

    again = await _ingest_with(pipeline, model="openai/text-embedding-3-large")

    assert again.status == "indexed"
    assert again.embedding_model == "openai/text-embedding-3-large"
    same = await _ingest_with(pipeline, model="openai/text-embedding-3-large")
    assert same.status == "unchanged"


async def test_a_record_from_before_the_model_was_recorded_stays_current() -> None:
    """An engine upgrade alone must not re-embed a knowledge base; a caller
    that names a model gets the document embedded with it."""
    pipeline, registry = _embedding_pipeline()
    stored = await _ingest_with(pipeline)
    await registry.upsert(stored.model_copy(update={"embedding_model": None}))

    assert (await _ingest_with(pipeline)).status == "unchanged"
    named = await _ingest_with(pipeline, model=get_settings().embedding_model)
    assert named.status == "indexed"
    assert named.embedding_model == get_settings().embedding_model


async def test_a_received_document_is_embedded_once_the_engine_can() -> None:
    """Recorded while there was no key to embed with; answered `unchanged`
    after one is set, it would never become searchable."""
    without, _, _ = _pipeline()
    assert (await _ingest(without)).status == "received"

    pipeline, _ = _embedding_pipeline()
    again = await _ingest_with(pipeline)

    assert again.status == "indexed"


async def test_a_failed_re_embedding_keeps_the_version_already_indexed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider outage mid re-index: the record says `failed` and why, and
    the version indexed before still answers."""
    pipeline, registry = _embedding_pipeline()
    first = await _ingest_with(pipeline)
    before = sorted(open_vector_store().get(include=["documents"])["documents"])

    monkeypatch.setattr(DeterministicFakeEmbedding, "embed_documents", _outage)
    with pytest.raises(RuntimeError, match="provider down"):
        await _ingest_with(pipeline, data=TEXT + b" Two bags on Flexible fares.")

    stored = await registry.get(project_id="support", doc_id=first.doc_id)
    assert stored is not None and stored.status == "failed"
    assert stored.error == "provider down"
    assert count_chunks() == first.chunk_count
    assert sorted(open_vector_store().get(include=["documents"])["documents"]) == before


# --- a document is read within bounds -------------------------------------------


async def _ingest_pdf(pipeline: DocumentIngestPipeline, data: bytes):
    return await pipeline.ingest(
        project_id="support",
        external_id="manual.pdf",
        filename="manual.pdf",
        mimetype="application/pdf",
        data=data,
        chunking_strategy="page",
    )


async def test_a_pdf_is_read_in_a_bounded_process_and_indexed_by_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same reader a file sent in a chat has, with the index's bounds."""
    calls: list[dict] = []
    original = pipeline_module.read_bounded

    def spying(data, mimetype, **bounds):
        calls.append(bounds)
        return original(data, mimetype, **bounds)

    monkeypatch.setattr(pipeline_module, "read_bounded", spying)
    pipeline, _, _ = _pipeline()

    record = await _ingest_pdf(pipeline, _pdf_pages(["Cabin bags", "Checked bags"]))

    settings = get_settings()
    assert calls == [
        {
            "max_chars": settings.index_max_chars,
            "timeout_s": settings.index_read_timeout_s,
        }
    ]
    assert record.chunk_count == 2
    assert record.error is None


async def test_a_pdf_that_takes_too_long_is_a_failed_record_that_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENGINE_INDEX_READ_TIMEOUT_S", "0.001")
    get_settings.cache_clear()
    pipeline, _, registry = _pipeline()

    with pytest.raises(DocumentRejectedError, match="took longer than"):
        await _ingest_pdf(pipeline, _pdf_pages(["Cabin bags"]))

    [stored] = await registry.list(project_id="support")
    assert stored.status == "failed"
    assert stored.error is not None and "took longer than 0.001 seconds" in stored.error


async def test_a_pdf_made_to_inflate_is_a_failed_record_that_says_so() -> None:
    """20 KB that would be 20 MB of text, refused where it is, not read."""
    pipeline, _, registry = _pipeline()

    with pytest.raises(DocumentRejectedError, match="inflates past 8 MB"):
        await _ingest_pdf(pipeline, _pdf_pages(["x" * 20_000_000], compress=True))

    [stored] = await registry.list(project_id="support")
    assert stored.status == "failed"


async def test_a_document_past_the_index_cap_is_refused_rather_than_cut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Indexed in part, the rest would look indexed and answer nothing."""
    monkeypatch.setenv("ENGINE_INDEX_MAX_CHARS", "100")
    get_settings.cache_clear()
    pipeline, chunker, _ = _pipeline()

    with pytest.raises(DocumentRejectedError, match="more text than one document"):
        await _ingest(pipeline)

    assert chunker.produced == [], "nothing was cut from the part that was read"
