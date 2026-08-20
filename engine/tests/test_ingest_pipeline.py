"""The pipeline directly, for what HTTP cannot see.

The chunks are dropped until a vector store exists, so the metadata on them --
the basis of every later citation -- is only observable from in here.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document

from chatbot_engine.documents.registry import InMemoryDocumentRegistry
from chatbot_engine.rag.pipeline import DocumentIngestPipeline, doc_id_for
from chatbot_engine.rag.splitter import DocumentChunker

TEXT = ("Cabin baggage is one bag up to 8 kg. " * 12).encode()


class RecordingChunker(DocumentChunker):
    """The real splitter, keeping the chunks it produced for inspection."""

    def __init__(self) -> None:
        super().__init__(chunk_size=80, chunk_overlap=10)
        self.produced: list[Document] = []

    def chunk(self, documents: list[Document]) -> list[Document]:
        self.produced = super().chunk(documents)

        return self.produced


def _pipeline() -> tuple[
    DocumentIngestPipeline, RecordingChunker, InMemoryDocumentRegistry
]:
    chunker = RecordingChunker()
    registry = InMemoryDocumentRegistry()

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


def test_doc_id_is_stable_for_the_same_inputs() -> None:
    assert doc_id_for("support", "baggage.md") == doc_id_for("support", "baggage.md")


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


async def test_every_chunk_knows_where_it_sits_in_the_file() -> None:
    """A citation points at a place in a file, not just the file."""
    pipeline, chunker, _ = _pipeline()

    await _ingest(pipeline)

    offsets = [chunk.metadata["start_index"] for chunk in chunker.produced]
    assert offsets == sorted(offsets)
    assert offsets[0] == 0


async def test_the_reported_count_matches_the_chunks_produced() -> None:
    pipeline, chunker, _ = _pipeline()

    record = await _ingest(pipeline)

    assert record.chunk_count == len(chunker.produced)


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
    pipeline, _, registry = _pipeline()

    first = await _ingest(pipeline)
    second = await _ingest(pipeline, data=TEXT + b" Flexible fares allow two.")

    assert second.created_at == first.created_at
    assert second.updated_at is not None
    assert first.updated_at is not None
    assert second.updated_at >= first.updated_at
