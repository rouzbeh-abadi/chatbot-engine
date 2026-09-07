"""Chunk extracted documents for RAG indexing.

How a document is cut up decides what retrieval can find, and the right answer
depends on the document. A policy manual with headings retrieves better when a
chunk is a section; a scanned PDF retrieves better when a chunk never spans a
page. So the strategy is configuration, not a constant.

Every strategy ends with the same size cap. Splitting on structure alone leaves
whatever the author wrote -- a forty-page chapter is one "chunk" -- which
embeds badly and drowns the answer in irrelevant text. Structure decides the
boundaries; the size cap keeps the pieces usable.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from chatbot_engine.documents.models import ExtractedDocument

#: How a document is cut into chunks.
#:
#: - `size`: fixed-length windows with overlap. Works on anything, respects no
#:   structure. The safe default.
#: - `headings`: split at Markdown headings, so a chunk is a section and
#:   carries its heading trail. Only meaningful for Markdown; falls back to
#:   `size` when the text has no headings.
#: - `page`: never let a chunk span a page boundary, and label each with its
#:   page number. Only meaningful for paged formats (PDF); falls back to `size`
#:   when the document has no pages.
from chatbot_engine.models.documents import ChunkStrategy
from chatbot_engine.settings import get_settings

#: The same values at runtime, to check a string that came off the wire.
CHUNK_STRATEGIES: frozenset[str] = frozenset(("size", "headings", "page"))

#: The heading levels `headings` splits on. Deeper levels stay inside the
#: section: splitting on every `####` would produce chunks of a sentence or two.
_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


class DocumentChunker:
    """Split an extracted document into chunks, by the configured strategy."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        strategy: ChunkStrategy | None = None,
    ) -> None:
        settings = get_settings()

        self._strategy: ChunkStrategy = strategy or settings.chunk_strategy
        if self._strategy not in CHUNK_STRATEGIES:
            # Falling through to `size` would silently index the document a
            # different way than the caller asked for -- the kind of wrong that
            # only shows up later as poor retrieval.
            raise ValueError(
                f"unknown chunking strategy {self._strategy!r}; "
                f"expected one of {sorted(CHUNK_STRATEGIES)}"
            )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size if chunk_size is not None else settings.chunk_size,
            chunk_overlap=(
                chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
            ),
            add_start_index=True,
        )

    @property
    def strategy(self) -> ChunkStrategy:
        return self._strategy

    def chunk(
        self,
        extracted: ExtractedDocument,
        metadata: dict[str, object],
    ) -> list[Document]:
        """Cut one extracted document into chunks, each carrying `metadata`.

        `metadata` is the document's identity (doc_id, project_id, source,
        filename); a strategy may add to it -- the heading trail, or the page
        number -- but never removes from it, because citations are built from it.
        """
        if self._strategy == "page":
            return self._by_page(extracted, metadata)
        if self._strategy == "headings":
            return self._by_headings(extracted, metadata)

        return self._by_size(extracted.text, metadata)

    def _by_size(self, text: str, metadata: dict[str, object]) -> list[Document]:
        """Fixed-length windows. Also the last step of every other strategy."""
        return self._splitter.split_documents(
            [Document(page_content=text, metadata=dict(metadata))]
        )

    def _by_page(
        self, extracted: ExtractedDocument, metadata: dict[str, object]
    ) -> list[Document]:
        """One page at a time, so no chunk straddles a page boundary.

        Long pages are still size-split, but only within the page, so every
        chunk can name the page it came from -- which is what makes a citation
        useful for a PDF.
        """
        if not extracted.pages:
            # Not a paged format. Cutting arbitrary text into "pages" would be
            # a lie, so fall back rather than invent boundaries.
            return self._by_size(extracted.text, metadata)

        chunks: list[Document] = []
        for number, page_text in enumerate(extracted.pages, start=1):
            if not page_text.strip():
                continue
            chunks.extend(self._by_size(page_text, {**metadata, "page": number}))

        return chunks

    def _by_headings(
        self, extracted: ExtractedDocument, metadata: dict[str, object]
    ) -> list[Document]:
        """Split at Markdown headings, keeping the heading trail on each chunk."""
        sections = MarkdownHeaderTextSplitter(
            headers_to_split_on=_HEADERS, strip_headers=False
        ).split_text(extracted.text)

        # A document with no headings yields a single section -- the whole text.
        # That is not a heading split, so treat it as the size strategy instead.
        if len(sections) <= 1:
            return self._by_size(extracted.text, metadata)

        chunks: list[Document] = []
        for section in sections:
            # The splitter puts the heading trail in metadata; keep it, so a
            # chunk knows which section it came from.
            merged = {**metadata, **section.metadata}
            chunks.extend(self._by_size(section.page_content, merged))

        return chunks
