"""Chunk extracted documents for RAG indexing.

How a document is cut up decides what retrieval can find, and the right answer
depends on the document. A policy manual with headings retrieves better when a
chunk is a section; a scanned PDF retrieves better when a chunk never spans a
page. So the strategy is configuration, not a constant.

Every strategy ends with the same size cap. Splitting on structure alone leaves
whatever the author wrote -- a forty-page chapter is one "chunk" -- which
embeds badly and drowns the answer in irrelevant text. Structure decides the
boundaries; the size cap keeps the pieces usable.

A Markdown heading is never a chunk on its own. Cut off from its text (as one
followed by a paragraph longer than the size cap is), it carries a title and
nothing else, takes a place among the chunks a question retrieves, and is
cited with nothing to show, while the text it introduces loses its title. Such
a heading goes to the start of what follows it instead.
"""

from __future__ import annotations

import re
from itertools import pairwise

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


class ChunkingError(ValueError):
    """Chunking settings that cannot work together, such as an overlap as
    long as the chunk. The caller's mistake: the routes answer 422."""


#: The heading levels `headings` splits on. Deeper levels stay inside the
#: section: splitting on every `####` would produce chunks of a sentence or two.
_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]

#: A Markdown heading line at any level: up to three spaces, one to six `#`,
#: then its title or nothing. The trailing spaces the header splitter leaves at
#: the end of a line are allowed.
_HEADING_LINE = re.compile(r" {0,3}#{1,6}(?:[ \t].*)?")


def _only_headings(text: str) -> bool:
    """Whether every line of `text` that holds anything is a heading."""
    lines = [line for line in text.splitlines() if line.strip()]
    return bool(lines) and all(_HEADING_LINE.fullmatch(line) for line in lines)


def _fold_headings(pieces: list[Document], text: str) -> list[Document]:
    """`pieces`, cut from `text` in order, with each run of pieces that are
    only headings joined to the start of the piece after it.

    The joined piece starts where the heading did and may run a heading's
    length past the size cap. A heading with nothing after it joins the piece
    before it instead, and a text of nothing but headings keeps them as cut.
    """
    folded: list[Document] = []
    waiting: list[Document] = []
    for piece in pieces:
        if _only_headings(piece.page_content):
            waiting.append(piece)
            continue
        folded.append(_joined([*waiting, piece], text) if waiting else piece)
        waiting = []

    if waiting and folded:
        folded[-1] = _joined([folded[-1], *waiting], text)
    elif waiting:
        folded = waiting

    return folded


def _joined(pieces: list[Document], text: str) -> Document:
    """Consecutive pieces of `text` as one, with the first one's metadata."""
    parts = [pieces[0].page_content]
    for before, after in pairwise(pieces):
        parts.extend((_between(before, after, text), after.page_content))

    return Document(page_content="".join(parts), metadata=dict(pieces[0].metadata))


def _between(before: Document, after: Document, text: str) -> str:
    """What separates two consecutive pieces in `text`: the whitespace between
    them as written, or a blank line when their places do not say."""
    start = before.metadata.get("start_index")
    then = after.metadata.get("start_index")
    if isinstance(start, int) and isinstance(then, int) and start >= 0:
        end = start + len(before.page_content)
        gap = text[end:then]
        if end <= then and not gap.strip():
            return gap
    return "\n\n"


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
        self._size = chunk_size if chunk_size is not None else settings.chunk_size
        self._overlap = (
            chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        )
        # At most half: each piece then starts at least half a piece after
        # the one before, so a document is embedded at most about twice.
        # Just under the size, text with no spaces moved one character per
        # piece, a hundred times the document (docs/review-2026-10.md,
        # INGEST-1).
        if self._overlap * 2 > self._size:
            raise ChunkingError(
                f"chunk_overlap ({self._overlap}) can be at most half of "
                f"chunk_size ({self._size})"
            )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._size,
            chunk_overlap=self._overlap,
            add_start_index=True,
        )

    @property
    def strategy(self) -> ChunkStrategy:
        return self._strategy

    @property
    def size(self) -> int:
        """The size cap in characters, after defaults."""
        return self._size

    @property
    def overlap(self) -> int:
        """The overlap between neighbouring chunks in characters, after defaults."""
        return self._overlap

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
        """Fixed-length windows. Also the last step of every other strategy.

        A window that holds only headings joins the one after it, the text
        it introduces (`_fold_headings`).
        """
        pieces = self._splitter.split_documents(
            [Document(page_content=text, metadata=dict(metadata))]
        )
        return _fold_headings(pieces, text)

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

        # A section that is only headings (one whose heading is followed
        # straight away by the next at its level, say) has no text of its own:
        # it goes to the start of the section after it, or, at the very end,
        # to the one before it. The header splitter ends each line it joins
        # with two spaces; a folded heading is joined the same way.
        texts: list[tuple[str, dict[str, object]]] = []
        waiting: list[str] = []
        for section in sections:
            if _only_headings(section.page_content):
                waiting.append(section.page_content)
                continue
            # The splitter puts the heading trail in metadata; keep it, so a
            # chunk knows which section it came from.
            merged = {**metadata, **section.metadata}
            texts.append(("  \n".join([*waiting, section.page_content]), merged))
            waiting = []

        if not texts:
            # Nothing but headings: no section to give them to.
            return self._by_size(extracted.text, metadata)
        if waiting:
            text, merged = texts[-1]
            texts[-1] = ("  \n".join([text, *waiting]), merged)

        chunks: list[Document] = []
        for text, merged in texts:
            chunks.extend(self._by_size(text, merged))

        return chunks
