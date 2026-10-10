"""Extract searchable text from uploaded documents.

The ingestion pipeline uses these extractors to normalize supported document
types into plain text and reject MIME types that cannot be indexed. A reader
that needs only the start of a document (a file sent in a chat) gives
`max_chars`, and the extractor stops once it has that much.
"""

import re
from io import BytesIO
from typing import Any, Protocol

from pypdf import PageObject, PdfReader
from pypdf.generic import IndirectObject

from chatbot_engine.documents.models import ExtractedDocument


class DocumentExtractor(Protocol):
    """Define how raw document bytes are converted into extracted text."""

    def extract_text(
        self,
        *,
        data: bytes,
        mimetype: str,
        max_chars: int | None = None,
    ) -> ExtractedDocument:
        """Extract text from one uploaded document.

        Args:
            data: Raw bytes of the original uploaded document.
            mimetype: MIME type describing the document format.
            max_chars: Stop once this much text is gathered; the rest of the
                document is not read. None reads it all.
        """
        ...


class UnsupportedDocumentTypeError(ValueError):
    """Raised when no extractor supports the provided document MIME type."""


class TextDocumentExtractor(DocumentExtractor):
    """Extract UTF-8 text from plain-text and Markdown documents."""

    def extract_text(
        self,
        *,
        data: bytes,
        mimetype: str,
        max_chars: int | None = None,
    ) -> ExtractedDocument:
        """Decode one text-based document into normalized text.

        Args:
            data: Raw bytes of the uploaded text-based document.
            mimetype: MIME type of the uploaded document.
            max_chars: Keep only this much of the text.
        """
        text = data.decode("utf-8")
        truncated = max_chars is not None and len(text) > max_chars
        if truncated:
            text = text[:max_chars]
        return ExtractedDocument(text=text, truncated=truncated)


#: An operator that can put text on a page: `BT` opens a text object, and
#: `Do` draws a form, which may hold one. A page with neither shows no text,
#: however long its content, so it is not parsed. A token stands between
#: PDF delimiters.
_SHOWS_TEXT = re.compile(
    rb"(?<![^\x00\t\n\x0c\r ()<>\[\]{}/%])(?:BT|Do)(?![^\x00\t\n\x0c\r ()<>\[\]{}/%])"
)


#: The most forms looked into for one page, nested ones included.
_MAX_FORMS = 256


def _content_of(page: PageObject, budget: int | None) -> bytes:
    """A page's content, and that of the forms it can draw, inflated; read
    only until it passes `budget` bytes, when one is given."""
    found: list[bytes] = []
    size = 0
    seen: set[object] = set()
    pending: list[Any] = [page]
    while pending and (budget is None or size <= budget):
        holder = pending.pop()
        if holder is page:
            contents = page.get_contents()
            data = contents.get_data() if contents is not None else b""
        else:
            data = holder.get_data()
        found.append(data)
        size += len(data)
        resources = holder.get("/Resources")
        xobjects = resources.get_object().get("/XObject") if resources else None
        for ref in xobjects.get_object().values() if xobjects else ():
            # Known by its object number, so a form that draws itself is
            # looked into once.
            key = (
                (ref.idnum, ref.generation)
                if isinstance(ref, IndirectObject)
                else id(ref)
            )
            form = ref.get_object()
            if key in seen or len(seen) >= _MAX_FORMS:
                continue
            if form.get("/Subtype") == "/Form":
                seen.add(key)
                pending.append(form)
    return b"".join(found)


def _measure(page: PageObject, budget: int | None) -> tuple[bool, int]:
    """Whether a page can show text, and how much content parsing it reads.
    A page whose content cannot be measured is parsed, as every page was."""
    try:
        content = _content_of(page, budget)
    except Exception:
        return True, 0
    return _SHOWS_TEXT.search(content) is not None, len(content)


class PdfDocumentExtractor(DocumentExtractor):
    """Extract text from PDF documents.

    `max_content_bytes` bounds the parse, as `max_chars` bounds the text:
    once the pages parsed hold that much content, the rest is not read. A
    page with no text costs as much to parse as one with it, and `max_chars`
    never stops a reader that finds none (docs/review-2026-10.md, INGEST-12).
    """

    def __init__(self, max_content_bytes: int | None = None) -> None:
        self.max_content_bytes = max_content_bytes

    def extract_text(
        self,
        *,
        data: bytes,
        mimetype: str,
        max_chars: int | None = None,
    ) -> ExtractedDocument:
        """Extract text from the pages of one PDF document.

        Args:
            data: Raw bytes of the uploaded PDF document.
            mimetype: MIME type of the uploaded document.
            max_chars: Stop reading pages once this much text is gathered, so
                a document made to inflate into gigabytes of text costs no
                more than the start of it.
        """
        reader = PdfReader(BytesIO(data))

        # Kept per page as well as joined: the page chunking strategy needs the
        # boundaries, and joining first would destroy them irrecoverably.
        pages: list[str] = []
        gathered = 0
        parsed = 0
        budget = self.max_content_bytes
        truncated = False
        for page in reader.pages:
            # The break between pages counts too, so the joined text never
            # passes the bound; and a page past it is not parsed at all, so
            # the bound bounds the work as well as the text.
            separator = 2 if pages else 0
            room = None if max_chars is None else max_chars - gathered - separator
            if room is not None and room <= 0:
                truncated = True
                break
            shows, size = _measure(page, None if budget is None else budget - parsed)
            if not shows:
                continue
            if budget is not None:
                parsed += size
                if parsed > budget:
                    truncated = True
                    break
            page_text = page.extract_text()
            if not page_text:
                continue
            if room is not None:
                if len(page_text) > room:
                    truncated = True
                    page_text = page_text[:room]
                gathered += len(page_text) + separator
            pages.append(page_text)

        return ExtractedDocument(
            text="\n\n".join(pages),
            pages=tuple(pages),
            truncated=truncated,
            page_count=len(reader.pages),
        )


def select_extractor(
    mimetype: str,
) -> DocumentExtractor:
    """Select an extractor that supports the provided MIME type.

    Args:
        mimetype: MIME type of the uploaded document.

    Raises:
        UnsupportedDocumentTypeError: If the document type is unsupported.
    """
    if mimetype in {
        "text/plain",
        "text/markdown",
    }:
        return TextDocumentExtractor()

    if mimetype == "application/pdf":
        return PdfDocumentExtractor()

    raise UnsupportedDocumentTypeError(f"Unsupported document type: {mimetype}")
