"""Extract searchable text from uploaded documents.

The ingestion pipeline uses these extractors to normalize supported document
types into plain text and reject MIME types that cannot be indexed. A reader
that needs only the start of a document (a file sent in a chat) gives
`max_chars`, and the extractor stops once it has that much.
"""

from io import BytesIO
from typing import Protocol

from pypdf import PdfReader

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


class PdfDocumentExtractor(DocumentExtractor):
    """Extract text from PDF documents."""

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
