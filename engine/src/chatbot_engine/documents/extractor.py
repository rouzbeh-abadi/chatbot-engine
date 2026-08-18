from io import BytesIO
from typing import Protocol

from chatbot_engine.documents.models import ExtractedDocument
from pypdf import PdfReader


class DocumentExtractor(Protocol):
    """Define how raw document bytes are converted into extracted text."""

    def extract_text(
        self,
        *,
        data: bytes,
        mimetype: str,
    ) -> ExtractedDocument:
        """Extract text from one uploaded document.

        Args:
            data: Raw bytes of the original uploaded document.
            mimetype: MIME type describing the document format.
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
    ) -> ExtractedDocument:
        """Decode one text-based document into normalized text.

        Args:
            data: Raw bytes of the uploaded text-based document.
            mimetype: MIME type of the uploaded document.
        """
        return ExtractedDocument(
            text=data.decode("utf-8"),
        )


class PdfDocumentExtractor(DocumentExtractor):
    """Extract text from PDF documents."""

    def extract_text(
        self,
        *,
        data: bytes,
        mimetype: str,
    ) -> ExtractedDocument:
        """Extract text from all pages of one PDF document.

        Args:
            data: Raw bytes of the uploaded PDF document.
            mimetype: MIME type of the uploaded document.
        """
        reader = PdfReader(BytesIO(data))

        text = "\n\n".join(
            page_text
            for page in reader.pages
            if (page_text := page.extract_text())
        )

        return ExtractedDocument(text=text)


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

    raise UnsupportedDocumentTypeError(
        f"Unsupported document type: {mimetype}"
    )