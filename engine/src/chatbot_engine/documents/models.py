from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """Represent normalized text extracted from one uploaded document.

    Attributes:
        text: Plain text extracted from the original document.
        pages: The same text split at the original page boundaries, when the
            format has pages. Empty for formats that do not (plain text,
            Markdown). Kept alongside `text` rather than replacing it, because
            most chunking strategies want the whole document and only the
            page strategy needs the boundaries.
        truncated: Whether the reading stopped at a bound the caller gave
            (`max_chars`), so `text` is the start of the document, not all
            of it.
        page_count: How many pages the document has, for a format with pages,
            read or not; 0 otherwise.
    """

    text: str
    pages: tuple[str, ...] = field(default=())
    truncated: bool = False
    page_count: int = 0
