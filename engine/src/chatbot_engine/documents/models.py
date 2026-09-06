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
    """

    text: str
    pages: tuple[str, ...] = field(default=())
