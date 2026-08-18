from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """Represent normalized text extracted from one uploaded document.

    Attributes:
        text: Plain text extracted from the original document.
    """

    text: str