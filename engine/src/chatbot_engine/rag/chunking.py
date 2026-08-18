from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """Represent one searchable piece of an extracted document.

    Attributes:
        content: Text contained in this chunk.
        index: Position of the chunk inside the original document.
    """

    content: str
    index: int


class TextChunker:
    """Split extracted document text into overlapping chunks."""

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 150,
    ) -> None:
        """Initialize the chunker.

        Args:
            chunk_size: Maximum number of characters in each chunk.
            overlap: Number of characters shared between neighboring chunks.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        if overlap < 0:
            raise ValueError("overlap cannot be negative")

        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self._chunk_size = chunk_size
        self._overlap = overlap

    def chunk(
        self,
        text: str,
    ) -> list[DocumentChunk]:
        """Split one extracted document into searchable chunks.

        Args:
            text: Extracted plain text from one document.

        Returns:
            Ordered chunks created from the document text.
        """
        if not text.strip():
            return []

        chunks: list[DocumentChunk] = []
        start = 0
        index = 0

        while start < len(text):
            end = min(
                start + self._chunk_size,
                len(text),
            )

            content = text[start:end].strip()

            if content:
                chunks.append(
                    DocumentChunk(
                        content=content,
                        index=index,
                    )
                )
                index += 1

            if end == len(text):
                break

            start = end - self._overlap

        return chunks