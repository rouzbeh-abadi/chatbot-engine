from chatbot_engine.settings import get_settings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    """Split LangChain documents into overlapping chunks for RAG indexing."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        """Create a recursive text splitter."""
        settings = get_settings()

        if chunk_size is None:
            chunk_size = settings.chunk_size
        if chunk_overlap is None:
            chunk_overlap = settings.chunk_overlap

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )

    def chunk(
        self,
        documents: list[Document],
    ) -> list[Document]:
        """Split documents while preserving their metadata."""
        return self._splitter.split_documents(documents)
