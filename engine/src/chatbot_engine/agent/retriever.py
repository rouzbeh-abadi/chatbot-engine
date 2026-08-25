"""Find the chunks most like the question."""

from __future__ import annotations

from chatbot_engine.models.chat import ChatRequest
from chatbot_engine.models.events import SourceRef
from chatbot_engine.rag.vector_store import load_vector_store
from langchain_core.documents import Document

Hit = tuple[Document, float]


async def retrieve(request: ChatRequest) -> list[Hit]:
    """Search this project's chunks for the question.

    The `project_id` filter is not optional: without it one project's documents
    would answer another project's questions, with no error to notice.

    Uses `load_vector_store`, so an unseeded engine raises instead of answering
    from an empty store.
    """
    return await load_vector_store().asimilarity_search_with_score(
        request.message,
        k=request.project.top_k,
        filter={"project_id": request.project.project_id},
    )


def to_source_refs(hits: list[Hit]) -> list[SourceRef]:
    """Describe each hit for the UI to cite."""
    return [
        SourceRef(
            doc_id=chunk.metadata.get("doc_id", ""),
            source=chunk.metadata.get("source", "unknown"),
            score=_similarity(distance),
            excerpt=" ".join(chunk.page_content.split())[:240],
        )
        for chunk, distance in hits
    ]


def to_context(hits: list[Hit]) -> str:
    """Number the chunks, so the model can cite one by number.

    The numbers line up with the order of `to_source_refs`, which is what lets
    the UI turn a `[2]` in the answer into a chip naming the file.
    """
    return "\n\n".join(
        f"[{index}] {chunk.metadata.get('source', 'unknown')}\n{chunk.page_content}"
        for index, (chunk, _) in enumerate(hits, start=1)
    )


def _similarity(distance: float) -> float:
    """Chroma returns cosine distance, the UI shows a percentage.

    Distance runs 0 (identical) to 2 (opposite), so halve it and invert.
    """
    return max(0.0, 1.0 - distance / 2.0)
