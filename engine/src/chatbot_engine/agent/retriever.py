"""Find the chunks most like the question."""

from __future__ import annotations

from chatbot_engine.agent.client import build_chat_model
from chatbot_engine.models.chat import ChatRequest
from chatbot_engine.models.events import SourceRef
from chatbot_engine.rag.vector_store import load_vector_store
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

Hit = tuple[Document, float]

REWRITE_SYSTEM = """\
Rewrite the user's latest message into standalone search queries for a knowledge
base.

- Resolve references using the conversation: pronouns like "that", "it" or "the
  second one" become the thing they point to.
- If the message asks about several separate things, write one query for each.
  Otherwise write a single query.
- Keep to what the user asked. Do not add topics they did not raise.
- Put each query on its own line, and output nothing else: no numbering, quotes,
  or preamble.
- If the latest message already stands on its own, return it unchanged."""


async def rewrite_queries(request: ChatRequest) -> list[str]:
    """Turn a follow-up into one or more self-contained search queries.

    A follow-up like "and what about the taxes?" retrieves badly on its own
    words; with earlier turns to lean on, the model rewrites it into something
    the vector search can match, and splits a message that asks several things
    into one query each. Skipped when there is no history, so a first-turn
    question costs no extra model call.
    """
    if not request.history:
        return [request.message]

    history = "\n".join(
        f"{turn.role}: {turn.content}" for turn in request.history
    )
    model = build_chat_model(
        request.project.model_copy(update={"temperature": 0.0})
    )
    reply = await model.ainvoke(
        [
            SystemMessage(REWRITE_SYSTEM),
            HumanMessage(
                f"Conversation so far:\n{history}\n\n"
                f"Latest message: {request.message}\n\nSearch queries:"
            ),
        ]
    )

    queries = [line.strip() for line in str(reply.content).splitlines()]
    return [query for query in queries if query] or [request.message]


async def retrieve(request: ChatRequest) -> list[Hit]:
    """Search this project's chunks for the question.

    The message is first rewritten against the history, so a follow-up searches
    on what it means rather than the few words it was typed with, and a message
    asking several things searches for each and merges the results.

    The `project_id` filter is not optional: without it one project's documents
    would answer another project's questions, with no error to notice.

    Uses `load_vector_store`, so an unseeded engine raises instead of answering
    from an empty store.
    """
    queries = await rewrite_queries(request)
    store = load_vector_store()
    top_k = request.project.top_k
    where = {"project_id": request.project.project_id}

    if len(queries) == 1:
        return await store.asimilarity_search_with_score(
            queries[0], k=top_k, filter=where
        )

    # Several questions: search each, then keep every chunk once at its best
    # (lowest) distance, and return the closest `top_k` across them all.
    best: dict[str, Hit] = {}
    for query in queries:
        hits = await store.asimilarity_search_with_score(
            query, k=top_k, filter=where
        )
        for document, distance in hits:
            key = document.page_content
            if key not in best or distance < best[key][1]:
                best[key] = (document, distance)

    return sorted(best.values(), key=lambda hit: hit[1])[:top_k]


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
