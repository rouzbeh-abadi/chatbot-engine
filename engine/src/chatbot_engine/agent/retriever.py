"""Find the chunks most like the question.

Retrieval is a pipeline with two configurable stages:

    rewrite the question against the history
      -> for each query: vector search, and keyword search when `hybrid`
      -> fuse the rankings (reciprocal rank fusion)
      -> merge across queries, best score per chunk
      -> rerank with the model, when enabled
      -> keep the top `top_k`

Vector search finds what a chunk is about; keyword search finds exact terms
the embedding blurs. Fusing the two is what lets "is the Basic fare
refundable?" find the chunk that names the Basic fare rather than one that
discusses refunds in general. Reranking then lets the model judge relevance
directly, at the cost of one call, for assistants where that is worth it.

Every hit carries a score in [0, 1], higher is better, whatever produced it,
so a citation's confidence means the same thing in every mode.

The rewrite and the rerank are model calls. `retrieve_with_usage` returns
their token counts alongside the hits, so an agent can fold them into the
turn's reported usage; `retrieve` is the same without the bookkeeping.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from chatbot_engine.agent.client import (
    Totals,
    add_usage,
    build_chat_model,
    empty_totals,
)
from chatbot_engine.models.chat import AssistantConfig, ChatRequest
from chatbot_engine.models.events import SourceRef
from chatbot_engine.rag import sparse
from chatbot_engine.rag.rerank import rerank
from chatbot_engine.rag.vector_store import load_vector_store
from chatbot_engine.settings import get_settings
from chatbot_engine.tracing import run_config

#: A retrieved chunk and its score: 1.0 is the best match in the result set.
Hit = tuple[Document, float]

#: The constant in reciprocal rank fusion. 60 is the value from the original
#: paper and the one every implementation uses; it damps the advantage of a
#: first-place rank so that a chunk ranked well by both searches beats one
#: ranked first by a single search.
RRF_K = 60

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


def utility_config(project: AssistantConfig) -> AssistantConfig:
    """The assistant config for the small calls a turn makes around the answer.

    `ENGINE_UTILITY_MODEL` swaps in a cheaper model for the rewrite and the
    rerank; neither writes a word the customer reads. Temperature zero, since
    both want the same output for the same input. The assistant's
    `max_output_tokens` is for the answer the customer reads; a tight cap
    there must not truncate a rerank list, so it is lifted here.
    """
    settings = get_settings()
    return project.model_copy(
        update={
            "model": settings.utility_model or project.model,
            "temperature": 0.0,
            "max_output_tokens": None,
        }
    )


async def rewrite_queries(
    request: ChatRequest, totals: Totals | None = None
) -> list[str]:
    """Turn a follow-up into one or more self-contained search queries.

    A follow-up like "and what about the taxes?" retrieves badly on its own
    words; with earlier turns to lean on, the model rewrites it into something
    the search can match, and splits a message that asks several things into
    one query each. Skipped when there is no history, so a first-turn question
    costs no extra model call.
    """
    if not request.history:
        return [request.message]

    history = "\n".join(f"{turn.role}: {turn.content}" for turn in request.history)
    model = build_chat_model(utility_config(request.project))
    reply = await model.ainvoke(
        [
            SystemMessage(REWRITE_SYSTEM),
            HumanMessage(
                f"Conversation so far:\n{history}\n\n"
                f"Latest message: {request.message}\n\nSearch queries:"
            ),
        ],
        config=run_config(request, name="rewrite"),
    )

    if totals is not None:
        add_usage(totals, reply)

    queries = [line.strip() for line in str(reply.content).splitlines()]
    return [query for query in queries if query] or [request.message]


async def retrieve(request: ChatRequest) -> list[Hit]:
    """Search this project's chunks for the question. See `retrieve_with_usage`."""
    hits, _ = await retrieve_with_usage(request)
    return hits


async def retrieve_with_usage(
    request: ChatRequest, *, queries: list[str] | None = None
) -> tuple[list[Hit], Totals]:
    """Search this project's chunks for the question, and count what it cost.

    The `project_id` filter is not optional: without it one project's documents
    would answer another project's questions, with no error to notice.

    Uses `load_vector_store`, so an unseeded engine raises instead of answering
    from an empty store.

    `queries` are the search queries to run, for a caller that has already
    rewritten the message with `rewrite_queries` and wants to reuse the result
    rather than pay for the rewrite twice. Left out, the rewrite happens here.
    """
    project = request.project
    totals = empty_totals()
    settings = get_settings()
    mode = project.retrieval or settings.retrieval
    candidates = project.retrieval_candidates or settings.retrieval_candidates
    do_rerank = project.rerank if project.rerank is not None else settings.rerank

    store = load_vector_store(project.embedding_model)
    if queries is None:
        queries = await rewrite_queries(request, totals)

    # Across queries, a chunk keeps its best score. Keyed by content: the same
    # chunk can come back from both searches and from several queries.
    best: dict[str, Hit] = {}
    for query in queries:
        dense = await _dense(store, project.project_id, query, candidates)
        if mode == "hybrid":
            keyword = sparse.sparse_index(store, project.project_id).search(
                query, candidates
            )
            fused = _fuse(dense, keyword)
        else:
            fused = _normalise(dense)

        for document, score in fused:
            key = document.page_content
            if key not in best or score > best[key][1]:
                best[key] = (document, score)

    ranked = sorted(best.values(), key=lambda hit: hit[1], reverse=True)
    ranked = ranked[: max(candidates, project.top_k)]

    if do_rerank:
        # The reranker orders; the fused score stays as the evidence for each
        # chunk, so a citation's confidence is not an artefact of its position.
        by_content = {document.page_content: score for document, score in ranked}
        ordered = await rerank(
            request, queries[0], [document for document, _ in ranked], totals
        )
        ranked = [(document, by_content[document.page_content]) for document in ordered]

    return ranked[: project.top_k], totals


async def _dense(
    store, project_id: str, query: str, k: int
) -> list[tuple[Document, float]]:
    """Vector search: chunks with their cosine similarity, best first."""
    hits = await store.asimilarity_search_with_score(
        query, k=k, filter={"project_id": project_id}
    )
    return [(document, _similarity(distance)) for document, distance in hits]


def _fuse(*rankings: list[tuple[Document, float]]) -> list[Hit]:
    """Reciprocal rank fusion, normalised so the best chunk scores 1.0.

    Each ranking contributes 1 / (RRF_K + rank) per chunk. Ranks rather than
    scores, because a cosine similarity and a BM25 score are not on a common
    scale and any attempt to put them on one is a guess.
    """
    fused: dict[str, float] = {}
    documents: dict[str, Document] = {}

    for ranking in rankings:
        for rank, (document, _) in enumerate(ranking, start=1):
            key = document.page_content
            documents[key] = document
            fused[key] = fused.get(key, 0.0) + 1.0 / (RRF_K + rank)

    if not fused:
        return []
    top = max(fused.values())
    return sorted(
        ((documents[key], score / top) for key, score in fused.items()),
        key=lambda hit: hit[1],
        reverse=True,
    )


def _normalise(dense: list[tuple[Document, float]]) -> list[Hit]:
    """Vector-only results, already scored in [0, 1] and ordered."""
    return list(dense)


def to_source_refs(hits: list[Hit]) -> list[SourceRef]:
    """Describe each hit for the UI to cite.

    Carries whatever the chunking strategy recorded: the heading trail or the
    page number. Dropping them here would leave a citation able to name only
    the file, whichever strategy indexed it.
    """
    return [
        SourceRef(
            doc_id=chunk.metadata.get("doc_id", ""),
            source=chunk.metadata.get("source", "unknown"),
            score=score,
            heading=_heading_trail(chunk.metadata),
            page=_page(chunk.metadata),
            excerpt=" ".join(chunk.page_content.split())[:240],
        )
        for chunk, score in hits
    ]


def _heading_trail(metadata: dict) -> str | None:
    """`h1 > h2 > h3`, from whichever levels the chunk carries."""
    levels = [metadata.get(key) for key in ("h1", "h2", "h3")]
    trail = [str(level) for level in levels if level]

    return " > ".join(trail) or None


def _page(metadata: dict) -> int | None:
    page = metadata.get("page")

    return int(page) if isinstance(page, int) else None


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
    """Chroma returns cosine distance, 0 (identical) to 2 (opposite)."""
    return max(0.0, 1.0 - distance / 2.0)
