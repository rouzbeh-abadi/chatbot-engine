"""Review: retrieval, vector store, keyword search, rerank (slice `retrieval`).

Every test asserts the behaviour the engine should have (or that ChatFrom's
docs say it has), and fails on 0.1.26. Offline: a real Chroma in a temp dir
with fake vectors (engine/tests/review/conftest.py); no model is called.
"""

from __future__ import annotations

import gc
import random
import threading
import time
import tracemalloc
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage

from chatbot_engine.agent import retriever
from chatbot_engine.agent.retriever import retrieve
from chatbot_engine.models.chat import AssistantConfig, ChatRequest
from chatbot_engine.rag import embeddings as embeddings_module
from chatbot_engine.rag import rerank as rerank_module
from chatbot_engine.rag import sparse
from chatbot_engine.rag import vector_store as vector_store_module
from chatbot_engine.rag.embeddings import resolve_embedding_model
from chatbot_engine.rag.vector_store import open_vector_store, reset_vector_store

CHUNKS = {
    "fares.md": "The Basic fare is non-refundable. Flexible fares can be refunded in full.",
    "baggage.md": "Cabin baggage is one bag up to eight kilograms.",
    "checkin.md": "Online check-in opens 24 hours before departure.",
    "seats.md": "Seat selection is available in My Trips after booking.",
}

LARGE = "openai/text-embedding-3-large"


def _put(
    client: TestClient,
    name: str,
    text: str,
    *,
    project_id: str = "support",
    **form: str,
) -> dict:
    response = client.put(
        "/documents",
        data={
            "project_id": project_id,
            "external_id": name,
            "chunking_strategy": "size",
            **form,
        },
        files={"file": (name, text.encode(), "text/markdown")},
    )
    assert response.status_code == 201, response.text
    return response.json()


def _seed(client: TestClient, chunks: dict[str, str] = CHUNKS) -> None:
    for name, text in chunks.items():
        _put(client, name, text)


def _request(message: str = "is the Basic fare refundable?", **config) -> ChatRequest:
    project = AssistantConfig(
        **{
            "project_id": "support",
            "name": "S",
            "system_prompt": "s",
            "top_k": 5,
            **config,
        }
    )
    return ChatRequest(project=project, message=message)


def _scored(**similarities: float):
    """A `_dense` that gives each chunk the similarity named for its file
    (0.1 when unnamed), as engine/tests/test_retrieval.py does."""
    original = retriever._dense

    async def dense(store, project_id, query, k):
        found = await original(store, project_id, query, k)
        return sorted(
            (
                (document, similarities.get(document.metadata["source"], 0.1))
                for document, _ in found
            ),
            key=lambda hit: hit[1],
            reverse=True,
        )

    return patch.object(retriever, "_dense", dense)


# --- claim 5: "Passages under a vector similarity of 0.4 never reach the model" ----


@pytest.mark.xfail(
    strict=True, reason="RETRIEVAL-3 in docs/review-2026-10.md: fails until it is fixed"
)
async def test_claim5_no_passage_under_the_floor_reaches_the_model_under_hybrid(
    client: TestClient,
) -> None:
    """ChatFrom sends `min_score: 0.4` and every chatbot defaults to `hybrid`
    (supabase/migrations/0002_assistants.sql:13). docs/ethics.md says passages
    under 0.4 never reach the model. Under hybrid, each query's best keyword
    match is kept whatever its similarity once any chunk clears the bar
    (retriever.py:279-285)."""
    _seed(client)

    # baggage.md is on topic (0.7); fares.md is far (0.1) but is the best
    # keyword match for "Basic fare".
    with _scored(**{"baggage.md": 0.7, "fares.md": 0.1}):
        hits = await retrieve(_request(retrieval="hybrid", min_score=0.4))

    under = {
        document.metadata["source"]: score for document, score in hits if score < 0.4
    }
    assert under == {}, f"passages under the 0.4 floor reached the model: {under}"


@pytest.mark.xfail(
    strict=True, reason="RETRIEVAL-3 in docs/review-2026-10.md: fails until it is fixed"
)
async def test_claim5_a_keyword_only_hit_with_no_vector_score_reaches_the_model(
    client: TestClient,
) -> None:
    """A chunk outside the vector search's candidates is never vector-scored:
    it is cited with 0.0 and still reaches the model as an extract."""
    _seed(client)

    async def baggage_only(store, project_id, query, k):
        found = await store.asimilarity_search_with_score(
            "Cabin baggage", k=10, filter={"project_id": project_id}
        )
        baggage = next(d for d, _ in found if d.metadata["source"] == "baggage.md")
        return [(baggage, 0.7)]

    with patch.object(retriever, "_dense", baggage_only):
        hits = await retrieve(_request(retrieval="hybrid", min_score=0.4))

    context = retriever.to_context(hits)
    assert "Basic fare" not in context, (
        "a chunk with vector similarity 0.0 reached the model's extracts: "
        f"{[(d.metadata['source'], s) for d, s in hits]}"
    )


# --- one collection per model: the embedder bound to it ------------------------


def _two_models(monkeypatch: pytest.MonkeyPatch):
    """Two embedders that cannot be confused: different dimensions."""
    default = DeterministicFakeEmbedding(size=64)
    large = DeterministicFakeEmbedding(size=32)

    def pick(model: str | None = None, *args, **kwargs):
        return large if resolve_embedding_model(model) == LARGE else default

    monkeypatch.setattr(embeddings_module, "get_embeddings", pick)
    monkeypatch.setattr(vector_store_module, "get_embeddings", pick)
    reset_vector_store()
    return default, large


@pytest.mark.xfail(
    strict=True, reason="RETRIEVAL-2 in docs/review-2026-10.md: fails until it is fixed"
)
async def test_a_collection_found_by_the_delete_sweep_keeps_its_own_models_embedder(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a restart, the first upload or delete by ANY project sweeps every
    collection (`_every_store`, vector_store.py:192-198) and caches each one
    it had not opened yet with the DEFAULT model's embedder. A project that
    embeds with another model then searches its collection with the wrong
    embedder until the next restart."""
    _, large = _two_models(monkeypatch)
    _put(
        client,
        "baggage.md",
        CHUNKS["baggage.md"],
        project_id="beta",
        embedding_model=LARGE,
    )

    reset_vector_store()  # the engine restarts (a deploy)
    # Another tenant uploads a document under the default model.
    _put(client, "fares.md", CHUNKS["fares.md"], project_id="alpha")

    store = open_vector_store(LARGE)
    assert store.embeddings is large, (
        "the large model's collection is now bound to the default embedder"
    )


@pytest.mark.xfail(
    strict=True, reason="RETRIEVAL-2 in docs/review-2026-10.md: fails until it is fixed"
)
async def test_after_the_sweep_a_project_on_another_model_can_still_retrieve(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same, seen from the visitor: every turn of project `beta` fails."""
    _two_models(monkeypatch)
    _put(
        client,
        "baggage.md",
        CHUNKS["baggage.md"],
        project_id="beta",
        embedding_model=LARGE,
    )

    reset_vector_store()
    _put(client, "fares.md", CHUNKS["fares.md"], project_id="alpha")

    request = _request(
        "how heavy can cabin baggage be?",
        project_id="beta",
        embedding_model=LARGE,
        retrieval="vector",
    )
    try:
        hits = await retrieve(request)
    except Exception as exc:
        pytest.fail(f"retrieval for a project on {LARGE} failed: {exc!r}")
    assert [d.metadata["source"] for d, _ in hits] == ["baggage.md"]


@pytest.mark.xfail(
    strict=True, reason="RETRIEVAL-2 in docs/review-2026-10.md: fails until it is fixed"
)
async def test_after_the_sweep_a_project_on_another_model_can_still_upload(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _two_models(monkeypatch)
    _put(
        client,
        "baggage.md",
        CHUNKS["baggage.md"],
        project_id="beta",
        embedding_model=LARGE,
    )

    reset_vector_store()
    _put(client, "fares.md", CHUNKS["fares.md"], project_id="alpha")

    try:
        response = client.put(
            "/documents",
            data={
                "project_id": "beta",
                "external_id": "seats.md",
                "embedding_model": LARGE,
            },
            files={"file": ("seats.md", CHUNKS["seats.md"].encode(), "text/markdown")},
        )
    except Exception as exc:
        pytest.fail(f"an upload for a project on {LARGE} failed: {exc!r}")
    assert response.status_code == 201 and response.json()["status"] == "indexed", (
        response.status_code,
        response.text[:300],
    )


# --- the keyword index -------------------------------------------------------


@pytest.mark.xfail(
    strict=True, reason="RETRIEVAL-1 in docs/review-2026-10.md: fails until it is fixed"
)
def test_concurrent_searches_on_a_cold_index_build_it_once(client: TestClient) -> None:
    """Every turn of a project whose index is older than 60 s (or pushed out
    by 32 others) rebuilds it, reading and tokenising every chunk. Turns that
    arrive together each build their own copy (sparse.py:106-115): N copies
    of the project's text in memory and N worker threads of the default
    executor, the one every project's vector search also runs on."""
    _seed(client)
    sparse.invalidate()
    store = open_vector_store()
    builds = 0
    lock = threading.Lock()
    original = sparse._build

    def slow_build(store, project_id, now):
        nonlocal builds
        with lock:
            builds += 1
        time.sleep(0.3)  # a large project's build
        return original(store, project_id, now)

    with patch.object(sparse, "_build", slow_build), ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda _: sparse.sparse_index(store, "support"), range(8)))

    assert builds == 1, f"{builds} concurrent builds of one project's index"


@pytest.mark.xfail(
    strict=True, reason="RETRIEVAL-1 in docs/review-2026-10.md: fails until it is fixed"
)
def test_a_projects_keyword_index_holds_no_more_than_a_few_times_its_text(
    client: TestClient,
) -> None:
    """The index keeps every chunk as a Document plus a term-count dict per
    chunk (sparse.py:139-149, rank_bm25), with no bound on a project's size;
    32 are kept whatever their size (MAX_INDEXES, sparse.py:48), and an
    expired one stays until 32 others push it out. ChatFrom's engine runs
    under `mem_limit: 3g`, and a Pro owner may hold 500 MB of knowledge."""
    rng = random.Random(1)
    vocab = [
        "".join(
            rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(2, 10))
        )
        for _ in range(20_000)
    ]
    weights = [1 / (rank + 1) for rank in range(len(vocab))]
    texts = [" ".join(rng.choices(vocab, weights, k=170))[:1000] for _ in range(2_000)]
    store = open_vector_store()
    for start in range(0, len(texts), 500):
        batch = range(start, start + 500)
        store.add_texts(
            [texts[i] for i in batch],
            metadatas=[
                {
                    "doc_id": "d" * 32,
                    "project_id": "big",
                    "source": "page.md",
                    "filename": "page.md",
                    "start_index": i,
                }
                for i in batch
            ],
            ids=[f"big:{i}" for i in batch],
        )
    text_bytes = sum(len(text.encode()) for text in texts)
    sparse.invalidate()

    gc.collect()
    tracemalloc.start()
    try:
        index = sparse.sparse_index(store, "big")
        retained, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert len(index.documents) == 2_000
    assert retained <= 3 * text_bytes, (
        f"{text_bytes / 1e6:.1f} MB of text held as {retained / 1e6:.0f} MB "
        f"({retained / text_bytes:.1f}x), {peak / 1e6:.0f} MB at the peak of the build"
    )


@pytest.mark.xfail(
    strict=True, reason="RETRIEVAL-5 in docs/review-2026-10.md: fails until it is fixed"
)
async def test_keyword_search_matches_across_unicode_normalisation_forms(
    client: TestClient,
) -> None:
    """Text from a PDF or a Mac is often decomposed (NFD: `e` + U+0301); a
    visitor types composed (NFC: `é`). `\\w+` on the raw text splits the
    decomposed word at the combining mark, so the exact term never matches."""
    chunks = {
        "menu.md": unicodedata.normalize(
            "NFD", "Le café crème coûte trois euros au comptoir."
        ),
        "hours.md": "Opening hours are nine to five on weekdays.",
        "parking.md": "Parking is free for customers after six.",
        "wifi.md": "The wifi password is printed on the receipt.",
    }
    _seed(client, chunks)

    index = sparse.sparse_index(open_vector_store(), "support")
    found = index.search(unicodedata.normalize("NFC", "prix du café crème"), 3)

    assert [d.metadata["source"] for d, _ in found][:1] == ["menu.md"], found


# --- the rerank prompt and the prompt budget ----------------------------------


@pytest.mark.xfail(
    strict=True, reason="RETRIEVAL-4 in docs/review-2026-10.md: fails until it is fixed"
)
async def test_the_rerank_prompt_is_kept_within_the_prompt_budget(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ENGINE_PROMPT_CHARS` is "the most characters one model call's prompt
    may hold" (settings.py:266). The rerank is a model call whose prompt
    holds every candidate whole (rerank.py:210-224), up to
    `retrieval_candidates` (200) chunks of up to 8,000 characters."""
    from chatbot_engine.api.dependencies import reset_dependency_cache

    monkeypatch.setenv("ENGINE_PROMPT_CHARS", "10000")
    reset_dependency_cache()
    # One document cut into 30 chunks of about 1,000 characters.
    text = "\n\n".join(f"Page {n}. " + "lorem ipsum " * 80 for n in range(30))
    _put(client, "pages.md", text, chunk_size="1000", chunk_overlap="0")

    prompts: list[str] = []

    class Reranker:
        async def ainvoke(self, messages, config=None):
            prompts.append("".join(m.content for m in messages))
            return AIMessage(content='{"ranking": [1]}')

    with patch.object(rerank_module, "build_chat_model", return_value=Reranker()):
        await retrieve(
            _request(
                "lorem ipsum", rerank=True, retrieval="vector", retrieval_candidates=30
            )
        )

    assert prompts, "the rerank ran"
    assert len(prompts[0]) <= 10_000, f"rerank prompt of {len(prompts[0]):,} characters"


# --- a search under a model nothing was indexed with ---------------------------


@pytest.mark.xfail(
    strict=True, reason="RETRIEVAL-6 in docs/review-2026-10.md: fails until it is fixed"
)
async def test_a_search_under_a_model_nothing_was_indexed_with_creates_no_collection(
    client: TestClient,
) -> None:
    """`embedding_model` is any string the caller sends with a chat turn
    (models/chat.py:112). Retrieval opens its collection with
    `get_or_create` (vector_store.py:127, langchain_chroma), so every new
    string leaves a collection on disk and a wrapper in `_stores` for the
    life of the process, and every later write or delete, by any project,
    sweeps all of them (`_every_store`)."""
    from chatbot_engine.rag.vector_store import _chroma_client

    open_vector_store()
    before = {collection.name for collection in _chroma_client().list_collections()}

    for n in range(5):
        await retrieve(
            _request(embedding_model=f"nobody/model-{n}", retrieval="vector")
        )

    after = {collection.name for collection in _chroma_client().list_collections()}
    assert after == before, f"{len(after - before)} collections created by searches"
