"""Review: deletion end to end, from the app's side (INGEST-4, INGEST-17).

ChatFrom deletes a chatbot (app/api/assistants/[id]/route.ts DELETE) or a whole
account (libs/account-data.ts deleteAccount) the only way the engine allows:
`GET /documents?project_id=X`, then `DELETE /documents/{doc_id}` for every
record listed, then it drops its own row. The engine has no project-level
delete, and no way to enumerate projects.

Every test asserts the behaviour the engine should have, and fails on 0.1.26.
Offline: a real Chroma in a temp dir with fake vectors (engine/tests/conftest.py),
driven over HTTP through the real app with httpx's ASGI transport, so the upload
and the app's delete flow interleave on one event loop as they do in uvicorn.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator

import httpx
import pytest

from chatbot_engine.api.dependencies import reset_dependency_cache
from chatbot_engine.rag.pipeline import doc_id_for
from chatbot_engine.rag.vector_store import ChromaChunkStore, open_vector_store
from chatbot_engine.settings import get_settings

PROJECT = "6f1c0a52-0000-4000-8000-00000000000x"  # a chatbot's uuid in ChatFrom
PAGE = "https-example-com-pricing"  # slugFor(page.url), as the crawl names a page
SECRET = "Our wholesale price to ACME is 4.10 EUR per unit."


@pytest.fixture
async def engine(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[httpx.AsyncClient]:
    """The real app, open (no key), on this test's event loop."""
    monkeypatch.delenv("ENGINE_API_KEY", raising=False)
    reset_dependency_cache()
    from chatbot_engine.app import create_app

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(
        transport=transport, base_url="http://engine:8100", timeout=30
    ) as client:
        yield client
    reset_dependency_cache()


def _slow_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[asyncio.Event, asyncio.Event]:
    """Hold the embedding of the page carrying SECRET until released: what a
    large PDF or a slow provider does for seconds to a minute."""
    at_gate, released = asyncio.Event(), asyncio.Event()
    original = ChromaChunkStore._add

    async def gated(self, store, documents, ids):
        if any(SECRET in d.page_content for d in documents):
            at_gate.set()
            await released.wait()
        return await original(self, store, documents, ids)

    monkeypatch.setattr(ChromaChunkStore, "_add", gated)
    return at_gate, released


async def _put(engine: httpx.AsyncClient) -> httpx.Response:
    """libs/engine/client.ts putDocument, for one crawled page."""
    body = f"# Pricing\n\n{SECRET}\n\n" + "Volume discounts apply. " * 40
    return await engine.put(
        "/documents",
        data={"project_id": PROJECT, "external_id": PAGE},
        files={"file": ("Pricing.md", body.encode(), "text/markdown")},
    )


async def _app_deletes_the_chatbot(engine: httpx.AsyncClient) -> list[str]:
    """`for (const doc of await listDocuments(id)) await deleteDocument(id, doc.doc_id)`:
    the chatbot DELETE route and deleteAccount both do exactly this."""
    listed = (await engine.get("/documents", params={"project_id": PROJECT})).json()
    for doc in listed:
        gone = await engine.delete(
            f"/documents/{doc['doc_id']}", params={"project_id": PROJECT}
        )
        assert gone.status_code in (200, 404)
    return [doc["doc_id"] for doc in listed]


def _what_the_engine_still_holds() -> dict[str, object]:
    """Everything of PROJECT on the engine's volume: registry rows, chunks, blob."""
    settings = get_settings()
    with sqlite3.connect(settings.registry_db) as db:
        rows = db.execute(
            "SELECT doc_id, external_id, status FROM documents WHERE project_id = ?",
            (PROJECT,),
        ).fetchall()
    chunks = open_vector_store().get(
        where={"project_id": PROJECT}, include=["documents"]
    )
    doc_id = doc_id_for(PROJECT, PAGE)
    blob = settings.blob_dir / doc_id
    return {
        "registry rows": rows,
        "chunks": len(chunks["ids"]),
        "chunks quoting the page": sum(SECRET in t for t in chunks["documents"]),
        "blob": blob.exists(),
    }


# --- INGEST-4: a first upload in flight is invisible, so it survives ------


async def test_a_document_being_indexed_for_the_first_time_is_listed(
    engine: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /documents is the only way a caller can find what the engine holds
    for a project. While a first upload is embedding, its original is already
    on disk (pipeline.py:385-388) but the record is written only once the
    vectors land (pipeline.py:416-431), so the listing says the project holds
    nothing."""
    at_gate, released = _slow_embedding(monkeypatch)
    upload = asyncio.create_task(_put(engine))
    await asyncio.wait_for(at_gate.wait(), 10)
    try:
        listed = (await engine.get("/documents", params={"project_id": PROJECT})).json()
        on_disk = _what_the_engine_still_holds()["blob"]
    finally:
        released.set()
        assert (await upload).status_code == 201

    assert on_disk, "precondition: the original is stored while it embeds"
    assert [d["external_id"] for d in listed] == [PAGE], (
        f"the page's original is on the engine's disk, yet GET /documents lists {listed}"
    )


async def test_deleting_a_chatbot_during_its_first_upload_leaves_nothing_of_it(
    engine: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The owner deletes a chatbot (or the account) while a crawled page or an
    uploaded file is being indexed for the first time. The app lists the
    chatbot's documents, deletes each, reports {deleted: true} and drops its
    row. The page was not listed, so no DELETE named it; when its embedding
    finishes, its record, its chunks and its original stay on the engine's
    volume under a project id no caller will ever name again."""
    at_gate, released = _slow_embedding(monkeypatch)
    upload = asyncio.create_task(_put(engine))
    await asyncio.wait_for(at_gate.wait(), 10)

    deleted = await _app_deletes_the_chatbot(engine)  # the app now says deleted: true

    released.set()
    # Since 0.1.28 a first upload is recorded before it is embedded, so the
    # app's listing names it and its delete cancels the upload, which keeps
    # nothing and answers 409. Before, the listing saw nothing (deleted == [])
    # and the upload answered 201; either way, what the engine holds after
    # is the finding.
    answered = (await upload).status_code
    assert (len(deleted), answered) in ((0, 201), (1, 409)), (deleted, answered)

    held = _what_the_engine_still_holds()
    assert held == {
        "registry rows": [],
        "chunks": 0,
        "chunks quoting the page": 0,
        "blob": False,
    }, f"after the app deleted the chatbot, the engine still holds {held}"


# --- INGEST-17: a deleted document's text stays in Chroma's write log -----


@pytest.mark.xfail(
    strict=True, reason="INGEST-17 in docs/review-2026-10.md: fails until it is fixed"
)
async def test_a_deleted_documents_text_is_gone_from_the_vector_store_file(
    engine: httpx.AsyncClient,
) -> None:
    """The app deletes a chatbot whose page was indexed: every listed document
    is deleted, its chunks are gone from every query. Chroma keeps each write
    in its log (`embeddings_queue`, the chunk text in `metadata`) and purges
    the log only below the vector segment's last persisted sequence id, which
    advances every `hnsw:sync_threshold` (1,000) records per collection. Until
    then the deleted page's words stay in chroma.sqlite3 on the volume."""
    assert (await _put(engine)).status_code == 201
    await _app_deletes_the_chatbot(engine)
    assert _what_the_engine_still_holds()["chunks"] == 0, "precondition: deleted"

    with sqlite3.connect(get_settings().chroma_dir / "chroma.sqlite3") as db:
        kept = db.execute(
            "SELECT seq_id FROM embeddings_queue WHERE metadata LIKE ?",
            (f"%{SECRET}%",),
        ).fetchall()
    assert kept == [], (
        f"the deleted page's text is still in chroma.sqlite3 (embeddings_queue rows {kept})"
    )
