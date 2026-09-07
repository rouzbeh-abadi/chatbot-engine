"""Chunking config, from the HTTP request down to the stored vectors.

`test_chunk_strategies.py` covers what each strategy does to a document. This
covers the wiring: that a strategy named on the request is the one actually
used, that its metadata survives all the way into the vector store, and that a
bad value is refused instead of quietly indexing the document some other way.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from chatbot_engine.rag.vector_store import open_vector_store

MARKDOWN = (
    b"# Baggage\nOne cabin bag up to 8 kg.\n\n"
    b"## Checked\nOne 23 kg bag on Plus fares.\n\n"
    b"# Refunds\nBasic fares are non-refundable.\n"
)


def _upload(client: TestClient, content: bytes, external_id: str, **form: object):
    mimetype = "application/pdf" if content[:4] == b"%PDF" else "text/markdown"
    return client.put(
        "/documents",
        data={"project_id": "support", "external_id": external_id, **form},
        files={"file": (external_id, content, mimetype)},
    )


def _stored(doc_id: str) -> list[dict]:
    """The metadata of every chunk stored for one document."""
    got = open_vector_store().get(where={"doc_id": doc_id}, include=["metadatas"])
    return list(got["metadatas"])


# --- the strategy on the request is the one that runs -------------------------


def test_headings_requested_over_http_reaches_the_stored_chunks(
    client: TestClient,
) -> None:
    response = _upload(client, MARKDOWN, "baggage.md", chunking_strategy="headings")
    assert response.status_code == 201

    metadatas = _stored(response.json()["doc_id"])

    assert len(metadatas) == 3, "one chunk per heading section"
    assert {m.get("h1") for m in metadatas} == {"Baggage", "Refunds"}


def test_chunk_size_from_the_request_changes_how_many_chunks_are_stored(
    client: TestClient,
) -> None:
    big = _upload(
        client, MARKDOWN, "a.md", chunking_strategy="size", chunk_size=2000
    ).json()
    small = _upload(
        client,
        MARKDOWN,
        "b.md",
        chunking_strategy="size",
        chunk_size=100,
        chunk_overlap=0,
    ).json()

    assert small["chunk_count"] > big["chunk_count"]


# --- a bad value is refused, not silently reinterpreted -----------------------


def test_an_unknown_strategy_is_rejected(client: TestClient) -> None:
    """Quietly size-chunking instead would only surface later as bad retrieval."""
    response = _upload(client, MARKDOWN, "bad.md", chunking_strategy="banana")

    assert response.status_code == 422
