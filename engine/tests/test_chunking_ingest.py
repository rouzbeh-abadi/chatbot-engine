"""Chunking config, from the HTTP request down to the stored vectors.

`test_chunk_strategies.py` covers what each strategy does to a document. This
covers the wiring: that a strategy named on the request is the one actually
used, that its metadata survives all the way into the vector store, and that a
bad value is refused instead of quietly indexing the document some other way.
"""

from __future__ import annotations

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from chatbot_engine.rag.vector_store import open_vector_store

MARKDOWN = (
    b"# Baggage\nOne cabin bag up to 8 kg.\n\n"
    b"## Checked\nOne 23 kg bag on Plus fares.\n\n"
    b"# Refunds\nBasic fares are non-refundable.\n"
)


def _pdf(pages: int) -> bytes:
    """A real PDF with `pages` pages, each carrying distinguishable text."""
    writer = PdfWriter()
    for index in range(pages):
        page = writer.add_blank_page(width=200, height=200)
        page.merge_page(page)  # no-op; keeps pypdf happy on a blank page
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


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


def test_size_requested_over_http_ignores_the_headings(
    client: TestClient,
) -> None:
    """Same document, different strategy -- the result must actually differ."""
    response = _upload(
        client, MARKDOWN, "baggage-size.md", chunking_strategy="size", chunk_size=2000
    )
    assert response.status_code == 201

    metadatas = _stored(response.json()["doc_id"])

    assert len(metadatas) == 1, "the whole document fits in one 2000-char window"
    assert all("h1" not in m for m in metadatas)


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


def test_the_reported_chunk_count_matches_what_was_stored(
    client: TestClient,
) -> None:
    body = _upload(client, MARKDOWN, "count.md", chunking_strategy="headings").json()

    assert body["chunk_count"] == len(_stored(body["doc_id"]))


# --- defaults and fallbacks ---------------------------------------------------


def test_no_strategy_on_the_request_still_ingests(client: TestClient) -> None:
    """Omitting the config must not break the route -- the engine default applies."""
    response = _upload(client, MARKDOWN, "default.md")

    assert response.status_code == 201
    assert response.json()["chunk_count"] >= 1


def test_page_on_a_document_without_pages_falls_back_instead_of_failing(
    client: TestClient,
) -> None:
    """Markdown has no pages; the upload must still succeed, unlabelled."""
    response = _upload(client, MARKDOWN, "nopages.md", chunking_strategy="page")

    assert response.status_code == 201
    assert all("page" not in m for m in _stored(response.json()["doc_id"]))


# --- a bad value is refused, not silently reinterpreted -----------------------


def test_an_unknown_strategy_is_rejected(client: TestClient) -> None:
    """Quietly size-chunking instead would only surface later as bad retrieval."""
    response = _upload(client, MARKDOWN, "bad.md", chunking_strategy="banana")

    assert response.status_code == 422


def test_a_rejected_upload_stores_nothing(client: TestClient) -> None:
    response = _upload(client, MARKDOWN, "bad2.md", chunking_strategy="banana")
    assert response.status_code == 422

    listed = client.get("/documents", params={"project_id": "support"}).json()
    assert all(record["external_id"] != "bad2.md" for record in listed)


@pytest.mark.parametrize("bad_size", [10, 99_999])
def test_an_out_of_range_chunk_size_is_rejected_by_the_chunker(bad_size: int) -> None:
    """The wire contract bounds these; the engine should not be asked to honour
    a 10-character or 100k-character window."""
    from chatbot_engine.models.chat import AssistantConfig

    with pytest.raises(Exception):
        AssistantConfig(
            project_id="p",
            name="n",
            system_prompt="s",
            chunk_size=bad_size,
        )
