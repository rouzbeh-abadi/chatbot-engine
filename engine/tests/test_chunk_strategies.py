"""How a document is cut up, and what each strategy promises.

The strategy is chosen per project, so the guarantees have to hold for whatever
document arrives -- including one the strategy does not suit. Each fallback
below is deliberate: producing nonsense boundaries would be worse than
producing plain size-based ones.
"""

from __future__ import annotations

from chatbot_engine.documents.models import ExtractedDocument
from chatbot_engine.rag.splitter import DocumentChunker

METADATA = {"doc_id": "d1", "project_id": "support", "filename": "policy.md"}

MARKDOWN = ExtractedDocument(
    text=(
        "# Baggage\nCabin bag up to 8kg.\n\n"
        "## Checked\nOne 23kg bag on Plus fares.\n\n"
        "# Refunds\nBasic fares are non-refundable.\n"
    )
)
PAGED = ExtractedDocument(
    text="page one text\n\npage two text",
    pages=("page one text", "page two text"),
)


def _chunk(extracted: ExtractedDocument, strategy: str, size: int = 200):
    return DocumentChunker(chunk_size=size, chunk_overlap=0, strategy=strategy).chunk(
        extracted, METADATA
    )


# --- every strategy -----------------------------------------------------------


def test_every_strategy_keeps_the_document_identity_on_each_chunk() -> None:
    """Citations are built from this metadata, so no strategy may drop it."""
    for strategy in ("size", "headings", "page"):
        for chunk in _chunk(PAGED, strategy):
            assert chunk.metadata["doc_id"] == "d1"
            assert chunk.metadata["filename"] == "policy.md"


# --- headings -----------------------------------------------------------------


def test_headings_splits_at_markdown_headings() -> None:
    chunks = _chunk(MARKDOWN, "headings")

    assert len(chunks) == 3


def test_headings_records_the_heading_trail() -> None:
    """A chunk should know which section it came from, not just which file."""
    trails = [
        (c.metadata.get("h1"), c.metadata.get("h2"))
        for c in _chunk(MARKDOWN, "headings")
    ]

    assert trails == [("Baggage", None), ("Baggage", "Checked"), ("Refunds", None)]


def test_headings_falls_back_when_there_are_no_headings() -> None:
    """A PDF has no Markdown headings; one giant 'section' would be a lie."""
    plain = ExtractedDocument(text="no headings here, just prose. " * 20)

    assert len(_chunk(plain, "headings", size=100)) > 1


# --- page ---------------------------------------------------------------------


def test_page_never_lets_a_chunk_span_two_pages() -> None:
    chunks = _chunk(PAGED, "page")

    assert [c.metadata["page"] for c in chunks] == [1, 2]
    assert [c.page_content for c in chunks] == ["page one text", "page two text"]


def test_a_long_page_is_still_split_but_stays_on_its_page() -> None:
    """The page bound is the promise; the size cap still applies inside it."""
    long_page = ExtractedDocument(text="x", pages=("word " * 200, "short second page"))

    chunks = _chunk(long_page, "page", size=100)

    assert len(chunks) > 2, "the long first page should have been split further"
    # Every chunk still names exactly one page.
    assert set(c.metadata["page"] for c in chunks) == {1, 2}


def test_page_falls_back_for_a_format_without_pages() -> None:
    """Markdown has no pages -- inventing boundaries would be arbitrary."""
    chunks = _chunk(MARKDOWN, "page")

    assert chunks
    assert all("page" not in c.metadata for c in chunks)


# --- size ---------------------------------------------------------------------


def test_size_ignores_structure_and_just_fills_windows() -> None:
    chunks = _chunk(MARKDOWN, "size", size=60)

    assert len(chunks) > 1
    assert all("h1" not in c.metadata for c in chunks)


# --- a heading is never a chunk on its own --------------------------------------

#: A heading, a subheading, and a paragraph longer than the size cap: the
#: shape that used to leave the two headings as a chunk of their own.
FARES = (
    "# Fares\n\n## Flexible fare\n\n"
    + "Flexible fares can be changed before departure. " * 25
    + "\n\n## Basic fare\n\nBasic fares cannot be changed.\n"
)


def _only_headings(chunk) -> bool:
    return all(
        line.lstrip().startswith("#")
        for line in chunk.page_content.splitlines()
        if line.strip()
    )


def test_a_heading_cut_off_from_its_text_leads_it_instead() -> None:
    """Alone, it takes a place among the retrieved chunks and is cited with
    nothing to show, while the text it introduces loses its title."""
    for strategy in ("size", "headings"):
        chunks = _chunk(ExtractedDocument(text=FARES), strategy, size=400)

        assert not any(_only_headings(chunk) for chunk in chunks), strategy
        first = chunks[0].page_content
        assert first.startswith("# Fares"), strategy
        assert "## Flexible fare" in first, strategy
        assert "Flexible fares can be changed" in first, strategy
        assert chunks[0].metadata["start_index"] == 0, strategy


def test_the_joined_heading_keeps_the_text_as_written() -> None:
    chunks = _chunk(ExtractedDocument(text=FARES), "size", size=400)

    assert chunks[0].page_content.startswith(
        "# Fares\n\n## Flexible fare\n\nFlexible fares can be changed"
    )


def test_a_section_that_is_only_headings_joins_the_one_after_it() -> None:
    """`## Overview` with `## Fees` straight after it: the header splitter
    makes `# Policy` and `## Overview` a section with no text of its own."""
    text = "# Policy\n\n## Overview\n\n## Fees\n\nA change costs 30 EUR.\n"

    chunks = _chunk(ExtractedDocument(text=text), "headings")

    assert len(chunks) == 1
    assert chunks[0].page_content.startswith("# Policy")
    assert chunks[0].page_content.endswith("A change costs 30 EUR.")
    assert (chunks[0].metadata["h1"], chunks[0].metadata["h2"]) == ("Policy", "Fees")


def test_a_heading_with_nothing_after_it_joins_the_text_before_it() -> None:
    text = "Cabin bags weigh up to 8 kg. " * 20 + "\n\n## Notes\n"

    chunks = _chunk(ExtractedDocument(text=text), "size", size=300)

    assert not any(_only_headings(chunk) for chunk in chunks)
    assert chunks[-1].page_content.endswith("## Notes")


def test_a_document_of_nothing_but_headings_keeps_them() -> None:
    """There is no text to give them to, and dropping text is never the fix."""
    text = "# One\n\n# Two\n\n# Three\n"

    chunks = _chunk(ExtractedDocument(text=text), "headings")

    assert "".join(chunk.page_content for chunk in chunks).count("#") == 3


# --- what reaches a citation --------------------------------------------------


def test_source_refs_carry_the_page_and_the_heading_trail() -> None:
    """The chunker records these so a citation can name a page or a section.
    Dropping them on the way to the UI would make that promise false."""
    from langchain_core.documents import Document

    from chatbot_engine.agent.retriever import to_source_refs

    paged = Document(page_content="p", metadata={"source": "a.pdf", "page": 12})
    sectioned = Document(
        page_content="s",
        metadata={"source": "a.md", "h1": "Refunds", "h2": "Basic fares"},
    )
    plain = Document(page_content="x", metadata={"source": "b.md"})

    refs = to_source_refs([(paged, 0.1), (sectioned, 0.2), (plain, 0.3)])

    assert refs[0].page == 12 and refs[0].heading is None
    assert refs[1].heading == "Refunds > Basic fares" and refs[1].page is None
    assert refs[2].page is None and refs[2].heading is None


def test_each_extract_says_which_section_it_is_from() -> None:
    """The heading trail travels in the extract's header, so a chunk whose
    text does not name its subject still says what it is about, and a
    document indexed before the header carried it has it too."""
    from langchain_core.documents import Document

    from chatbot_engine.agent.retriever import to_context

    sectioned = Document(
        page_content="Changes are free until departure.",
        metadata={"source": "faq.md", "h1": "Fares", "h2": "Flexible fare"},
    )
    plain = Document(page_content="Bags up to 8 kg.", metadata={"source": "bags.md"})

    context = to_context([(sectioned, 0.6), (plain, 0.5)])

    assert context == (
        "[1] faq.md > Fares > Flexible fare\nChanges are free until departure.\n\n"
        "[2] bags.md\nBags up to 8 kg."
    )
