# Chunking strategies

A document is not retrieved whole. It is cut into chunks, each embedded and
searched independently, and the chunk is what the model finally reads. Where the
cuts fall therefore decides what retrieval can find, and the right place to cut
depends on the document.

The strategy is project configuration, not an engine constant.

## The three strategies

| Strategy | Cuts at | Adds to each chunk | Suits |
| --- | --- | --- | --- |
| `size` | fixed-length windows, with overlap | nothing | anything; the safe default |
| `headings` | Markdown headings (`#`, `##`, `###`) | the heading trail (`h1`, `h2`, `h3`) | manuals, policies, any structured Markdown |
| `page` | page boundaries | the page number (`page`) | PDFs, especially scanned or paginated ones |

**`size`** respects no structure. It is predictable and works on every format,
at the cost of cutting mid-sentence and mid-section.

**`headings`** makes a chunk a section, so a retrieved chunk is a coherent unit
of the author's own making. Each chunk also carries the headings above it, which
tells the model (and a citation) which section it came from.

**`page`** guarantees no chunk spans two pages, and labels every chunk with its
page number. This is what lets a citation say *page 12* rather than only naming
the file.

## Every strategy ends with the size cap

`headings` and `page` choose the boundaries; `chunk_size` still applies inside
them. Without that cap a forty-page chapter would be a single chunk, which
embeds poorly and buries the answer in unrelated text.

So a long PDF page is still split, but only within the page, and every piece
still names that page. Structure decides the boundaries; the cap keeps the
pieces usable.

## A strategy that does not suit the document falls back

The strategy is set once per project, but documents vary. Rather than invent
boundaries that are not there, a strategy that cannot apply degrades to `size`:

- `headings` on a document with no Markdown headings (a PDF, plain text)
- `page` on a format with no pages (Markdown, plain text)

The upload still succeeds; the chunks simply carry no heading trail or page
number. Uploading a mixed corpus under one strategy is therefore safe.

## Configuration

Set it on the project, beside the model and embedding model:

```yaml
# examples/backend/src/support_agent/projects/support.yaml
chunking_strategy: headings    # size | headings | page
chunk_size: 1000
chunk_overlap: 200
```

The backend sends these with every upload. Omit them and the engine's own
defaults apply (`ENGINE_CHUNK_STRATEGY`, `ENGINE_CHUNK_SIZE`,
`ENGINE_CHUNK_OVERLAP`), so an engine with no per-project config still runs.

Uploading directly to the engine takes the same three fields:

```bash
curl -X PUT localhost:8100/documents \
  -F project_id=support \
  -F external_id=baggage.md \
  -F chunking_strategy=headings \
  -F chunk_size=1000 \
  -F "file=@examples/backend/knowledge/baggage.md;type=text/markdown"
```

An unknown strategy is rejected with `422`, and nothing is stored. It is not
quietly reinterpreted as `size`: a document indexed differently than requested
fails silently at upload and only shows up later as poor retrieval.

## What a document remembers

Every document record carries the settings it was indexed with,
`chunking_strategy`, `chunk_size` and `chunk_overlap`, after the engine's
defaults filled in what the caller left out. `GET /documents` returns them, so
a caller can compare each document with its current settings and list the
ones that are out of date.

Re-uploading identical bytes answers `unchanged` and does no work only when
the settings match too. The same file sent with a different size or strategy
is cut again. A document indexed by an engine before 0.1.10 has no recorded
settings; it counts as current unless the upload names chunking explicitly,
so upgrading the engine never re-embeds a knowledge base by itself.

## Changing the strategy means re-indexing

Chunking is applied when a document is ingested, so changing any of these
settings has no effect on documents already indexed. The existing vectors keep
whatever boundaries they were built with.

Re-index a document from the original the engine kept, without uploading it
again:

```bash
curl -X POST "localhost:8100/documents/$DOC_ID/reindex?project_id=support" \
  -H 'content-type: application/json' \
  -d '{"chunking_strategy": "headings", "chunk_size": 800, "chunk_overlap": 100}'
```

Settings work as on upload. An empty body rebuilds with the engine's current
defaults, after a change to `ENGINE_CHUNK_SIZE` for example, and a field left
out takes the engine's default. The response is the rebuilt record. `404`
means no such document, `501` means the engine keeps no originals, and `422`
means settings that cannot work together: `chunk_size` runs from 100 to 8000
characters, `chunk_overlap` from 0 to 2000 and must be smaller than the size.
Both routes answer `422` for those, and nothing is stored.

To rebuild the example knowledge base from source instead, `make seed`
re-ingests it.

## Choosing one

- **Markdown knowledge base with headings**: `headings`. The sections are
  already the author's own units of meaning.
- **PDFs, or anything where "which page" matters**: `page`. It is the only
  strategy that makes page citations possible.
- **Mixed, unstructured, or unknown**: `size`. It always works, and the other
  two fall back to it anyway.

Tuning `chunk_size` matters more than the strategy for retrieval quality: too
small and a chunk loses the context that makes it answerable, too large and the
embedding blurs across several topics. The RAG evaluation
(`POST /admin/eval/rag`) is the way to tell whether a change actually helped:
the RAGAS scores are shown in the admin dashboard.
