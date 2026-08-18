# rag/ — yours to write

Two halves: getting documents in, and getting relevant chunks back out.

## Ingestion — satisfies `ports.documents.IngestPipeline`

```python
async def ingest(self, *, project_id, external_id, filename, mimetype, data) -> DocumentRecord: ...
```

extract → chunk → embed → store. Plus two things worth doing from the start: hash
the bytes and skip everything when the hash is unchanged, and derive a stable
`doc_id` from `project_id` + `external_id` so a re-upload overwrites instead of
duplicating.

The demo knowledge base is 9 Markdown files, 89 `##` sections, median 325
characters and 568 at the longest — so splitting on `##` gives one self-contained
chunk per section and no overlap is needed. Keep each chunk's heading; it is what
makes a citation readable.

## Retrieval

Embed the question, search, return enough provenance to fill a `SourceRef`:
`doc_id`, `source`, `score`, `heading`, `excerpt`.

## Storage — `ports.documents.BlobStore` and `DocumentRegistry`

The blob store keeps the original files, so changing chunk size or embedding
model becomes a re-index rather than a re-upload. The registry answers "what is
indexed, is it current, delete it" — which a vector store does badly.

Chroma is declared as the `chroma` extra and runs embedded: a directory on disk,
no server. Pin the embedding model per collection; changing it means a full
re-index, because vectors from different models are not comparable.

Register these in `chatbot_engine/api/deps.py` → `get_ingest_pipeline()`,
`get_registry()`.
