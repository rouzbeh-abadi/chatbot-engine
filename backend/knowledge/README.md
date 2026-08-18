# Knowledge base

Source documents for the assistant. Drop PDFs, Markdown, or text files here,
start the app, and upload them:

```bash
make dev    # one terminal
make seed   # another
```

Only this `README.md` is skipped. Everything else in this folder — including
`.md` files — is treated as a document and indexed.

The path relative to this directory becomes the document's `external_id`, so
re-running `make seed` is idempotent: unchanged files are skipped by content
hash, changed files replace their own chunks. Subfolders work too
(`billing/refunds.md` becomes that document's id).

Files are sent to the engine as **raw bytes**. Do not extract text here —
extraction, chunking and embedding belong to the engine, and the original bytes
carry page numbers and layout that plain text has already lost.
