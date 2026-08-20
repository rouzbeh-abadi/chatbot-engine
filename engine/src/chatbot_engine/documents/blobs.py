"""The original uploaded bytes, addressed by `doc_id`.

`DocumentRecord` has nowhere to keep the URI that `BlobStore.put` returns, and
adding a field would put a server filesystem path on the wire. It does not need
one: the key *is* the `doc_id`, so the URI is a function of the root and the id,
and can be recomputed whenever it is needed.

Using `doc_id` as the key also closes a hole. `external_id` comes from the caller,
and a `../` in it would escape the blob root; a `doc_id` is 32 hex characters.
"""

from __future__ import annotations

from pathlib import Path

from chatbot_engine.documents.storage import LocalBlobStore
from chatbot_engine.ports.documents import BlobStore


class DocumentBlobs:
    """Keeps one file per document, so re-indexing needs no re-upload."""

    def __init__(self, root: Path, store: BlobStore | None = None) -> None:
        self._root = root
        self._store = store or LocalBlobStore(root)

    def _uri(self, doc_id: str) -> str:
        """Where `write` put it. Pinned by a test against `LocalBlobStore.put`."""
        return str(self._root / doc_id)

    async def write(self, *, doc_id: str, data: bytes, mimetype: str) -> str:
        return await self._store.put(key=doc_id, data=data, mimetype=mimetype)

    async def read(self, *, doc_id: str) -> bytes:
        return await self._store.get(uri=self._uri(doc_id))

    async def delete(self, *, doc_id: str) -> None:
        """Silent when the file is already gone -- delete is idempotent."""
        await self._store.delete(uri=self._uri(doc_id))
