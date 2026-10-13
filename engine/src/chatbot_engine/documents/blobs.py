"""The original uploaded bytes, addressed by `doc_id`.

`DocumentRecord` has nowhere to keep the URI that `BlobStore.put` returns, and
adding a field would put a server filesystem path on the wire. It does not need
one: the key *is* the `doc_id`, so the URI is a function of the root and the id,
and can be recomputed whenever it is needed.

Using `doc_id` as the key also closes a hole. `external_id` comes from the caller,
and a `../` in it would escape the blob root; a `doc_id` is 32 hex characters.
"""

from __future__ import annotations

import re
from pathlib import Path

from chatbot_engine.documents.storage import LocalBlobStore
from chatbot_engine.ports.documents import BlobStore

#: What a `doc_id` looks like, and so the only keys `stored` names: a bucket
#: or a directory may hold other things, which are never the engine's to touch.
_DOC_ID = re.compile(r"[0-9a-f]{32}")


class DocumentBlobs:
    """Keeps one file per document, so re-indexing needs no re-upload.

    Composes a `BlobStore`: local files by default, an object store when the
    engine runs as replicas that must share their uploads.
    """

    def __init__(
        self, root: Path | None = None, store: BlobStore | None = None
    ) -> None:
        if store is None:
            if root is None:
                raise ValueError("DocumentBlobs needs a root directory or a store")
            store = LocalBlobStore(root)
        self._store = store

    def _uri(self, doc_id: str) -> str:
        """Where `write` put it: the store's own addressing, from the id alone."""
        return self._store.uri_for(doc_id)

    async def write(self, *, doc_id: str, data: bytes, mimetype: str) -> str:
        return await self._store.put(key=doc_id, data=data, mimetype=mimetype)

    async def read(self, *, doc_id: str) -> bytes:
        return await self._store.get(uri=self._uri(doc_id))

    async def delete(self, *, doc_id: str) -> None:
        """Silent when the file is already gone -- delete is idempotent."""
        await self._store.delete(uri=self._uri(doc_id))

    async def stored(self) -> list[str] | None:
        """The ids of the documents whose originals are kept; None from a
        store that cannot list what it holds (one of the caller's own)."""
        keys = getattr(self._store, "keys", None)
        if keys is None:
            return None
        return sorted(key for key in await keys() if _DOC_ID.fullmatch(key))
