"""Local storage for original document uploads.

The blob store persists raw uploaded bytes on the filesystem so ingestion can
retain the source document for later reads, cleanup, or reindexing workflows.
"""

import asyncio
import os
from pathlib import Path

from chatbot_engine.ports.documents import BlobStore


def _write_whole(path: Path, data: bytes) -> None:
    """`data` at `path`, all of it or, should the write fail, what was there."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = _part(path)
    try:
        with part.open("wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(part, path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def _part(path: Path) -> Path:
    """Where `put` writes before the file takes its place: one per key, so a
    write cut short leaves at most one, which the next write replaces."""
    return path.with_name(f".{path.name}.part")


class LocalBlobStore(BlobStore):
    """Store original uploaded documents as raw bytes on the local filesystem.

    Each method operates on one document at a time, while the store itself can
    manage many documents through their individual keys and filesystem paths.
    """

    def __init__(self, root: Path) -> None:
        """Initialize the local blob store.

        Args:
            root: Directory where the original document bytes are stored.
        """
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def uri_for(self, key: str) -> str:
        return str(self._root / key)

    async def put(
        self,
        *,
        key: str,
        data: bytes,
        mimetype: str,
    ) -> str:
        """Store one original document as raw bytes and return its filesystem path.

        The file is written beside its place, flushed to the disk, and only
        then moved into it, so a process killed mid-write, or a disk that
        fills, leaves the version stored before whole rather than part of
        the new one (docs/review-2026-10.md, INGEST-15).

        Args:
            key: Relative path used to identify the document.
            data: Raw bytes of the original uploaded document.
            mimetype: MIME type of the uploaded document.
        """
        path = self._root / key
        # On a thread: flushing 25 MB to a slow disk takes long enough to
        # stall every request the event loop is serving.
        await asyncio.to_thread(_write_whole, path, data)
        return str(path)

    async def get(
        self,
        *,
        uri: str,
    ) -> bytes:
        """Read one original document from storage as raw bytes.

        Args:
            uri: Filesystem path returned when the document was stored.
        """
        return await asyncio.to_thread(Path(uri).read_bytes)

    async def delete(
        self,
        *,
        uri: str,
    ) -> None:
        """Delete one original document from storage if it exists.

        Args:
            uri: Filesystem path of the stored document to delete.
        """
        path = Path(uri)
        path.unlink(missing_ok=True)
        _part(path).unlink(missing_ok=True)
