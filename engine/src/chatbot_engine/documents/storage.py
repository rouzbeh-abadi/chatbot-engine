from pathlib import Path

from chatbot_engine.ports.documents import BlobStore


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

    async def put(
        self,
        *,
        key: str,
        data: bytes,
        mimetype: str,
    ) -> str:
        """Store one original document as raw bytes and return its filesystem path.

        Args:
            key: Relative path used to identify the document.
            data: Raw bytes of the original uploaded document.
            mimetype: MIME type of the uploaded document.
        """
        path = self._root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

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
        return Path(uri).read_bytes()

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

        if path.exists():
            path.unlink()