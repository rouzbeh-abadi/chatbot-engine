"""The free-space floor that writes to the engine's volumes keep above.

Embedded Chroma does not fail cleanly on a full disk: a write can hang inside
it holding the interpreter, so every route stops, and a write cut short can
leave the index every tenant shares unreadable after a restart
(docs/review-2026-10.md, DISK-3 and DISK-4). So indexing checks the room
first, and refuses well before the disk is full.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from chatbot_engine.errors import StorageFullError
from chatbot_engine.settings import get_settings


def _local_paths() -> set[Path]:
    """The directories the engine writes to on its own disk."""
    settings = get_settings()
    paths = {settings.checkpoint_db.parent}
    if not settings.chroma_url:
        paths.add(settings.chroma_dir)
    if not settings.blob_s3_bucket:
        paths.add(settings.blob_dir)
    if not settings.registry_url:
        paths.add(settings.registry_db.parent)
    return paths


def _existing(path: Path) -> Path:
    """`path`, or its nearest parent that exists yet."""
    path = path.resolve()
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def ensure_room() -> None:
    """Raise `StorageFullError` when a volume the engine writes to has less
    free space than `ENGINE_MIN_FREE_MB`. A cheap call (one `statvfs` per
    directory), made before every write of an upload."""
    floor = get_settings().min_free_mb * 1024 * 1024
    if floor == 0:
        return
    for path in _local_paths():
        free = shutil.disk_usage(_existing(path)).free
        if free < floor:
            raise StorageFullError(
                f"the engine's volume has {free // (1024 * 1024)} MB free, under "
                f"the {floor // (1024 * 1024)} MB it keeps (ENGINE_MIN_FREE_MB), "
                "so the upload was not indexed -- make room, then try again"
            )
