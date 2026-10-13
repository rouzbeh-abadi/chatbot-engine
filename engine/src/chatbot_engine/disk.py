"""The free-space floor that writes to the engine's volumes keep above.

Embedded Chroma does not fail cleanly on a full disk: a write can hang inside
it holding the interpreter, so every route stops, and a write cut short can
leave the index every tenant shares unreadable after a restart
(docs/review-2026-10.md, DISK-3 and DISK-4). So indexing checks the room
first, and refuses well before the disk is full. With the floor at zero
there is no such check, and a full volume can freeze the engine again.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from chatbot_engine.errors import StorageFullError
from chatbot_engine.settings import get_settings

logger = logging.getLogger(__name__)


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


def disk_low() -> bool:
    """Whether a volume the engine writes to has less than twice the free
    space `ensure_room` keeps: the warning before the refusals. Never with
    the floor off (`ENGINE_MIN_FREE_MB=0`)."""
    floor = get_settings().min_free_mb * 1024 * 1024
    if floor == 0:
        return False
    try:
        return any(
            shutil.disk_usage(_existing(path)).free < 2 * floor
            for path in _local_paths()
        )
    except OSError as exc:
        # /health answers whenever the engine can answer at all: a volume
        # that cannot be read just now is a warning in the log, not a 500.
        logger.warning("disk: free space could not be read: %s", exc)
        return False


def ensure_room(refused: str = "the document was not indexed") -> None:
    """Raise `StorageFullError` when a volume the engine writes to has less
    free space than `ENGINE_MIN_FREE_MB`. A cheap call (one `statvfs` per
    directory), made before every write of an upload, a move or a rebuild.

    `refused` is what the refusal stops, in the message's words: an upload
    is not the only thing that writes, and a message about one would send
    whoever reads it after the wrong thing. Deletes and purges never ask:
    they are how room is made.
    """
    floor = get_settings().min_free_mb * 1024 * 1024
    if floor == 0:
        return
    for path in _local_paths():
        free = shutil.disk_usage(_existing(path)).free
        if free < floor:
            raise StorageFullError(
                f"the engine's volume has {free // (1024 * 1024)} MB free, under "
                f"the {floor // (1024 * 1024)} MB it keeps (ENGINE_MIN_FREE_MB), "
                f"so {refused} -- make room, then try again"
            )
