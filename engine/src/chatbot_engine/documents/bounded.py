"""Read a file into text in a process of its own, within a time and a size.

A file a stranger sends in a chat is read by pypdf, which is pure Python: a
PDF made to inflate into gigabytes of text, or to take minutes to parse, would
otherwise hold a worker thread and the engine's memory for as long as it
likes, since a running Python thread cannot be stopped. A process can. The
reading runs in a short-lived process with a deadline; past it, the process
is killed and the file is refused. The process also bounds itself, so a
reader orphaned by a parent that died keeps to the same time, and pypdf's
inflation caps are lowered in it, so one compressed stream cannot blow up on
its own.
"""

from __future__ import annotations

import contextlib
import math
import multiprocessing
from dataclasses import dataclass
from multiprocessing.connection import Connection

from chatbot_engine.documents.extractor import PdfDocumentExtractor, select_extractor
from chatbot_engine.documents.models import ExtractedDocument
from chatbot_engine.errors import DocumentRejectedError

#: The most one compressed stream of a PDF may inflate to in the reading
#: process, whichever filter packed it. pypdf allows 75 MB; a chat file's text
#: is cut long before that.
INFLATE_MAX_BYTES = 8 * 1024 * 1024

#: The most memory the reading process may take for its data, where the
#: system enforces it (Linux): a page made of hundreds of megabytes of
#: operators ends in a MemoryError there, not in the engine's memory.
MEMORY_MAX_BYTES = 1024 * 1024 * 1024


class ReadTimeoutError(DocumentRejectedError):
    """The file took longer to read than a chat allows."""


@dataclass
class ReadFailed(Exception):
    """The reading process raised: the exception's kind and its words."""

    kind: str
    message: str


def _bound_self(timeout_s: float) -> None:
    """Hold the reading process to its time and memory, where the system can.

    The parent kills it at the deadline; this is for a reader the parent is
    no longer there to kill. A limit the environment refuses, or has already
    set lower, is left as it is.
    """
    try:
        import resource
    except ImportError:  # pragma: no cover - not on this platform
        return
    ceiling = math.ceil(timeout_s) + 1
    for name, wanted in (("RLIMIT_CPU", ceiling), ("RLIMIT_DATA", MEMORY_MAX_BYTES)):
        limit = getattr(resource, name, None)
        if limit is None:
            continue
        try:
            soft, hard = resource.getrlimit(limit)
            cap = wanted if hard == resource.RLIM_INFINITY else min(wanted, hard)
            if soft == resource.RLIM_INFINITY or soft > cap:
                resource.setrlimit(limit, (cap, hard))
        except (ValueError, OSError):
            continue


def _worker(
    conn: Connection,
    data: bytes,
    mimetype: str,
    max_chars: int,
    timeout_s: float,
    max_content_bytes: int | None,
) -> None:
    """Read `data` and send the result, or the failure, back; never raises."""
    try:
        _bound_self(timeout_s)
        import pypdf.filters

        # pypdf types its caps as the constants they start as; lowering them
        # is what its documentation says to do.
        for cap in (
            "ZLIB_MAX_OUTPUT_LENGTH",
            "LZW_MAX_OUTPUT_LENGTH",
            "RUN_LENGTH_MAX_OUTPUT_LENGTH",
        ):
            setattr(pypdf.filters, cap, INFLATE_MAX_BYTES)
        extractor = select_extractor(mimetype)
        if isinstance(extractor, PdfDocumentExtractor):
            extractor.max_content_bytes = max_content_bytes
        document = extractor.extract_text(
            data=data, mimetype=mimetype, max_chars=max_chars
        )
        conn.send(
            (
                "ok",
                document.text,
                document.pages,
                document.truncated,
                document.page_count,
            )
        )
    except BaseException as exc:  # reported to the parent, which decides
        with contextlib.suppress(OSError):  # the parent may be gone
            conn.send(("error", type(exc).__name__, str(exc)))
    finally:
        conn.close()


def read_bounded(
    data: bytes,
    mimetype: str,
    *,
    max_chars: int,
    timeout_s: float,
    max_content_bytes: int | None = None,
) -> ExtractedDocument:
    """The document's text, read in a process that is killed at `timeout_s`,
    and for a PDF no further than `max_content_bytes` of page content.

    Raises `ReadTimeoutError` when the deadline passes, `ReadFailed` with the
    exception the reader raised (a damaged PDF, text that is not UTF-8), and
    `DocumentRejectedError` when the process died without a word (killed by
    the system for its memory, say).
    """
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker,
        args=(child, data, mimetype, max_chars, timeout_s, max_content_bytes),
        daemon=True,
    )
    process.start()
    child.close()
    try:
        if not parent.poll(timeout_s):
            raise ReadTimeoutError("the file took too long to read")
        try:
            result = parent.recv()
        except EOFError as exc:
            raise DocumentRejectedError("the file could not be read") from exc
    finally:
        if process.is_alive():
            process.kill()
        process.join(5)
        parent.close()

    if result[0] == "ok":
        return ExtractedDocument(
            text=result[1],
            pages=tuple(result[2]),
            truncated=result[3],
            page_count=result[4],
        )
    raise ReadFailed(kind=result[1], message=result[2])
