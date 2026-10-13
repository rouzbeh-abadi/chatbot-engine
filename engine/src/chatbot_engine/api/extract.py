"""Read a file into text without keeping it: what a chat attachment needs.

A person sends a file in the conversation. The caller asks for its text here
and sends that text with the turns that follow (`ChatRequest.attachments`).
Nothing is stored, embedded or indexed. A document is read without any model,
in a process of its own that is killed at a deadline (documents/bounded.py),
and only as far as a chat file's text goes; it is metered as a reading, apart
from chat. An image is read by a vision model (agent/vision.py), which is
billed, so it is metered as a chat turn is and the answer says what it cost.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from chatbot_engine.agent.vision import (
    IMAGE_TYPES,
    ImageReadCutOff,
    ImageReading,
    describe_image,
)
from chatbot_engine.api.auth import CallerDep
from chatbot_engine.api.dependencies import SettingsDep
from chatbot_engine.api.documents import MAX_UPLOAD_BYTES
from chatbot_engine.api.forms import only_fields
from chatbot_engine.api.rate_limit import limit_chat, limit_extract
from chatbot_engine.documents.bounded import ReadFailed, read_bounded
from chatbot_engine.documents.extractor import select_extractor
from chatbot_engine.errors import DocumentRejectedError
from chatbot_engine.models.chat import MAX_ATTACHMENT_CHARS
from chatbot_engine.models.documents import ExtractedText, ExtractUsage

router = APIRouter(tags=["documents"])

#: The readings running right now, bounded by `extract_concurrency`; rebuilt
#: when the setting changes, as the rate limiters are.
_slots: dict[int, asyncio.Semaphore] = {}

#: How long a reading waits for a slot before it is refused. Most readings
#: take well under a second, so a burst is served in turn rather than turned
#: away, and a few files sent together cannot keep out everyone else's.
SLOT_WAIT_S = 5.0
#: How many readings may wait at once, per slot; past it a file is refused at
#: once, so a flood holds no more uploads in memory than this.
WAITING_PER_SLOT = 2
_waiting = 0


@dataclass
class _Caller:
    """One caller's readings: one runs at a time, the rest wait their turn."""

    #: Held while one of the caller's readings runs.
    turn: asyncio.Lock = field(default_factory=asyncio.Lock)
    #: Its readings in line, for its turn or for a slot.
    waiting: int = 0
    #: Its readings running or in line; the record goes with the last.
    readings: int = 0


#: The callers with a reading running or in line, by the API key's name and
#: the form's `project_id` (empty when none was sent): one key may serve
#: every chatbot of an application, and the project tells them apart.
_callers: dict[tuple[str, str], _Caller] = {}


def _reading_slot(capacity: int) -> asyncio.Semaphore:
    slot = _slots.get(capacity)
    if slot is None:
        _slots.clear()
        slot = _slots[capacity] = asyncio.Semaphore(capacity)
    return slot


def _share(room: int) -> int:
    """How much of the line one caller may take: half, so another caller
    always finds a place in it."""
    return max(1, room // 2)


def _busy() -> HTTPException:
    return HTTPException(
        status_code=503,
        detail="as many files are being read as the engine allows at once",
        headers={"Retry-After": "2"},
    )


def _no_room(who: tuple[str, str], slot: asyncio.Semaphore, room: int) -> bool:
    """Whether a reading by `who` would have to wait with no place in line:
    the line is full, or the caller already has its share of it."""
    caller = _callers.get(who)
    if not slot.locked() and (caller is None or not caller.turn.locked()):
        return False
    return _waiting >= room or (caller is not None and caller.waiting >= _share(room))


@contextlib.asynccontextmanager
async def _reading(
    who: tuple[str, str], slot: asyncio.Semaphore, room: int
) -> AsyncIterator[None]:
    """A reading slot for `who`, within `SLOT_WAIT_S`, or a 503.

    Each caller reads one file at a time and may take only its share of the
    line, so no caller can hold every slot and the line with files made to
    read slowly (a page of drawing a few megabytes long, just under the
    parse budget) while another caller's file is refused
    (docs/review-2026-10.md, INGEST-12)."""
    global _waiting

    caller = _callers.setdefault(who, _Caller())
    caller.readings += 1
    try:
        waits = caller.turn.locked() or slot.locked()
        if waits:
            if _no_room(who, slot, room):  # filled while the allowance was checked
                raise _busy()
            _waiting += 1
            caller.waiting += 1
        try:
            async with asyncio.timeout(SLOT_WAIT_S if waits else None):
                await caller.turn.acquire()
                try:
                    await slot.acquire()
                except BaseException:
                    caller.turn.release()
                    raise
        except TimeoutError:
            raise _busy() from None
        finally:
            if waits:
                _waiting -= 1
                caller.waiting -= 1
        try:
            yield
        finally:
            slot.release()
            caller.turn.release()
    finally:
        caller.readings -= 1
        if caller.readings == 0 and _callers.get(who) is caller:
            del _callers[who]


@router.post(
    "/extract",
    dependencies=[
        Depends(only_fields("file", "model", "provider_api_key", "project_id"))
    ],
    responses={
        400: {"description": "The file is empty."},
        413: {"description": "The file is larger than an upload may be."},
        415: {"description": "No extractor reads this type."},
        422: {
            "description": "The file is damaged, not UTF-8, has no text, or took too long to read."
        },
        429: {"description": "Over the reading rate; an image, over the chat rate."},
        501: {"description": "An image, and no model provider key is configured."},
        502: {
            "description": "An image, and the vision model refused or failed. "
            "One cut off at its deadline carries `usage` too, estimated: the "
            "provider may still bill it."
        },
        503: {
            "description": "As many documents are being read as the engine "
            "allows at once, or as this caller may have in line."
        },
    },
)
async def extract(
    caller: CallerDep,
    settings: SettingsDep,
    file: UploadFile = File(...),
    model: str | None = Form(default=None),
    provider_api_key: str | None = Form(default=None),
    project_id: str | None = Form(default=None, min_length=1, max_length=256),
) -> ExtractedText:
    """One file's text: a PDF, text or Markdown, or what an image says and shows.

    `model` reads an image and must see images; unset, the utility model
    does, or the engine's chat model when there is none. `provider_api_key`
    pays for that call instead of the engine's own key, as
    `AssistantConfig.provider_api_key` does for a turn. A document ignores
    both, and is read only as far as a chat file's text goes
    (`MAX_ATTACHMENT_CHARS`). `project_id`, the chatbot the file was sent
    to, shares the reading slots out between chatbots served on one API
    key: each reads one document at a time.
    """
    # One byte past the limit at most: enough to know it is too large.
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413, detail=f"file exceeds {MAX_UPLOAD_BYTES} bytes"
        )
    mimetype = file.content_type or "application/octet-stream"
    name = file.filename or "the file"
    key = provider_api_key or None

    if mimetype in IMAGE_TYPES:
        # No key, no reading: said before the chat allowance is charged.
        settings.require_provider_key(key)
        # A model call, so counted as a chat turn is.
        await limit_chat(caller, settings)
        try:
            reading = await describe_image(
                data, mimetype, model=model, provider_api_key=key
            )
        except ImageReadCutOff as exc:
            return JSONResponse(  # ty: ignore[invalid-return-type]
                status_code=502,
                content={
                    "detail": str(exc),
                    "usage": _usage(exc.reading).model_dump(),
                },
            )
        # The call was made and billed whatever the model said, so the caller
        # hears what it cost even for an image it said nothing about.
        return ExtractedText(
            text=reading.text, chars=len(reading.text), usage=_usage(reading)
        )

    # Before any reading, so an unknown type is a 415 (app.py), not a 422.
    select_extractor(mimetype)
    # A place to read in, or a short wait for one, one file at a time for
    # each caller. With no place in line for it, the file is refused at once,
    # before the allowance is charged.
    slot = _reading_slot(settings.extract_concurrency)
    room = settings.extract_concurrency * WAITING_PER_SLOT
    who = (caller.name, project_id or "")
    if _no_room(who, slot, room):
        raise _busy()
    await limit_extract(caller, settings)

    async with _reading(who, slot, room):
        try:
            # In a process of its own, off the event loop and killed at the
            # deadline: a PDF made to take minutes would otherwise hold a
            # worker thread, and every request behind it, for as long as it
            # liked.
            extracted = await run_in_threadpool(
                read_bounded,
                data,
                mimetype,
                max_chars=MAX_ATTACHMENT_CHARS,
                timeout_s=settings.extract_timeout_s,
                max_content_bytes=settings.extract_parse_mb * 1024 * 1024,
            )
        except ReadFailed as exc:
            if exc.kind == "UnicodeDecodeError":
                raise DocumentRejectedError("the file is not UTF-8 text") from exc
            # pypdf raises its own errors for a damaged or encrypted PDF.
            raise DocumentRejectedError("the file could not be read") from exc

    if not extracted.text.strip():
        raise DocumentRejectedError(
            "no text could be extracted from the file -- "
            "a scanned document needs OCR before it can be read"
        )

    del name  # the file's name takes no part in reading it, and is never echoed
    return ExtractedText(
        text=extracted.text,
        pages=extracted.page_count,
        chars=len(extracted.text),
        truncated=extracted.truncated,
    )


def _usage(reading: ImageReading) -> ExtractUsage:
    return ExtractUsage(
        input_tokens=reading.usage.input_tokens,
        output_tokens=reading.usage.output_tokens,
        cost_usd=reading.usage.cost_usd,
        model=reading.usage.model,
        estimated=reading.estimated,
    )
