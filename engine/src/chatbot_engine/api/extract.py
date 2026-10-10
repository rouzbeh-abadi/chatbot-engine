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

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from chatbot_engine.agent.vision import IMAGE_TYPES, describe_image
from chatbot_engine.api.auth import CallerDep
from chatbot_engine.api.dependencies import SettingsDep
from chatbot_engine.api.documents import MAX_UPLOAD_BYTES
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


def _reading_slot(capacity: int) -> asyncio.Semaphore:
    slot = _slots.get(capacity)
    if slot is None:
        _slots.clear()
        slot = _slots[capacity] = asyncio.Semaphore(capacity)
    return slot


def _busy() -> HTTPException:
    return HTTPException(
        status_code=503,
        detail="as many files are being read as the engine allows at once",
        headers={"Retry-After": "2"},
    )


async def _take(slot: asyncio.Semaphore, room: int) -> None:
    """A reading slot, within `SLOT_WAIT_S`, or a 503."""
    global _waiting

    if not slot.locked():
        await slot.acquire()
        return
    if _waiting >= room:  # filled while the allowance was checked
        raise _busy()
    _waiting += 1
    try:
        await asyncio.wait_for(slot.acquire(), SLOT_WAIT_S)
    except TimeoutError:
        raise _busy() from None
    finally:
        _waiting -= 1


@router.post(
    "/extract",
    responses={
        400: {"description": "The file is empty."},
        413: {"description": "The file is larger than an upload may be."},
        415: {"description": "No extractor reads this type."},
        422: {
            "description": "The file is damaged, not UTF-8, has no text, or took too long to read."
        },
        429: {"description": "Over the reading rate; an image, over the chat rate."},
        501: {"description": "An image, and no model provider key is configured."},
        502: {"description": "An image, and the vision model refused or failed."},
        503: {
            "description": "As many documents are being read as the engine allows at once."
        },
    },
)
async def extract(
    caller: CallerDep,
    settings: SettingsDep,
    file: UploadFile = File(...),
    model: str | None = Form(default=None),
    provider_api_key: str | None = Form(default=None),
) -> ExtractedText:
    """One file's text: a PDF, text or Markdown, or what an image says and shows.

    `model` reads an image and must see images; unset, the utility model
    does, or the engine's chat model when there is none. `provider_api_key`
    pays for that call instead of the engine's own key, as
    `AssistantConfig.provider_api_key` does for a turn. A document ignores
    both, and is read only as far as a chat file's text goes
    (`MAX_ATTACHMENT_CHARS`).
    """
    data = await file.read()
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
        text, usage = await describe_image(
            data, mimetype, model=model, provider_api_key=key
        )
        # The call was made and billed whatever the model said, so the caller
        # hears what it cost even for an image it said nothing about.
        return ExtractedText(
            text=text,
            chars=len(text),
            usage=ExtractUsage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost_usd=usage.cost_usd,
                model=usage.model,
            ),
        )

    # Before any reading, so an unknown type is a 415 (app.py), not a 422.
    select_extractor(mimetype)
    # A place to read in, or a short wait for one. With the line full, the
    # file is refused at once, before the allowance is charged.
    slot = _reading_slot(settings.extract_concurrency)
    room = settings.extract_concurrency * WAITING_PER_SLOT
    if slot.locked() and _waiting >= room:
        raise _busy()
    await limit_extract(caller, settings)
    await _take(slot, room)

    try:
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
    finally:
        slot.release()

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
