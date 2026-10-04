"""Read a file into text without keeping it: what a chat attachment needs.

A person sends a file in the conversation. The caller asks for its text here
and sends that text with the turns that follow (`ChatRequest.attachments`).
Nothing is stored, embedded or indexed. A document is read without any model,
so it costs nothing and is not metered; an image is read by a vision model
(agent/vision.py), which is billed, so it is metered as a chat turn is and
the answer says what it cost.
"""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from chatbot_engine.agent.vision import IMAGE_TYPES, describe_image
from chatbot_engine.api.auth import CallerDep
from chatbot_engine.api.dependencies import SettingsDep
from chatbot_engine.api.documents import MAX_UPLOAD_BYTES
from chatbot_engine.api.rate_limit import limit_chat
from chatbot_engine.documents.extractor import select_extractor
from chatbot_engine.errors import DocumentRejectedError
from chatbot_engine.models.documents import ExtractedText, ExtractUsage

router = APIRouter(tags=["documents"])


@router.post(
    "/extract",
    responses={
        413: {"description": "The file is larger than an upload may be."},
        415: {"description": "No extractor reads this type."},
        422: {"description": "The file is damaged, not UTF-8, or has no text."},
        429: {"description": "An image, over the chat rate."},
        502: {"description": "An image, and the vision model refused or failed."},
    },
)
async def extract(
    caller: CallerDep,
    settings: SettingsDep,
    file: UploadFile = File(...),
    model: str | None = Form(default=None),
) -> ExtractedText:
    """One file's text: a PDF, text or Markdown, or what an image says and shows.

    `model` reads an image and must see images; unset, the utility model
    does, or the engine's chat model when there is none. A document ignores it.
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

    if mimetype in IMAGE_TYPES:
        # A model call, so counted as a chat turn is.
        await limit_chat(caller, settings)
        text, usage = await describe_image(data, mimetype, model=model)
        if not text:
            raise DocumentRejectedError(f"{name!r} could not be read")
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
    extractor = select_extractor(mimetype)

    try:
        # CPU-bound, so off the event loop: a slow PDF would stall every
        # request in flight.
        extracted = await run_in_threadpool(
            extractor.extract_text, data=data, mimetype=mimetype
        )
    except UnicodeDecodeError as exc:
        raise DocumentRejectedError(f"{name!r} is not UTF-8 text") from exc
    except Exception as exc:
        # pypdf raises its own errors for a damaged or encrypted PDF.
        raise DocumentRejectedError(f"{name!r} could not be read") from exc

    if not extracted.text.strip():
        raise DocumentRejectedError(
            f"no text could be extracted from {name!r} -- "
            "a scanned document needs OCR before it can be read"
        )

    return ExtractedText(
        text=extracted.text, pages=len(extracted.pages), chars=len(extracted.text)
    )
