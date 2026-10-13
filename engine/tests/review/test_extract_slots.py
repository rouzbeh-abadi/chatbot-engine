"""Review: one caller can hold every `/extract` reading slot with a small PDF
(INGEST-12).

The reading slots are one engine-wide semaphore (`extract_concurrency`, 2 by
default), and a PDF page with no text is parsed whole before `max_chars`
applies, so a 53 KB file of graphics operators holds its slot for the whole
deadline. These tests assert the behaviour the engine should have.
"""

from __future__ import annotations

import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from chatbot_engine.api import extract as extract_module
from chatbot_engine.api.dependencies import reset_dependency_cache
from chatbot_engine.documents.bounded import (
    INFLATE_MAX_BYTES,
    ReadTimeoutError,
    read_bounded,
)
from chatbot_engine.models.chat import MAX_ATTACHMENT_CHARS

#: A page content stream made only of graphics operators (q/cm/rg/Q). pypdf
#: must tokenise every one, but none of them shows text, so the page yields
#: zero characters -- `max_chars` sees nothing and never stops the reader.
_UNIT = b"q 1 0 0 1 0 0 cm 0 0 0 rg Q\n"


def _nontext_stream(inflated_target: int) -> bytes:
    """Decompressed content a little under `inflated_target` bytes."""
    reps = inflated_target // len(_UNIT)
    return _UNIT * reps


def _pdf(page_streams: list[bytes]) -> bytes:
    """A PDF with one page per stream, each stream FlateDecode-compressed."""
    objects: list[bytes] = [b"<< /Type /Catalog /Pages 2 0 R >>", b""]
    kids: list[bytes] = []
    font = 3 + 2 * len(page_streams)
    for i, content in enumerate(page_streams):
        page_no = 3 + 2 * i
        body = zlib.compress(content, 9)
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>"
            % (font, page_no + 1)
        )
        objects.append(
            b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(body)
            + body
            + b"\nendstream"
        )
        kids.append(b"%d 0 R" % page_no)
    objects[1] = (
        b"<< /Type /Pages /Kids ["
        + b" ".join(kids)
        + b"] /Count %d >>" % len(page_streams)
    )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)


#: Each stream a little under the 8 MiB inflate cap, so it is NOT refused as an
#: inflate bomb -- it is read, slowly. Three pages parse for ~35 s on this box;
#: the deadline, not the file, decides when the reader is killed.
_PER_PAGE_INFLATED = INFLATE_MAX_BYTES - 1024 * 1024  # ~7 MiB, under the 8 MiB cap
_SLOW_PDF = _pdf([_nontext_stream(_PER_PAGE_INFLATED)] * 3)

#: ChatFrom caps a chat file at 5 MiB (libs/chat-files.ts FILE_MAX_BYTES).
_APP_FILE_CAP = 5 * 1024 * 1024


@pytest.fixture(autouse=True)
def _fresh_slots() -> None:
    """Drop any semaphore a sibling test left in the module global."""
    extract_module._slots.clear()


# --- added in verification ---------------------------------------------------


#: How long each of the burst's files holds its reading here. Under the parse
#: budget the worst a PDF can do is a page of drawing operators a few
#: megabytes long, read to its end in about 2.7 s; the reader is slowed to
#: stand for it, so the test does not hang on this box's speed.
_HELD_S = 1.5


@pytest.mark.parametrize(
    "burst,other",
    [
        # Two API keys: two callers.
        (("secret-a", None), ("secret-b", None)),
        # One key serving two chatbots, each named by the form's project.
        (("secret-a", "bot-1"), ("secret-a", "bot-2")),
    ],
    ids=["two-keys", "two-projects-one-key"],
)
def test_one_callers_crafted_files_do_not_lock_out_another_caller(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    burst: tuple[str, str | None],
    other: tuple[str, str | None],
) -> None:
    """With the parse budget, a crafted PDF is read in seconds, not to the
    deadline; but one caller sending a burst of them still took both slots
    and the whole line behind them, and another caller's two-word text file
    was refused with 503. Each caller now reads one file at a time and has
    half the line at most: the rest of the burst is refused, and the other
    caller's file is read at once in the slot left free."""
    monkeypatch.setenv("ENGINE_API_KEYS", "tenant_a:secret-a,tenant_b:secret-b")
    monkeypatch.setenv("ENGINE_EXTRACT_CONCURRENCY", "2")
    reset_dependency_cache()
    extract_module._slots.clear()
    real = extract_module.read_bounded

    def slowly(data, mimetype, **kwargs):
        if mimetype == "application/pdf":
            time.sleep(_HELD_S)
        return real(data, mimetype, **kwargs)

    monkeypatch.setattr(extract_module, "read_bounded", slowly)

    def send(who: tuple[str, str | None], name: str, data: bytes, kind: str):
        key, project = who
        return client.post(
            "/extract",
            headers={"X-API-Key": key},
            data={"project_id": project} if project else None,
            files={"file": (name, data, kind)},
        )

    def attack(_: int) -> int:
        return send(burst, "a.pdf", _SLOW_PDF, "application/pdf").status_code

    sender = ("tenant_a", burst[1] or "")
    with ThreadPoolExecutor(max_workers=6) as pool:
        files = [pool.submit(attack, i) for i in range(6)]
        # Until the burst holds what it may: one reading and two in line.
        until = time.perf_counter() + 5
        while time.perf_counter() < until:
            held = extract_module._callers.get(sender)
            if held is not None and held.turn.locked() and held.waiting == 2:
                break
            time.sleep(0.02)
        assert held is not None and held.waiting == 2, "harness: the burst is held"

        answer = send(other, "b.txt", b"order 4521", "text/plain")
        codes = sorted(f.result() for f in files)

    # The other caller's file is read, at once.
    assert answer.status_code == 200, (answer.status_code, answer.text)
    # The burst got one reading and its share of the line; the rest went.
    assert codes.count(503) == 3, codes


def test_a_small_pdf_with_no_text_does_not_hold_a_slot_for_the_whole_deadline() -> None:
    deadline = 4.0
    started = time.perf_counter()
    try:
        read_bounded(
            _SLOW_PDF,
            "application/pdf",
            max_chars=MAX_ATTACHMENT_CHARS,
            timeout_s=deadline,
        )
        outcome = "read"
    except ReadTimeoutError:
        outcome = "timeout"
    except Exception as exc:  # refused for its content: also acceptable
        outcome = type(exc).__name__
    elapsed = time.perf_counter() - started
    # CORRECT behaviour: a 54 KB file is decided (read or refused for what it
    # is) in a fraction of the deadline, not by the deadline itself.
    assert outcome != "timeout" and elapsed < deadline / 2, (outcome, elapsed)
