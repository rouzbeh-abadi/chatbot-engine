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


@pytest.mark.xfail(
    strict=True, reason="INGEST-12 in docs/review-2026-10.md: fails until it is fixed"
)
def test_one_callers_crafted_files_do_not_lock_out_another_caller(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    deadline = 3.0
    monkeypatch.setenv("ENGINE_API_KEYS", "tenant_a:secret-a,tenant_b:secret-b")
    monkeypatch.setenv("ENGINE_EXTRACT_TIMEOUT_S", str(deadline))
    monkeypatch.setenv("ENGINE_EXTRACT_CONCURRENCY", "2")
    reset_dependency_cache()
    extract_module._slots.clear()

    def attack() -> int:
        return client.post(
            "/extract",
            headers={"X-API-Key": "secret-a"},
            files={"file": ("a.pdf", _SLOW_PDF, "application/pdf")},
        ).status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        fillers = [pool.submit(attack) for _ in range(2)]
        slot = extract_module._reading_slot(2)
        until = time.perf_counter() + deadline
        while not slot.locked() and time.perf_counter() < until:
            time.sleep(0.02)
        assert slot.locked(), "harness: the crafted reads did not take both slots"

        other = client.post(
            "/extract",
            headers={"X-API-Key": "secret-b"},
            files={"file": ("b.txt", b"order 4521", "text/plain")},
        )
        attack_codes = [f.result() for f in fillers]

    # Harness sanity: the attacker's reads ran to the deadline and were refused.
    assert attack_codes == [422, 422], attack_codes
    # CORRECT behaviour: a different caller's two-word text file is read.
    assert other.status_code == 200, (other.status_code, other.text)


@pytest.mark.xfail(
    strict=True, reason="INGEST-12 in docs/review-2026-10.md: fails until it is fixed"
)
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
