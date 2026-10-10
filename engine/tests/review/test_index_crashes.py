"""Review: what a stop, a kill or a failed update leaves of a document.

The engine keeps the only original of every upload (ChatFrom keeps none), and
`_index` writes it first, then the chunks, then the record. These tests check
what that order leaves behind when an update fails, when the process is stopped
the way Docker stops it (SIGTERM, then SIGKILL after the grace period), and when
the knowledge base outgrows one Chroma upsert.

No model or provider is called: embeddings are DeterministicFakeEmbedding
(slowed down in the subprocess so a stop lands mid-index).
"""

from __future__ import annotations

import hashlib
import os
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding

from chatbot_engine.api.dependencies import reset_dependency_cache
from chatbot_engine.documents.storage import LocalBlobStore
from chatbot_engine.rag import embeddings as embeddings_module
from chatbot_engine.rag import vector_store as vector_store_module
from chatbot_engine.rag.pipeline import doc_id_for
from chatbot_engine.rag.vector_store import open_vector_store
from chatbot_engine.settings import get_settings

PROJECT = "support"


# --------------------------------------------------------------------------- helpers


def _pdf(text: str | None) -> bytes:
    """A one-page PDF with `text` in Helvetica, or a blank (scanned-like) page."""
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
    ]
    content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode() if text else b""
    objects.append(
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream"
    )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
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


def _put(
    client: TestClient | httpx.Client,
    external_id: str,
    data: bytes,
    mimetype: str,
    **form: str,
):
    return client.put(
        "/documents",
        data={"project_id": PROJECT, "external_id": external_id, **form},
        files={"file": (external_id, data, mimetype)},
    )


def _chunk_texts(doc_id: str) -> list[str]:
    found = open_vector_store().get(
        where={"$and": [{"doc_id": doc_id}, {"project_id": PROJECT}]},
        include=["documents"],
    )
    return list(found["documents"] or [])


def _blob_path(doc_id: str) -> Path:
    return Path(get_settings().blob_dir) / doc_id


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# The engine, started the way engine/Dockerfile starts it (uvicorn's CLI, so its
# own signal handling), with an embedder that hangs on the texts marked
# "VERSION TWO" or "FIRST UPLOAD": a large document whose embedding takes
# longer than Docker's grace period. Everything else is the real engine.
_LAUNCHER = textwrap.dedent(
    """
    import sys, time, pathlib
    marker = pathlib.Path(sys.argv[1])
    port = sys.argv[2]

    from langchain_core.embeddings import DeterministicFakeEmbedding
    import chatbot_engine.rag.embeddings as E
    import chatbot_engine.rag.vector_store as V

    class Slow(DeterministicFakeEmbedding):
        def embed_documents(self, texts):
            if any("VERSION TWO" in t or "FIRST UPLOAD" in t for t in texts):
                marker.write_text("embedding started")
                time.sleep(600)
            return super().embed_documents(texts)

    fake = Slow(size=64)
    E.get_embeddings = lambda *a, **k: fake
    V.get_embeddings = lambda *a, **k: fake

    import uvicorn
    sys.argv = ["uvicorn", "chatbot_engine.app:app", "--host", "127.0.0.1",
                "--port", port, "--proxy-headers"]
    sys.exit(uvicorn.main())
    """
)


class _Engine:
    """A real uvicorn engine process on the test's directories."""

    def __init__(self, tmp_path: Path) -> None:
        self.marker = tmp_path / "embedding-started"
        script = tmp_path / "launch_engine.py"
        script.write_text(_LAUNCHER)
        self.port = _free_port()
        env = dict(os.environ)
        env.pop("ENGINE_API_KEY", None)
        env.pop("ENGINE_ENV", None)
        env.update(
            RAGAS_DO_NOT_TRACK="true",
            ANONYMIZED_TELEMETRY="False",
            ENGINE_TRACING="off",
        )
        self.log = tmp_path / "engine.log"
        self._log = self.log.open("wb")
        self.proc = subprocess.Popen(
            [sys.executable, "-I", str(script), str(self.marker), str(self.port)],
            env=env,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            cwd=tmp_path,
        )
        self.url = f"http://127.0.0.1:{self.port}"
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            assert self.proc.poll() is None, self.log.read_text()[-3000:]
            try:
                if httpx.get(f"{self.url}/health", timeout=1).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        raise AssertionError("engine did not start:\n" + self.log.read_text()[-3000:])

    def put_in_background(self, external_id: str, data: bytes, mimetype: str) -> None:
        def run() -> None:
            try:
                with httpx.Client(base_url=self.url, timeout=900) as c:
                    _put(c, external_id, data, mimetype)
            except httpx.HTTPError:
                pass  # the connection dies with the process

        threading.Thread(target=run, daemon=True).start()

    def wait_for_embedding(self) -> None:
        deadline = time.monotonic() + 60
        while not self.marker.exists():
            assert time.monotonic() < deadline, self.log.read_text()[-3000:]
            time.sleep(0.05)

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGKILL)
        self.proc.wait(timeout=30)
        self._log.close()


@pytest.fixture
def local_client(monkeypatch: pytest.MonkeyPatch):
    """An in-process engine on the same directories, opened only when asked
    (after any subprocess that used them is dead)."""

    def open_client() -> TestClient:
        monkeypatch.delenv("ENGINE_API_KEY", raising=False)
        reset_dependency_cache()
        from chatbot_engine.app import create_app

        return TestClient(create_app(), raise_server_exceptions=False)

    return open_client


# ------------------------------------------------- INGEST-14: failed update


@pytest.mark.xfail(
    strict=True, reason="INGEST-14 in docs/review-2026-10.md: fails until it is fixed"
)
def test_a_rejected_update_keeps_the_original_of_the_version_still_answering(
    client: TestClient,
) -> None:
    """v1 is indexed; the owner sends a new version under the same name and the
    engine refuses it (a scanned PDF). The docs promise that v1 keeps
    answering; it does. Its original must still be the one the engine keeps,
    since the engine is the only place it is kept."""
    v1 = _pdf("Refunds are paid within fourteen days")
    v2 = _pdf(None)  # a scanned page: no text
    doc_id = doc_id_for(PROJECT, "handbook.pdf")

    first = _put(client, "handbook.pdf", v1, "application/pdf")
    assert first.status_code == 201, first.text
    assert first.json()["status"] == "indexed"

    second = _put(client, "handbook.pdf", v2, "application/pdf")
    assert second.status_code == 422, second.text

    # As documented: the version indexed before keeps answering.
    assert any("fourteen days" in t for t in _chunk_texts(doc_id))

    # The only stored original of that version.
    stored = _blob_path(doc_id).read_bytes()
    assert stored == v1, (
        "v1 still answers visitors, but its only stored original was replaced "
        f"by the rejected upload ({len(stored)} bytes, sha {_sha(stored)[:12]}, "
        f"is v2: {stored == v2})"
    )


@pytest.mark.xfail(
    strict=True, reason="INGEST-14 in docs/review-2026-10.md: fails until it is fixed"
)
def test_after_a_rejected_update_reindex_still_rebuilds_the_version_that_answers(
    client: TestClient,
) -> None:
    """The app's re-split (POST /reindex) of that document must rebuild v1,
    the version visitors are getting, from the original the engine kept."""
    v1 = b"# Refunds\n\nRefunds are paid within fourteen days.\n"
    v2 = b"# Refunds\n\nR\xe9funds are paid within thirty days.\n"  # Latin-1, not UTF-8
    doc_id = doc_id_for(PROJECT, "refunds.md")

    assert _put(client, "refunds.md", v1, "text/markdown").status_code == 201
    rejected = _put(client, "refunds.md", v2, "text/markdown")
    assert rejected.status_code == 422, rejected.text
    assert any("fourteen days" in t for t in _chunk_texts(doc_id))

    rebuilt = client.post(
        f"/documents/{doc_id}/reindex",
        params={"project_id": PROJECT},
        json={"chunking_strategy": "size", "chunk_size": 500, "chunk_overlap": 0},
    )
    assert rebuilt.status_code == 200, (
        "the version still answering can no longer be re-indexed: "
        f"{rebuilt.status_code} {rebuilt.json().get('detail')!r}"
    )


# ------------------------------- INGEST-15: stop or kill during an update


@pytest.mark.xfail(
    strict=True, reason="INGEST-15 in docs/review-2026-10.md: fails until it is fixed"
)
def test_a_stop_during_an_update_does_not_leave_another_versions_original_under_the_record(
    tmp_path: Path, local_client
) -> None:
    """Docker's stop: SIGTERM, then SIGKILL after the grace period (10 s, the
    compose default; ChatFrom's engine service sets none). v2's embedding is
    still running, so uvicorn (no --timeout-graceful-shutdown) waits for it and
    the process is killed. What is left must be one consistent version."""
    v1 = b"# Refunds\n\nVERSION ONE: refunds are paid within fourteen days.\n"
    v2 = b"# Refunds\n\nVERSION TWO: refunds are paid within thirty days.\n"
    doc_id = doc_id_for(PROJECT, "refunds.md")

    engine = _Engine(tmp_path)
    try:
        with httpx.Client(base_url=engine.url, timeout=120) as c:
            first = _put(c, "refunds.md", v1, "text/markdown")
        assert first.status_code == 201, first.text
        assert first.json()["status"] == "indexed"

        engine.put_in_background("refunds.md", v2, "text/markdown")
        engine.wait_for_embedding()

        engine.proc.send_signal(signal.SIGTERM)
        time.sleep(10)  # Docker's default stop_grace_period
        still_running = engine.proc.poll() is None
    finally:
        engine.kill()

    # Characterisation of the stop: uvicorn was still waiting on the upload.
    assert still_running, "uvicorn exited on SIGTERM within the grace period"

    with local_client() as client:
        records = client.get("/documents", params={"project_id": PROJECT}).json()
        assert len(records) == 1
        record = records[0]
        stored = _blob_path(doc_id).read_bytes()
        chunks_after_kill = _chunk_texts(doc_id)

        # The owner re-sends the version they still have and see listed (v1),
        # then the app re-splits the knowledge base (SplittingSettings).
        resent = _put(client, "refunds.md", v1, "text/markdown")
        rebuilt = client.post(
            f"/documents/{doc_id}/reindex",
            params={"project_id": PROJECT},
            json={"chunking_strategy": "size", "chunk_size": 500, "chunk_overlap": 0},
        )
        chunks_after_reindex = _chunk_texts(doc_id)
        final = client.get("/documents", params={"project_id": PROJECT}).json()[0]

    assert record["status"] == "indexed" and record["content_hash"] == _sha(v1)
    assert any("VERSION ONE" in t for t in chunks_after_kill)
    assert any("VERSION ONE" in t for t in chunks_after_reindex) and not any(
        "VERSION TWO" in t for t in chunks_after_reindex
    ), (
        "after the kill the record and chunks said v1 but the stored original "
        f"was {'v2' if stored == v2 else 'other'}; re-sending v1 answered "
        f"{resent.json().get('status')!r}, the re-index answered "
        f"{rebuilt.status_code}, and visitors now get {chunks_after_reindex!r} "
        f"under a record whose content_hash is still v1's "
        f"({final['content_hash'] == _sha(v1)})"
    )


@pytest.mark.parametrize("how", ["killed", "disk_full"])
@pytest.mark.xfail(
    strict=True, reason="INGEST-15 in docs/review-2026-10.md: fails until it is fixed"
)
def test_a_kill_while_the_original_is_written_leaves_one_whole_version(
    tmp_path: Path, how: str
) -> None:
    """LocalBlobStore.put is `path.write_bytes(data)`: the old file is truncated
    first and the new one written in place. A process that dies inside it
    ("killed": SIGXFSZ at its default action, standing in for SIGKILL or the
    OOM killer at that moment), or a write that fails part-way ("disk_full":
    CPython ignores SIGXFSZ, so the write raises as ENOSPC would), must leave
    either the old original or the new one, not a piece."""
    root = tmp_path / "blobs"
    store = LocalBlobStore(root)
    v1 = b"A" * 300_000
    import asyncio

    asyncio.run(store.put(key="doc", data=v1, mimetype="text/plain"))

    child = textwrap.dedent(
        f"""
        import asyncio, resource, signal
        from pathlib import Path
        if {how!r} == "killed":
            signal.signal(signal.SIGXFSZ, signal.SIG_DFL)
        from chatbot_engine.documents.storage import LocalBlobStore
        resource.setrlimit(resource.RLIMIT_FSIZE, (1_000_000, 1_000_000))
        asyncio.run(LocalBlobStore(Path({str(root)!r})).put(
            key="doc", data=b"B" * 5_000_000, mimetype="text/plain"))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-I", "-c", child], capture_output=True, timeout=60
    )
    assert proc.returncode != 0, "the writer was expected to die mid-write"

    stored = (root / "doc").read_bytes()
    assert stored in (v1, b"B" * 5_000_000), (
        f"the writer died mid-write (exit {proc.returncode}) and the stored "
        f"original is neither version: {len(stored)} bytes, "
        f"starts {stored[:1]!r}"
    )


# ---------------------- INGEST-16: first upload killed, file kept forever


@pytest.mark.xfail(
    strict=True, reason="INGEST-16 in docs/review-2026-10.md: fails until it is fixed"
)
def test_a_first_upload_killed_mid_index_leaves_no_file_nothing_can_delete(
    tmp_path: Path, local_client
) -> None:
    """A first upload's file is written before any record exists. If the
    engine dies while it is embedded, the file must still be reachable by the
    caller's delete (or not kept at all): ChatFrom deletes a chatbot's
    knowledge by listing it and deleting each document."""
    data = b"# Customers\n\nFIRST UPLOAD: jane@example.com, +44 20 7946 0000\n"
    doc_id = doc_id_for(PROJECT, "customers.md")

    engine = _Engine(tmp_path)
    try:
        engine.put_in_background("customers.md", data, "text/markdown")
        engine.wait_for_embedding()
    finally:
        engine.kill()

    with local_client() as client:
        listed = client.get("/documents", params={"project_id": PROJECT}).json()
        # What the app does when the chatbot is deleted: list, delete each.
        for record in listed:
            client.delete(
                f"/documents/{record['doc_id']}", params={"project_id": PROJECT}
            )
        # And a delete aimed at the exact id, which the app never has.
        direct = client.delete(f"/documents/{doc_id}", params={"project_id": PROJECT})

    assert not _blob_path(doc_id).exists(), (
        f"listed={listed!r}; direct delete answered {direct.json()!r}; the "
        f"uploaded file is still on the volume: {_blob_path(doc_id).read_bytes()!r}"
    )


# ---------------------- INGEST-13: more chunks than one Chroma upsert


@pytest.mark.xfail(
    strict=True, reason="INGEST-13 in docs/review-2026-10.md: fails until it is fixed"
)
def test_a_document_of_more_than_5461_chunks_is_indexed_or_refused_before_embedding(
    monkeypatch: pytest.MonkeyPatch, local_client
) -> None:
    """An FAQ export of 6,000 short questions, uploaded with ChatFrom's FAQ
    preset (`headings`, 600/0: sections are never joined). 600 KB, well inside the
    2,000,000-character bound the docs give. It must be indexed, or refused
    before anything is paid for."""
    embedded: list[int] = []

    class Counting(DeterministicFakeEmbedding):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            embedded.append(len(texts))
            return super().embed_documents(texts)

    fake = Counting(size=16)
    monkeypatch.setattr(embeddings_module, "get_embeddings", lambda *a, **k: fake)
    monkeypatch.setattr(vector_store_module, "get_embeddings", lambda *a, **k: fake)

    faq = "".join(
        f"## Question {i}\n\nAnswer number {i} is in the handbook.\n\n"
        for i in range(6000)
    ).encode()
    assert len(faq) < 2_000_000

    with local_client() as client:
        response = _put(
            client,
            "faq.md",
            faq,
            "text/markdown",
            # ChatFrom's "FAQ" preset (libs/chunking.ts DOC_KINDS).
            chunking_strategy="headings",
            chunk_size="600",
            chunk_overlap="0",
        )
        listed = client.get("/documents", params={"project_id": PROJECT}).json()

    assert response.status_code == 201, (
        f"{response.status_code}: {response.text[:200]!r}; record "
        f"{[(r['status'], (r['error'] or '')[:120]) for r in listed]!r}; "
        f"chunks embedded (paid) before the failure: {sum(embedded)}"
    )


# ------------- WORKFLOW-8: in the container uvicorn is PID 1 (WORKFLOW-8)

_ASK_WORKFLOW = {
    "start": "phone",
    "nodes": [
        {
            "id": "phone",
            "type": "ask",
            "prompt": "What number should we call?",
            "input": "phone",
            "var": "phone",
        },
        {"id": "done", "type": "reply", "text": "Thanks."},
    ],
    "edges": [{"from": "phone", "to": "done"}],
}


def _children(pid: int) -> list[int]:
    found: list[int] = []
    for task in Path(f"/proc/{pid}/task").iterdir():
        text = (task / "children").read_text().split()
        found.extend(int(c) for c in text)
    return found


@pytest.mark.skipif(
    os.geteuid() != 0 or not Path("/usr/bin/unshare").exists(),
    reason="needs root and unshare to run the engine as PID 1",
)
@pytest.mark.parametrize(
    "pause_store_opened",
    [
        pytest.param(False, id="no_pause_store"),
        pytest.param(
            True,
            id="pause_store_opened",
            marks=pytest.mark.xfail(
                strict=True,
                reason="WORKFLOW-8 in docs/review-2026-10.md: fails until it is fixed",
            ),
        ),
    ],
)
def test_the_engine_as_pid_1_exits_on_sigterm_with_nothing_in_flight(
    tmp_path: Path, pause_store_opened: bool
) -> None:
    """engine/Dockerfile runs uvicorn as the container's PID 1 (exec-form CMD,
    no init). uvicorn re-raises SIGTERM after its graceful shutdown, but a
    PID-namespace init ignores a signal it raises at its default action, so
    the interpreter then exits normally and waits on every non-daemon thread.
    With nothing in flight, a stop must not need Docker's SIGKILL."""
    marker = tmp_path / "unused-marker"
    script = tmp_path / "launch_engine.py"
    script.write_text(_LAUNCHER)
    port = _free_port()
    env = dict(os.environ)
    env.pop("ENGINE_API_KEY", None)
    env.update(RAGAS_DO_NOT_TRACK="true", ANONYMIZED_TELEMETRY="False")
    log = (tmp_path / "engine.log").open("wb")
    outer = subprocess.Popen(
        [
            "unshare",
            "-p",
            "-f",
            "--mount-proc",
            sys.executable,
            "-I",
            str(script),
            str(marker),
            str(port),
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        cwd=tmp_path,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 90
        while True:
            assert time.monotonic() < deadline, "engine did not start"
            try:
                if httpx.get(f"{url}/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.2)
        (engine_pid,) = _children(outer.pid)
        nspid = next(
            line.split()[1:]
            for line in Path(f"/proc/{engine_pid}/status").read_text().splitlines()
            if line.startswith("NSpid:")
        )
        assert nspid[-1] == "1", f"the engine is not PID 1 in its namespace: {nspid}"

        if pause_store_opened:
            # A visitor's answer to a question that is no longer waiting: the
            # workflow opens its pause store and says resume_expired. No model.
            with httpx.Client(base_url=url, timeout=60) as c:
                body = c.post(
                    "/chat",
                    json={
                        "project": {
                            "project_id": PROJECT,
                            "name": "S",
                            "system_prompt": "You are helpful.",
                            "model": "openai/gpt-5-mini",
                            "agent": "workflow",
                            "workflow": _ASK_WORKFLOW,
                        },
                        "message": "+44 20 7946 0958",
                        "session_id": "s1",
                        "resume": {"thread_id": f"{PROJECT}:0123", "value": "+44"},
                    },
                ).text
            assert "resume_expired" in body, body[:500]

        os.kill(engine_pid, signal.SIGTERM)  # what `docker stop` sends PID 1
        start = time.monotonic()
        while outer.poll() is None and time.monotonic() - start < 10:
            time.sleep(0.1)
        exited = outer.poll() is not None
        took = time.monotonic() - start
    finally:
        if outer.poll() is None:
            for pid in _children(outer.pid):
                os.kill(pid, signal.SIGKILL)
            outer.wait(timeout=30)
        log.close()

    assert exited, (
        f"pause store opened={pause_store_opened}: still running {took:.1f}s "
        "after SIGTERM with nothing in flight; Docker ends this with SIGKILL. "
        "Log tail: " + (tmp_path / "engine.log").read_text()[-400:]
    )
