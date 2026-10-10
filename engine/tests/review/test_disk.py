"""Review: one tenant fills or churns the shared volume (DISK-1 to DISK-4).

ChatFrom runs one engine for every owner, with Chroma (one collection shared by
every chatbot), the SQLite registry, the uploaded originals and the workflow
checkpoints on ONE volume (deploy/docker-compose.yml: engine_data). These tests
ask what one owner can make the engine write there, whether deleting gives it
back, and what a full volume does to everyone else.

Every test asserts the behaviour the engine should have and fails on 0.1.26.
Offline: Chroma 1.5.9 for real, embeddings are DeterministicFakeEmbedding at
1536 dimensions (the size of ChatFrom's model, openai/text-embedding-3-small),
no provider is called. The full-volume tests mount a small tmpfs, so they need
root; elsewhere they skip.
"""

from __future__ import annotations

import asyncio
import os
import random
import socket
import subprocess
import sys
import textwrap
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import httpx
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from chatbot_engine.api.dependencies import reset_dependency_cache
from chatbot_engine.rag import embeddings as embeddings_module
from chatbot_engine.rag import vector_store as vector_store_module
from chatbot_engine.settings import get_settings

#: openai/text-embedding-3-small, ChatFrom's (and the engine's default) model.
DIMS = 1536
WORDS = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "sigma",
    "omega",
]


# --------------------------------------------------------------------------- helpers


def _fake_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = DeterministicFakeEmbedding(size=DIMS)
    monkeypatch.setattr(embeddings_module, "get_embeddings", lambda *a, **k: fake)
    monkeypatch.setattr(vector_store_module, "get_embeddings", lambda *a, **k: fake)
    reset_dependency_cache()


def _words(count: int, seed: int) -> bytes:
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(count)).encode()


def _du(*paths: Path) -> int:
    """Bytes the files under `paths` take on disk."""
    total = 0
    for root in paths:
        if root.is_file():
            total += root.stat().st_blocks * 512
            continue
        for directory, _, files in os.walk(root):
            for name in files:
                total += (Path(directory) / name).stat().st_blocks * 512
    return total


def _volume() -> int:
    """What engine_data holds: vectors, registry, originals, checkpoints."""
    s = get_settings()
    extra = [
        p for p in s.registry_db.parent.glob("*.sqlite3*") if p.parent != s.chroma_dir
    ]
    return _du(s.chroma_dir, s.blob_dir, *extra)


@pytest.fixture
async def engine(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[httpx.AsyncClient]:
    """The real app, open (no key), on this test's event loop, as under uvicorn."""
    monkeypatch.delenv("ENGINE_API_KEY", raising=False)
    _fake_vectors(monkeypatch)
    from chatbot_engine.app import create_app

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(
        transport=transport, base_url="http://engine:8100", timeout=120
    ) as client:
        yield client
    reset_dependency_cache()


# ------------------- DISK-1: a delete gives nothing back to the volume


@pytest.mark.xfail(
    strict=True, reason="DISK-1 in docs/review-2026-10.md: fails until it is fixed"
)
async def test_deleting_a_tenants_documents_gives_its_space_back(
    engine: httpx.AsyncClient,
) -> None:
    """An owner adds a 9 KB source with the chunking ChatFrom lets them pick
    (Length, piece size 100, overlap 99: the app's bounds are 100-8000 and
    0-2000, overlap below size), then removes it, four times over. The
    knowledge base is empty after every round, so the app's own quota (the
    `size_bytes` of the sources listed) never sees anything. The engine's
    volume should be back where it was after the first round."""
    owner = "6f1c0a52-0000-4000-8000-00000000000b"
    after: list[int] = []
    chunks: list[int] = []
    for round_ in range(4):
        data = _words(1500, seed=round_)
        put = await engine.put(
            "/documents",
            data={
                "project_id": owner,
                "external_id": f"notes-{round_}.txt",
                "chunking_strategy": "size",
                "chunk_size": "100",
                "chunk_overlap": "99",
            },
            files={"file": (f"notes-{round_}.txt", data, "text/plain")},
        )
        assert put.status_code == 201, put.text
        chunks.append(put.json()["chunk_count"])
        gone = await engine.delete(
            f"/documents/{put.json()['doc_id']}", params={"project_id": owner}
        )
        assert gone.json()["deleted"] is True
        listed = (await engine.get("/documents", params={"project_id": owner})).json()
        assert listed == [], "precondition: the owner holds nothing"
        after.append(_volume())

    grown = after[-1] - after[0]
    assert grown < 2_000_000, (
        f"{len(data):,}-byte sources cut into {chunks} chunks; with nothing left "
        f"indexed the volume holds {after[0]:,} B after the first round and "
        f"{after[-1]:,} B after the fourth: +{grown:,} B kept for documents that "
        "no longer exist"
    )


async def test_a_forgotten_paused_turn_gives_its_space_back(tmp_path: Path) -> None:
    """WORKFLOW-4's turn: a tool on the owner's own server answers with 1 MB,
    six reply steps, then an ask step that pauses. The pause is then
    forgotten (what a finished, stopped or expired turn does: `forget`).
    The engine keeps the checkpoint connection open for its whole life, as
    here; the file it holds on the volume should shrink back."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine" / "tests"))
    pytest.importorskip("langgraph_agent.workflow")
    from test_workflow_ask import (  # type: ignore[import-not-found]
        Tools,
        _asked,
        _request,
        _run,
    )

    from langgraph_agent.pauses import Pauses

    class BigResult(Tools):
        """A tool on the owner's own MCP server that answers with one megabyte."""

        async def list_tools(self, config):
            return [
                {
                    "server": "s",
                    "name": "list_slots",
                    "description": "d",
                    "input_schema": {},
                }
            ]

        async def call_tool(self, *, name, arguments, **kwargs):
            return "x" * 1_000_000

    spec = {
        "start": "slots",
        "nodes": [
            {"id": "slots", "type": "tool", "tool": "list_slots", "var": "slots"},
            *({"id": f"say{i}", "type": "reply", "text": "."} for i in range(6)),
            {
                "id": "mail",
                "type": "ask",
                "prompt": "Email?",
                "input": "email",
                "var": "email",
            },
        ],
        "edges": [
            {"from": "slots", "to": "say0"},
            *({"from": f"say{i}", "to": f"say{i + 1}"} for i in range(5)),
            {"from": "say5", "to": "mail"},
        ],
    }
    path = tmp_path / "checkpoints.sqlite3"
    pauses = Pauses.sqlite(path, ttl_s=86_400)
    try:
        for k in range(3):
            thread = _asked(
                await _run(
                    _request(spec, "hi", session_id=f"s{k}"), pauses, BigResult()
                )
            ).thread_id
        peak = _du(*tmp_path.glob("checkpoints.sqlite3*"))
        for row in await (
            await pauses._conn.execute("SELECT thread_id FROM workflow_pauses")
        ).fetchall():
            await pauses.forget(row[0])
        left = await (
            await pauses._conn.execute("SELECT COUNT(*) FROM checkpoints")
        ).fetchone()
        kept = _du(*tmp_path.glob("checkpoints.sqlite3*"))
    finally:
        if pauses._conn is not None:
            await pauses._conn.close()

    assert left[0] == 0, "precondition: every checkpoint was deleted"
    assert kept < 2_000_000, (
        f"three paused turns took {peak:,} B; with every one forgotten and no "
        f"checkpoint left ({thread[:12]}... was the last), the checkpoint files "
        f"still hold {kept:,} B of the volume"
    )


# ------- DISK-3: writing vectors holds the GIL, so every tenant waits


async def test_an_upload_does_not_stop_the_engine_answering_while_its_vectors_are_written(
    engine: httpx.AsyncClient,
) -> None:
    """One owner indexes a 2 MB handbook (the engine's own `index_max_chars`)
    with ChatFrom's defaults (Sections, 1000/200): 2,000 sections, 2,000 chunks. While
    it is written, the engine should go on answering everyone else: here a
    /health probe every 20 ms on the same event loop, standing for the tokens
    of every other chatbot's answer in flight."""
    import chromadb.api.models.Collection as collection_module

    original = collection_module.Collection.upsert
    upserts: list[tuple[int, float]] = []

    def timed_upsert(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return original(self, *args, **kwargs)
        finally:
            upserts.append(
                (len(kwargs.get("ids") or args[0]), time.perf_counter() - started)
            )

    collection_module.Collection.upsert = timed_upsert  # type: ignore[method-assign]
    try:
        handbook = ("## Section\n\n" + _words(160, 1).decode() + "\n\n") * 2_000
        handbook = handbook[:2_000_000]
        done = asyncio.Event()
        worst = 0.0

        async def probe() -> None:
            # A request every 20 ms; what counts is how long the loop went
            # without serving one, including time it could not wake from the
            # sleep (the event loop thread waiting for the GIL).
            nonlocal worst
            while not done.is_set():
                started = time.perf_counter()
                assert (await engine.get("/health")).status_code == 200
                await asyncio.sleep(0.02)
                worst = max(worst, time.perf_counter() - started - 0.02)

        prober = asyncio.create_task(probe())
        await asyncio.sleep(0.2)
        put = await engine.put(
            "/documents",
            data={
                "project_id": "6f1c0a52-0000-4000-8000-00000000000a",
                "external_id": "handbook.md",
                "chunking_strategy": "headings",
                "chunk_size": "1000",
                "chunk_overlap": "200",
            },
            files={"file": ("handbook.md", handbook.encode(), "text/markdown")},
        )
        done.set()
        await prober
    finally:
        collection_module.Collection.upsert = original  # type: ignore[method-assign]

    assert put.status_code == 201, put.text
    assert worst < 0.5, (
        f"while {put.json()['chunk_count']} chunks were written (Chroma upserts "
        f"{[(n, round(s, 2)) for n, s in upserts]} as (chunks, seconds)), the "
        f"engine did not answer /health for {worst:.2f} s"
    )


# ----------------------------------------------- a real engine on a small volume

_LAUNCHER = textwrap.dedent(
    """
    import sys
    port = sys.argv[1]

    from langchain_core.embeddings import DeterministicFakeEmbedding
    import chatbot_engine.rag.embeddings as E
    import chatbot_engine.rag.vector_store as V

    fake = DeterministicFakeEmbedding(size=1536)
    E.get_embeddings = lambda *a, **k: fake
    V.get_embeddings = lambda *a, **k: fake

    import uvicorn
    sys.argv = ["uvicorn", "chatbot_engine.app:app", "--host", "127.0.0.1",
                "--port", port, "--proxy-headers"]
    sys.exit(uvicorn.main())
    """
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def small_volume(tmp_path: Path) -> Iterator[Path]:
    """engine_data as a 128 MB tmpfs: a volume that can be filled in a test."""
    if os.geteuid() != 0:
        pytest.skip("mounting a size-limited tmpfs needs root")
    volume = tmp_path / "engine_data"
    volume.mkdir()
    mounted = subprocess.run(
        ["mount", "-t", "tmpfs", "-o", "size=128m", "tmpfs", str(volume)],
        capture_output=True,
        text=True,
    )
    if mounted.returncode:
        pytest.skip(f"cannot mount a tmpfs here: {mounted.stderr.strip()}")
    yield volume
    subprocess.run(["umount", "-l", str(volume)], capture_output=True)


class _Engine:
    """The engine as the image starts it (uvicorn CLI), on `volume`."""

    def __init__(self, tmp_path: Path, volume: Path) -> None:
        script = tmp_path / "launch_engine.py"
        script.write_text(_LAUNCHER)
        self.port = _free_port()
        env = dict(os.environ)
        env.pop("ENGINE_API_KEY", None)
        env.pop("ENGINE_ENV", None)
        env.update(
            ANONYMIZED_TELEMETRY="False",
            ENGINE_TRACING="off",
            # deploy/docker-compose.yml: everything on the one volume.
            ENGINE_CHROMA_DIR=str(volume / "chroma"),
            ENGINE_REGISTRY_DB=str(volume / "documents.sqlite3"),
            ENGINE_BLOB_DIR=str(volume / "blobs"),
            ENGINE_CHECKPOINT_DB=str(volume / "checkpoints.sqlite3"),
            # A floor to the scale of the test's small volume; the default
            # (512 MB) would refuse every upload on it from the start.
            ENGINE_MIN_FREE_MB="8",
        )
        self.log = tmp_path / "engine.log"
        self._log = self.log.open("wb")
        self.proc = subprocess.Popen(
            [sys.executable, "-I", str(script), str(self.port)],
            env=env,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            cwd=tmp_path,
        )
        self.url = f"http://127.0.0.1:{self.port}"
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            assert self.proc.poll() is None, self.log.read_text()[-3000:]
            if self.answers(timeout=1):
                return
            time.sleep(0.2)
        raise AssertionError("engine did not start:\n" + self.log.read_text()[-3000:])

    def answers(self, timeout: float = 5) -> bool:
        try:
            return httpx.get(f"{self.url}/health", timeout=timeout).status_code == 200
        except httpx.HTTPError:
            return False

    def put(self, project: str, name: str, data: bytes, **form: str) -> str:
        try:
            r = httpx.put(
                f"{self.url}/documents",
                data={"project_id": project, "external_id": name, **form},
                files={"file": (name, data, "text/plain")},
                timeout=30,
            )
        except httpx.HTTPError as exc:
            return f"no answer ({type(exc).__name__})"
        return f"{r.status_code} {r.text[:120]}"

    def delete(self, project: str, doc_id: str) -> str:
        try:
            r = httpx.delete(
                f"{self.url}/documents/{doc_id}",
                params={"project_id": project},
                timeout=30,
            )
        except httpx.HTTPError as exc:
            return f"no answer ({type(exc).__name__})"
        return f"{r.status_code} {r.text[:120]}"

    def listed(self, project: str) -> list[dict]:
        r = httpx.get(
            f"{self.url}/documents", params={"project_id": project}, timeout=30
        )
        return r.json()

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait(timeout=30)
        self._log.close()


def _fill(volume: Path, leave: int) -> Path:
    """What the rest of one tenant's writes do: leave `leave` bytes free."""
    filler = volume / "other-tenant-bytes"
    with filler.open("wb") as f:
        try:
            while True:
                st = os.statvfs(volume)
                if st.f_bavail * st.f_frsize <= leave:
                    break
                f.write(b"\0" * 4096)
                f.flush()
        except OSError:
            pass
    return filler


TRICK = {"chunking_strategy": "size", "chunk_size": "100", "chunk_overlap": "99"}
B = "6f1c0a52-0000-4000-8000-00000000000b"  # the tenant who fills the volume
A = "6f1c0a52-0000-4000-8000-00000000000a"  # everyone else
C = "6f1c0a52-0000-4000-8000-00000000000c"


# --------- DISK-2: a full volume freezes the engine, for every tenant


def _reindex(engine: _Engine, project: str, doc_id: str) -> str:
    try:
        r = httpx.post(
            f"{engine.url}/documents/{doc_id}/reindex",
            params={"project_id": project},
            json=TRICK | {"chunk_size": 100, "chunk_overlap": 99},
            timeout=30,
        )
    except httpx.HTTPError as exc:
        return f"no answer ({type(exc).__name__})"
    return f"{r.status_code} {r.text[:120]}"


@pytest.mark.xfail(
    strict=True, reason="DISK-3 in docs/review-2026-10.md: fails until it is fixed"
)
def test_a_full_volume_does_not_freeze_the_engine_for_every_tenant(
    tmp_path: Path, small_volume: Path
) -> None:
    """Tenant A has sources indexed; tenant B indexes and removes one, then
    the volume fills up. A keeps working on its sources (re-index, remove),
    which must fail while there is no room, but the engine must keep
    answering every tenant (here GET /health, which needs nothing but the
    event loop), and must work again once room is made."""
    engine = _Engine(tmp_path, small_volume)
    steps: list[str] = []
    frozen = None
    came_back = None
    try:
        for k in range(6):
            steps.append(
                f"A put: {engine.put(A, f'a{k}.txt', _words(1200, k), **TRICK)[:12]}"
            )
        steps.append(
            f"B put: {engine.put(B, 'big.txt', _words(3000, 7), **TRICK)[:12]}"
        )
        for doc in engine.listed(B):
            steps.append(f"B delete: {engine.delete(B, doc['doc_id'])[:12]}")
        docs = [d["doc_id"] for d in engine.listed(A)]
        filler = _fill(
            small_volume, leave=int(os.environ.get("REVIEW_LEAVE", "1048576"))
        )
        ops = []
        for round_ in range(3):
            for doc_id in docs:
                ops.append(
                    (
                        f"{round_} A reindex {doc_id[:6]}",
                        lambda d=doc_id: _reindex(engine, A, d),
                    )
                )
        ops += [
            (f"A delete {d[:6]}", lambda d=d: engine.delete(A, d)) for d in docs[:3]
        ]
        for name, op in ops:
            out = op()
            steps.append(f"{name}: {out[:60]}")
            if not engine.answers(timeout=10):
                frozen = name
                break
        filler.unlink()
        if frozen:
            deadline = time.monotonic() + 30
            came_back = False
            while time.monotonic() < deadline and not came_back:
                came_back = engine.answers(timeout=5)
        else:
            for doc_id in docs[3:]:
                out = _reindex(engine, A, doc_id)
                steps.append(f"after free A reindex: {out[:40]}")
                if not engine.answers(timeout=10):
                    frozen = f"after free, A reindex {doc_id[:6]}"
                    came_back = False
                    break
        alive = engine.proc.poll() is None
    finally:
        engine.kill()

    assert frozen is None, (
        f"the engine (process alive: {alive}) stopped answering after: {frozen}; "
        f"30 s after the room was made it "
        f"{'answered again' if came_back else 'still did not answer'}. Steps: {steps}"
    )


# --- DISK-4: a write that meets a full volume corrupts the shared collection


def _fill_then_fail_an_upload(engine: _Engine, volume: Path) -> list[str]:
    """C has a source indexed; B adds and removes one; then the volume fills
    and A adds a source, which fails. Then room is made."""
    steps = [f"C put: {engine.put(C, 'hours.txt', _words(300, 3), **TRICK)[:12]}"]
    steps.append(f"B put: {engine.put(B, 'big.txt', _words(3000, 7), **TRICK)[:12]}")
    for doc in engine.listed(B):
        steps.append(f"B delete: {engine.delete(B, doc['doc_id'])[:12]}")
    filler = _fill(volume, leave=1_048_576)
    steps.append(
        f"A put while full: {engine.put(A, 'price-list.txt', _words(2500, 1), **TRICK)[:40]}"
    )
    filler.unlink()
    return steps


async def _search(monkeypatch: pytest.MonkeyPatch, volume: Path, project: str):
    """The vector half of a chat turn's retrieval (agent/retriever.py `_dense`),
    in a fresh process state on `volume`: what a restarted engine does."""
    from chatbot_engine.agent.retriever import _dense
    from chatbot_engine.rag.vector_store import open_vector_store

    monkeypatch.setenv("ENGINE_CHROMA_DIR", str(volume / "chroma"))
    _fake_vectors(monkeypatch)
    try:
        hits = await _dense(open_vector_store(), project, "opening hours", 5)
        return f"{len(hits)} hits"
    except Exception as exc:
        return f"{type(exc).__name__}: {str(exc)[:200]}"
    finally:
        reset_dependency_cache()


@pytest.mark.xfail(
    strict=True, reason="DISK-4 in docs/review-2026-10.md: fails until it is fixed"
)
async def test_a_write_that_meets_a_full_volume_leaves_every_tenants_knowledge_readable(
    tmp_path: Path, small_volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tenant A's upload meets the full volume and fails; room is made; the
    engine is restarted (a deploy, a crash, the container's memory limit, or
    an operator clearing the freeze above). Tenant C never touched anything:
    its chatbot must still find its source, and it must still be able to add
    one."""
    engine = _Engine(tmp_path, small_volume)
    try:
        steps = _fill_then_fail_an_upload(engine, small_volume)
    finally:
        engine.kill()

    restarted = _Engine(tmp_path, small_volume)
    try:
        added = restarted.put(C, "faq.txt", b"We open at nine.")
    finally:
        restarted.kill()
    found = await _search(monkeypatch, small_volume, C)

    assert found == "5 hits" and added.startswith("201"), (
        f"{steps}; after the restart, tenant C's search: {found!r}; tenant C "
        f"adding a source: {added[:160]!r}"
    )


# ------ DISK-5: a failed upload's chunks stay, under no record


@pytest.mark.xfail(
    strict=True, reason="INGEST-4 in docs/review-2026-10.md: fails until it is fixed"
)
def test_an_upload_refused_for_lack_of_room_leaves_no_chunks_behind(
    tmp_path: Path, small_volume: Path
) -> None:
    """The upload that met the full volume was answered 500, so the app tells
    the owner it failed and lists nothing (it lists what GET /documents
    lists). Whatever the engine kept of it must be listed, so that the app can
    show and delete it, or not be kept at all."""
    engine = _Engine(tmp_path, small_volume)
    try:
        steps = _fill_then_fail_an_upload(engine, small_volume)
        # Room again; C adds a source, which also rewrites the vector
        # index's metadata (without it the collection cannot be read at all).
        steps.append(f"C put: {engine.put(C, 'faq.txt', b'We open at nine.')[:12]}")
        listed = engine.listed(A)
    finally:
        engine.kill()

    import chromadb

    client = chromadb.PersistentClient(path=str(small_volume / "chroma"))
    (collection,) = client.list_collections()
    held = collection.get(where={"project_id": A}, include=["metadatas"])
    names = sorted({m["source"] for m in held["metadatas"]})

    assert not held["ids"] or listed, (
        f"{steps}; GET /documents lists {listed} for tenant A, yet the shared "
        f"collection holds {len(held['ids'])} of its chunks (of {names}), "
        "which no DELETE can reach: the engine deletes only what its registry names"
    )


# ---- DISK-6: a turn that meets the full volume keeps its answers forever


@pytest.mark.xfail(
    strict=True, reason="WORKFLOW-5 in docs/review-2026-10.md: fails until it is fixed"
)
async def test_a_turn_that_meets_a_full_volume_keeps_no_answers_after_the_ttl(
    tmp_path: Path,
) -> None:
    """A visitor answers an Ask step with their email; the rest of the turn
    meets the full volume and fails. Room is made, the engine goes on serving
    workflow turns, two days pass (the pause TTL is one). Nothing of the
    failed turn may still hold the visitor's email."""
    if os.geteuid() != 0:
        pytest.skip("mounting a size-limited tmpfs needs root")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine" / "tests"))
    pytest.importorskip("langgraph_agent.workflow")
    from unittest.mock import patch

    from test_workflow_ask import (  # type: ignore[import-not-found]
        Tools,
        _asked,
        _request,
        _run,
    )

    from langgraph_agent.pauses import Pauses

    class BigResult(Tools):
        async def list_tools(self, config):
            return [
                {
                    "server": "s",
                    "name": "list_slots",
                    "description": "d",
                    "input_schema": {},
                }
            ]

        async def call_tool(self, *, name, arguments, **kwargs):
            return "x" * 1_000_000

    spec = {
        "start": "mail",
        "nodes": [
            {
                "id": "mail",
                "type": "ask",
                "prompt": "Email?",
                "input": "email",
                "var": "email",
            },
            {"id": "slots", "type": "tool", "tool": "list_slots", "var": "slots"},
            *({"id": f"say{i}", "type": "reply", "text": "."} for i in range(8)),
            {
                "id": "when",
                "type": "ask",
                "prompt": "When?",
                "input": "text",
                "var": "when",
            },
        ],
        "edges": [
            {"from": "mail", "to": "slots"},
            {"from": "slots", "to": "say0"},
            *({"from": f"say{i}", "to": f"say{i + 1}"} for i in range(7)),
            {"from": "say7", "to": "when"},
        ],
    }
    email = "ann.private@example.com"
    volume = tmp_path / "engine_data"
    volume.mkdir()
    mounted = subprocess.run(
        ["mount", "-t", "tmpfs", "-o", "size=24m", "tmpfs", str(volume)],
        capture_output=True,
        text=True,
    )
    if mounted.returncode:
        pytest.skip(f"cannot mount a tmpfs here: {mounted.stderr.strip()}")
    try:
        filler = volume / "other-tenant-bytes"
        filler.write_bytes(b"\0" * 10_000_000)
        path = volume / "checkpoints.sqlite3"
        pauses = Pauses.sqlite(path, ttl_s=86_400)
        try:
            thread = _asked(
                await _run(_request(spec, "hi", session_id="v1"), pauses, BigResult())
            ).thread_id
            try:
                await _run(
                    _request(
                        spec,
                        email,
                        {"thread_id": thread, "value": email},
                        session_id="v1",
                    ),
                    pauses,
                    BigResult(),
                )
                failed = "no"
            except Exception as exc:  # the turn fails: no room
                failed = f"{type(exc).__name__}: {exc}"
            filler.unlink()
            # The engine goes on: another chatbot's workflow turn pauses, two
            # days later, which is when expired pauses are pruned.
            with patch(
                "langgraph_agent.pauses.time.time",
                return_value=time.time() + 2 * 86_400,
            ):
                other = await _run(
                    _request(spec, "hello", project_id="other", session_id="v2"),
                    pauses,
                    Tools(),
                )
            assert _asked(other).node == "mail", "precondition: the store works again"
        finally:
            if pauses._conn is not None:
                await pauses._conn.close()
        import sqlite3

        db = sqlite3.connect(path)
        kept = db.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ? AND instr(checkpoint, ?) > 0",
            (thread, email.encode()),
        ).fetchone()[0]
        db.close()
    finally:
        subprocess.run(["umount", "-l", str(volume)], capture_output=True)

    assert kept == 0, (
        f"the turn failed ({failed[:80]}); two days later {kept} checkpoints of it "
        "still hold the visitor's email, under no pause record, so no prune will ever "
        "reach them"
    )
