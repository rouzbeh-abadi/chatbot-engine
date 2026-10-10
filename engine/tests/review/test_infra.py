"""Review repros for the infra / docs / contract slice.

Each test asserts the behaviour the engine's own docs promise (or that a
multi-tenant deployment needs) and fails on 0.1.26.
"""

from __future__ import annotations

import pytest

from chatbot_engine import tracing
from chatbot_engine.api import documents as documents_api
from chatbot_engine.api import extract as extract_api
from chatbot_engine.api.dependencies import reset_dependency_cache
from chatbot_engine.models.chat import AssistantConfig, ChatRequest, TracingConfig

# --- helpers ----------------------------------------------------------------

BOUNDARY = "reviewboundary1234"


def _multipart(fields: dict[str, str], file_bytes: bytes, mimetype: str) -> bytes:
    parts = []
    for name, value in fields.items():
        parts.append(
            f"--{BOUNDARY}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode()
        )
    parts.append(
        f"--{BOUNDARY}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="big.md"\r\n'
        f"Content-Type: {mimetype}\r\n\r\n".encode()
        + file_bytes
        + b"\r\n"
    )
    parts.append(f"--{BOUNDARY}--\r\n".encode())
    return b"".join(parts)


async def _send_counted(app, method: str, path: str, body: bytes, chunk: int):
    """Drive the ASGI app directly; return (status, bytes the app pulled)."""
    chunks = [body[i : i + chunk] for i in range(0, len(body), chunk)]
    consumed = 0
    index = 0
    status: list[int] = []

    async def receive():
        nonlocal consumed, index
        if index >= len(chunks):
            return {"type": "http.disconnect"}
        piece = chunks[index]
        index += 1
        consumed += len(piece)
        return {
            "type": "http.request",
            "body": piece,
            "more_body": index < len(chunks),
        }

    async def send(message):
        if message["type"] == "http.response.start":
            status.append(message["status"])

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", f"multipart/form-data; boundary={BOUNDARY}".encode()),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
    }
    await app(scope, receive, send)
    return (status[0] if status else None), consumed


@pytest.fixture
def open_app(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ENGINE_API_KEY", raising=False)
    reset_dependency_cache()
    from chatbot_engine.app import create_app

    yield create_app()
    reset_dependency_cache()


# --- INFRA-1: the upload limit is checked after the whole body is taken -------

CAP = 64 * 1024  # the 25 MB limit, scaled down so the test stays fast
SLACK = 256 * 1024  # what a streaming check may read past the cap


@pytest.mark.parametrize(
    ("path", "fields", "module"),
    [
        ("/documents", {"project_id": "p", "external_id": "big"}, documents_api),
        ("/extract", {}, extract_api),
    ],
)
@pytest.mark.xfail(
    strict=True, reason="API-1 in docs/review-2026-10.md: fails until it is fixed"
)
async def test_an_upload_over_its_limit_is_refused_before_the_whole_body_is_read(
    open_app, monkeypatch, path, fields, module
) -> None:
    """DEPLOYMENT.md: "PUT /documents and POST /extract keep their own limit of
    25 MB"; body_limit.py: the uploads are "spooled to disk rather than held".
    A body many times the limit should be refused once it passes the limit,
    not after all of it was spooled to disk and then read into memory."""
    monkeypatch.setattr(module, "MAX_UPLOAD_BYTES", CAP)
    body = _multipart(fields, b"a" * (4 * 1024 * 1024), "text/markdown")

    status, consumed = await _send_counted(
        open_app, "PUT" if path == "/documents" else "POST", path, body, 16 * 1024
    )

    assert status == 413
    assert consumed <= CAP + SLACK, (
        f"the engine read {consumed} bytes of a {len(body)}-byte upload before "
        f"refusing it at a {CAP}-byte limit"
    )


@pytest.mark.xfail(
    strict=True, reason="API-1 in docs/review-2026-10.md: fails until it is fixed"
)
async def test_a_40_mib_upload_is_refused_near_the_real_25_mb_limit(open_app) -> None:
    """The same with the shipped constant, so a fix that reads the limit at
    import time is still held to it: refused within a MiB of 25 MB."""
    limit = documents_api.MAX_UPLOAD_BYTES
    body = _multipart(
        {"project_id": "p", "external_id": "big"},
        b"a" * (40 * 1024 * 1024),
        "text/markdown",
    )

    status, consumed = await _send_counted(
        open_app, "PUT", "/documents", body, 1024 * 1024
    )

    assert status == 413
    assert consumed <= limit + 1024 * 1024, (
        f"the engine took {consumed} bytes of a {len(body)}-byte upload "
        f"before refusing it at {limit}"
    )


# --- INFRA-2: unknown multipart fields are accepted silently -----------------


@pytest.mark.xfail(
    strict=True, reason="API-6 in docs/review-2026-10.md: fails until it is fixed"
)
def test_a_misspelled_upload_field_is_refused_not_ignored(client) -> None:
    """docs/backend-integration.md: "Unknown fields are rejected ... A
    misspelled field is a 422 that names it". On the multipart routes a
    misspelled chunking field is dropped and the document is indexed with the
    engine defaults instead."""
    response = client.put(
        "/documents",
        data={
            "project_id": "support",
            "external_id": "faq",
            "chunking_stratgy": "headings",  # misspelled
        },
        files={"file": ("faq.md", b"# Returns\n\nThirty days.", "text/markdown")},
    )

    assert response.status_code == 422, (
        f"{response.status_code}: indexed with chunking_strategy="
        f"{response.json().get('chunking_strategy')!r}"
    )


@pytest.mark.xfail(
    strict=True, reason="API-6 in docs/review-2026-10.md: fails until it is fixed"
)
def test_a_misspelled_extract_field_is_refused_not_ignored(client) -> None:
    """A misspelled `provider_api_key` on /extract would bill an image read to
    the engine's own key without a word; for a document it is ignored too."""
    response = client.post(
        "/extract",
        data={"provider_api_kee": "sk-or-customer"},  # misspelled
        files={"file": ("note.txt", b"hello there", "text/plain")},
    )

    assert response.status_code == 422, response.status_code


# --- INFRA-3: one tenant's Langfuse public key decides another's destination -


def _with_tracing(project_id: str, public_key: str, secret: str, host: str):
    return ChatRequest(
        project=AssistantConfig(
            project_id=project_id,
            name=project_id,
            system_prompt=".",
            tracing=TracingConfig(public_key=public_key, secret_key=secret, host=host),
        ),
        message="hi",
        session_id="s",
        user_id="u",
    )


def test_an_assistant_traces_to_its_own_host_whoever_used_its_public_key_first():
    """DEPLOYMENT.md: `project.tracing` "is how a multi-tenant product lets each
    customer keep their traces on their own Langfuse". The handler is cached
    by public key alone, and Langfuse keeps the first client registered under a
    key, so a tenant that sends another tenant's public key first, with its own
    host, receives that tenant's traces (prompts, messages, extracts)."""
    pytest.importorskip("langfuse")
    shared = "pk-lf-review-victim-0001"

    # The attacker's chatbot is the first after a restart to name the key.
    tracing.run_config(
        _with_tracing("attacker", shared, "sk-anything", "https://attacker.example"),
        name="answer",
    )
    victim = tracing.run_config(
        _with_tracing("victim", shared, "sk-victim", "https://victim.example"),
        name="answer",
    )["callbacks"][0]

    resources = victim._langfuse_client._resources
    assert resources.base_url == "https://victim.example", (
        f"the victim's traces go to {resources.base_url}"
    )
    assert resources.secret_key == "sk-victim"
