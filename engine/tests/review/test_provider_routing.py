"""Review: no model call carries OpenRouter provider routing (TURN-10).

ChatFrom's DPA and privacy policy promise that a visitor's message goes to the
chosen model's own provider, "never to a provider you did not choose".
OpenRouter routes a model with no `provider` object to any host that serves
it, with fallbacks on. These tests run every model call the engine makes
against a fake OpenAI-compatible server on 127.0.0.1 and read the bodies it
receives; no real provider is contacted.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from chatbot_engine.api.dependencies import reset_dependency_cache
from chatbot_engine.settings import get_settings

# A 1x1 transparent PNG.
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"
)

CHUNKS = {
    "fares.md": "# Fares\n\nThe Basic fare is not refundable. Flex fares are.\n",
    "baggage.md": "# Baggage\n\nEvery fare includes one cabin bag of 8 kg.\n",
    "pets.md": "# Pets\n\nSmall pets travel in the cabin for a fee.\n",
}


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.lock = threading.Lock()

    def chat(self) -> list[dict]:
        return [body for path, body in self.calls if path.endswith("/chat/completions")]

    def embeddings(self) -> list[dict]:
        return [body for path, body in self.calls if path.endswith("/embeddings")]


def _handler(recorder: _Recorder) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: object) -> None:  # quiet
            pass

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            with recorder.lock:
                recorder.calls.append((self.path, body))
            model = body.get("model", "m")
            usage = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}

            if self.path.endswith("/embeddings"):
                inputs = body.get("input")
                count = len(inputs) if isinstance(inputs, list) else 1
                data = [
                    {"object": "embedding", "index": i, "embedding": [0.1] * 8}
                    for i in range(count)
                ]
                payload = {
                    "object": "list",
                    "data": data,
                    "model": model,
                    "usage": usage,
                }
                self._send(200, json.dumps(payload).encode(), "application/json")
                return

            system = ""
            for message in body.get("messages", []):
                if message.get("role") == "system" and isinstance(
                    message.get("content"), str
                ):
                    system += message["content"]
            text = '{"ranking": [2, 1]}' if "ranking" in system else "basic fare refund"

            if body.get("stream"):
                chunks = [
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": text},
                                "finish_reason": None,
                            }
                        ]
                    },
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                    {"choices": [], "usage": usage},
                ]
                out = b""
                for chunk in chunks:
                    chunk.update(
                        {
                            "id": "c",
                            "object": "chat.completion.chunk",
                            "created": 0,
                            "model": model,
                        }
                    )
                    out += b"data: " + json.dumps(chunk).encode() + b"\n\n"
                out += b"data: [DONE]\n\n"
                self._send(200, out, "text/event-stream")
                return

            payload = {
                "id": "c",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage,
            }
            self._send(200, json.dumps(payload).encode(), "application/json")

    return Handler


@pytest.fixture
def upstream(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Recorder]:
    """A fake OpenRouter on 127.0.0.1; the engine's base URL points at it.

    The utility model is ChatFrom's production one (deploy/docker-compose.yml).
    """
    recorder = _Recorder()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(recorder))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv(
        "ENGINE_OPENROUTER_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1"
    )
    monkeypatch.setenv("ENGINE_UTILITY_MODEL", "openai/gpt-4.1-mini")
    monkeypatch.setenv("ENGINE_PROVIDER_MAX_RETRIES", "0")
    reset_dependency_cache()
    yield recorder
    server.shutdown()
    server.server_close()
    reset_dependency_cache()


def _routing_keeps_to_maker(body: dict) -> bool:
    """True when the body carries OpenRouter routing that cannot leave the maker.

    Either `provider.only` / `provider.order` with fallbacks off. A body with no
    `provider` object routes by OpenRouter's defaults: any host, fallbacks on.
    """
    provider = body.get("provider")
    if not isinstance(provider, dict):
        return False
    pinned = bool(provider.get("only") or provider.get("order"))
    return pinned and provider.get("allow_fallbacks") is False


def _seed(client: TestClient) -> None:
    for name, text in CHUNKS.items():
        response = client.put(
            "/documents",
            data={
                "project_id": "support",
                "external_id": name,
                "chunking_strategy": "size",
            },
            files={"file": (name, text.encode(), "text/markdown")},
        )
        assert response.status_code == 201, response.text


def _turn(client: TestClient) -> list[dict]:
    """One ChatFrom-shaped turn with history (rewrite) and rerank on."""
    _seed(client)
    response = client.post(
        "/chat",
        json={
            "project": {
                "project_id": "support",
                "name": "Support",
                "system_prompt": "You are a fixture.",
                "model": "anthropic/claude-haiku-4.5",
                "retrieval": "hybrid",
                "rerank": True,
                "min_score": 0.0,
                "top_k": 2,
            },
            "message": "and is the Basic one refundable?",
            "history": [
                {"role": "user", "content": "Tell me about fares."},
                {"role": "assistant", "content": "There are Basic and Flex fares."},
            ],
        },
    )
    assert response.status_code == 200, response.text
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert any(e.get("type") == "done" for e in events), events
    return events


def _kind(body: dict) -> str:
    system = " ".join(
        m["content"]
        for m in body.get("messages", [])
        if m.get("role") == "system" and isinstance(m.get("content"), str)
    )
    if "standalone search queries" in system:
        return "rewrite"
    if "ranking" in system:
        return "rerank"
    return "answer"


# --- what reaches OpenRouter ---------------------------------------------------


def test_the_turn_makes_all_three_calls(
    client: TestClient, upstream: _Recorder
) -> None:
    """Sanity: the fake upstream sees the rewrite, the rerank and the answer,
    each naming the model ChatFrom chose (the utility model for the small calls)."""
    _turn(client)
    kinds = {_kind(b): b["model"] for b in upstream.chat()}
    assert kinds == {
        "rewrite": "openai/gpt-4.1-mini",
        "rerank": "openai/gpt-4.1-mini",
        "answer": "anthropic/claude-haiku-4.5",
    }, kinds


@pytest.mark.parametrize("kind", ["answer", "rewrite", "rerank"])
@pytest.mark.xfail(
    strict=True, reason="TURN-10 in docs/review-2026-10.md: fails until it is fixed"
)
def test_every_turn_call_is_kept_with_the_models_maker(
    client: TestClient, upstream: _Recorder, kind: str
) -> None:
    """CORRECT behaviour: the answer to an anthropic/* chatbot, and the small
    calls on ChatFrom's openai/* utility model, carry routing that keeps them
    with the model's maker (fallbacks off), as ethics.md and the DPA promise."""
    _turn(client)
    bodies = [b for b in upstream.chat() if _kind(b) == kind]
    assert bodies, f"no {kind} call was made"
    for body in bodies:
        assert _routing_keeps_to_maker(body), (
            f"{kind} call for {body['model']} has no provider routing; keys: {sorted(body)}"
        )


@pytest.mark.xfail(
    strict=True, reason="TURN-10 in docs/review-2026-10.md: fails until it is fixed"
)
def test_an_image_read_is_kept_with_the_models_maker(
    client: TestClient, upstream: _Recorder
) -> None:
    """POST /extract as ChatFrom sends it (no `model`: the utility model reads)."""
    response = client.post("/extract", files={"file": ("shot.png", PNG, "image/png")})
    assert response.status_code == 200, response.text
    bodies = upstream.chat()
    assert len(bodies) == 1 and bodies[0]["model"] == "openai/gpt-4.1-mini"
    assert _routing_keeps_to_maker(bodies[0]), (
        f"image read has no provider routing; keys: {sorted(bodies[0])}"
    )


@pytest.mark.xfail(
    strict=True, reason="TURN-10 in docs/review-2026-10.md: fails until it is fixed"
)
def test_an_embedding_is_kept_with_the_models_maker(upstream: _Recorder) -> None:
    """The embedder the engine builds for indexing and search.

    `check_embedding_ctx_length` is switched off only so tiktoken does not
    download its vocabulary here; it changes how the input is tokenized, not
    which top-level keys the body has.
    """
    from chatbot_engine.rag.embeddings import build_embeddings

    embedder = build_embeddings(get_settings())
    embedder.check_embedding_ctx_length = False
    embedder.embed_query("is the Basic fare refundable?")
    bodies = upstream.embeddings()
    assert len(bodies) == 1 and bodies[0]["model"] == "openai/text-embedding-3-small"
    assert _routing_keeps_to_maker(bodies[0]), (
        f"embedding call has no provider routing; keys: {sorted(bodies[0])}"
    )


# --- no way for the app to ask for it ------------------------------------------


@pytest.mark.xfail(
    strict=True, reason="TURN-10 in docs/review-2026-10.md: fails until it is fixed"
)
def test_the_app_can_send_a_routing_preference(
    client: TestClient, upstream: _Recorder
) -> None:
    """CORRECT behaviour: the caller can pin the host for a chatbot. Any field
    would do; OpenRouter's own name, `provider`, is used here. Today the
    request model forbids unknown fields, so it is a 422."""
    response = client.post(
        "/chat",
        json={
            "project": {
                "project_id": "support",
                "name": "Support",
                "system_prompt": "You are a fixture.",
                "model": "anthropic/claude-haiku-4.5",
                "provider": {"only": ["anthropic"], "allow_fallbacks": False},
            },
            "message": "hi",
        },
    )
    assert response.status_code == 200, response.text[:300]


# --- added in verification ---------------------------------------------------


def test_the_engine_never_switches_model_within_a_turn(
    client: TestClient, upstream: _Recorder
) -> None:
    _turn(client)
    models = {b["model"] for b in upstream.chat()}
    # Only the chatbot's model and the utility model ever appear.
    assert models == {"anthropic/claude-haiku-4.5", "openai/gpt-4.1-mini"}, models
    for body in upstream.chat():
        assert "models" not in body and "route" not in body
