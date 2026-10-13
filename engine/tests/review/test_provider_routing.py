"""Review: every model call carries OpenRouter provider routing (TURN-10).

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


def _turn(client: TestClient, provider: dict | None = None) -> list[dict]:
    """One ChatFrom-shaped turn with history (rewrite) and rerank on, with
    `provider` as the request's routing when given."""
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
                **({"provider": provider} if provider is not None else {}),
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


# --- what the app may ask for ------------------------------------------------


def test_the_app_can_send_a_routing_preference(
    client: TestClient, upstream: _Recorder
) -> None:
    """CORRECT behaviour: the caller can pin the host for a chatbot, with
    OpenRouter's own object, `provider`."""
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


def test_the_hosts_a_request_names_serve_the_chatbots_model_only(
    client: TestClient, upstream: _Recorder
) -> None:
    """Hosts named for an anthropic/* chatbot do not serve the utility model's
    rewrite and rerank, which keep to their own maker rather than fail."""
    _turn(client, provider={"only": ["anthropic"], "allow_fallbacks": False})
    for body in upstream.chat():
        maker = "anthropic" if body["model"].startswith("anthropic/") else "openai"
        assert body["provider"]["only"] == [maker], (_kind(body), body["provider"])


def test_a_request_that_names_no_host_keeps_each_call_with_its_maker(
    client: TestClient, upstream: _Recorder
) -> None:
    _turn(client, provider={"data_collection": "deny"})
    for body in upstream.chat():
        assert _routing_keeps_to_maker(body), (_kind(body), body.get("provider"))
        assert body["provider"]["data_collection"] == "deny"


def test_open_weight_models_are_not_kept_to_their_author() -> None:
    """No OpenAI host serves gpt-oss: kept to `openai`, it would find none."""
    from chatbot_engine.agent.routing import routing_for

    assert routing_for("openai/gpt-oss-120b") is None
    assert routing_for("google/gemma-3-27b-it") is None
    assert routing_for("openai/gpt-5-mini") == {
        "only": ["openai"],
        "allow_fallbacks": False,
    }


async def test_ragas_calls_follow_the_requests_routing() -> None:
    """The judge's calls in a RAG evaluation, on another model than the
    chatbot's, keep what of the request's routing holds for any model."""
    import httpx

    from chatbot_engine.agent.routing import RoutedTransport
    from chatbot_engine.models.chat import ProviderRouting

    seen: list[dict] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={})

    routed = RoutedTransport(
        httpx.MockTransport(answer),
        requested=ProviderRouting(only=["anthropic"], data_collection="deny"),
        model="anthropic/claude-haiku-4.5",
    )
    async with httpx.AsyncClient(transport=routed, base_url="http://h/v1") as http:
        await http.post(
            "/chat/completions", json={"model": "anthropic/claude-haiku-4.5"}
        )
        await http.post("/chat/completions", json={"model": "openai/gpt-4.1-mini"})
    assert seen[0]["provider"] == {"only": ["anthropic"], "data_collection": "deny"}
    assert seen[1]["provider"] == {
        "only": ["openai"],
        "allow_fallbacks": False,
        "data_collection": "deny",
    }


async def test_ragas_calls_go_through_the_proxy_the_environment_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """httpx reads no proxy from the environment for a client given a
    transport of its own, so the judge's calls of `/eval/rag`, carried by
    `RoutedTransport`, went straight out on a host that reaches the
    internet only through `HTTPS_PROXY` or `HTTP_PROXY`."""
    import httpx

    from chatbot_engine.agent.routing import RoutedTransport

    recorder = _Recorder()
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), _handler(recorder))
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    for scheme in ("http", "https", "all", "no"):
        monkeypatch.delenv(f"{scheme}_proxy", raising=False)
        monkeypatch.delenv(f"{scheme.upper()}_PROXY", raising=False)
    monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{proxy.server_port}")

    routed = RoutedTransport()
    try:
        async with httpx.AsyncClient(
            transport=routed, base_url="http://openrouter.invalid/api/v1"
        ) as http:
            answer = await http.post(
                "/chat/completions",
                json={"model": "openai/gpt-4.1-mini", "messages": []},
            )
    finally:
        proxy.shutdown()

    assert answer.status_code == 200
    # Asked of the proxy in absolute form, routing and all.
    path, body = recorder.calls[0]
    assert path == "http://openrouter.invalid/api/v1/chat/completions"
    assert body["provider"] == {"only": ["openai"], "allow_fallbacks": False}


def test_a_models_variant_is_not_kept_to_its_maker() -> None:
    """A `:free` model (or any `:variant`) is served by the hosts OpenRouter
    has for that variant, often not the maker: kept to the maker, a call to
    one would find no host at all."""
    from chatbot_engine.agent.routing import routing_for

    assert routing_for("openai/gpt-oss-20b:free") is None
    assert routing_for("anthropic/claude-haiku-4.5:free") is None
    assert routing_for("openai/gpt-5-mini:nitro") is None
    assert routing_for("anthropic/claude-haiku-4.5") == {
        "only": ["anthropic"],
        "allow_fallbacks": False,
    }


def test_a_utility_model_or_judge_on_the_requests_own_model_keeps_its_hosts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request's routing is sent for its model as it resolves: with no
    `model`, the engine's chat model. The rewrite, the rerank, a reading or
    a judge on that same model was taken for another model, and lost the
    hosts the request named."""
    from chatbot_engine.agent.retriever import utility_config
    from chatbot_engine.eval.prompt_evaluation import judge_config
    from chatbot_engine.models.chat import AssistantConfig, ProviderRouting

    monkeypatch.setenv("ENGINE_CHAT_MODEL", "anthropic/claude-haiku-4.5")
    monkeypatch.setenv("ENGINE_UTILITY_MODEL", "anthropic/claude-haiku-4.5")
    get_settings.cache_clear()
    routing = ProviderRouting(only=["anthropic"], data_collection="deny")
    project = AssistantConfig(
        project_id="p", name="n", system_prompt="s", provider=routing
    )
    try:
        assert utility_config(project).provider == routing
        assert judge_config(project, "anthropic/claude-haiku-4.5").provider == routing
        assert judge_config(project).provider == routing
        other = judge_config(project, "openai/gpt-5-mini").provider
        assert other == ProviderRouting(data_collection="deny")
    finally:
        get_settings.cache_clear()


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
