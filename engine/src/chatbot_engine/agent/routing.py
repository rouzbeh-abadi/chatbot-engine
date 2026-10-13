"""Which hosts OpenRouter may serve a model from.

OpenRouter serves a model from every host that offers it, and falls back to
another when one fails: a call to a Claude model can be served by Amazon
Bedrock or Google Vertex, one to a GPT model by Azure. A caller that tells
the people behind its chatbots their messages go to the model's own maker
needs every call kept there (docs/review-2026-10.md, TURN-10). So, unless
told otherwise (`ENGINE_PROVIDER_ROUTING`, or a request's `project.provider`),
each call names its model's maker as the only host, with no fallback.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from chatbot_engine.models.chat import ProviderRouting
from chatbot_engine.settings import get_settings

#: The hosts on OpenRouter that are a model's own maker, by the author in its
#: id (`anthropic/claude-haiku-4.5`). A maker that serves no model there, as
#: DeepSeek served none of its V4 models in October 2026, is not listed: its
#: models go by OpenRouter's defaults unless a request names hosts.
MAKER_HOSTS: dict[str, tuple[str, ...]] = {
    "anthropic": ("anthropic",),
    "openai": ("openai",),
    "google": ("google-ai-studio", "google-vertex"),
    "mistralai": ("mistral",),
    "x-ai": ("xai",),
}

#: Models a listed maker publishes as open weights, which other companies
#: serve and the maker may not (no OpenAI host serves `openai/gpt-oss-120b`).
#: Kept to the maker, a call to one would find no host at all, so they go by
#: OpenRouter's defaults unless a request names hosts.
OPEN_WEIGHTS: tuple[str, ...] = (
    "openai/gpt-oss",
    "google/gemma",
    "mistralai/mistral-7b",
    "mistralai/mixtral",
)

#: The fields of a routing that name hosts, and so hold for one model only.
_NAMES_HOSTS = ("only", "order")


def _maker(model: str) -> dict[str, Any] | None:
    """`model`'s maker as its only host, or None for OpenRouter's defaults.

    A variant (`…:free`, `…:nitro`, any id with a `:` suffix) is not pinned:
    OpenRouter serves it from the hosts it has for that variant, often not
    the maker (a free one only from others), and a variant that orders
    hosts by price or speed is routing of its own."""
    if get_settings().provider_routing != "maker" or model.startswith(OPEN_WEIGHTS):
        return None
    if ":" in model:
        return None
    hosts = MAKER_HOSTS.get(model.split("/", 1)[0])
    if hosts is None:
        return None
    return {"only": list(hosts), "allow_fallbacks": False}


def routing_for(
    model: str, requested: ProviderRouting | None = None
) -> dict[str, Any] | None:
    """OpenRouter's `provider` object for a call to `model`, or None for its
    defaults. A request that names hosts (`only`, `order`) is followed as it
    is; what a request asks without naming any (`ignore`, `allow_fallbacks`,
    `data_collection`) is added to the maker's own hosts."""
    default = _maker(model)
    asked = requested.model_dump(exclude_none=True) if requested else {}
    if not asked:
        return default
    if default is not None and not asked.keys() & set(_NAMES_HOSTS):
        return {**default, **asked}
    return asked


def elsewhere(requested: ProviderRouting | None) -> ProviderRouting | None:
    """A request's routing for a call on another model than the one it was
    sent for (the utility model's rewrite and rerank, a judge, the embedding
    model): the hosts it names serve that one model, so they go and the other
    model's maker applies; what holds for any model stays."""
    if requested is None:
        return None
    rest = requested.model_copy(update=dict.fromkeys(_NAMES_HOSTS))
    return rest if rest.model_dump(exclude_none=True) else None


class RoutedTransport(httpx.AsyncBaseTransport):
    """Adds the routing for its model to every chat and embedding call it
    carries: for a client the engine cannot pass a body field through, as
    RAGAS builds its own calls. `requested` is the request's routing, sent
    for `model`; a call on any other model gets what of it holds there.

    Without an `inner` transport, the calls go out as a plain client's
    would, through the proxy the environment names (`HTTPS_PROXY`,
    `HTTP_PROXY`, `ALL_PROXY`, except to the hosts `NO_PROXY` names): httpx
    reads none of these for a client given a transport of its own, so a
    client of this one's picks the proxy for each address."""

    def __init__(
        self,
        inner: httpx.AsyncBaseTransport | None = None,
        *,
        requested: ProviderRouting | None = None,
        model: str | None = None,
    ) -> None:
        self._inner = inner
        self._client = None if inner is not None else httpx.AsyncClient(trust_env=True)
        self._requested = requested
        self._model = model

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith(
            ("/chat/completions", "/embeddings")
        ):
            body = json.loads(request.content or b"{}")
            called = str(body.get("model", ""))
            asked = (
                self._requested if called == self._model else elsewhere(self._requested)
            )
            routing = routing_for(called, asked)
            if routing is not None and "provider" not in body:
                body["provider"] = routing
                headers = [
                    (name, value)
                    for name, value in request.headers.multi_items()
                    if name.lower() != "content-length"
                ]
                request = httpx.Request(
                    request.method,
                    request.url,
                    headers=headers,
                    content=json.dumps(body).encode(),
                    extensions=request.extensions,
                )
        if self._inner is not None:
            return await self._inner.handle_async_request(request)
        assert self._client is not None
        # Sent as it is, the client adding nothing but the transport it picks
        # for the address; the caller's client reads the answer.
        return await self._client.send(request, stream=True)

    async def aclose(self) -> None:
        if self._inner is not None:
            await self._inner.aclose()
        if self._client is not None:
            await self._client.aclose()
