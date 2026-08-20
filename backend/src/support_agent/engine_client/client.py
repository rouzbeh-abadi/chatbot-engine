"""HTTP client for the AI engine service.

The only module in this backend that knows the engine exists over the network.
Everything else deals in `engine_client.models`.

Two error concerns shape this file:

* the engine is a *remote* dependency now, so "unreachable" is a normal outcome
  and must not surface as a 500 from an unhandled `ConnectError`;
* a streaming call has to fail *before* the backend starts its own 200 response,
  which is why `start_chat()` is awaited rather than being an async generator.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
from pydantic import TypeAdapter, ValidationError

from support_agent.engine_client.models import (
    ChatEvent,
    DeleteResult,
    DocumentRecord,
    EngineChatRequest,
)

_EVENT = TypeAdapter(ChatEvent)
_RECORDS = TypeAdapter(list[DocumentRecord])


class EngineError(RuntimeError):
    """Base for every failure talking to the engine."""


class EngineUnavailable(EngineError):
    """The engine could not be reached at all."""


class EngineNotImplemented(EngineError):
    """The engine answered 501: that capability has no implementation yet."""


class EngineRejected(EngineError):
    """The engine answered 4xx.

    Carries the status because two very different things land here: a 415 means
    the caller's document was unusable, a 401 means this backend is
    misconfigured. Only the first kind should reach the caller unchanged.
    """

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class EngineFailed(EngineError):
    """The engine answered 5xx."""


class EngineClient:
    """Calls the engine over HTTP.

    One client per request is fine at this scale and keeps lifetimes simple. If
    the extra connection setup ever shows up in a profile, hold a single
    `httpx.AsyncClient` on the app lifespan and inject it here instead.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        # An empty key means "not configured", which is what an unset
        # BACKEND_ENGINE_API_KEY in a .env file actually looks like.
        self._headers = {"X-API-Key": api_key} if api_key else {}
        # A read timeout is per-chunk, not per-response, so a long answer that
        # keeps producing tokens will not be cut off by it.
        self._timeout = httpx.Timeout(timeout_s, connect=10.0)
        #: Only tests pass this, so they can serve canned responses without a
        #: socket. Cheaper than patching a private method from outside.
        self._transport = transport

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._headers,
            timeout=self._timeout,
            transport=self._transport,
        )

    # --- chat ---------------------------------------------------------------

    async def start_chat(self, request: EngineChatRequest) -> AsyncIterator[ChatEvent]:
        """Begin a turn and return its event stream.

        Awaiting this sends the request and checks the status, so a 501 or an
        unreachable engine raises here -- while our own response status can still
        be set. The returned iterator owns the connection and closes it when it is
        exhausted or the caller stops early.
        """
        client = self._client()
        payload = request.model_dump(mode="json", exclude_none=True)

        try:
            http_request = client.build_request("POST", "/chat", json=payload)
            response = await client.send(http_request, stream=True)
        except httpx.HTTPError as exc:
            await client.aclose()
            raise EngineUnavailable(f"engine unreachable at {self._base_url}: {exc}") from exc

        if response.is_error:
            await response.aread()
            error = self._error_for(response)
            await response.aclose()
            await client.aclose()
            raise error

        return _events(response, client)

    # --- documents ----------------------------------------------------------

    async def ingest_document(
        self,
        *,
        project_id: str,
        external_id: str,
        filename: str,
        mimetype: str,
        data: bytes,
    ) -> DocumentRecord:
        """Send raw bytes for indexing.

        Bytes, not text: extraction is the engine's job, and the original file
        carries page numbers and layout that extracted text has already lost.
        """
        async with self._client() as client:
            response = await self._request(
                client,
                "PUT",
                "/documents",
                data={"project_id": project_id, "external_id": external_id},
                files={"file": (filename, data, mimetype)},
            )
            return DocumentRecord.model_validate(response.json())

    async def list_documents(self, *, project_id: str) -> list[DocumentRecord]:
        async with self._client() as client:
            response = await self._request(
                client, "GET", "/documents", params={"project_id": project_id}
            )
            return _RECORDS.validate_python(response.json())

    async def delete_document(self, *, project_id: str, doc_id: str) -> DeleteResult:
        async with self._client() as client:
            response = await self._request(
                client,
                "DELETE",
                f"/documents/{doc_id}",
                params={"project_id": project_id},
            )
            return DeleteResult.model_validate(response.json())

    # --- plumbing -----------------------------------------------------------

    async def _request(
        self, client: httpx.AsyncClient, method: str, url: str, **kwargs: object
    ) -> httpx.Response:
        try:
            response = await client.request(method, url, **kwargs)  # type: ignore[arg-type]
        except httpx.HTTPError as exc:
            raise EngineUnavailable(
                f"engine unreachable at {self._base_url}: {exc}"
            ) from exc

        if response.is_error:
            raise self._error_for(response)
        return response

    def _error_for(self, response: httpx.Response) -> EngineError:
        detail = _detail(response)
        if response.status_code == 501:
            return EngineNotImplemented(detail)
        if response.status_code < 500:
            return EngineRejected(
                f"engine rejected the request ({response.status_code}): {detail}",
                status_code=response.status_code,
            )
        return EngineFailed(f"engine failed ({response.status_code}): {detail}")


async def _events(
    response: httpx.Response, client: httpx.AsyncClient
) -> AsyncIterator[ChatEvent]:
    """Parse the engine's NDJSON stream into events, then clean up.

    The `finally` is what stops a browser that navigates away mid-answer from
    leaking a connection: Starlette closes this generator on disconnect, which
    unwinds through here.
    """
    try:
        async for line in response.aiter_lines():
            if not line.strip():
                continue
            try:
                yield _EVENT.validate_json(line)
            except ValidationError as exc:
                raise EngineFailed(f"engine sent an unreadable event: {line!r}") from exc
    finally:
        await response.aclose()
        await client.aclose()


def _detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text.strip() or f"HTTP {response.status_code}"
    if isinstance(body, dict) and "detail" in body:
        return str(body["detail"])
    return str(body)
