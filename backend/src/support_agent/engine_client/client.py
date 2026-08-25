"""HTTP client for communicating with the chatbot engine.

All backend-to-engine HTTP communication goes through `EngineClient`. It sends
chat, document, and evaluation requests, parses responses into typed models,
and converts network or engine failures into `EngineError` exceptions.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
from pydantic import TypeAdapter, ValidationError

from support_agent.engine_client.models import (
    AssistantConfig,
    ChatEvent,
    DeleteResult,
    DocumentRecord,
    EngineChatRequest,
)
from support_agent.evals.models import JudgeReport, RagReport

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

    Carries the status code because the cause varies: a 415 means the caller's
    document was unusable (pass it back), a 401 means this backend is
    misconfigured (do not). The app layer decides based on `status_code`.
    """

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class EngineFailed(EngineError):
    """The engine answered 5xx."""


class EngineClient:
    """Send HTTP requests to the chatbot engine and parse its responses."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        # An empty/unset key means "no auth"; send the header only when we have one.
        self._headers = {"X-API-Key": api_key} if api_key else {}
        # The read timeout applies per chunk, not to the whole response, so a
        # long streamed answer is not cut off as long as tokens keep arriving.
        self._timeout = httpx.Timeout(timeout_s, connect=10.0)
        # Only tests set this, to serve canned responses without a real socket.
        self._transport = transport

    def _client(self, timeout_s: float | None = None) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._headers,
            timeout=self._timeout
            if timeout_s is None
            else httpx.Timeout(timeout_s, connect=10.0),
            transport=self._transport,
        )

    # chat

    async def start_chat(self, request: EngineChatRequest) -> AsyncIterator[ChatEvent]:
        """Send a chat request from the backend to the engine and return its event stream.

        The request is sent to the engine's `POST /chat` endpoint. If successful,

        the response stays open and is read as streamed `ChatEvent`s until the turn

        finishes or the caller stops reading."""
      
        client = self._client()
        payload = request.model_dump(mode="json", exclude_none=True)

        try:
            http_request = client.build_request("POST", "/chat", json=payload)
            response = await client.send(http_request, stream=True)
        except httpx.HTTPError as exc:
            await client.aclose()
            raise EngineUnavailable(
                f"engine unreachable at {self._base_url}: {exc}"
            ) from exc

        if response.is_error:
            await response.aread()
            error = self._error_for(response)
            await response.aclose()
            await client.aclose()
            raise error

        return _events(response, client)

    # documents

    async def ingest_document(
        self,
        *,
        project_id: str,
        external_id: str,
        filename: str,
        mimetype: str,
        data: bytes,
    ) -> DocumentRecord:
        """Upload one document's raw bytes to the engine for indexing.

        Raw bytes, not extracted text: extraction is the engine's job, and the
        original file carries layout that extracted text would have lost.
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
        """List the documents indexed for a project."""
        async with self._client() as client:
            response = await self._request(
                client, "GET", "/documents", params={"project_id": project_id}
            )
            return _RECORDS.validate_python(response.json())

    async def delete_document(self, *, project_id: str, doc_id: str) -> DeleteResult:
        """Delete one document from a project's knowledge base."""
        async with self._client() as client:
            response = await self._request(
                client,
                "DELETE",
                f"/documents/{doc_id}",
                params={"project_id": project_id},
            )
            return DeleteResult.model_validate(response.json())

    # evaluation

    async def judge(
        self,
        *,
        project: AssistantConfig,
        judge_prompt: str,
        cases: list[dict[str, object]],
        timeout_s: float = 1800.0,
    ) -> JudgeReport:
        """Run an evaluation: the engine answers every case, then grades them.

        Both halves run in the engine because only it holds model credentials.
        The engine owns the case shape and validates it, so `cases` is forwarded
        as-is. Uses a much longer timeout than a chat turn, since one request
        covers dozens of model calls (every case answered, then judged).
        """
        async with self._client(timeout_s) as client:
            response = await self._request(
                client,
                "POST",
                "/judge",
                json={
                    "project": project.model_dump(mode="json", exclude_none=True),
                    "judge_prompt": judge_prompt,
                    "cases": cases,
                },
            )
            return JudgeReport.model_validate(response.json())

    async def evaluate_rag(
        self,
        *,
        project: AssistantConfig,
        cases: list[dict[str, object]],
        timeout_s: float = 1800.0,
    ) -> RagReport:
        """Score retrieval with RAGAS: the engine answers each case, then grades it.

        Runs in the engine, like `judge`, because only it holds model
        credentials and the RAGAS dependency. The engine owns the case shape and
        validates it, so `cases` is forwarded as-is. Uses a long timeout: every
        case is answered and then graded by several metric model calls.
        """
        async with self._client(timeout_s) as client:
            response = await self._request(
                client,
                "POST",
                "/eval/rag",
                json={
                    "project": project.model_dump(mode="json", exclude_none=True),
                    "cases": cases,
                },
            )
            return RagReport.model_validate(response.json())

    # plumbing

    async def _request(
        self, client: httpx.AsyncClient, method: str, url: str, **kwargs: object
    ) -> httpx.Response:
        """Send a non-streaming request and convert HTTP failures to engine errors."""
        try:
            response = await client.request(method, url, **kwargs)  # type: ignore[arg-type]
        except httpx.TimeoutException as exc:
            # str() on an httpx timeout is empty, so build a message ourselves.
            raise EngineUnavailable(
                f"engine did not answer within {client.timeout.read:.0f}s "
                f"at {self._base_url}"
            ) from exc
        except httpx.HTTPError as exc:
            raise EngineUnavailable(
                f"engine unreachable at {self._base_url}: {exc}"
            ) from exc

        if response.is_error:
            raise self._error_for(response)
        return response

    def _error_for(self, response: httpx.Response) -> EngineError:
        """Convert an unsuccessful engine response into the appropriate error."""

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
    """Read the engine's NDJSON stream and convert each line into a `ChatEvent`.

    The HTTP connection is always closed when streaming ends.
    """

    try:
        async for line in response.aiter_lines():
            if not line.strip():
                continue
            try:
                yield _EVENT.validate_json(line)
            except ValidationError as exc:
                raise EngineFailed(
                    f"engine sent an unreadable event: {line!r}"
                ) from exc
    finally:
        await response.aclose()
        await client.aclose()


def _detail(response: httpx.Response) -> str:
    """Extract a human-readable error message from an engine response body."""

    try:
        body = response.json()
    except ValueError:
        return response.text.strip() or f"HTTP {response.status_code}"
    if isinstance(body, dict) and "detail" in body:
        return str(body["detail"])
    return str(body)
