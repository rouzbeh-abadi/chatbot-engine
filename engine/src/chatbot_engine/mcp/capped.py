"""A bound on what one tool server's answer may be, in bytes, gzip included.

The MCP SDK reads a response whole, and the HTTP client inflates gzip on its
own, so a tool server could answer with gigabytes, or with a 65 KB gzip body
that becomes 64 MB, and take the engine's memory for every tenant
(docs/review-2026-10.md, MCP-3). This transport sits under the client: it
asks only for encodings it can inflate itself, inflates them a bounded piece
at a time, and ends the answer once its decoded bytes pass
`ENGINE_MCP_MAX_RESPONSE_BYTES`. The call then fails like any other.
"""

from __future__ import annotations

import zlib
from collections.abc import AsyncIterator
from typing import cast

import httpx2

#: The encodings inflated here. Brotli and zstd are not asked for, so a
#: server that sends them anyway is refused rather than inflated unbounded.
ACCEPTED_ENCODINGS = "gzip, deflate"

#: zlib's window bits for "a gzip or a zlib header, whichever it is".
_AUTO_HEADER = 32 + zlib.MAX_WBITS


class McpResponseRefusedError(httpx2.TransportError):
    """A tool server's answer was not read: past the byte limit, or in an
    encoding the engine did not ask for."""


class _CappedStream(httpx2.AsyncByteStream):
    def __init__(
        self, raw: httpx2.AsyncByteStream, *, inflate: bool, limit: int, host: str
    ) -> None:
        self._raw = raw
        self._inflater = zlib.decompressobj(_AUTO_HEADER) if inflate else None
        self._limit = limit
        self._host = host
        self._seen = 0

    def _count(self, data: bytes) -> bytes:
        self._seen += len(data)
        if self._seen > self._limit:
            raise McpResponseRefusedError(
                f"the tool server at {self._host} answered with more than "
                f"{self._limit:,} bytes (ENGINE_MCP_MAX_RESPONSE_BYTES); "
                "the rest was not read"
            )
        return data

    async def __aiter__(self) -> AsyncIterator[bytes]:
        inflater = self._inflater
        async for chunk in self._raw:
            if inflater is None:
                yield self._count(chunk)
                continue
            data = chunk
            while data:
                # At most one byte past the limit per step, so a small body
                # that inflates to gigabytes never does in memory.
                out = inflater.decompress(data, self._limit - self._seen + 1)
                yield self._count(out)
                data = inflater.unconsumed_tail
        if inflater is not None:
            yield self._count(inflater.flush())

    async def aclose(self) -> None:
        await self._raw.aclose()


class CappedTransport(httpx2.AsyncBaseTransport):
    """`AsyncHTTPTransport`, with every response bounded to `limit` decoded bytes."""

    def __init__(
        self, limit: int, inner: httpx2.AsyncBaseTransport | None = None
    ) -> None:
        self._inner = inner or httpx2.AsyncHTTPTransport()
        self._limit = limit

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        request.headers["Accept-Encoding"] = ACCEPTED_ENCODINGS
        response = await self._inner.handle_async_request(request)
        encoding = response.headers.get("content-encoding", "identity").strip().lower()
        if encoding not in ("identity", "gzip", "deflate"):
            await response.aclose()
            raise McpResponseRefusedError(
                f"the tool server at {request.url.host} answered in an encoding "
                f"it was not asked for ({encoding!r})"
            )
        headers = response.headers.copy()
        if encoding != "identity":
            # Inflated here, so the client must not inflate it again.
            del headers["content-encoding"]
            headers.pop("content-length", None)
        return httpx2.Response(
            response.status_code,
            headers=headers,
            stream=_CappedStream(
                cast(httpx2.AsyncByteStream, response.stream),
                inflate=encoding != "identity",
                limit=self._limit,
                host=request.url.host,
            ),
            extensions=response.extensions,
            request=request,
        )

    async def aclose(self) -> None:
        await self._inner.aclose()
