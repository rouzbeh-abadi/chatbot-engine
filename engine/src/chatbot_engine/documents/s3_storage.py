"""Original uploads in an S3-compatible bucket, for replicas that share them.

The local blob store is a directory, and a directory belongs to one host. An
object store is reachable from every replica, which is what lets any of them
re-index a document another one ingested. Works against AWS S3 and against
anything speaking its API, such as MinIO, through `endpoint_url`.

`boto3` is synchronous; every call runs in a thread so the event loop keeps
serving. The `s3` extra provides it.
"""

from __future__ import annotations

import asyncio
from typing import Any

from chatbot_engine.ports.documents import BlobStore


class S3BlobStore(BlobStore):
    """One object per document under `prefix`, addressed as `s3://bucket/key`."""

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "",
        endpoint_url: str | None = None,
        region: str | None = None,
        client: Any | None = None,
    ) -> None:
        import boto3

        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._client = client or boto3.client(
            "s3", endpoint_url=endpoint_url, region_name=region
        )

    def _key(self, key: str) -> str:
        return f"{self._prefix}/{key}" if self._prefix else key

    def uri_for(self, key: str) -> str:
        return f"s3://{self._bucket}/{self._key(key)}"

    def _parse(self, uri: str) -> tuple[str, str]:
        """`s3://bucket/key` back into its parts; only our own URIs arrive here."""
        rest = uri.removeprefix("s3://")
        bucket, _, key = rest.partition("/")
        return bucket, key

    async def put(self, *, key: str, data: bytes, mimetype: str) -> str:
        await asyncio.to_thread(
            self._client.put_object,
            Bucket=self._bucket,
            Key=self._key(key),
            Body=data,
            ContentType=mimetype,
        )
        return self.uri_for(key)

    async def get(self, *, uri: str) -> bytes:
        bucket, key = self._parse(uri)
        response = await asyncio.to_thread(
            self._client.get_object, Bucket=bucket, Key=key
        )
        return await asyncio.to_thread(response["Body"].read)

    async def delete(self, *, uri: str) -> None:
        bucket, key = self._parse(uri)
        await asyncio.to_thread(self._client.delete_object, Bucket=bucket, Key=key)

    def ensure_bucket(self) -> None:
        """Create the bucket when it does not exist. For a fresh MinIO in the
        scale overlay; a real deployment provisions its bucket itself."""
        from botocore.exceptions import ClientError

        try:
            self._client.head_bucket(Bucket=self._bucket)
        except ClientError:
            self._client.create_bucket(Bucket=self._bucket)
