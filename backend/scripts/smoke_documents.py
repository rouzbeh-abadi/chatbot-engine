"""Prove the document path works, against the services actually running.

The test suite covers all of this with a fake embedder. This does it for real:
your key, your OpenRouter account, real vectors on disk. Run it when you have
changed configuration and want to know the whole path still works.

Goes through the backend rather than straight at the engine, so it exercises the
same validation and project resolution a real client hits.

    make dev            # engine and backend
    make smoke-docs     # in another terminal

Cleans up after itself: everything it uploads, it deletes.
"""

from __future__ import annotations

import sys

import httpx

BASE_URL = "http://localhost:8000"
EXTERNAL_ID = "__smoke__/probe.md"

LONG = ("Cabin baggage is one bag up to eight kilograms, plus one personal "
        "item that fits under the seat in front of you. ").encode() * 12
SHORT = b"# Probe\n\nOne bag.\n"

_passed = 0
_failed = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global _passed, _failed
    if ok:
        _passed += 1
        print(f"  ok    {label}{f'  ({detail})' if detail else ''}")
    else:
        _failed += 1
        print(f"  FAIL  {label}{f'  ({detail})' if detail else ''}")


def upload(
    client: httpx.Client,
    content: bytes,
    mimetype: str = "text/markdown",
    external_id: str = EXTERNAL_ID,
) -> httpx.Response:
    return client.put(
        "/documents",
        data={"external_id": external_id},
        files={"file": ("probe.md", content, mimetype)},
    )


def main() -> int:
    with httpx.Client(base_url=BASE_URL, timeout=120.0) as client:
        try:
            client.get("/health").raise_for_status()
        except httpx.HTTPError:
            print(f"backend is not running at {BASE_URL} -- start it with 'make dev'",
                  file=sys.stderr)
            return 1

        print("\ningest")
        first = upload(client, LONG)
        body = first.json() if first.status_code == 201 else {}
        check("upload accepted", first.status_code == 201, f"HTTP {first.status_code}")
        check("status is indexed", body.get("status") == "indexed",
              str(body.get("status")))
        check("chunks were produced", body.get("chunk_count", 0) > 1,
              f"{body.get('chunk_count')} chunks")

        doc_id = body.get("doc_id")
        if doc_id is None:
            print("\ncannot continue without a doc_id")
            return 1

        print("\nidempotency")
        again = upload(client, LONG).json()
        check("identical bytes do no work", again.get("status") == "unchanged",
              str(again.get("status")))
        check("same document, not a second one", again.get("doc_id") == doc_id)

        print("\nreplace")
        second = upload(client, SHORT).json()
        check("a changed file is re-indexed", second.get("status") == "indexed")
        check("chunk count follows the new content",
              second.get("chunk_count", 99) < body.get("chunk_count", 0),
              f"{body.get('chunk_count')} -> {second.get('chunk_count')}")

        print("\nlisting")
        listed = client.get("/documents").json()
        mine = [record for record in listed if record["doc_id"] == doc_id]
        check("the document is listed", len(mine) == 1)
        check("its chunk count is reported",
              mine and mine[0]["chunk_count"] == second.get("chunk_count"))

        print("\nrejections")
        unreadable = upload(client, b"x", mimetype="application/msword",
                            external_id="__smoke__/probe.docx")
        check("an unreadable type is 415", unreadable.status_code == 415,
              f"HTTP {unreadable.status_code}")
        blank = upload(client, b"   \n\n  \n", external_id="__smoke__/blank.md")
        check("a document with no text is 422", blank.status_code == 422,
              f"HTTP {blank.status_code}")
        failed = [
            record for record in client.get("/documents").json()
            if record["external_id"] == "__smoke__/blank.md"
        ]
        check("the failure is recorded, not silent",
              bool(failed) and failed[0]["status"] == "failed")

        print("\ncleanup")
        for external_id in ("__smoke__/blank.md", EXTERNAL_ID):
            target = [
                record for record in client.get("/documents").json()
                if record["external_id"] == external_id
            ]
            for record in target:
                client.delete(f"/documents/{record['doc_id']}")
        remaining = [
            record for record in client.get("/documents").json()
            if record["external_id"].startswith("__smoke__/")
        ]
        check("everything uploaded here is gone", not remaining,
              f"{len(remaining)} left")

    print(f"\n{_passed} passed, {_failed} failed\n")

    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
