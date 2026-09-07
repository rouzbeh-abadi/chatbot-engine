"""Upload the demo knowledge base through the backend.

Talks to the backend's own `PUT /documents`, which forwards to the engine. Going
through the backend rather than straight at the engine keeps one ingestion path,
so the validation and project resolution that a real client hits are exercised
here too.

Each file's path becomes its `external_id`, so re-running is idempotent: the
engine skips unchanged files by content hash and replaces changed ones.

    make dev     # both services, in one terminal
    make seed    # in another
"""

from __future__ import annotations

import mimetypes
import sys

import httpx
from support_agent.assistant import KNOWLEDGE_DIR

BASE_URL = "http://localhost:8000"

#: Housekeeping files, skipped by exact name. Everything else is a document --
#: including Markdown, which is a perfectly good knowledge-base format. Only the
#: folder's own README is excluded, not `.md` as a whole.
SKIP = {"README.md", "readme.md", ".gitkeep", ".DS_Store"}


def main() -> int:
    knowledge = KNOWLEDGE_DIR

    files = sorted(
        path
        for path in knowledge.glob("**/*")
        if path.is_file() and path.name not in SKIP
    )
    if not files:
        print(f"no documents found in {knowledge}")
        print("drop some PDFs, Markdown or text files there, then run this again.")
        return 0

    with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
        try:
            client.get("/health").raise_for_status()
        except httpx.HTTPError:
            print(f"app is not running at {BASE_URL} -- start it with 'make dev'",
                  file=sys.stderr)
            return 1

        for path in files:
            mimetype, _ = mimetypes.guess_type(path.name)
            response = client.put(
                "/documents",
                data={"external_id": path.relative_to(knowledge).as_posix()},
                files={
                    "file": (
                        path.name,
                        path.read_bytes(),
                        mimetype or "application/octet-stream",
                    )
                },
            )
            if response.status_code == 501:
                print(
                    "the engine has no IngestPipeline yet -- register one in "
                    "chatbot_engine/api/deps.py::get_ingest_pipeline()",
                    file=sys.stderr,
                )
                return 1
            if response.status_code == 503:
                print(
                    "the engine service is not reachable -- start it with "
                    "'make engine'",
                    file=sys.stderr,
                )
                return 1
            if response.is_error:
                print(f"{path.name}: FAILED {response.status_code} {response.text}",
                      file=sys.stderr)
                return 1

            body = response.json()
            print(f"{body['external_id']}: {body['status']} ({body['size_bytes']} bytes)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
