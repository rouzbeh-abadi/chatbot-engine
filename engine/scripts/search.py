"""Ask the vector store what it would retrieve for a question.

A diagnostic, not the retriever: no prompt, no model call, no `SourceRef`. It
embeds the question with the configured model and prints the nearest chunks, so
you can see whether ingestion produced something worth retrieving.

    make search Q="how much hand luggage can I bring?"

Reads the same `ENGINE_*` settings the engine does, so it looks at the same store.
Safe to run while the engine is up -- it only reads.
"""

from __future__ import annotations

import sys

from chatbot_engine.rag.vector_store import (
    EmptyVectorStoreError,
    count_chunks,
    load_vector_store,
)
from chatbot_engine.settings import get_settings

PROJECT_ID = "support"
TOP_K = 4


def main() -> int:
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print('usage: make search Q="your question"', file=sys.stderr)
        return 1

    settings = get_settings()
    if settings.openrouter_api_key is None:
        print("ENGINE_OPENROUTER_API_KEY is not set -- the question cannot be "
              "embedded, so there is nothing to compare against", file=sys.stderr)
        return 1

    try:
        store = load_vector_store()
    except EmptyVectorStoreError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"\n{count_chunks()} chunks in {settings.chroma_dir}")
    print(f"embedding with {settings.embedding_model}")
    print(f'\n  "{question}"\n')

    hits = store.similarity_search_with_score(
        question, k=TOP_K, filter={"project_id": PROJECT_ID}
    )
    if not hits:
        print(f"  nothing found for project {PROJECT_ID!r}")
        return 1

    for rank, (chunk, distance) in enumerate(hits, start=1):
        source = chunk.metadata.get("source", "?")
        offset = chunk.metadata.get("start_index", "?")
        excerpt = " ".join(chunk.page_content.split())[:220]
        print(f"  {rank}. {source}  (distance {distance:.3f}, at character {offset})")
        print(f"     {excerpt}...\n")

    print("Lower distance is closer. The first hit is what the model would be "
          "shown first.\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
