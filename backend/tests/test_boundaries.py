"""The separation this refactor exists to create.

Each of these is invisible in review and easy to undo with one import.
"""

from __future__ import annotations

import ast
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
SOURCES = (BACKEND / "src", BACKEND / "scripts")

#: The engine is a separate service. Importing its package would collapse the two
#: back into one deployable, which is exactly what the HTTP boundary prevents.
ENGINE_PACKAGE = "chatbot_engine"

#: Prompts, retrieval, embeddings and model calls belong to the engine.
AI_LIBRARIES = {
    "langchain",
    "langchain_core",
    "langchain_openai",
    "openai",
    "anthropic",
    "tiktoken",
    "chromadb",
    "qdrant_client",
    "sentence_transformers",
}

#: Only the client layer may speak HTTP to the engine. A stray `httpx` call in a
#: route would route around the client's error handling and contract models.
HTTP_CLIENT = "httpx"
#: `client.py` is the one route to the engine from the app itself. The two
#: scripts are operator tools that drive the running API from outside, which
#: is the whole point of them.
ALLOWED_HTTP_FILES = {"client.py", "seed_knowledge.py", "smoke_documents.py"}


def _modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def _all_sources() -> list[Path]:
    return [p for root in SOURCES for p in sorted(root.rglob("*.py"))]


def test_the_backend_never_imports_the_engine_package() -> None:
    offenders = [
        f"{path.name} imports {module}"
        for path in _all_sources()
        for module in _modules(path)
        if module.split(".")[0] == ENGINE_PACKAGE
    ]

    assert not offenders, (
        "the engine is a separate service -- call it through engine_client: "
        + "; ".join(offenders)
    )


def test_the_backend_imports_no_ai_libraries() -> None:
    offenders = [
        f"{path.name} imports {module}"
        for path in _all_sources()
        for module in _modules(path)
        if module.split(".")[0] in AI_LIBRARIES
    ]

    assert not offenders, "AI logic leaked into the backend: " + "; ".join(offenders)


def test_only_the_engine_client_speaks_http_to_the_engine() -> None:
    offenders = [
        f"{path.name} imports {module}"
        for path in _all_sources()
        for module in _modules(path)
        if module.split(".")[0] == HTTP_CLIENT and path.name not in ALLOWED_HTTP_FILES
    ]

    assert not offenders, (
        "keep engine HTTP calls inside engine_client/client.py: " + "; ".join(offenders)
    )
