"""Layering rules that the separation depends on.

All three are easy to break by accident and invisible in code review.
"""

from __future__ import annotations

import ast
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1] / "src" / "chatbot_engine"

#: `models/` and `ports/` are the reusable core: contracts and interfaces. A
#: framework or vendor SDK in there would drag the transport into every
#: implementation, and would stop a non-HTTP caller from using them.
CORE_DIRS = ("models", "ports")
FORBIDDEN_IN_CORE = {
    "fastapi",
    "starlette",
    "uvicorn",
    "httpx",
    "mcp",
    "chromadb",
    "sqlalchemy",
    "langchain",
    "langchain_core",
    "langchain_openai",
    "openai",
}

#: The engine is a separate service. If it imports the backend, the whole point
#: of the split is gone.
BACKEND_PACKAGE = "support_agent"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def _core_files() -> list[Path]:
    return [p for d in CORE_DIRS for p in sorted((ENGINE / d).rglob("*.py"))]


def test_models_and_ports_import_no_framework_or_vendor_sdk() -> None:
    offenders = [
        f"{path.relative_to(ENGINE)} imports {module}"
        for path in _core_files()
        for module in _imports(path)
        if module.split(".")[0] in FORBIDDEN_IN_CORE
    ]

    assert not offenders, (
        "models/ and ports/ are the framework-free core: " + "; ".join(offenders)
    )


def test_models_and_ports_do_not_depend_on_api_or_services() -> None:
    """Contracts must not know about their transport or their callers."""
    offenders = [
        f"{path.relative_to(ENGINE)} imports {module}"
        for path in _core_files()
        for module in _imports(path)
        if module.startswith(("chatbot_engine.api", "chatbot_engine.services"))
    ]

    assert not offenders, "dependency arrow is backwards: " + "; ".join(offenders)


def test_the_engine_never_imports_the_backend() -> None:
    offenders = [
        f"{path.relative_to(ENGINE)} imports {module}"
        for path in sorted(ENGINE.rglob("*.py"))
        for module in _imports(path)
        if module.split(".")[0] == BACKEND_PACKAGE
    ]

    assert not offenders, (
        "the engine is a separate service and must not import the backend: "
        + "; ".join(offenders)
    )
