"""RAGAS loads through the compatibility shim, in the order that matters.

ragas imports a LangChain Vertex AI path that no longer exists; the shim stubs
it and must run first. An import sorter once moved the shim below the ragas
imports, which nothing caught until a real evaluation returned a 500.
"""

from __future__ import annotations

import importlib.util

import pytest

# `find_spec`, not `importorskip`: importing ragas bare is the very failure the
# shim exists to prevent, so it would skip this test everywhere.
if importlib.util.find_spec("ragas") is None:
    pytest.skip("the eval extra is not installed", allow_module_level=True)


def test_the_metrics_can_be_built(monkeypatch: pytest.MonkeyPatch) -> None:
    """Building the metrics is what performs the ragas imports."""
    from chatbot_engine.api.dependencies import reset_dependency_cache
    from chatbot_engine.eval.rag_evaluation import _build_metrics

    monkeypatch.setenv("ENGINE_OPENROUTER_API_KEY", "sk-or-fake")
    reset_dependency_cache()

    metrics = _build_metrics()

    assert len(metrics) == 4
    reset_dependency_cache()
