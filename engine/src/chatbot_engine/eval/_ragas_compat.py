"""Make `import ragas` work on this project's LangChain version.

ragas 0.4.3 unconditionally imports a Vertex AI path (`langchain_community.
chat_models.vertexai.ChatVertexAI` and `langchain_community.llms.VertexAI`) at
package load. LangChain v1 moved Vertex out to a separate package and dropped
that path, so `import ragas` raises `ModuleNotFoundError` before any of our
code runs. No ragas release (0.3.x-0.4.3) avoids the import, and no LangChain
version has both the Vertex path and core 1.x, so it cannot be resolved by
pinning.

We never use Vertex, so this module registers empty stand-ins for the two
symbols. Importing it (once, before ragas) is enough; it is idempotent and a
no-op if the real symbols are ever present.
"""

from __future__ import annotations

import sys
import types

_VERTEX_MODULE = "langchain_community.chat_models.vertexai"

if _VERTEX_MODULE not in sys.modules:
    _stub = types.ModuleType(_VERTEX_MODULE)
    _stub.ChatVertexAI = type("ChatVertexAI", (), {})  # ty: ignore[unresolved-attribute]
    sys.modules[_VERTEX_MODULE] = _stub

import langchain_community.llms as _llms

if not hasattr(_llms, "VertexAI"):
    _llms.VertexAI = type("VertexAI", (), {})
