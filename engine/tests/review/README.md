# Review tests, October 2026

The tests behind [`docs/review-2026-10.md`](../../../docs/review-2026-10.md),
a review of release 0.1.26.

- A test marked `xfail(strict=True, reason="<ID> …")` confirms finding `<ID>`.
  It asserts the behaviour the engine should have, so it fails today and
  `pytest` counts it as expected. When a fix makes it pass, the suite turns
  red (strict): drop the marker, and the test stays as a regression test.
- A test with no marker checks something the review found sound, or is a
  control for a test beside it.

They run with the rest of the suite (`uv run pytest`), on the suite's own
fixtures (`engine/tests/conftest.py`). Nothing calls a real model provider,
tool server or tracer: providers and servers are fakes on 127.0.0.1 or
in-process transports.

| File | Findings |
| --- | --- |
| `test_api.py` | API-1 to API-6 |
| `test_ingest.py` | INGEST-1, -3 to -10; API-1, API-6 |
| `test_extract_slots.py` | INGEST-12 |
| `test_index_crashes.py` | INGEST-13 to -16; WORKFLOW-8 |
| `test_deletion.py` | INGEST-4, INGEST-17 |
| `test_usage.py` | TURN-8, INGEST-18, INGEST-19 |
| `test_retrieval.py` | RETRIEVAL-1 to -6 |
| `test_turn.py` | TURN-1, -2, -4, -5, -6, -8; API-4, API-5, MCP-6 |
| `test_provider_routing.py` | TURN-10 |
| `test_mcp.py` | MCP-2 to MCP-6, TURN-5, API-5 |
| `test_mcp_redirects.py` | MCP-2 |
| `test_workflow.py` | TURN-1, WORKFLOW-2 to -9 |
| `test_evaltrace.py` | EVALTRACE-1 to -4, -7, -8; API-2, API-5, WORKFLOW-7 |
| `test_metric_labels.py` | EVALTRACE-10, EVALTRACE-11 |
| `test_infra.py` | API-1, API-6, EVALTRACE-1 |
| `test_demo.py` | DEMO-1 to DEMO-8 |

Some tests start a real `uvicorn` process and stop it with SIGTERM and
SIGKILL, so the folder takes a few minutes. The test that runs the engine as
PID 1 (`test_index_crashes.py`) needs root and `unshare`, and is skipped
without them, as on GitHub's runners.
