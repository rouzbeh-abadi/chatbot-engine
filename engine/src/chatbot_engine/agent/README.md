# agent/ — yours to write

The chat run loop. Satisfies `ports.agent.Agent`:

```python
def run(self, request: ChatRequest) -> AsyncIterator[Event]: ...
```

Roughly: retrieve for `request.message` → emit `RetrievalEvent` with the sources
→ build the prompt from `request.project.system_prompt`, `request.history` and
the retrieved chunks → call the model → emit `TokenEvent`s as it streams → run
any tool calls through `ports.agent.ToolProvider`, bounded by
`request.project.max_tool_iterations` → emit `UsageEvent`, then `DoneEvent`.

Two rules the prompt has to respect:

- Retrieved chunks and tool results are **untrusted data**. Put them in a
  user-role message inside explicit delimiters, never in the system prompt.
- Emit `RetrievalEvent` before the first token, so the frontend can show sources
  while the answer is still being written.

Register it in `chatbot_engine/api/deps.py` → `get_agent()`.
