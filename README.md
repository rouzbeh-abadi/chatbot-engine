# chatbot-engine

[![CI](https://github.com/rouzbeh-abadi/chatbot-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/rouzbeh-abadi/chatbot-engine/actions/workflows/ci.yml)
[![Release](https://github.com/rouzbeh-abadi/chatbot-engine/actions/workflows/release.yml/badge.svg)](https://github.com/rouzbeh-abadi/chatbot-engine/actions/workflows/release.yml)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

chatbot-engine is an HTTP service for building chatbots that answer from your
own documents and call your own tools. It implements the AI part of a chatbot,
which is retrieval-augmented generation (RAG) over a knowledge base, prompting
a language model, streaming the answer, and the loop in which the model calls
tools and reads their results. It is written in Python with FastAPI and
LangChain, stores vectors in Chroma, reaches models through OpenRouter, and
calls tools over the Model Context Protocol (MCP).

The engine is stateless with respect to your product. The system prompt, the
model, the allowed tools and the retrieval settings are sent by your backend
with every request. As a result, there is no configuration to migrate, a
prompt change takes effect on the next request, and one engine instance can
serve many assistants with separate knowledge bases.

The engine is the `engine/` package. The `examples/` directory contains a demo
application and an agent plugin that show it in use. Nothing under `examples/`
is required to run the engine.

## Features

- **Retrieval-augmented answers:** Documents are chunked and embedded. Each
  question is answered with hybrid retrieval, which fuses vector similarity
  with BM25 keyword search, followed by an optional rerank by the model. See
  [docs/retrieval.md](docs/retrieval.md).
- **Tool calling over MCP:** The model can call tools that run in your
  backend. Your backend exposes them as an MCP server and lists the allowed
  tool names in the request. The engine calls only those.
- **Streaming:** The response is a stream of JSON events with the sources, the
  answer tokens, tool calls, token usage and cost. A model stream that fails
  before its first token is retried. A failure after the first token is
  reported as an event inside the stream, so the caller never receives a
  duplicated answer.
- **Bounded turns:** `max_output_tokens` caps the length of one reply and
  `max_tool_iterations` caps the number of tool rounds. The final `done` event
  reports which limit ended the turn, as `length` or `tool_limit`, so the
  interface can inform the user instead of the turn failing after text has
  streamed.
- **Any model:** The model is selected per request. OpenAI, Anthropic, Google,
  DeepSeek and others are available through OpenRouter.
- **Any language:** Retrieval works across languages, so a question in one
  language is answered in that language from the same knowledge base. The
  reply language is controlled by the system prompt, which your backend owns.
- **Document ingestion:** Upload a file and the engine extracts the text,
  chunks it, embeds it and stores it. Markdown, plain text and PDF are
  supported. Identical bytes uploaded again are detected by content hash and
  skipped.
- **Pluggable agents:** The built-in agent is a plain LangChain tool loop with
  no framework dependency. Other agents are Python packages installed next to
  the engine, registered through an entry point and selected by name in the
  request. See [docs/agents.md](docs/agents.md).
- **Workflows:** With the LangGraph plugin, a request can carry a `workflow`
  that defines the turn as a graph of steps (retrieve, model, condition, tool,
  reply, hand-off and end) with edges between them. The engine builds and runs
  the LangGraph for that turn. See the Workflows section of
  [docs/agents.md](docs/agents.md).
- **Tracing:** With `ENGINE_TRACING` set, every model call in a turn is sent to
  LangSmith or Langfuse, hosted or self-hosted, tagged with the request,
  project, session and user ids. An assistant can also carry its own Langfuse
  keys. See the Tracing section of [DEPLOYMENT.md](DEPLOYMENT.md).
- **Chunking strategies:** Documents can be split by fixed size, by Markdown
  heading or by page. The strategy is set per project, and the page strategy
  lets PDF citations carry page numbers. See
  [docs/chunking.md](docs/chunking.md).
- **Production settings:** API keys per caller, rate limits on the routes that
  cost money, a production mode that refuses to start with development
  defaults, and external stores (a Chroma server, Postgres, S3, Redis) through
  environment variables. See [DEPLOYMENT.md](DEPLOYMENT.md).
- **Evaluation:** A judge-model harness scores the assistant's behaviour
  against a rubric, and a RAGAS harness scores retrieval on faithfulness,
  answer relevancy, context precision and context recall.
- **Guardrails:** Tools are allowlisted per request. Retrieved text and tool
  results are placed in the context as data, never as instructions.

## How to use the engine

The engine is called from your own backend over HTTP. The steps below go from
an empty engine to a chatbot that answers from your documents and calls your
tools. [docs/backend-integration.md](docs/backend-integration.md) documents
every field in detail.

### Step 1. Run it

You need Docker and an [OpenRouter API key](https://openrouter.ai/keys). The
published image takes the key as an environment variable and keeps its data
(vectors, the document registry and the uploaded files) on one volume. The
three directory settings point the stores at that volume. Images after 0.1.7
use these paths by default.

```bash
docker run -d --name engine -p 8100:8100 \
  -e ENGINE_OPENROUTER_API_KEY=sk-or-... \
  -e ENGINE_CHROMA_DIR=/var/lib/chatbot-engine/chroma \
  -e ENGINE_REGISTRY_DB=/var/lib/chatbot-engine/documents.sqlite3 \
  -e ENGINE_BLOB_DIR=/var/lib/chatbot-engine/blobs \
  -v engine-data:/var/lib/chatbot-engine \
  ghcr.io/rouzbeh-abadi/chatbot-engine/engine-langgraph:0.1.7
```

The readiness endpoint reports whether a provider key is set, whether the
vector store answers, and which agents are installed.

```bash
curl localhost:8100/health/ready
```

```json
{"ready": true, "model_provider": true, "vector_store": true, "agents": ["graph", "loop", "workflow"]}
```

### Step 2. Add documents

Upload each document as multipart form data. `project_id` identifies the
knowledge base and is sent again with every question. `external_id` is your
own stable identifier for the document, so uploading it again replaces the
previous version.

```bash
curl -X PUT localhost:8100/documents \
  -F project_id=docs \
  -F external_id=faq.md \
  -F "file=@faq.md;type=text/markdown"
```

The engine extracts the text, chunks it, embeds it and stores it, and returns
a record with the document's status and chunk count. Uploading the same bytes
twice is a no-op.

### Step 3. Ask a question

Send the message together with the assistant definition. The engine stores no
configuration, so the definition is part of every request. The minimum is a
project id, a name and a system prompt.

```bash
curl -N -X POST localhost:8100/chat -H 'Content-Type: application/json' -d '{
  "project": {
    "project_id": "docs",
    "name": "Docs assistant",
    "system_prompt": "Answer from the documents. If they do not cover the question, say so."
  },
  "message": "How do I reset my password?"
}'
```

The `project` block also accepts `model`, `temperature`, `top_k` (how many
passages reach the model), `retrieval` (`vector` or `hybrid`), `rerank`,
`max_output_tokens` and `max_tool_iterations`. Previous turns go in `history`
as a list of `role` and `content` objects. `session_id` and `user_id` identify
the conversation and the person, and both are forwarded to your tool server.

### Step 4. Read the answer

The response is newline-delimited JSON, one event per line. The retrieval
event comes first, so an interface can show the sources while the model is
still generating. The answer tokens follow, then the usage, then `done`.

```text
{"type": "retrieval", "query": "How do I reset my password?", "sources": [{"doc_id": "…", "source": "faq.md", "score": 0.91, "excerpt": "…"}]}
{"type": "token", "text": "Open "}
{"type": "token", "text": "Settings, then "}
{"type": "usage", "input_tokens": 812, "output_tokens": 24, "total_tokens": 836, "cost_usd": 0.0004, "model": "openai/gpt-5-mini"}
{"type": "done", "finish_reason": "stop"}
```

Every turn ends with `done`. Its `finish_reason` is `stop` for a normal end,
`length` when the reply reached `max_output_tokens`, `tool_limit` when the
model was still requesting tools after `max_tool_iterations` rounds, and
`error` when an `error` event preceded it. A backend usually converts these
lines into server-sent events for the browser.

### Step 5. Give it tools

A tool is a function that runs in your backend and returns text, such as an
order lookup or a ticket creation. Expose your tools as an MCP server and list
that server in the request with the tool names the assistant may call. The
engine discovers the tools, passes them to the model, and executes the calls
the model makes. It forwards `user_id` and `session_id` as the `X-User-Id` and
`X-Session-Id` headers, so your tool server can scope reads and writes to the
right person.

```json
"project": {
  "project_id": "docs",
  "name": "Docs assistant",
  "system_prompt": "…",
  "mcp_servers": [
    {"name": "shop", "url": "http://tools:8200/mcp", "allowed_tools": ["get_order_status"]}
  ]
}
```

Each call appears in the stream as a `tool_call_started` event and a
`tool_call_finished` event with the result, duration and any error. A failed
tool does not end the turn. The error text is returned to the model, which can
answer accordingly. Section 7 of
[docs/backend-integration.md](docs/backend-integration.md) shows how to write
a tool server.

### Step 6. Choose how a turn runs

By default, the engine runs each turn with LangChain. The built-in `loop`
agent retrieves the relevant passages, calls the model, executes any tools the
model asks for, and repeats until the model gives its final answer. Nothing
needs to be configured for this, and it is enough for most chatbots.

If you need workflows and automation, LangGraph is also supported. The
`engine-langgraph` image includes a LangGraph plugin with two agents, selected
by setting `agent` in the project block.

- `agent: "graph"` runs the standard turn as a LangGraph state machine. It is
  a reference implementation for anyone who wants to build their own graph.
- `agent: "workflow"` runs a graph that you define in the request. You send
  the steps and the edges between them, using a fixed set of step types
  (retrieve, model, condition, tool, reply, hand-off and end), and the engine
  builds and executes the LangGraph for that turn. Use it when a turn needs
  branching, for example to detect that a visitor is asking for a person and
  hand over instead of answering.

The request format and the event stream are identical for every agent, so
switching agents requires no change in your backend. The example below runs
a turn with one condition in it.

```json
"project": {
  "project_id": "docs",
  "name": "Docs assistant",
  "system_prompt": "…",
  "agent": "workflow",
  "workflow": {
    "start": "retrieve",
    "nodes": [
      {"id": "retrieve", "type": "retrieve"},
      {"id": "wants_person", "type": "condition",
       "question": "Does the visitor ask for a human?",
       "branches": {"no": "answer", "yes": "handoff"}},
      {"id": "answer", "type": "model"},
      {"id": "handoff", "type": "handoff", "message": "A colleague will get back to you shortly."}
    ],
    "edges": [{"from": "retrieve", "to": "wants_person"}]
  }
}
```

The Workflows section of [docs/agents.md](docs/agents.md) documents every
step type.

### Step 7. Run it in production

Set an API key and enable production mode. The engine then requires the
`X-API-Key` header on every request except the health endpoints, and refuses
to start if any setting still has a development default. Tracing is enabled
with `ENGINE_TRACING` and the keys of the tracing backend.

```bash
docker run -d --name engine -p 8100:8100 \
  -e ENGINE_ENV=production \
  -e ENGINE_API_KEY=a-long-random-string \
  -e ENGINE_OPENROUTER_API_KEY=sk-or-... \
  -e ENGINE_TRACING=langfuse \
  -e ENGINE_LANGFUSE_PUBLIC_KEY=pk-lf-... \
  -e ENGINE_LANGFUSE_SECRET_KEY=sk-lf-... \
  -v engine-data:/var/lib/chatbot-engine \
  ghcr.io/rouzbeh-abadi/chatbot-engine/engine-langgraph:0.1.7
```

[DEPLOYMENT.md](DEPLOYMENT.md) covers the remaining topics, including rate
limits, the network layout, TLS, the external stores for vectors and files,
and running several engine instances.

## Architecture

The system has three services with separate responsibilities. The engine is
the AI service. The backend and the frontend can be implemented in any
language or framework, because the interfaces between them are HTTP and JSON.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/architecture-dark.png" />
  <img alt="Who owns what: a generic engine, a product-specific backend, and a frontend that can be any stack" src="docs/images/architecture.png" />
</picture>

- **The engine** (`:8100`) implements retrieval, prompt construction, the model
  call and the tool loop. It holds no product configuration, because the
  configuration arrives with every request.
- **The backend** (`:8000`) implements the product. It owns users, the
  assistant definitions (prompt, model and tools), the document endpoints and
  the domain tools. It contains no AI logic.
- **The frontend** (`:5173`) is the chat interface. It renders the streamed
  answer with sources and cost, and communicates only with the backend.

## How a chat turn flows

A request travels from the browser to the backend to the engine. The engine
reads the vector store and calls the model, and calls the backend's tool
server over MCP only when the model requests a tool. Events stream back along
the same path.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/chat-workflow-dark.png" />
  <img alt="One chat turn: request over HTTP to the engine, which retrieves, calls the model, and calls a tool over MCP when needed, then streams events back" src="docs/images/chat-workflow.png" />
</picture>

1. The browser sends the message to the backend.
2. The backend loads the assistant definition and calls the engine.
3. The engine retrieves the relevant chunks from the vector store and streams
   the answer from the model.
4. If the model requests a tool, the engine calls the backend's tool server
   over MCP, which queries the database.
5. The events stream back to the backend, beginning with the sources, then
   the tokens, then the usage, and ending with `done`.
6. The backend forwards them to the browser as server-sent events.

## The example application

`examples/` contains a complete demo built on the engine. It has a React chat
interface, a FastAPI backend with a Postgres database, an MCP tool server and
a small knowledge base. It shows one way of wrapping the engine, including the
features a product adds on top, such as long-term memory, conversation export
and an admin page for running the evaluation. It is not a dependency of the
engine.

To run it, install the dependencies, put your OpenRouter key in the generated
`.env` file, and start the stack with Docker Compose.

```bash
make setup
make up             # frontend, backend, engine, tools and Postgres
make migrate        # create the tables
make seed-db        # load the example data
make seed           # load the knowledge base
```

Then open **http://localhost:5173**. Running `make` without arguments lists
every target, including the ones that run the services locally with reload.
[examples/backend/README.md](examples/backend/README.md) and
[examples/frontend/README.md](examples/frontend/README.md) describe the demo.

## Writing your own agent

An agent is a Python class with a `run` method that takes a chat request and
yields events. The engine discovers agents through the
`chatbot_engine.agents` entry-point group, so an agent is an installable
package and requires no change to the engine.

```toml
# in your package, not in the engine
[project.entry-points."chatbot_engine.agents"]
my-agent = "my_package.agent:build"
```

```console
$ pip install ./my-agent      # wherever the engine runs
$ curl localhost:8100/agents
["loop", "my-agent"]
```

The agent is selected with `agent: "my-agent"` in the project block.

`examples/langgraph-agent/` is a complete plugin that registers the `graph`
and `workflow` agents. Both use the engine's own functions for streaming with
retry, running tool calls and pricing a turn, so their event stream is
identical to the built-in loop's. It is the recommended starting point, and
[docs/agents.md](docs/agents.md) specifies the event contract an agent must
follow.

Every release publishes two engine images. `engine` contains the built-in
loop only. `engine-langgraph` is the same engine with the LangGraph plugin
installed.

## Evaluation

Two evaluation harnesses run inside the engine, because only the engine holds
the model credentials. The caller supplies the dataset.

**System prompt.** A judge model scores the assistant's behaviour against a
rubric, checking that it refuses what it should, stays grounded in the
documents and does not invent policies.

```bash
make eval                       # score the system prompt (the stack must be running)
make eval ARGS=--show           # print the last run without new model calls
```

**Retrieval.** RAGAS scores the retrieval pipeline on faithfulness (is the
answer grounded in the retrieved context), answer relevancy (does it address
the question), and context precision and recall (did retrieval return the
relevant chunks).

```bash
make eval-rag                        # score retrieval (needs the engine's eval extra)
make eval-rag ARGS="--only follow_up"
```

## Project layout

```text
engine/     the engine package; see engine/README.md
examples/
  backend/          an example backend with a database, domain tools and an admin page
  frontend/         an example chat interface
  langgraph-agent/  the LangGraph agent plugin
docs/       guides
tests/      the contract-parity test between the engine and the example backend
```

The engine has no dependency on anything under `examples/`.

## Documentation

- [docs/backend-integration.md](docs/backend-integration.md) documents the
  HTTP contract field by field, from sending a message and reading the stream
  to uploading documents and exposing tools.
- [DEPLOYMENT.md](DEPLOYMENT.md) covers production deployment, including API
  keys, rate limits, the network layout, TLS, tracing, external stores and
  scaling out.
- [docs/agents.md](docs/agents.md) covers the agent contract, the LangGraph
  plugin, workflows and how to install your own agent.
- [docs/engine-architecture.md](docs/engine-architecture.md) maps the code,
  from the entry points through the layers a request passes to the module
  responsible for each step.
- [docs/retrieval.md](docs/retrieval.md) covers hybrid search, rank fusion,
  reranking and how to evaluate a retrieval change.
- [docs/chunking.md](docs/chunking.md) covers the chunking strategies and why
  changing one requires re-indexing.
- [docs/memory.md](docs/memory.md) describes how the example backend
  implements long-term memory. The engine has no memory concept of its own.
- [engine/README.md](engine/README.md) covers the engine package.

## Licence

Apache License 2.0. See [LICENSE](LICENSE).

Apache 2.0 was chosen over MIT for the explicit patent grant in section 3,
under which an adopter receives a licence to any patents covering this code
and loses it if they sue over them. The example backend and frontend are under
the same licence.

## Tests

```bash
make test           # the Python suites for the engine and the backend, and the frontend suite
make lint           # ruff, formatting and ty, as CI runs them
```
