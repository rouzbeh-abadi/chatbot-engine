# A RAG chatbot engine with tool calling

[![CI](https://github.com/rouzbeh-abadi/chatbot-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/rouzbeh-abadi/chatbot-engine/actions/workflows/ci.yml)
[![Release](https://github.com/rouzbeh-abadi/chatbot-engine/actions/workflows/release.yml/badge.svg)](https://github.com/rouzbeh-abadi/chatbot-engine/actions/workflows/release.yml)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

A reusable engine for building domain-specific chatbots. It does the AI:
retrieval from a knowledge base, prompting, streaming the answer, and the loop
that calls tools and feeds their results back to the model. It knows nothing
about any particular product.

What makes a chatbot *yours* (the system prompt, the model, the tools, the
documents) is supplied by a backend you own, and arrives with **every request**.
So the engine holds no state to migrate, a prompt change takes effect on the next
request, and one engine can power many different assistants at once.

`engine/` is the product. Everything else in this repository lives under
`examples/` and exists to show it working: a travel-support assistant with a
React UI, a backend with its own database and tools, a knowledge base, and an
agent plugin. None of it is a dependency of the engine, and you are meant to
replace all of it.

## What it does

- **Retrieval-augmented answers.** Documents are chunked and embedded, then
  searched per question with hybrid retrieval: vector similarity fused with
  keyword search, and an optional rerank by the model. See
  [docs/retrieval.md](docs/retrieval.md).
- **Tool calling over MCP.** The model calls the backend's own tools when it
  needs live data; the engine only ever sees the tools a request allowlists.
- **Streaming.** The answer appears token by token, with tool activity and token
  cost shown as they happen. A stream that breaks before its first token is
  retried; one that breaks after it reports the error in the stream, so the
  caller never sees a duplicated answer.
- **Bounded turns.** `max_output_tokens` caps one reply and
  `max_tool_iterations` caps the tool rounds; the `done` event says which
  bound ended the turn (`length`, `tool_limit`) so the UI can tell the reader,
  instead of the turn failing after text has streamed.
- **Multi-model.** The model is chosen per request (OpenAI, Anthropic, Google,
  and more), all through OpenRouter.
- **Multi-language.** Ask in any language and the assistant replies in the same
  one, still grounded in the same knowledge base, since retrieval works across
  languages. This comes from the backend's system prompt, not the engine, since
  the backend owns the prompt.
- **Document ingestion.** Upload a file and the engine extracts, chunks, embeds,
  and stores it; re-uploading identical bytes is skipped by content hash.
- **Pluggable agents.** The engine ships one plain tool loop and no agent
  framework. Anything else, including the bundled LangGraph agent, is a package
  you install: register a factory under an entry point and name it in the
  project config, no fork required. See [docs/agents.md](docs/agents.md).
- **Workflows.** A request can carry a `workflow`: the turn as a graph of
  steps from a fixed library (retrieve, model, condition, tool, reply,
  hand-off, end) with edges between them. The bundled `workflow` agent builds
  a LangGraph from it per request, so a caller composes the turn as data and
  never ships code to the engine. See the Workflows section of
  [docs/agents.md](docs/agents.md).
- **Tracing.** With `ENGINE_TRACING` set, every model call in a turn is
  recorded in LangSmith or Langfuse (cloud or self-hosted), tagged with the
  request, project, session and user ids so a trace lines up with the log
  lines. An assistant can name its own Langfuse instead of the engine's. See
  the Tracing section of [DEPLOYMENT.md](DEPLOYMENT.md).
- **Chunking strategies.** Cut documents by fixed size, by Markdown heading, or
  by page. Chosen per project, so PDFs can carry page numbers into their
  citations. See [docs/chunking.md](docs/chunking.md).
- **Deployable.** Rate limits on the routes that cost money, one seam for real
  authentication, and a startup check that refuses to serve a production
  deployment with development defaults. See [DEPLOYMENT.md](DEPLOYMENT.md).
- **Evaluation.** An LLM-as-judge harness grades the assistant's behaviour
  against a rubric, and a RAGAS harness scores the retrieval (faithfulness,
  answer relevancy, context precision and recall).
- **Guardrails.** A tool allowlist, prompt-injection handling, and untrusted
  document and tool text treated as data, never as instructions.

The example application under `examples/` adds what a product adds on top,
and shows where each belongs:

- **Long-term memory.** The backend keeps notes about a customer and uses
  them in later conversations, scoped to the person. The engine has no
  memory concept; see [docs/memory.md](docs/memory.md) for one way to build
  it in a backend.
- **Conversation export.** Download a transcript as JSON, CSV, or PDF.
- **Admin dashboard.** Inspect the application data and run the evaluation
  from the browser, behind a shared operator key (`BACKEND_ADMIN_KEY`).

## Architecture

Three services, each owning one thing. The engine is the fixed AI service; the
frontend and backend can be any stack, because everything crosses a plain
HTTP + JSON boundary.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/architecture-dark.png" />
  <img alt="Who owns what: a generic engine, a product-specific backend, and a frontend that can be any stack" src="docs/images/architecture.png" />
</picture>

- **Engine** (`:8100`) is the AI. It owns retrieval, prompt construction, the
  model call, and the tool loop. It holds no product configuration, since that
  arrives with every request. This is the reusable part.
- **Backend** (`:8000`) is the product. It owns users, the assistant
  configuration (prompt, model, tools), the documents API, and the domain tools.
  It owns no AI logic, and can be any language. *(The example is a travel-support
  app in FastAPI.)*
- **Frontend** (`:5173`) is the chat interface. It streams the answer, shows
  sources and cost, and talks only to the backend. Any UI framework.

> Connecting a backend to the engine (the endpoints, the request shape, reading
> the stream, exposing your tools over MCP) is written up separately in
> **[docs/backend-integration.md](docs/backend-integration.md)**.

## How a chat turn flows

A request runs right to left (browser to backend to engine); a stream of events
runs back to the screen. The engine reaches down to the vector store and the
model, and sideways over MCP only when the model needs live data.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/chat-workflow-dark.png" />
  <img alt="One chat turn: request over HTTP to the engine, which retrieves, calls the model, and calls a tool over MCP when needed, then streams events back" src="docs/images/chat-workflow.png" />
</picture>

1. **Frontend to Backend.** The browser sends the message.
2. **Backend to Engine.** The backend loads its config and calls the engine.
3. **Engine to vector store and model.** Retrieve the nearest chunks, then stream
   the answer from the model.
4. **Engine to MCP to database.** If the model needs live data, it calls the
   backend's tool server, which queries the database.
5. **Engine to Backend.** Events stream back (sources, tokens, usage, done).
6. **Backend to Frontend.** Reframed as server-sent events; the UI paints tokens
   as they arrive.

## Running it

You need an [OpenRouter API key](https://openrouter.ai/keys). For the engine
alone, Docker is enough. For the example application as well, either Docker or
Python 3.13 plus Node plus a local Postgres.

### Just the engine

The published image, one key, one volume for its data (the three directory
settings point the vector store, the document registry and the uploads at
that volume; images after 0.1.7 default to them):

```bash
docker run -d --name engine -p 8100:8100 \
  -e ENGINE_OPENROUTER_API_KEY=sk-or-... \
  -e ENGINE_CHROMA_DIR=/var/lib/chatbot-engine/chroma \
  -e ENGINE_REGISTRY_DB=/var/lib/chatbot-engine/documents.sqlite3 \
  -e ENGINE_BLOB_DIR=/var/lib/chatbot-engine/blobs \
  -v engine-data:/var/lib/chatbot-engine \
  ghcr.io/rouzbeh-abadi/chatbot-engine/engine-langgraph:0.1.7
curl localhost:8100/health/ready
```

Give it a document, then ask about it. The assistant is described in the
request; the engine stores nothing about it:

```bash
curl -X PUT localhost:8100/documents \
  -F project_id=docs -F external_id=baggage.md \
  -F "file=@examples/backend/knowledge/baggage.md;type=text/markdown"

curl -N -X POST localhost:8100/chat -H 'Content-Type: application/json' -d '{
  "project": {"project_id": "docs", "name": "Docs",
              "system_prompt": "Answer from the documents, briefly."},
  "message": "What is the cabin baggage allowance?"
}'
```

The answer streams back as one JSON event per line: the sources it used, the
tokens, the usage, then `done`. That is the whole contract; the rest of this
README is the example application that wraps it, and
[docs/backend-integration.md](docs/backend-integration.md) is the guide to
wrapping it yourself. Without an `ENGINE_API_KEY` the engine is open, which is
fine on a laptop and refused in production mode; see
[DEPLOYMENT.md](DEPLOYMENT.md).

### One-time setup for the example application

```bash
make setup          # install dependencies and create .env
```

Then put your key in `.env`:

```
ENGINE_OPENROUTER_API_KEY=sk-or-...
```

### The whole stack, in Docker

```bash
make up             # frontend, backend, engine, tools, and Postgres
make migrate        # create the tables
make seed-db        # load the example data
make seed           # load the knowledge base
```

Open **http://localhost:5173**.

### Or run it locally for development

```bash
make db && make migrate && make seed-db   # Postgres in Docker, seeded
make dev            # engine (:8100) and backend (:8000), with reload
make tools          # the MCP tool server (:8200)
make frontend       # the UI (:5173)
make seed           # load the knowledge base
```

Run `make` on its own to see every command.

## Trying the example

The included travel-support app answers from its knowledge base, or calls a tool
when you give it a booking reference:

- *What is the cabin baggage allowance?* answers from the documents, with a
  citation.
- *Is my flight delayed? My booking is AB12CD.* chains two tools.
- *Can I get a refund on a Basic fare?* gives a grounded policy answer.
- *Wie viel Handgepäck darf ich mitnehmen?* answers in German from the same
  English documents.

Switch the model from the dropdown, export the chat, or open the **Admin
dashboard** to view the data and run the evaluation.

## Deploying it

Everything above is set up for a laptop: every service is published to the host,
the database password is in the compose file, and nothing is authenticated.

For anything other people can reach, set both services to production:

```
BACKEND_ENV=production
ENGINE_ENV=production
```

They will then refuse to start on a default that is only safe locally, naming
each variable to set, rather than serve with one. There is a compose overlay
that does the rest (unpublishes the internal ports, demands every secret):

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

**[DEPLOYMENT.md](DEPLOYMENT.md)** is the full guide: secrets, the network
shape, TLS, rate limits, migrations, and an honest list of what is
authenticated and what is not, which is worth reading before you put this in
front of users.

## Bringing your own agent

The engine ships one plain tool loop and no agent framework. Anything else is a
package you install, discovered through a Python entry point:

```toml
# in YOUR package, not the engine
[project.entry-points."chatbot_engine.agents"]
my-agent = "my_package.agent:build"
```

```console
$ pip install ./my-agent      # into wherever the engine runs
$ curl localhost:8100/agents
["loop", "my-agent"]
```

Then select it with `agent: my-agent` in the project config, or from the UI's
dropdown. No fork, no change to engine code.

`examples/langgraph-agent/` is a complete working one. It registers two
agents: `graph`, a fixed LangGraph state machine of four nodes and one
conditional edge, and `workflow`, which assembles a LangGraph per request from
the `workflow` block in the assistant config. Both reuse the engine's own
pieces (the streaming retry, the tool runner, the pricing), so a turn through
them is reported exactly as a turn through the loop, and their own code is the
graph and one small runner. The demo stack installs the plugin the same way
yours would be installed, so `agent: graph` in the picker really is an
injected plugin rather than something built in. Copy it as your starting
point, and see **[docs/agents.md](docs/agents.md)** for the contract your
agent owes.

Every release publishes two engine images: `engine`, with the loop agent only,
and `engine-langgraph`, the same engine with this plugin installed. Pick the
second to get `graph` and `workflow` without a build.

## Evaluation

Two harnesses. Both run in the engine (only it holds the model credentials);
the backend just sends the dataset.

**System prompt.** A judge model grades the assistant's behaviour against a
rubric: does it refuse what it should, stay grounded, and never invent a policy?
Run it from the Admin dashboard, or the command line:

```bash
make eval                       # score the system prompt (needs the stack running)
make eval ARGS=--show           # re-read the last run, no model calls
```

**Retrieval (RAGAS).** Scores the search itself: is the answer grounded in the
retrieved context (faithfulness), does it address the question (answer
relevancy), and did retrieval find relevant, sufficient chunks (context
precision and recall)? Run it from the Admin dashboard, or the command line:

```bash
make eval-rag                        # score retrieval (needs the engine eval extra)
make eval-rag ARGS="--only follow_up"
```

## Project layout

```text
engine/     the reusable AI engine. This is the product   -- see engine/README.md
examples/   everything built *with* it:
  backend/          an example product backend: database, domain tools, admin
  frontend/         an example chat UI
  langgraph-agent/  an agent plugin, installed into the engine
docs/       guides
tests/      the contract-parity test, where engine and backend meet
```

The split is the point. `engine/` depends on nothing under `examples/`, and holds
no opinion about your product, your users, or even which agent framework you use.
Everything under `examples/` is one way to use it, and you are meant to replace
all of it.

## Documentation

- **[docs/engine-architecture.md](docs/engine-architecture.md)** maps the
  engine's structure: entry points, the layers a request passes through, and the
  file responsible for each step.
- **[DEPLOYMENT.md](DEPLOYMENT.md)** covers running this where other people can
  reach it: secrets, the network shape, TLS, rate limits, and what is and is not
  authenticated.
- **[docs/backend-integration.md](docs/backend-integration.md)** shows how to
  connect a backend to the engine.
- **[docs/agents.md](docs/agents.md)** covers the agent contract, the bundled
  LangGraph plugin, workflows described in the request, and how to install
  your own agent.
- **[docs/memory.md](docs/memory.md)** covers the example backend's long-term
  memory: what is stored, why reading is injected rather than a tool, and why
  the unauthenticated owner id partitions notes without protecting them. The
  engine itself has no memory concept; this is a pattern for the backend.
- **[docs/retrieval.md](docs/retrieval.md)** covers retrieval: hybrid search,
  rank fusion, reranking, and how to evaluate a change.
- **[docs/chunking.md](docs/chunking.md)** explains the chunking strategies:
  what each cuts at, when to use it, and why changing one means re-indexing.
- **[engine/README.md](engine/README.md)** covers the engine itself.
- **[examples/backend/README.md](examples/backend/README.md)** and
  **[examples/frontend/README.md](examples/frontend/README.md)** cover the example
  application, and **[examples/langgraph-agent/](examples/langgraph-agent)** is a
  working agent plugin to copy.

## Licence

Apache License 2.0. See [LICENSE](LICENSE).

Chosen over MIT for the explicit patent grant in section 3: an adopter receives
a licence to any patents covering this code, and loses it if they sue over
them. Corporate legal review commonly requires that clause, and it costs a
permissive licence nothing.

The example backend and frontend are under the same licence. Take them, change
them, ship them. Attribution and the notice in section 4 are all that is asked.

## Tests

```bash
make test           # Python (engine + backend) and the frontend suite
make lint           # ruff, formatting, and ty, as CI runs them
```
