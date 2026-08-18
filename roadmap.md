# Roadmap

Detailed version, with file paths:
[docs/architecture.md §10](docs/architecture.md#10-where-to-implement-each-piece).

Done:

- [x] Set up the engine folder
- [x] Migrate files from backend to the engine
- [x] Initial setup of Docker and docker compose
- [x] Define the communication standard between the engine and the backend
      (HTTP + NDJSON; see docs/architecture.md §3)
- [x] Define config file for the engine and a way for the backend to expose it
      (`projects/support.yaml`, sent inline with every request)
- [x] Wire the backend to the engine (`engine_client/`)

Open:

- [ ] Set up the LLM call
- [ ] Validation of the LLM call
- [ ] Set up the vector database for the engine
- [ ] Add two users to the sample backend, to prove there is no user leakage and
      no room for prompt injection
