# chatbot-engine

A domain-specialised AI customer support assistant built with FastAPI,
LangChain, OpenRouter, RAG, tool calling, and a vector database.

## Architecture

``` text
Frontend
   ↓
FastAPI Backend
   ↓
Chat Service
   ↓
LangChain
   ├── RAG
   ├── Tools
   └── LLM
   ↓
OpenRouter
```

## Layout

``` text
backend/
└── src/
    └── support_agent/
        ├── api/
        ├── chains/
        ├── models/
        ├── services/
        ├── rag/
        ├── tools/
        ├── config.py
        ├── constants.py
        ├── llm.py
        └── app.py

docs/
tests/
```

## Setup

Install dependencies:

``` bash
uv sync
```

Copy `.env.example` to `.env` and add the required environment
variables:

``` bash
cp .env.example .env
```

`.env` is ignored by Git.

## Run

Start the FastAPI development server from the project root:

``` bash
uv run fastapi dev backend/src/support_agent/app.py
```

Then open:

``` text
http://127.0.0.1:8000/docs
```

to test the backend through FastAPI's Swagger interface.

## Current Status

Currently implemented:

-   FastAPI backend structure
-   Typed request and response models
-   LangChain integration
-   OpenRouter LLM integration
-   Structured LLM responses
-   Basic customer support chat flow

Planned:

-   Customer support knowledge base
-   RAG ingestion and retrieval
-   Query translation
-   Tool calling
-   Vector database
-   Frontend
-   Evaluation and security improvements

## Test

``` bash
uv run pytest
```
