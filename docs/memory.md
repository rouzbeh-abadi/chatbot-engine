# Long-term memory

The assistant stores facts about a customer and applies them in later
conversations. Notes persist across conversations: starting a new chat clears
the transcript, not the memory.

Memory is implemented in the backend. The engine holds no memory state and
defines no memory concept.

## Scope

A note belongs to a user, identified by `user_id`. It does not belong to a
conversation.

| Action | Effect on memory |
| --- | --- |
| New chat | none; notes persist |
| Forget everything (UI), `DELETE /memory` | all of the caller's notes are erased |

`session_id` is recorded on each row to show where a note originated. No query
filters on it.

## Identity

The owner of a note is resolved in
[`api/identity.py`](../examples/backend/src/support_agent/api/identity.py) by
the `resolve_memory_owner` dependency.

| `BACKEND_TRUST_USER_HEADER` | Owner | Header | Forgeable |
| --- | --- | --- | --- |
| `1` (proxy authenticates callers) | the authenticated user | `X-User-Id`, set by the proxy | no |
| unset (default) | the browser's client id | `X-Client-Id`, set by the browser | yes |

The two headers are deliberately distinct. `X-User-Id` is the proxy's: the
bundled nginx clears it on every request, so a browser can never reach the
backend with one. `X-Client-Id` is the browser's own, passes through the proxy
untouched, and is never treated as authentication. The example UI generates it
once and stores it in `localStorage`, which gives each browser a stable
identity across chats and restarts.

**The client id partitions data; it does not protect it.** Any client can send
any other client's id and read its notes. Two constraints follow:

1. Do not key authorisation on `user_id` while the default configuration is in
   use. Memory is the only consumer of it today, which is what makes the current
   arrangement acceptable.
2. Set `BACKEND_TRUST_USER_HEADER=1` behind an authenticating proxy before
   serving real users. No schema or application change is required.

`resolve_memory_owner` is separate from `resolve_user_id` so that the relaxed
rule applies to memory alone and does not affect routes that require an
authenticated caller.

## Write path

Writing is performed by the model through the `remember` MCP tool, which takes a
`subject` and `content`. The model determines what is worth storing; the system
prompt defines the criteria and the exclusions.

The owner is read from the `X-User-Id` header that the engine forwards. It is
never taken from a tool argument, so the model cannot select whose memory it
writes to. A call without that header is rejected.

The tool does not report storage to the customer. The system prompt instructs
the assistant to store a note and continue answering.

## Read path

Reading is performed by the backend, not by the model. Before a request is sent
to the engine, `recall_for_prompt` loads the caller's notes and appends them to
the system prompt.

```
POST /chat  X-Client-Id: u-123
      |
      +-- load notes for u-123
      +-- append to the system prompt
      +-- send the assistant configuration to the engine
```

Recall is not exposed as a tool. A tool-based read depends on the model electing
to call it, which produces intermittent failures that are indistinguishable to
the customer from the assistant having no memory at all. Injection makes recall
unconditional.

At most `PROMPT_LIMIT` (20) notes are injected, most recently updated first.

## Engine responsibilities

The engine forwards two headers to every MCP tool call and assigns no meaning to
either:

| Header | Value |
| --- | --- |
| `X-User-Id` | the caller supplied by the backend |
| `X-Session-Id` | the conversation supplied by the backend |

A tool server requires both to scope its reads and writes. Neither is available
to the model.

## Storage

Table `memories`:

| Column | Notes |
| --- | --- |
| `user_id` | owner; every query filters on it |
| `project_id` | assistant that stored the note |
| `session_id` | conversation of origin; nullable, not queried |
| `subject` | short label, maximum 120 characters |
| `content` | the fact, maximum 2000 characters |
| `created_at`, `updated_at` | timestamps |

Indexes:

- `ix_memories_user_project` on `(user_id, project_id)`
- `uq_memories_user_project_subject`, unique, on
  `(user_id, project_id, subject)`

The unique constraint makes a repeated subject an update rather than an insert,
so a preference revised in a later conversation replaces the earlier value.

## Excluded content

The system prompt prohibits storing payment details, passwords, identity
document numbers, health information beyond what a request requires, booking
references, and anything the customer asks not to be kept.

This is a prompt-level control and is therefore a default rather than a
guarantee. Deployments handling real customer data should validate tool input
before it is written.

## Notes are untrusted input

Note content originates from customer input and is reinserted into the system
prompt on later turns. It is treated as the same class of input as a retrieved
document. The injected block states that the notes are information rather than
instructions, and the system prompt's rule covering tool results applies to
them.

Because notes cross conversations, content injected in one conversation reaches
all later conversations for that user.

## API

```http
GET    /memory     notes belonging to the caller (X-Client-Id, or the proxy's X-User-Id)
DELETE /memory     erase them
```

There is no write endpoint. Writing is available only to the assistant through
its tool, so a client cannot insert content that the assistant will later treat
as its own notes.

Both routes resolve the caller through `resolve_memory_owner` and are therefore
subject to the identity constraints above.

## Tests

[`examples/backend/tests/test_memory.py`](../examples/backend/tests/test_memory.py)
covers two properties:

- a note written in one conversation is readable from another
- notes are confined to their owner, in the read path and in the `remember` tool

Each test seeds an unrelated owner that no assertion queries. Without it, a
query missing its owner filter returns an empty result on an empty database and
the test passes regardless.

## Production checklist

1. Enable authentication and set `BACKEND_TRUST_USER_HEADER=1`.
2. Define retention and deletion beyond the current erase-all control, as notes
   are personal data.
3. Validate tool input rather than relying on the system prompt.
