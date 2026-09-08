# Retrieval

Retrieval decides which chunks the model reads. It runs once per turn, before
the model is called, and the same pipeline serves chat turns and the RAGAS
evaluation.

```text
question
  -> rewritten against the conversation history
  -> per query: vector search, and keyword search under `hybrid`
  -> rankings fused (reciprocal rank fusion)
  -> merged across queries, best score per chunk
  -> reranked by the model, when `rerank` is on
  -> the top `top_k` reach the model
```

Every retrieved chunk carries a score in [0, 1], where 1.0 is the best match
in the result set. The score means the same thing in every mode, which is what
lets the UI show it as a confidence.

## Modes

| `retrieval` | Finds chunks by | Suits |
| --- | --- | --- |
| `vector` | embedding similarity | questions phrased differently from the text |
| `hybrid` | embedding similarity fused with BM25 keyword match | the above, plus exact terms: fare names, codes, product numbers |

`hybrid` is the engine default. A vector search finds what a chunk is *about*;
it is poor at exact terms that appear once and carry little semantic weight,
which are common in support material. BM25 ranks by term match and is good at
precisely those. Fusing the two catches both.

Fusion uses reciprocal rank fusion: each search contributes `1 / (60 + rank)`
per chunk. Ranks rather than scores, because a cosine similarity and a BM25
score are not on a common scale. A chunk found by both searches outranks one
found first by only one of them.

## The keyword index

The BM25 index is built from the chunks already in the vector store, per
project, so there is one source of truth and nothing extra to persist. It is
rebuilt when the project's chunk count changes or 60 seconds have passed, and
dropped immediately when the process that holds it ingests or deletes a
document. With several engine replicas, a document ingested through one
replica is keyword-searchable on the others within a minute.

Tokenisation is word characters on lowercased text. That serves languages
that separate words with spaces. For languages that do not, the keyword half
contributes little and the vector half carries the query.

## Reranking

With `rerank: true`, the fused candidates are shown to the assistant's model
with the question, and it returns them in order of relevance. The top `top_k`
are then taken from that order. The score on each chunk stays the fused
retrieval score, so a citation's confidence is not an artefact of position.

Reranking costs one model call per turn, with a prompt containing every
candidate. It is off by default. It is worth enabling for an assistant whose
answers are read carefully and where retrieval quality matters more than
latency, and the RAGAS harness is how to find out whether it helps on a given
knowledge base.

A rerank can degrade but cannot lose a turn or a candidate: a reply that is
not a ranking, or a failed call, keeps the fused order, with a warning in the
log.

## Configuration

Per assistant, in the project configuration:

```yaml
top_k: 5                   # chunks that reach the model
retrieval: hybrid          # vector | hybrid
rerank: false              # one extra model call per turn when true
retrieval_candidates: 20   # per search, before fusion and reranking
```

Omitted fields fall back to the engine's `ENGINE_RETRIEVAL`, `ENGINE_RERANK`
and `ENGINE_RETRIEVAL_CANDIDATES`.

`retrieval_candidates` bounds recall: a chunk outside the top candidates of
both searches cannot be retrieved. Raising it improves recall at the cost of a
longer rerank prompt when reranking is on.

## Evaluating a change

`POST /eval/rag` scores retrieval with RAGAS on a dataset of questions with
reference answers. Run it before and after changing any of the settings above
and compare context precision and recall; the example backend's `make eval-rag`
does this for its own knowledge base.

The example dataset holds 55 cases over the nine knowledge documents, in three
categories: `single_turn` questions that stand alone, `follow_up` questions
whose subject is only in the history, and `negative` questions the knowledge
base does not answer, where the reference answer is that the assistant should
say so. The report groups scores by category, so a change that helps
follow-ups and hurts negatives is visible as such.

## The small calls, and what they cost

The query rewrite and the rerank are model calls. Both run on
`ENGINE_UTILITY_MODEL` when it is set, and on the assistant's own model
otherwise; a cheap, fast model there cuts latency and cost without touching
the answer, which neither call writes.

Their tokens are counted in the turn's `usage` event along with the answer's,
so the cost a caller sees is the whole turn's. They are priced at the answer
model's rate: when a cheaper utility model is in use, that overstates the cost
by a small, known amount rather than requiring per-call bookkeeping.

An agent gets the counts from `retrieve_with_usage()` and passes them to
`stream_completion(..., prior=...)`; both bundled agents do.
