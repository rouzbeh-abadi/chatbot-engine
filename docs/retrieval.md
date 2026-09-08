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

A run is sent a few cases per request (`make eval-rag` uses five), so no
request lasts long enough to hit a timeout and a failure part-way keeps every
batch already scored. The engine logs one line per case as it scores it.

Scoring is not free: RAGAS makes about thirty judge-model calls per case,
each carrying the question, the answer and the retrieved chunks. The judge
is `ENGINE_RAG_JUDGE_MODEL`, by default `google/gemini-2.5-flash-lite`, kept
in a different family than the answering model so nothing grades itself.
At that model's prices the 55-case dataset costs about $0.25 per run,
measured; a judge at Claude Haiku prices makes the same run about $7. Run
it when a retrieval-side change is on the table, not routinely.

A follow-up is scored against its rewritten, self-contained question rather
than the words the customer typed. The metrics see one message, not the
conversation, and "how long does it need to stay valid?" on its own would
make a correct answer about passport validity look off-topic. The rewrite is
the same one retrieval uses, made once and shared.

The example dataset holds 55 cases over the nine knowledge documents, in three
categories: `single_turn` questions that stand alone, `follow_up` questions
whose subject is only in the history, and `negative` questions the knowledge
base does not answer, where the reference answer is that the assistant should
say so. The report groups scores by category, so a change that helps
follow-ups and hurts negatives is visible as such.

## Baseline

Scored on 8 September 2026 with the example project as shipped: answers by
`openai/gpt-5-mini`, hybrid retrieval, no rerank, judged by
`google/gemini-2.5-flash-lite`. Cost: $0.28, 25 minutes.

| category    | cases | faithfulness | answer relevancy | context precision | context recall |
|-------------|------:|-------------:|-----------------:|------------------:|---------------:|
| single_turn |    42 |         0.86 |             0.70 |              0.82 |           0.98 |
| follow_up   |    10 |         0.90 |             0.62 |              0.76 |           0.95 |
| negative    |     3 |         0.35 |             0.21 |              0.00 |           0.00 |
| overall     |    55 |         0.83 |             0.66 |              0.76 |           0.92 |

An earlier run the same day, before the prompt asked for the answer in the
first sentence and before follow-ups were scored against their standalone
question, had overall answer relevancy at 0.60 and follow-ups at 0.49. The
retrieval columns did not move; the answer columns did.

How to read it:

- Context recall is the retrieval number. At 0.98 and 0.95 for the answerable
  categories, the chunk that holds the answer is almost always retrieved.
- Context precision below that means the answer's chunk arrives with
  neighbours that do not help. Reranking is the lever for it, and costs one
  more small call per turn.
- Answer relevancy is the weakest metric. RAGAS measures how directly the
  answer addresses the question, and multiplies by zero when it reads the
  answer as non-committal. An honest "the documents do not give a number"
  therefore scores zero even when it is right: `followup_passport_validity`
  is one, and the knowledge base really does not state a validity period.
- The negative cases score near zero by construction: they have no reference
  context and the right answer is a refusal, which these four metrics cannot
  reward. They are in the dataset to be read, not averaged.
- The cheap judge leaves the odd faithfulness cell empty where its output did
  not parse: 8 of 55 on this run. Empty cells are left out of the averages
  rather than counted as zero. Single cases move by 0.2 or more between
  otherwise identical runs, so read the category rows and treat one-case
  differences as noise.

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
