# Retrieval

Retrieval decides which chunks the model reads. It runs once per turn, before
the model is called, and the same pipeline serves chat turns and the RAGAS
evaluation.

```text
question
  -> rewritten against the conversation history, into at most four queries
  -> per query, all at once: vector search, and keyword search under `hybrid`
  -> rankings fused (reciprocal rank fusion)
  -> merged across queries, best score per chunk
  -> dropped below `min_score`, when set
  -> reranked by the model against every query, when `rerank` is on
  -> the top `top_k` reach the model
```

Every retrieved chunk is cited with its vector similarity to the question, a
number in [0, 1] that is the cosine of the two embeddings for a model like
OpenAI's (see [Scores](#scores)). The order is the one fusion or the rerank
gave; the score says how close the chunk is, in every mode. A chunk that only
the keyword search found has no vector score and is cited with 0. Before
0.1.26 a hybrid search cited the fused score instead, which is relative to the
result set, so its first chunk always showed 1.0.

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
per chunk. Ranks rather than scores, because a vector similarity and a BM25
score are not on a common scale. A chunk found by both searches outranks one
found first by only one of them.

Only the keyword search's three best matches take part (`KEYWORD_FUSED`).
Below those, a keyword "match" is mostly a common word the chunk shares with
the question ("to", "a", "have"), and because being in both rankings beats
being first in one, such matches would push out the chunk closest in meaning
whenever it shares no word with the question. "How long do I have to return a
lamp?" against a page that says "accepts returns within 30 days" is the case:
the tokenizer does not stem, so "returns" is not "return", and before this
limit that page lost its place in the top five to passages that merely
contained "to".

## The keyword index

The BM25 index is built from the chunks already in the vector store, per
project, so there is one source of truth and nothing extra to persist. It is
kept for 60 seconds and then built again, and dropped immediately when the
process that holds it ingests or deletes a document. Within those 60 seconds a
keyword search reads nothing from the vector store. With several engine
replicas, a document ingested through one replica is keyword-searchable on the
others within a minute.

Building the index reads every chunk the project has, and a search scores
every one of them, so both run in a worker thread rather than on the event
loop, where they would hold up every other request. Searches that find a
project's index cold while it is being built wait for that build instead of
starting their own. An engine keeps the indexes of the 32 projects searched
most recently, and no more than `ENGINE_KEYWORD_INDEX_MB` (512) of them; one
that comes back after that is read again, once. An index takes a little over
twice its project's text in memory. A project of more than 100,000 chunks gets
no keyword index and is searched by vector alone.

Tokenisation is word characters on lowercased text. That serves languages
that separate words with spaces. For languages that do not, the keyword half
contributes little and the vector half carries the query.

## Reranking

With `rerank: true`, the fused candidates are shown to the assistant's model
with every query the message became, one per line, and it returns them in
order of relevance. A message that asks two things is two queries, so a
passage that answers the second counts as much as one that answers the first.
The top `top_k` are then taken from that order. The score on each chunk stays
its vector similarity, so a citation's score is not an artefact of position.

Reranking costs one model call per turn, with a prompt containing every
candidate. It is off by default. It is worth enabling for an assistant whose
answers are read carefully and where retrieval quality matters more than
latency, and the RAGAS harness is how to find out whether it helps on a given
knowledge base.

A rerank can degrade but cannot lose a turn or a candidate: a reply that is
not a ranking, or a failed call, keeps the fused order, with a warning in the
log.

## A minimum score

Every turn searches, and without a bar the top `top_k` chunks reach the model
however far they are from the message. For "hi" or "thanks" that is a few
thousand characters of unrelated context in the answer's prompt, a rerank
call when reranking is on, and sources shown for small talk.

`min_score` sets that bar on vector similarity, the score a chunk is cited
with (see [Scores](#scores)). The fused score cannot serve: it is relative to
the result set, so its best chunk is always 1.0.

- When no chunk reaches `min_score`, nothing is retrieved: no context, no
  rerank call, no sources. The model answers from the conversation alone.
- Otherwise the chunks that reach it are kept, and under `hybrid` so is each
  query's best keyword match, since an exact term the embedding blurs is what
  hybrid retrieval is for. Only the best match: nearly every chunk shares a
  common word with a question.

The rewrite still runs on a turn with history, since the rewritten query is
what is compared.

A good value depends on the embedding model. Measured with
`openai/text-embedding-3-small` on a product help centre: small talk ("hi",
"thanks", "tell me a joke") scored 0.22 to 0.31 at best; product questions
scored 0.56 to 0.68. About 0.4 separates the two. A message that names
something the documents mention (a person, a product) will clear it, as it
should. Unset, every chunk is kept, as before. Check a new value with
`POST /eval/rag`, since a bar set too high shows up as lost context recall.

## Configuration

Per assistant, in the project configuration:

```yaml
top_k: 5                   # chunks that reach the model
retrieval: hybrid          # vector | hybrid
rerank: false              # one extra model call per turn when true
retrieval_candidates: 20   # per search, before fusion and reranking
min_score: 0.4             # least vector similarity; unset keeps every chunk
```

Omitted fields fall back to the engine's `ENGINE_RETRIEVAL`, `ENGINE_RERANK`
and `ENGINE_RETRIEVAL_CANDIDATES`.

`retrieval_candidates` bounds recall: a chunk outside the top candidates of
both searches cannot be retrieved. Raising it improves recall at the cost of a
longer rerank prompt when reranking is on.

## Scores

Chroma's collections are created in its default distance space, `l2`, and a
search returns the squared Euclidean distance between the question's
embedding and the chunk's. For embeddings of unit length, which OpenAI's are,
that distance is `2 - 2 cos`, so the engine's similarity, `1 - d / 2`, is the
cosine of the two embeddings, floored at zero. For a model whose vectors are
not of unit length it is still a closeness, but not a cosine, and values of
`min_score` measured with one model do not carry over to another. Moving the
collections to the cosine space would mean creating them again and embedding
every document anew, so the engine keeps `l2`.

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
At that model's prices the 55-case dataset costs about $0.28 per run, as
measured on the [baseline](#baseline) below; a judge at Claude Haiku prices
makes the same run about $7. Run it when a retrieval-side change is on the
table, not routinely.

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
follow-ups and hurts negatives is visible as such. Since 0.1.26 the report's
overall averages leave the `negative` cases out; they have their own category
row.

A metric that cannot be scored for a case, because the judge's reply did not
parse or a call failed, leaves the cell empty. Since 0.1.26 the engine logs
each one with its reason, and the report counts them per metric in
`unscored`, so an average taken over fewer cases than the others shows as
such.

The evaluated assistant runs without its MCP servers. Retrieval cases call no
tool, and an assistant that allowed tools but was offered none would be told
in its prompt that they are unavailable, a line no real turn has.

## Baseline

Scored on 8 September 2026 with engine 0.1.0 and the example project as
shipped: answers by `openai/gpt-5-mini`, hybrid retrieval, no rerank, judged
by `google/gemini-2.5-flash-lite`. Cost: $0.28, 25 minutes. The overall row
was averaged over all 55 cases, the three negative ones included, as the
engine did then; from 0.1.26 it averages the 52 answerable ones.

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
  reward. They are in the dataset to be read, not averaged, and since 0.1.26
  the overall averages leave them out.
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

The rewrite keeps at most four queries, each once, and may write at most
4,000 tokens, room for a reasoning model's thinking and a few short lines.
Neither bound shapes a normal answer; they stop a model that does not stop
from turning one message into dozens of searches. A rewrite cut off by the
token cap drops its unfinished last line.

The rewrite reads the last six turns of the history, a turn over 2,000
characters cut to its first 500 and last 1,500, since the end is where a list
just offered usually is. That is enough to resolve "it" or "the second one", and a
long conversation does not make every turn's rewrite cost more.

Their tokens are counted in the turn's `usage` event along with the answer's,
so the cost a caller sees is the whole turn's. When every call of the turn
reported what the provider billed (OpenRouter does), the cost is that sum,
exact for each model. Otherwise the whole turn is priced from `ENGINE_PRICING`
at the answer model's rate, which overstates the rewrite and the rerank by a
small, known amount when a cheaper utility model runs them.

An agent gets the counts from `retrieve_with_usage()` and passes them to
`stream_completion(..., prior=...)`; both bundled agents do.
