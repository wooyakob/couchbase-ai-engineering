# Chapter 4: Embeddings and the Vectorization Service

> *An embedding model turns text into coordinates in meaning-space; retrieval becomes geometry. The engineering question is not "how do embeddings work" but "who keeps them correct as data changes forever." Capella's Vectorization service answers: the database does.*

---

## 4.1 Embeddings in Five Minutes
An embedding model maps text or images to a fixed-length vector of floats such that semantically similar inputs land near each other.

Similarity is computed with **cosine similarity**, **dot product**, or **Euclidean (L2) distance**. Your index must be built with the same metric family the model was trained for (Ch. 5).

The three facts that drive engineering decisions:

1. **Dimensions are fixed per deployment**: `text-embedding-3-small` → 1536; `intfloat/e5-mistral-7b-instruct` is 4096-dim natively, but the Capella Model Service deployment used in this book serves 2048. The vector index's `dims` must match exactly — measure with one probe call (the notebooks do) rather than trusting a spec sheet.
2. **Vectors from different models are incompatible.** Query vectors must come from the *same model* as document vectors. Changing models = re-embedding everything.
3. **Embeddings go stale.** When the document changes, its vector is wrong until re-embedded. Staleness is silent; retrieval quality just degrades.

Fact 3 is why "generate embeddings" is a *service* problem, not a script problem.

---

## 4.2 Two Ways to Vectorize on Couchbase

### DIY: Embed in Your Pipeline
You call the embedding API and store the vector on the document (the `embed()` in Ch. 3's pipeline):

```python
from openai import OpenAI

client = OpenAI()  # or Capella Model Service, Ch. 8

def embed(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in resp.data]

vectors = embed([c["text"] for c in chunks])          # batch, always batch
for chunk, vec in zip(chunks, vectors):
    chunk["embedding"] = vec
    chunk["embedding_model"] = "text-embedding-3-small"
    chunks_coll.upsert(key_for(chunk), chunk)
```

You own batching, retries, rate limits, freshness (Eventing pattern, §3.6), and backfills. Full control, full responsibility.

### Managed: Capella Vectorization Workflows (Recommended)
Capella AI Services' **auto-vectorization workflows** run the whole pipeline as managed infrastructure. You configure a workflow in the Capella UI ([docs](https://docs.couchbase.com/ai/build/vectorization-service/vectorize-structured-data.html)):

1. **Source**: Couchbase collection (structured JSON) or an S3 bucket (unstructured files: PDFs and friends, which the Data Processing service converts to JSON first).
2. **Fields**: Which field(s) to embed.
3. **Chunking strategy**: Fixed-size, sentence, paragraph, or semantic (the default; see §3.3 for how to choose).
4. **Embedding model**: A Capella Model Service-hosted model or OpenAI (bring your API key).
5. **Destination + index**: Where embeddings are written, and the workflow creates the matching **Vector Search index** for you (dims and similarity set correctly for the chosen model, one whole class of config errors eliminated).

From then on the workflow watches the source: new and changed documents get (re)processed and (re)embedded automatically.

The staleness problem, the hard 20% of DIY, is the managed feature.

---

## 4.3 Reading a Vectorized Document
Either path produces documents like this in `ai.docs.chunks`:

```json
{
  "text": "To rotate credentials in Capella, open Settings → ...",
  "embedding": [0.013, -0.082, ...],
  "embedding_model": "intfloat/e5-mistral-7b-instruct",
  "metadata": {"source": "docs-manual", "product": "capella"}
}
```

Sanity checks worth automating (the notebook does):

```sql
-- coverage: chunks missing embeddings
SELECT COUNT(*) AS missing FROM ai.docs.chunks c WHERE c.embedding IS MISSING;

-- consistency: exactly one model in use?
SELECT c.embedding_model, COUNT(*) AS n, ARRAY_LENGTH(c.embedding) AS dims
FROM ai.docs.chunks c GROUP BY c.embedding_model, ARRAY_LENGTH(c.embedding);
```

Two models or two dims in that second result means a broken migration. Some queries are searching in a space where half the corpus doesn't live.

---

## 4.4 Choosing an Embedding Model

1. **Data locality**: If content can't leave your perimeter, Capella-hosted `e5-mistral` (Ch. 8) or another in-VPC model wins outright.
2. **Quality on your retrieval task**: Measure with Ragas `context_recall` on a labeled set (Ch. 13); leaderboard rank (MTEB) is a prior, not an answer.
3. **Dimensions = cost**: 4096-dim vectors cost ~2.7× the memory/storage of 1536-dim. At millions of chunks this is real money and real latency.
4. **Latency**: The query-time embedding call sits on your critical path (Ch. 6 measures it).

Migration discipline: embed-model changes are corpus rebuilds. Write the new vectors to a new field or collection, build the new index, A/B with Ragas, then cut over, never in place.

---

## 4.5 Query-Time Embedding

Retrieval needs the *query* embedded too, with the same model. Note for `e5`-family models: they're trained with instruction prefixes; queries should be embedded as `"query: ..."` (retrieval quality drops noticeably if you skip this; encode it in one helper and use it everywhere):

```python
def embed_query(text: str) -> list[float]:
    if EMBEDDING_MODEL.startswith("intfloat/e5"):
        text = f"query: {text}"
    return embed([text])[0]
```

---

## 4.6 Recap

- Embeddings are coordinates; same model everywhere, dims match the index, or nothing works.
- Freshness is the real problem: Capella vectorization workflows solve it managed; Eventing + workers solve it DIY.
- Model changes are migrations. Measure before switching.

Notebook: [`notebooks/02_vector_search_fundamentals.ipynb`](../notebooks/02_vector_search_fundamentals.ipynb) (embedding + indexing half).

Running into errors? See [Troubleshooting](troubleshooting.md).

Next: [Chapter 5: Vector Search](05-vector-search.md).
