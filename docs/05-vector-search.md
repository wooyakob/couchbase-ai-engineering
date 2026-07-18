# Chapter 5: Vector Search

> *Vector search is the retrieval engine of AI applications: given a query vector, find the K nearest documents in meaning-space, filtered, scored, and combined with full-text matching, inside the same database that holds everything else.*

---

## 5.1 How Vector Search Works in Couchbase
Couchbase Vector Search lives in the **Search Service** (FTS).

A Search index can contain `vector`-type fields alongside text fields; vector fields are indexed for approximate nearest-neighbor (ANN) search and queried through the SDK's `SearchRequest` API. As a result of this design, **one index and one query** can combine semantic similarity, keyword matching and metadata filtering.

Three parameters define a vector field:

- **`dims`**: Must equal your embedding model's output length exactly (1536 for `text-embedding-3-small`; hosted deployments can differ from a model's native size — Capella's `e5-mistral` here serves 2048 — so measure it with a probe call).
- **`similarity`**: `dot_product` (use for normalized/OpenAI-style embeddings; equivalent to cosine on unit vectors) or `l2_norm` (Euclidean).
- **`vector_index_optimized_for`**: `recall` or `latency`.

---

## 5.2 Creating a Vector Index

Indexes are JSON definitions. This is the definition used throughout this repo, collection-scoped on `ai.docs.chunks`, indexing `embedding` (vector), `text` (stored text for retrieval), and `metadata` (dynamic, for filters):

```json
{
  "type": "fulltext-index",
  "name": "chunks-vector-index",
  "sourceType": "gocbcore",
  "sourceName": "ai",
  "planParams": {"maxPartitionsPerPIndex": 1024, "indexPartitions": 1},
  "params": {
    "doc_config": {"mode": "scope.collection.type_field", "type_field": "type"},
    "mapping": {
      "default_analyzer": "standard",
      "default_mapping": {"dynamic": false, "enabled": false},
      "types": {
        "docs.chunks": {
          "dynamic": false,
          "enabled": true,
          "properties": {
            "embedding": {
              "enabled": true,
              "dynamic": false,
              "fields": [{
                "name": "embedding",
                "type": "vector",
                "index": true,
                "dims": 1536,
                "similarity": "dot_product",
                "vector_index_optimized_for": "recall"
              }]
            },
            "text": {
              "enabled": true,
              "dynamic": false,
              "fields": [{"name": "text", "type": "text", "index": true,
                          "store": true, "analyzer": "en"}]
            },
            "metadata": {"enabled": true, "dynamic": true}
          }
        }
      }
    },
    "store": {"indexType": "scorch", "segmentVersion": 16}
  },
  "sourceParams": {}
}
```

Create it from Python with the scoped index manager (scoped indexes are the modern form; the Capella UI's *Search* tab works too and vectorization workflows can create them automatically):

```python
import json
from couchbase.management.search import SearchIndex

scope = cluster.bucket("ai").scope("docs")
with open("indexes/chunks-vector-index.json") as f:
    index_def = json.load(f)

scope.search_indexes().upsert_index(SearchIndex.from_json(index_def))

# wait for ingest
count = scope.search_indexes().get_indexed_documents_count("chunks-vector-index")
```

Reusing a definition across environments? Patch `name`, `sourceName`, and the `types` key (`"<scope>.<collection>"`) before upserting. The notebooks include the helper.

---

## 5.3 Querying: The `SearchRequest` API

The modern search API (SDK 4.x): use this, not the legacy `cluster.search_query`, for anything involving vectors:

```python
import couchbase.search as search
from couchbase.options import SearchOptions
from couchbase.vector_search import VectorQuery, VectorSearch

def semantic_search(query: str, k: int = 5):
    qvec = embed_query(query)                    # same model as the corpus! (Ch. 4)
    req = search.SearchRequest.create(
        VectorSearch.from_vector_query(
            VectorQuery("embedding", qvec, num_candidates=k)
        )
    )
    result = scope.search("chunks-vector-index", req,
                          SearchOptions(limit=k, fields=["text", "metadata.source"]))
    return [
        {"id": row.id, "score": row.score, **row.fields}
        for row in result.rows()
    ]
```

Mechanics worth knowing:

- `num_candidates` is the vector K (how many neighbors the ANN search returns); `SearchOptions(limit=...)` caps the final result count.
- The vector must be `list[float]`, all floats (a stray int raises `InvalidArgumentException`), or a base64-encoded string.
- `fields=[...]` returns stored fields with each hit, saving a KV round-trip when the hit itself is enough. For full documents, follow up with `collection.get(row.id)`, sub-millisecond by key.
- `scope.search(...)` targets scoped indexes; `cluster.search(...)` targets cluster-level ones.

---

## 5.4 Hybrid Search: Vector + Text + Filters
Pure semantic search fumbles exact identifiers ("error CB-4012", product codes, names), things keyword search nails. Combine both in one request:

```python
req = search.SearchRequest.create(
    search.MatchQuery("credential rotation", field="text")     # keyword side
).with_vector_search(
    VectorSearch.from_vector_query(
        VectorQuery("embedding", qvec, num_candidates=10, boost=1.5)  # semantic side
    )
)
result = scope.search("chunks-vector-index", req, SearchOptions(limit=5))
```

Scores from both sides merge into one ranked list; `boost` tunes the balance. The full FTS query zoo (`ConjunctionQuery`, `NumericRangeQuery`, `DateRangeQuery`, …) composes on the text side.

Prefiltering restricts the ANN search itself to documents matching a condition, the right tool for tenancy and access control, because filtering happens *before* K is spent:

```python
only_capella = search.MatchQuery("capella", field="metadata.product")
vq = VectorQuery("embedding", qvec, num_candidates=5, prefilter=only_capella)
```

Prefer prefilters over post-filtering retrieved rows: post-filtering K=5 hits by tenant can leave a user zero results even when their tenant has plenty of relevant documents.

Multiple vector fields (e.g., text + image embeddings on the same document) combine with AND/OR:

```python
from couchbase.options import VectorSearchOptions
from couchbase.vector_search import VectorQueryCombination

vs = VectorSearch(
    [VectorQuery("text_embedding", tvec, num_candidates=5, boost=0.7),
     VectorQuery("image_embedding", ivec, num_candidates=5, boost=0.3)],
    VectorSearchOptions(vector_query_combination=VectorQueryCombination.OR),
)
```

---

## 5.5 Through LangChain Instead

When you're in LangChain-land (Chapter 6 onward), `langchain-couchbase` wraps all of the above:

```python
from langchain_couchbase.vectorstores import CouchbaseSearchVectorStore
from langchain_openai import OpenAIEmbeddings

vector_store = CouchbaseSearchVectorStore(
    cluster=cluster,
    bucket_name="ai", scope_name="docs", collection_name="chunks",
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    index_name="chunks-vector-index",
)
vector_store.add_texts(["...chunk text..."], batch_size=50)
hits = vector_store.similarity_search_with_score("how do I rotate credentials?", k=5)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
```

Convention: the store writes `text`, `embedding`, and `metadata` fields, which is exactly what the §5.2 index maps. (The class was renamed from `CouchbaseVectorStore` to `CouchbaseSearchVectorStore` in `langchain-couchbase` 0.3; update old snippets.) `CouchbaseSearchVectorStore` needs Couchbase Server / Capella **7.6+**. `langchain-couchbase` has a second class, `CouchbaseQueryVectorStore`, wrapping the GSI-based Hyperscale/Composite indexes (**8.0+**) instead of this Search Service one (see [Chapter 14 §14.8](14-vector-index-architectures.md#148-through-langchain-instead)).

Raw SDK vs. LangChain is control vs. convenience: the SDK exposes hybrid queries, prefilters, boosts, and multi-vector combinations; the vector store gets you a `retriever` in five lines. The apps in this repo use LangChain for plumbing and drop to the SDK where hybrid/prefilter control matters.

---

## 5.6 Tuning Retrieval Quality

- **K and candidates**: retrieve more than you'll use (e.g. `num_candidates=20`, keep 5 after reranking/filtering). Measure with Ragas `context_recall`/`context_precision` (Ch. 13) rather than guessing.
- **`recall` vs `latency` optimization** is an index-time choice; benchmark both on your corpus if p99 matters.
- **Similarity metric ↔ model**: OpenAI and most modern embeddings are normalized → `dot_product`. Mismatching metric degrades ranking quietly.
- **Score thresholds**: `similarity_search_with_score` scores are index-relative, not calibrated probabilities. Set thresholds empirically per index (the semantic-cache in Ch. 6 depends on this).
- **Consistency**: Search indexes ingest asynchronously. Read-your-own-writes tests should use `SearchOptions(consistent_with=MutationState(...))` or poll `get_indexed_documents_count`.

---

## 5.7 Recap
One Search index now serves semantic, keyword, filtered, and multi-vector retrieval over the same documents your app reads and writes operationally. That index is the foundation the next chapters build on: RAG (6), memory recall (9), and tool retrieval (10).

Notebook: [`notebooks/02_vector_search_fundamentals.ipynb`](../notebooks/02_vector_search_fundamentals.ipynb).

Running into errors? See [Troubleshooting](troubleshooting.md).

Past this scale, or for pure-similarity workloads with no text-search needs, see [Chapter 14: Vector Index Architectures](14-vector-index-architectures.md) for the GSI-based Hyperscale and Composite Vector Index alternatives.

Next: [Chapter 6: Retrieval-Augmented Generation](06-rag.md).
