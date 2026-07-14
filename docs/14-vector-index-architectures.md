# Chapter 14 — Vector Index Architectures: Hyperscale, Composite & Tuning

> *Chapter 5 built one vector index — a Search-service index that puts vectors next to text
> fields for hybrid retrieval. That's the right default for RAG. But Couchbase Server 8.0
> added two more ways to index a vector, built on the Global Secondary Index (GSI) engine
> instead of the Search service, purpose-built for pure-vector and filtered-vector workloads
> at much larger scale. This chapter is a deep dive into all three: when to pick which, and
> how the knobs — similarity metric, IVF centroids, scalar/product quantization — actually
> change recall, latency, and memory.*

**Requires Couchbase Server / Capella 8.0+** (GA October 2025). `CREATE VECTOR INDEX` and
`APPROX_VECTOR_DISTANCE` are Query-service (SQL++) features — check your cluster version
before running this chapter's notebook.

## 14.1 Three ways to index a vector in Couchbase

| | **Search Vector Index** (Ch. 5) | **Hyperscale Vector Index** | **Composite Vector Index** |
|---|---|---|---|
| Engine | Search service (FTS) | Query service (GSI) | Query service (GSI) |
| Best for | Hybrid: vector + full-text + geospatial | Pure vector similarity, huge corpora | Vector similarity filtered by scalar fields |
| Scale | ~100M documents | billions of vectors | billions of vectors |
| Filtering | Prefilter via FTS query (text/numeric/date) | None (evaluates the whole index) | Scalar leading keys prefilter *before* the vector scan |
| Memory footprint | Higher (inverted index + vectors) | Lowest (disk-resident, quantized) | Moderate |
| Query API | `SearchRequest` / `VectorQuery` (SDK) or LangChain | SQL++ `ORDER BY APPROX_VECTOR_DISTANCE(...)` or LangChain | SQL++ `WHERE ... ORDER BY APPROX_VECTOR_DISTANCE(...)` or LangChain |
| DDL | `SearchIndex` JSON | `CREATE VECTOR INDEX ... (field VECTOR)` | `CREATE INDEX ... (field VECTOR, scalar1, scalar2)` |
| `langchain-couchbase` class | `CouchbaseSearchVectorStore` (Server/Capella 7.6+) | `CouchbaseQueryVectorStore` (Server/Capella 8.0+) | `CouchbaseQueryVectorStore` (Server/Capella 8.0+) |

**Default guidance from Couchbase's own docs: start with Hyperscale.** Reach for Composite
only when a scalar predicate eliminates a large fraction of the dataset (tenant ID, region,
document type) — pre-filtering by GSI key is cheaper than evaluating the vector index and
throwing results away. Reach for the Search Vector Index (Ch. 5) only when you need hybrid
text+vector scoring in one query, geospatial, or you're already on `langchain-couchbase`'s
`CouchbaseSearchVectorStore` and RAG-scale (not billions of rows) is your ceiling.

One `langchain-couchbase` wrinkle worth calling out: `CouchbaseQueryVectorStore` is the single
LangChain class for *both* GSI-based index types in this table — which one you get is a
`create_index(index_type=...)` argument (`IndexType.HYPERSCALE` or `IndexType.COMPOSITE`), not
a different class. That's different from Ch. 5, where hybrid FTS search has its own class,
`CouchbaseSearchVectorStore`. See §14.8.

These are not mutually exclusive — a real system can have a Search index for the RAG
retriever (Ch. 5–6) *and* a Hyperscale index for a separate large-scale recommendation
or "more like this" feature over the same or a different collection.

## 14.2 Similarity metrics

All three index types support the same four distance functions; pick to match your
embedding model, same rule as Chapter 5:

- **`COSINE`** — normalizes both vectors first; the safe default for text embeddings whose
  magnitude isn't meaningful.
- **`DOT`** — raw dot product; equivalent to cosine when the model already emits unit-length
  vectors (OpenAI's `text-embedding-3-*`, most Sentence-Transformers), and cheaper to compute.
- **`L2` / `EUCLIDEAN`** — straight-line distance; matches models trained explicitly for
  Euclidean nearness (some image/audio embeddings), or spatial data that's a genuine vector
  of coordinates.
- **`L2_SQUARED` / `EUCLIDEAN_SQUARED`** — the same ranking as `L2` without the square root;
  faster, and the GSI default (`similarity` defaults to `L2_SQUARED` if omitted — always set
  it explicitly).

Mismatching metric and model is a silent quality bug: rankings still come back, they're just
wrong, and nothing errors. Verify empirically (Ch. 5 §5.6, Ch. 13) rather than assuming.

## 14.3 Index algorithms: Flat vs. IVF, and quantization

Underneath all three index types is the same lineage of vector-search algorithms:

- **Flat** — brute-force, compare the query against every vector. Exact, no training, no
  memory savings. Couchbase uses this automatically for tiny collections (fewer than ~1,000
  vectors) where an approximate index buys nothing.
- **IVF (Inverted File)** — partitions the vector space into `nlist` clusters ("Voronoi
  cells"), each with a centroid. A query only compares against the `nprobes` cells nearest
  the query vector, not the whole dataset. This is *how* vector search scales past millions
  of rows: fewer full-vector comparisons per query, at the cost of missing neighbors that
  landed in a cell you didn't probe. Couchbase uses IVF once a collection passes the small-set
  threshold, training centroids from a sample of your data (`train_list` controls the sample
  size).
- **Quantization** — compresses each vector so it takes less memory/disk and compares faster,
  at some cost to precision:
  - **Scalar Quantization (SQ)** — quantizes each dimension independently into a fixed number
    of bits: `SQ8` (8-bit, 256 levels — the default, and the right starting point for most
    workloads), `SQ6` (64 levels), `SQ4` (16 levels, for the largest collections where memory
    dominates and a few points of recall are an acceptable trade). Cheap to train.
  - **Product Quantization (PQ)** — splits each vector into `n` subspaces and quantizes each
    subspace to a shared codebook (`PQ{n}x{bits}`, e.g. `PQ32x8` = 32 subspaces × 256
    centroids each). Compresses further than SQ, especially for high-dimensional vectors, but
    costs more to train and more to query (lower QPS, higher latency, lower recall than SQ at
    a similar memory budget). Reach for PQ only after SQ8/SQ4 doesn't fit your memory budget.

All of this is configured through one string, the `description` option, shared by Hyperscale
and Composite indexes: `"IVF[nlist],QUANTIZATION"` — e.g. `"IVF,SQ8"` (auto centroid count,
default quantization — the default if you omit `description` entirely), `"IVF1024,SQ4"`
(1024 centroids, 4-bit scalar quantization), `"IVF,PQ32x8"` (auto centroids, product
quantization). Fewer centroids train faster and use less memory but leave more vectors per
cell (slower query when a cell is probed); more centroids do the reverse. Couchbase's
default centroid count is `vector_count / 1000`.

**Training decays.** IVF centroids and PQ/SQ codebooks are learned from a sample of the data
at build time. As a collection grows or its distribution shifts, stale centroids no longer
match reality — recall degrades quietly. Rebuild the index (or trigger a retrain, where
supported) periodically on a fast-growing or drifting collection; this is the vector-index
equivalent of the reindex maintenance you'd already plan for a GSI.

## 14.4 Creating a Hyperscale Vector Index

Hyperscale indexes one vector column, nothing else — the leanest, highest-scale option:

```sql
CREATE VECTOR INDEX `chunks_hyperscale_idx`
    ON `ai`.`docs`.`chunks`(`embedding` VECTOR)
    WITH {
        "dimension": 1536,
        "similarity": "COSINE",
        "description": "IVF,SQ8"
    };
```

Key `WITH` options (all optional except `dimension`):

| Option | Default | Purpose |
|---|---|---|
| `dimension` | *required* | Must equal your embedding model's output length, exactly as in Ch. 5 §5.1 |
| `similarity` | `L2_SQUARED` | See §14.2 — set explicitly |
| `description` | `IVF,SQ8` | Centroids + quantization, see §14.3 |
| `train_list` | 10% of vectors, or 10× centroid count | Sample size for centroid/codebook training (max 1,000,000) |
| `scan_nprobes` | 1 | Default cells probed per query — override per-query too (§14.6) |
| `persist_full_vector` | `true` | Keep the un-quantized vector on disk for reranking; disable only if disk is the binding constraint |
| `num_replica` | 1 | Same replication semantics as any GSI |
| `defer_build` | `false` | Build immediately vs. batch with other index builds |

The index builds asynchronously — check `SYSTEM:INDEXES` (or poll `get_indexed_documents_count`-style
row counts via SQL++) before querying it, the GSI analog of the Search-index ingest wait in
Ch. 5 §5.2.

## 14.5 Creating a Composite Vector Index

Composite indexes are a regular `CREATE INDEX` where one key is `VECTOR` and the rest are
ordinary scalar/array/object keys. The Query service evaluates the scalar predicate *first*,
then runs the (approximate) vector comparison only over the surviving rows:

```sql
CREATE INDEX `chunks_composite_idx`
    ON `ai`.`docs`.`chunks`(`embedding` VECTOR, `metadata`.`product`)
    WITH {
        "dimension": 1536,
        "similarity": "COSINE",
        "description": "IVF,SQ8"
    };
```

The vector key can lead (as above) or follow scalar leading keys — put the vector last when
the scalar predicate is highly selective (it becomes the primary GSI lookup path and the
vector comparison only runs over the tiny surviving set); put it first when the scalar filter
is closer to advisory. This is the same prefilter-vs-postfilter reasoning as Ch. 5 §5.4's FTS
prefilters, just enforced by index key order instead of a query-time `prefilter`.

## 14.6 Querying: `APPROX_VECTOR_DISTANCE` and `VECTOR_DISTANCE`

Both index types are queried the same way — order by the distance function, limit to K:

```sql
SELECT META().id, text
FROM `ai`.`docs`.`chunks`
WHERE metadata.product = "couchbase"          -- composite index only
ORDER BY APPROX_VECTOR_DISTANCE(embedding, $query_vector, "COSINE")
LIMIT 5;
```

`APPROX_VECTOR_DISTANCE` takes three more optional positional arguments that tune the
recall/latency trade at query time, without touching the index:

```sql
APPROX_VECTOR_DISTANCE(field, query_vector, metric, nprobes, rerank, topNScan)
```

- **`nprobes`** (default 1) — how many IVF cells this query probes. Raising it improves
  recall *non-linearly* (diminishing returns) while decreasing QPS and raising latency
  *linearly* — it's the single most direct recall/latency dial you have.
- **`rerank`** (boolean) — when probing multiple cells, re-score the candidates against their
  full (un-quantized) vectors before truncating to K, correcting for quantization error.
  Most valuable paired with aggressive quantization (`SQ4`, `PQ`) where the compressed
  distance is a rougher approximation.
- **`topNScan`** — how many candidates to carry into that rerank step.

`VECTOR_DISTANCE(field, query_vector, metric)` (no `APPROX_`) is the brute-force, exact
version — no index, no quantization error, full O(n) scan. It's too slow for production at
scale, but it's the ground truth you measure `APPROX_VECTOR_DISTANCE` recall against
(§14.7) — the GSI equivalent of comparing an ANN index to a linear scan.

## 14.7 Measuring and tuning recall

"Recall" here means: of the true top-K nearest neighbors (from `VECTOR_DISTANCE`), what
fraction did `APPROX_VECTOR_DISTANCE` actually return? Compute it directly rather than
guessing from the description string:

1. Pick a sample of realistic query vectors.
2. For each, run the exact `VECTOR_DISTANCE` top-K (brute force) and the approximate
   `APPROX_VECTOR_DISTANCE` top-K (through the index).
3. Recall@K = `|approx_ids ∩ exact_ids| / K`, averaged over the sample.

If recall is short of your target, in order of cost:
1. Raise `nprobes` first — cheapest lever, biggest non-linear recall gain.
2. Enable `rerank` (with a sufficient `topNScan`) if quantization (especially `SQ4`/`PQ`)
   is the suspected source of error rather than IVF cell coverage.
3. Raise `train_list` if centroids seem to be modeling the data poorly (common right after
   a large bulk load, before enough representative vectors existed to train against).
4. Only then consider a coarser index parameter change (lower centroid count, lighter
   quantization) — these require a rebuild, unlike 1–3 which are query-time or a config
   tweak.

This mirrors Ch. 5 §5.6 and Ch. 13's evaluation loop: don't tune blind, measure the metric
that matters (recall here; `context_recall`/`context_precision` there) and change one lever
at a time.

## 14.8 Through LangChain instead

`langchain-couchbase` wraps both GSI-based index types behind one class,
`CouchbaseQueryVectorStore` — the counterpart to Ch. 5 §5.5's `CouchbaseSearchVectorStore`,
but requiring Couchbase Server / Capella **8.0+** (vs. **7.6+** for the Search-based class,
since Hyperscale/Composite are Query-service features that didn't exist before 8.0):

```python
from langchain_couchbase.vectorstores import CouchbaseQueryVectorStore, IndexType, DistanceStrategy
from langchain_openai import OpenAIEmbeddings

vector_store = CouchbaseQueryVectorStore(
    cluster=cluster,
    bucket_name="ai", scope_name="docs", collection_name="chunks",
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    distance_metric=DistanceStrategy.COSINE,
)
vector_store.add_texts(["...chunk text..."])

# Index creation happens after documents are added, and picks Hyperscale vs. Composite
# by argument, not by class:
vector_store.create_index(index_type=IndexType.HYPERSCALE, index_description="IVF,SQ8")
# vector_store.create_index(index_type=IndexType.COMPOSITE, index_description="IVF,SQ8")

hits = vector_store.similarity_search_with_score("how do I rotate credentials?", k=5)
# Composite-only: prefilter via a where_str, the LangChain equivalent of §14.5's scalar leading key
filtered = vector_store.similarity_search("...", k=5, where_str="region = 'emea'")
```

Same tradeoff as choosing between the raw DDL in §14.4/§14.5: reach for
`CouchbaseQueryVectorStore` when you want the Hyperscale/Composite scale and quantization
story with LangChain's `retriever`/`add_documents` conveniences instead of hand-written SQL++;
reach for `CouchbaseSearchVectorStore` (Ch. 5 §5.5) for hybrid text+vector retrieval, or drop
to raw SQL++ (§14.4–§14.6) when you need index/query options this wrapper doesn't expose yet.

## 14.9 Recap

Chapter 5's Search Vector Index is still the right default for RAG retrieval — one query,
text + vector + filters, comfortably up to ~100M documents. Past that scale, or for
pure-similarity workloads (recommendations, "more like this," dedup) with no text-search
need, Hyperscale and Composite Vector Indexes hand you the same similarity math on the GSI
engine instead: lower memory via quantization, prefiltering via scalar keys, and query-time
recall/latency tuning (`nprobes`, `rerank`) independent of the index build. Reach for it via
raw SQL++ (§14.4–§14.6) or `langchain-couchbase`'s `CouchbaseQueryVectorStore` (§14.8),
depending on whether you need index/query options the wrapper doesn't expose.

Notebook: [`notebooks/11_vector_index_architectures.ipynb`](../notebooks/11_vector_index_architectures.ipynb).
Running into errors? See [Troubleshooting](troubleshooting.md).

Back to: [Chapter 5 — Vector Search](05-vector-search.md).
