# %% [markdown]
# # 11: Vector Index Architectures: Hyperscale, Composite & Tuning
#
# Companion to [Chapter 14](../docs/14-vector-index-architectures.md). Chapter 5 / notebook 02
# built a Search-service vector index for hybrid RAG retrieval. This notebook builds the two
# GSI-based alternatives (**Hyperscale** and **Composite** Vector Indexes) and measures how
# their tuning knobs (similarity metric, IVF centroids, scalar/product quantization,
# `nprobes`) actually move recall and latency, instead of taking the defaults on faith.
#
# 1. Seed a collection with vectors that have real nearest-neighbor structure
# 2. Create a Hyperscale Vector Index; query with `APPROX_VECTOR_DISTANCE`
# 3. Measure recall against brute-force `VECTOR_DISTANCE`; tune `nprobes`
# 4. Create a Composite Vector Index; prefilter by a scalar field
# 5. Compare quantization: `SQ8` vs. `SQ4` recall at the same `nprobes`
# 6. Same two index types, through LangChain's `CouchbaseQueryVectorStore` wrapper
#
# **Prerequisites:** notebook 01 (the `ai` bucket is provisioned); **Couchbase Server /
# Capella 8.0+**: `CREATE VECTOR INDEX` and `APPROX_VECTOR_DISTANCE` don't exist before that
# (GA October 2025), and `CouchbaseQueryVectorStore` (§6) needs it too, unlike the 7.6+
# `CouchbaseSearchVectorStore` from Ch. 5. This notebook uses **synthetic vectors**, not real
# embeddings, enough volume (5,000 vectors) for IVF/quantization to actually activate,
# without an embedding-API bill. The DDL and query syntax are identical for your real
# embeddings (Ch. 4–5). §6 needs `OPENAI_API_KEY` for a small number of real embedding calls.

# %%
%pip install -q couchbase python-dotenv

# %%
import math
import os
import random
import time
from datetime import timedelta

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(os.getenv("ENV_FILE", ".env"), usecwd=True))

from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.exceptions import (AuthenticationException,
                                  CollectionAlreadyExistsException,
                                  CouchbaseException)
from couchbase.options import ClusterOptions, KnownConfigProfiles

conn = os.getenv("CB_CONN_STRING")
CB_USERNAME, CB_PASSWORD, CB_BUCKET = (os.getenv("CB_USERNAME"), os.getenv("CB_PASSWORD"),
                                       os.getenv("CB_BUCKET"))
_missing = [n for n, v in [("CB_CONN_STRING", conn), ("CB_USERNAME", CB_USERNAME),
                          ("CB_PASSWORD", CB_PASSWORD), ("CB_BUCKET", CB_BUCKET)] if not v]
if _missing:
    raise RuntimeError(
        f"Missing required env var(s): {', '.join(_missing)}. Check ENV_FILE="
        f"{os.getenv('ENV_FILE', '.env')!r} is set and that file has these. "
        "See docs/troubleshooting.md."
    )

opts = ClusterOptions(PasswordAuthenticator(CB_USERNAME, CB_PASSWORD))
if conn.startswith("couchbases://"):
    opts.apply_profile(KnownConfigProfiles.WanDevelopment)
try:
    cluster = Cluster.connect(conn, opts)
    cluster.wait_until_ready(timedelta(seconds=10))
except AuthenticationException as e:
    raise RuntimeError(
        f"Couchbase rejected CB_USERNAME={CB_USERNAME!r} for {conn!r}, check "
        "CB_USERNAME/CB_PASSWORD. See docs/troubleshooting.md."
    ) from e
except CouchbaseException as e:
    raise RuntimeError(
        f"Couldn't connect to Couchbase at {conn!r}: {e}. See docs/troubleshooting.md."
    ) from e
bucket = cluster.bucket(CB_BUCKET)
docs_scope = bucket.scope("docs")

try:
    bucket.collections().create_collection("docs", "vec_arch")
except CollectionAlreadyExistsException:
    pass
vec_coll = docs_scope.collection("vec_arch")
cluster.query(f"CREATE PRIMARY INDEX IF NOT EXISTS ON `{CB_BUCKET}`.docs.vec_arch").execute()

# %% [markdown]
# ## 1. Synthetic vectors with real nearest-neighbor structure
#
# Uniform-random high-dimensional vectors are all roughly equidistant: useless for measuring
# an ANN index, since there's no real "nearest neighbor" to recover. Instead we generate a
# handful of random unit-vector **topics**, then scatter vectors as noisy copies of a topic,
# the same structure real embeddings have (documents cluster by meaning). Each vector also
# gets a `region` tag, the scalar field the Composite Index will prefilter on.

# %%
DIM = 128
N_TOPICS = 20
N_VECTORS = 5000
REGIONS = ["amer", "emea", "apac", "latam"]


def random_unit_vector(dim: int) -> list[float]:
    v = [random.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def noisy_copy(base: list[float], noise: float = 0.15) -> list[float]:
    v = [b + random.gauss(0, noise) for b in base]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


random.seed(7)  # rerun-stable corpus
topics = [random_unit_vector(DIM) for _ in range(N_TOPICS)]

existing = cluster.query(
    f"SELECT RAW COUNT(*) FROM `{CB_BUCKET}`.docs.vec_arch WHERE type = 'vec_arch_synth'"
).execute()[0]

if existing < N_VECTORS:
    for i in range(N_VECTORS):
        topic_id = i % N_TOPICS
        vec_coll.upsert(f"vec_arch::{i:05d}", {
            "type": "vec_arch_synth",
            "embedding": noisy_copy(topics[topic_id]),
            "topic_id": topic_id,
            "region": REGIONS[i % len(REGIONS)],
        })
    print(f"upserted {N_VECTORS} synthetic vectors ({N_TOPICS} topics, dim={DIM})")
else:
    print(f"reusing {existing} previously-upserted vectors")

# %% [markdown]
# ## 2. Create a Hyperscale Vector Index
#
# One vector column, nothing else (Ch. 14 §14.4). `description: "IVF,SQ8"` is the default:
# auto centroid count (`vector_count / 1000`), 8-bit scalar quantization.

# %%
HYPERSCALE_IDX = "vec_arch_hyperscale_idx"


def wait_online(index_name: str, timeout_s: int = 60):
    for _ in range(timeout_s):
        rows = cluster.query("SELECT raw state FROM system:indexes WHERE name = $name",
                             named_parameters={"name": index_name}).execute()
        if rows and rows[0] == "online":
            return
        time.sleep(1)
    raise TimeoutError(
        f"{index_name} did not come online within {timeout_s}s, check "
        f"`SELECT * FROM system:indexes WHERE name = \"{index_name}\"` for its actual "
        "state (building/pending/offline). A large train_list or centroid count "
        "(Ch. 14 §14.3) can extend build time; re-run this cell with a longer "
        "timeout_s if that's the case. See docs/troubleshooting.md."
    )


def create_vector_index(ddl: str, index_name: str):
    """Runs a CREATE VECTOR INDEX / CREATE INDEX(...VECTOR) statement, tolerating a
    rerun (index already exists) and translating the most common failure (a
    cluster older than Couchbase Server / Capella 8.0, where this syntax doesn't
    exist at all) into an actionable message instead of a raw parse error."""
    try:
        cluster.query(ddl).execute()
    except Exception as e:
        msg = str(e).lower()
        if "already exists" in msg:
            return
        if "syntax error" in msg or "parse" in msg:
            raise RuntimeError(
                f"Creating {index_name!r} failed with what looks like a syntax "
                f"error: {e}. Most likely your cluster predates Couchbase Server / "
                "Capella 8.0. CREATE VECTOR INDEX and APPROX_VECTOR_DISTANCE don't "
                "exist before that (Ch. 14). Check your server version, or use "
                "Chapter 5's Search Vector Index instead on older clusters. "
                "See docs/troubleshooting.md."
            ) from e
        raise RuntimeError(
            f"Creating {index_name!r} failed: {e}. See docs/troubleshooting.md."
        ) from e


create_vector_index(f"""
    CREATE VECTOR INDEX `{HYPERSCALE_IDX}`
    ON `{CB_BUCKET}`.`docs`.`vec_arch`(`embedding` VECTOR)
    WITH {{ "dimension": {DIM}, "similarity": "COSINE", "description": "IVF,SQ8" }}
""", HYPERSCALE_IDX)

wait_online(HYPERSCALE_IDX)
print(f"{HYPERSCALE_IDX} online")

# %% [markdown]
# ## 3. Query it, and measure recall against brute force
#
# `APPROX_VECTOR_DISTANCE` (indexed, fast, approximate) vs. `VECTOR_DISTANCE` (brute-force,
# exact): the ground truth to measure recall against (Ch. 14 §14.7). We probe a query vector
# built as a noisy copy of a known topic, so we know its true nearest neighbors share that
# `topic_id`.

# %%
def recall_at_k(query_vec, k: int, nprobes: int) -> float:
    exact = cluster.query(f"""
        SELECT RAW META(v).id
        FROM `{CB_BUCKET}`.docs.vec_arch v
        WHERE v.type = 'vec_arch_synth'
        ORDER BY VECTOR_DISTANCE(v.embedding, $qv, "COSINE")
        LIMIT {k}
    """, named_parameters={"qv": query_vec}).execute()

    approx = cluster.query(f"""
        SELECT RAW META(v).id
        FROM `{CB_BUCKET}`.docs.vec_arch v
        WHERE v.type = 'vec_arch_synth'
        ORDER BY APPROX_VECTOR_DISTANCE(v.embedding, $qv, "COSINE", {nprobes})
        LIMIT {k}
    """, named_parameters={"qv": query_vec}).execute()

    exact_ids, approx_ids = set(exact), set(approx)
    return len(exact_ids & approx_ids) / k


K = 10
query_vec = noisy_copy(topics[3], noise=0.1)  # a fresh point near topic 3

for nprobes in (1, 4, 16):
    t0 = time.perf_counter()
    r = recall_at_k(query_vec, K, nprobes)
    dt = time.perf_counter() - t0
    print(f"nprobes={nprobes:>2}  recall@{K}={r:.2f}  ({dt * 1000:.0f} ms)")

# %% [markdown]
# Recall climbs as `nprobes` rises (non-linearly, per Ch. 14 §14.7) while each query does
# more work. On a real cluster, watch latency rise alongside it; on a small single-node
# demo corpus the gap is subtle, but the direction is the lesson: **`nprobes` is the
# recall/latency dial, tunable per query, no rebuild required.**

# %% [markdown]
# ## 4. Composite Vector Index: prefilter by `region`
#
# Same vector column, plus a scalar leading key. The Query service filters `region` *before*
# running the (approximate) vector comparison: the GSI analog of Ch. 5's FTS prefilter.

# %%
COMPOSITE_IDX = "vec_arch_composite_idx"

create_vector_index(f"""
    CREATE INDEX `{COMPOSITE_IDX}`
    ON `{CB_BUCKET}`.`docs`.`vec_arch`(`embedding` VECTOR, `region`)
    WITH {{ "dimension": {DIM}, "similarity": "COSINE", "description": "IVF,SQ8" }}
""", COMPOSITE_IDX)

wait_online(COMPOSITE_IDX)

filtered = cluster.query(f"""
    SELECT v.region, v.topic_id
    FROM `{CB_BUCKET}`.docs.vec_arch v
    WHERE v.type = 'vec_arch_synth' AND v.region = 'emea'
    ORDER BY APPROX_VECTOR_DISTANCE(v.embedding, $qv, "COSINE")
    LIMIT 5
""", named_parameters={"qv": query_vec}).execute()

print("nearest neighbors restricted to region=emea:")
for row in filtered:
    print(" ", row)

# %% [markdown]
# Every hit above satisfies `region = "emea"`: the scalar predicate scoped the candidate
# set before the vector scan ran, not after. This is the tool for tenancy, access control, or
# any workload where a filter is known to eliminate most of the corpus (Ch. 14 §14.1): cheaper
# than running the full vector search and discarding results that fail the filter.

# %% [markdown]
# ## 5. Quantization tradeoff: `SQ8` vs. `SQ4`
#
# Build a second Hyperscale index with lighter (4-bit) scalar quantization and compare recall
# at the same `nprobes`: the memory-vs-accuracy lever from Ch. 14 §14.3, measured rather than
# assumed.

# %%
SQ4_IDX = "vec_arch_hyperscale_sq4_idx"

create_vector_index(f"""
    CREATE VECTOR INDEX `{SQ4_IDX}`
    ON `{CB_BUCKET}`.`docs`.`vec_arch`(`embedding` VECTOR)
    WITH {{ "dimension": {DIM}, "similarity": "COSINE", "description": "IVF,SQ4" }}
""", SQ4_IDX)

wait_online(SQ4_IDX)


def recall_with_index(index_name: str, query_vec, k: int, nprobes: int) -> float:
    # USE INDEX pins the query to a specific index so SQ8 vs. SQ4 are compared head-to-head
    exact = cluster.query(f"""
        SELECT RAW META(v).id FROM `{CB_BUCKET}`.docs.vec_arch v
        WHERE v.type = 'vec_arch_synth'
        ORDER BY VECTOR_DISTANCE(v.embedding, $qv, "COSINE") LIMIT {k}
    """, named_parameters={"qv": query_vec}).execute()

    approx = cluster.query(f"""
        SELECT RAW META(v).id FROM `{CB_BUCKET}`.docs.vec_arch v USE INDEX (`{index_name}`)
        WHERE v.type = 'vec_arch_synth'
        ORDER BY APPROX_VECTOR_DISTANCE(v.embedding, $qv, "COSINE", {nprobes}) LIMIT {k}
    """, named_parameters={"qv": query_vec}).execute()

    return len(set(exact) & set(approx)) / k


for label, idx in [("SQ8", HYPERSCALE_IDX), ("SQ4", SQ4_IDX)]:
    r = recall_with_index(idx, query_vec, K, nprobes=4)
    print(f"{label}: recall@{K} (nprobes=4) = {r:.2f}")

# %% [markdown]
# `SQ4` compresses each vector to a quarter of `SQ8`'s footprint: the right trade once a
# collection is large enough that memory, not accuracy, is the binding constraint (Ch. 14
# §14.3). Measure the recall drop on *your* data before committing to it; it depends on how
# separable your embeddings are, same caveat as every other index default in this book.

# %% [markdown]
# ## 6. The LangChain wrapper: `CouchbaseQueryVectorStore`
#
# Everything above used raw SQL++ DDL/queries to show what's actually happening. Ch. 14 §14.8 /
# Ch. 5 §5.5 wrap those same two GSI-based index types (Hyperscale and Composite) behind one
# LangChain class, the counterpart to `CouchbaseSearchVectorStore`'s Search-service index. Which
# index type you get is a `create_index()` argument, not a different class.

# %%
%pip install -q langchain-couchbase langchain-openai

# %%
from langchain_couchbase.vectorstores import (CouchbaseQueryVectorStore,
                                              DistanceStrategy, IndexType)
from langchain_openai import OpenAIEmbeddings

# CouchbaseQueryVectorStore checks its collection exists at construction time (unlike
# the raw-DDL section above, it won't create one for you) — provision it first, same
# as `vec_arch` at the top of this notebook.
try:
    bucket.collections().create_collection("docs", "vec_arch_lc")
except CollectionAlreadyExistsException:
    pass

lc_vector_store = CouchbaseQueryVectorStore(
    cluster=cluster,
    bucket_name=CB_BUCKET, scope_name="docs", collection_name="vec_arch_lc",
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    distance_metric=DistanceStrategy.COSINE,
)
lc_vector_store.add_texts(
    ["couchbase hyperscale vector index", "couchbase composite vector index",
     "couchbase search vector index for hybrid retrieval"],
    metadatas=[{"region": r} for r in ("amer", "emea", "apac")],
)

# create_index() must run after documents are added (it needs `dimension` from the embedded
# vectors), same asynchronous-build behavior as the raw DDL above.
lc_vector_store.create_index(index_type=IndexType.HYPERSCALE, index_description="IVF,SQ8")

hits = lc_vector_store.similarity_search_with_score("hyperscale vector search", k=2)
for doc, score in hits:
    print(f"[{score:.3f}] {doc.page_content!r} {doc.metadata}")

# %% [markdown]
# `create_index(index_type=IndexType.COMPOSITE, ...)` builds the scalar-prefiltered variant
# instead; `similarity_search(..., where_str="region = 'emea'")` is the LangChain equivalent of
# §4's `WHERE v.region = 'emea'` prefilter. Same guidance as §14.8: reach for this wrapper for
# `add_documents`/`retriever` convenience, drop to the raw SQL++ above when you need index or
# query options it doesn't expose yet.

# %% [markdown]
# ## Recap
#
# Same similarity math as Chapter 5, different engine: Hyperscale and Composite Vector
# Indexes trade the Search service's hybrid text+vector scoring for GSI-scale (billions of
# rows), lower memory via quantization, and scalar prefiltering baked into the index key
# order instead of a query-time filter. `nprobes` and `rerank` tune recall/latency without a
# rebuild; `description` (centroids + quantization) is the lever that does need one: measure
# recall (§3, §5 above) before and after any change, the same discipline as tuning the RAG
# pipeline in Ch. 13. Reach for it via raw SQL++ (§1–§5) or `langchain-couchbase`'s
# `CouchbaseQueryVectorStore` (§6), the same choice as Ch. 5's SDK-vs-LangChain tradeoff.
#
# Back to: [Chapter 5: Vector Search](../docs/05-vector-search.md). Next: [Chapter 6: Retrieval-Augmented Generation](../docs/06-rag.md).
