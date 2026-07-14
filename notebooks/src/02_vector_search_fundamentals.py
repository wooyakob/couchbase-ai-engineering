# %% [markdown]
# # 02 — Vector Search Fundamentals
#
# Companion to [Chapters 3–5](../docs/05-vector-search.md): chunk → embed → index → search.
#
# 1. Chunk a small corpus (data processing, Ch. 3)
# 2. Embed and upsert with lineage (vectorization, Ch. 4 — the DIY path)
# 3. Create a Search index with a vector field
# 4. Query: pure vector, hybrid (vector + keyword), and prefiltered
#
# **Prerequisites:** notebook 01 has been run (the `ai` bucket is provisioned);
# `OPENAI_API_KEY` in your `.env` (or Capella Model Service credentials — see the switch cell).

# %%
%pip install -q couchbase python-dotenv openai

# %%
import hashlib
import json
import os
import re
from datetime import timedelta

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(os.getenv("ENV_FILE", ".env"), usecwd=True))  # ENV_FILE lets .env.server / .env.capella run side by side

from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions, KnownConfigProfiles

opts = ClusterOptions(PasswordAuthenticator(os.getenv("CB_USERNAME"),
                                            os.getenv("CB_PASSWORD")))
conn = os.getenv("CB_CONN_STRING")
if conn.startswith("couchbases://"):
    opts.apply_profile(KnownConfigProfiles.WanDevelopment)
cluster = Cluster.connect(conn, opts)
cluster.wait_until_ready(timedelta(seconds=10))

bucket = cluster.bucket(os.getenv("CB_BUCKET"))
docs_scope = bucket.scope("docs")
chunks_coll = docs_scope.collection("chunks")

# %% [markdown]
# ## Embeddings: OpenAI by default, Capella Model Service by switch
#
# The embedding model determines `dims` in the index below. Keep them in sync via config.

# %%
from openai import OpenAI

USE_CAPELLA = bool(os.getenv("CAPELLA_AI_ENDPOINT"))

if USE_CAPELLA:
    import base64
    key = os.getenv("CAPELLA_AI_TOKEN") or base64.b64encode(
        f"{os.environ['CB_USERNAME']}:{os.environ['CB_PASSWORD']}".encode()).decode()
    ai_client = OpenAI(base_url=os.environ["CAPELLA_AI_ENDPOINT"], api_key=key)
    EMBEDDING_MODEL = os.getenv("CAPELLA_EMBEDDING_MODEL", "intfloat/e5-mistral-7b-instruct")
else:
    ai_client = OpenAI()
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def embed(texts: list[str]) -> list[list[float]]:
    resp = ai_client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def embed_query(text: str) -> list[float]:
    # e5-family models are trained with instruction prefixes (Ch. 4 §4.5)
    if EMBEDDING_MODEL.startswith("intfloat/e5"):
        text = f"query: {text}"
    return embed([text])[0]


# Measured, not hardcoded — dims vary by deployed model (Ch. 4/8), and the index
# definition below must match exactly.
EMBEDDING_DIM = len(embed(["probe"])[0])
print(f"embedding with {EMBEDDING_MODEL} ({EMBEDDING_DIM} dims)")

# %% [markdown]
# ## 1. A tiny corpus, chunked
#
# Structure-aware chunking (split at markdown headings, size-cap within sections) — Ch. 3 §3.3.

# %%
CORPUS = {
    "guide::vector-search": """# Vector Search in Couchbase
Couchbase Vector Search runs in the Search service. A Search index can contain vector
fields alongside text fields, so one query combines semantic similarity with keyword
matching and metadata filters.

# Index configuration
A vector field needs three settings: dims (must match your embedding model exactly),
similarity (dot_product for normalized embeddings, l2_norm for euclidean), and
vector_index_optimized_for (recall or latency).""",
    "guide::agent-memory": """# Agent memory on Couchbase
Short-term memory is a session document with turns appended via subdocument operations,
expiring automatically with a TTL. Long-term memory stores extracted facts with
embeddings, recalled by vector search with a user prefilter.

# Forgetting
Memory hygiene matters: decay by recency and access count, replace contradicted facts,
and implement GDPR deletion as a SQL++ DELETE by user_id.""",
    "guide::capella-credentials": """# Credential rotation in Capella
To rotate database credentials in Couchbase Capella, open Settings, choose Database
Access, create the new credential, deploy it to your applications, then revoke the old
credential. Rotate credentials at least quarterly.""",
}


def chunk_markdown(doc_id: str, text: str, max_chars: int = 600, overlap: int = 80):
    pieces = []
    for section in re.split(r"(?m)^(?=# )", text):
        section = section.strip()
        if not section:
            continue
        start = 0
        while start < len(section):
            pieces.append(section[start:start + max_chars])
            start += max_chars - overlap
    return pieces

# %% [markdown]
# ## 2. Embed and upsert with lineage
#
# The `content_hash` makes re-runs idempotent: unchanged chunks skip the embedding call.

# %%
from couchbase.exceptions import DocumentNotFoundException

for doc_id, text in CORPUS.items():
    pieces = chunk_markdown(doc_id, text)
    vectors = embed(pieces)  # batch the API call
    for i, (piece, vec) in enumerate(zip(pieces, vectors)):
        key = f"chunk::{doc_id}::{i:04d}"
        content_hash = hashlib.sha256(piece.encode()).hexdigest()[:16]
        try:
            if chunks_coll.get(key).content_as[dict]["lineage"]["content_hash"] == content_hash:
                continue
        except (DocumentNotFoundException, KeyError):
            pass
        chunks_coll.upsert(key, {
            "type": "chunk",
            "text": piece,
            "embedding": vec,
            "embedding_model": EMBEDDING_MODEL,
            "metadata": {"source": doc_id.split("::")[1], "product": "couchbase"},
            "lineage": {"doc_id": doc_id, "chunk_index": i, "content_hash": content_hash,
                        "pipeline_version": "notebook-02"},
        })
        print("upserted", key)

# %% [markdown]
# ## 3. Create the vector Search index
#
# The definition below is Chapter 5 §5.2: collection-scoped, `embedding` as a vector field,
# `text` stored for retrieval, `metadata` dynamic for filters. We patch `dims` from config
# so the OpenAI/Capella switch stays consistent.

# %%
from couchbase.management.search import SearchIndex

INDEX_NAME = "chunks-vector-index"

index_def = {
    "type": "fulltext-index",
    "name": INDEX_NAME,
    "sourceType": "gocbcore",
    "sourceName": os.getenv("CB_BUCKET"),
    "planParams": {"maxPartitionsPerPIndex": 1024, "indexPartitions": 1},
    "params": {
        "doc_config": {"mode": "scope.collection.type_field", "type_field": "type"},
        "mapping": {
            "default_analyzer": "standard",
            "default_mapping": {"dynamic": False, "enabled": False},
            "types": {
                "docs.chunks": {
                    "dynamic": False,
                    "enabled": True,
                    "properties": {
                        "embedding": {
                            "enabled": True, "dynamic": False,
                            "fields": [{
                                "name": "embedding", "type": "vector", "index": True,
                                "dims": EMBEDDING_DIM,
                                "similarity": "dot_product",
                                "vector_index_optimized_for": "recall",
                            }],
                        },
                        "text": {
                            "enabled": True, "dynamic": False,
                            "fields": [{"name": "text", "type": "text", "index": True,
                                        "store": True, "analyzer": "en"}],
                        },
                        # Explicit (not dynamic) so `fields=[...]` retrieval works below —
                        # the search API's `fields=["*"]`/named retrieval only returns
                        # *stored* fields, and dynamic mapping alone doesn't store them.
                        "metadata": {"enabled": True, "dynamic": False, "properties": {
                            "source": {"enabled": True, "dynamic": False,
                                      "fields": [{"name": "source", "type": "text",
                                                  "index": True, "store": True}]},
                            "product": {"enabled": True, "dynamic": False,
                                       "fields": [{"name": "product", "type": "text",
                                                   "index": True, "store": True}]},
                        }},
                    },
                }
            },
        },
        "store": {"indexType": "scorch", "segmentVersion": 16},
    },
    "sourceParams": {},
}

from couchbase.exceptions import SearchIndexNotFoundException

try:
    docs_scope.search_indexes().drop_index(INDEX_NAME)  # rerun-safe: replace any stale definition
except SearchIndexNotFoundException:
    pass
docs_scope.search_indexes().upsert_index(SearchIndex.from_json(index_def))
print("index upserted")

# %%
import time

from couchbase.exceptions import CouchbaseException

# wait for the index to ingest our chunks — right after upsert_index the Search
# service hasn't planned the index's partitions yet, so even get_indexed_documents_count
# can 500 for the first second or two; retry through that instead of failing on it.
n = 0
target = sum(len(chunk_markdown(d, t)) for d, t in CORPUS.items())
for _ in range(30):
    try:
        n = docs_scope.search_indexes().get_indexed_documents_count(INDEX_NAME)
    except CouchbaseException:
        n = 0
    if n >= target:
        break
    time.sleep(2)
print("indexed docs:", n)

# %% [markdown]
# ## 4a. Pure semantic search

# %%
import couchbase.search as search
from couchbase.options import SearchOptions
from couchbase.vector_search import VectorQuery, VectorSearch


def semantic_search(query: str, k: int = 3):
    req = search.SearchRequest.create(
        VectorSearch.from_vector_query(
            VectorQuery("embedding", embed_query(query), num_candidates=k)))
    result = docs_scope.search(INDEX_NAME, req,
                               SearchOptions(limit=k, fields=["text", "metadata.source"]))
    return [{"id": r.id, "score": round(r.score, 4), **(r.fields or {})}
            for r in result.rows()]


for hit in semantic_search("how do I change my database password?"):
    print(f"{hit['score']:>8}  {hit['id']}")
    print("          ", hit["text"][:90].replace("\n", " "), "…")

# %% [markdown]
# Note the top hit: nothing in the corpus says "password" — the credential-rotation chunk
# wins on *meaning*. That's the point of vector search.
#
# ## 4b. Hybrid: vector + keyword in one request
#
# Keyword search nails exact terms; vectors nail paraphrase. Merge both score streams:

# %%
def hybrid_search(query: str, k: int = 3, boost: float = 1.5):
    req = search.SearchRequest.create(
        search.MatchQuery(query, field="text")            # keyword side
    ).with_vector_search(
        VectorSearch.from_vector_query(
            VectorQuery("embedding", embed_query(query),
                        num_candidates=k * 2, boost=boost))  # semantic side
    )
    result = docs_scope.search(INDEX_NAME, req, SearchOptions(limit=k, fields=["text"]))
    return [{"id": r.id, "score": round(r.score, 4)} for r in result.rows()]


hybrid_search("TTL subdocument session")

# %% [markdown]
# ## 4c. Prefiltered search
#
# Restrict the ANN search itself — the right tool for tenancy/permissions, because
# filtering happens *before* K is spent (Ch. 5 §5.4):

# %%
only_memory_docs = search.MatchQuery("agent-memory", field="metadata.source")
req = search.SearchRequest.create(
    VectorSearch.from_vector_query(
        VectorQuery("embedding", embed_query("how should data expire?"),
                    num_candidates=3, prefilter=only_memory_docs)))
result = docs_scope.search(INDEX_NAME, req, SearchOptions(limit=3, fields=["text", "metadata.source"]))
for r in result.rows():
    # Dotted field paths come back flat, keyed exactly as requested — not nested
    print(round(r.score, 4), r.fields["metadata.source"], "—", r.fields["text"][:60], "…")

# %% [markdown]
# ## Sanity checks (Ch. 4 §4.3)
#
# One embedding model, one dimensionality, full coverage — or retrieval silently degrades.

# %%
from couchbase.options import QueryOptions

CB_BUCKET = os.getenv("CB_BUCKET")
cluster.query(f"CREATE PRIMARY INDEX IF NOT EXISTS ON `{CB_BUCKET}`.docs.chunks").execute()
for row in cluster.query(
    f"""SELECT c.embedding_model, ARRAY_LENGTH(c.embedding) AS dims, COUNT(*) AS n
        FROM `{CB_BUCKET}`.docs.chunks c GROUP BY c.embedding_model, ARRAY_LENGTH(c.embedding)"""
):
    print(row)

# %% [markdown]
# **Next:** [03 — A complete RAG pipeline](03_rag_pipeline.ipynb)
