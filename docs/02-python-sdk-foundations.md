# Chapter 2: Python SDK Foundations

> *Everything in this book, vectors, memory, catalogs, checkpoints, eval logs, is ultimately documents in collections, read and written through the Couchbase Python SDK. This chapter builds the small set of SDK skills the rest of the book stands on.*

---

## 2.1 Install and Connect

```bash
pip install couchbase   # 4.x; this book assumes >= 4.3
```

The canonical connection (imports matter, options come from `couchbase.options`):

```python
from datetime import timedelta

from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions

auth = PasswordAuthenticator("Administrator", "password")
cluster = Cluster.connect("couchbase://localhost", ClusterOptions(auth))
cluster.wait_until_ready(timedelta(seconds=10))
```

For **Capella**, switch the scheme to `couchbases://` and apply the WAN profile so timeouts suit a cluster that isn't on your LAN:

```python
from couchbase.options import ClusterOptions, KnownConfigProfiles

opts = ClusterOptions(PasswordAuthenticator(username, password))
opts.apply_profile(KnownConfigProfiles.WanDevelopment)   # relaxed timeouts for WAN
cluster = Cluster.connect(f"couchbases://{endpoint}", opts)
cluster.wait_until_ready(timedelta(seconds=10))
```

Every example in this repo funnels through one helper so notebooks and apps share it:

```python
# common pattern used across notebooks/ and apps/
def connect_cluster() -> Cluster:
    conn = os.environ["CB_CONN_STRING"]           # couchbase://... or couchbases://...
    opts = ClusterOptions(PasswordAuthenticator(os.environ["CB_USERNAME"],
                                                os.environ["CB_PASSWORD"]))
    if conn.startswith("couchbases://"):
        opts.apply_profile(KnownConfigProfiles.WanDevelopment)
    cluster = Cluster.connect(conn, opts)
    cluster.wait_until_ready(timedelta(seconds=10))
    return cluster
```

---

## 2.2 Buckets, Scopes, and Collections: The AI Data Model

Couchbase organizes documents as **bucket → scope → collection**, and this hierarchy is your friend in AI systems: it gives every kind of AI state a named home with its own indexes, TTL defaults and RBAC.

This book uses one bucket, `ai`:

| Keyspace | Holds |
|---|---|
| `ai.docs.chunks` | RAG chunks + embeddings |
| `ai.agent.sessions` | short-term memory (TTL'd) |
| `ai.agent.memories` | long-term memories + embeddings |
| `ai.agent.checkpoints`, `ai.agent.checkpoint_writes` | LangGraph checkpoints |
| `ai.evals.runs`, `ai.evals.samples` | Ragas evaluation results |

Provisioning is idempotent via the management API:

```python
from couchbase.exceptions import (CollectionAlreadyExistsException,
                                  ScopeAlreadyExistsException)

from couchbase.management.collections import CreateCollectionSettings

def ensure_collection(bucket, scope_name: str, collection_name: str,
                      max_expiry: timedelta | None = None):
    cm = bucket.collections()
    try:
        cm.create_scope(scope_name)
    except ScopeAlreadyExistsException:
        pass
    try:
        settings = CreateCollectionSettings(max_expiry=max_expiry) if max_expiry else None
        cm.create_collection(scope_name, collection_name, settings)
    except CollectionAlreadyExistsException:
        pass

bucket = cluster.bucket("ai")
ensure_collection(bucket, "docs", "chunks")
ensure_collection(bucket, "agent", "sessions", max_expiry=timedelta(days=7))
```

Note `max_expiry` on `agent.sessions`: a **collection-level TTL** means every session document expires automatically. This is short-term memory that forgets by design (Chapter 9).

---

## 2.3 Key Value Operations
KV is the fastest path to a document, sub-millisecond reads by key, no query engine involved. Agent loops are KV-heavy (read session, append turn, write session), so this is the workhorse API.

```python
collection = bucket.scope("agent").collection("sessions")

# Create / replace
collection.upsert("session::u42::2026-07-05", {
    "user_id": "u42",
    "turns": [],
    "created_at": "2026-07-05T10:00:00Z",
})

# Read
res = collection.get("session::u42::2026-07-05")
session = res.content_as[dict]

# Delete
collection.remove("session::u42::2026-07-05")
```

**Key design** is schema design. We use `type::natural-id` (`chunk::docs-manual::0017`, `memory::u42::a1b2c3`). Keys are sortable, debuggable and collision free.

### TTL: Memory That Expires

```python
from couchbase.options import UpsertOptions

collection.upsert(key, doc, UpsertOptions(expiry=timedelta(hours=24)))
collection.touch(key, timedelta(hours=24))       # extend on activity
```

### Subdocument: Append a Turn Without Rewriting the Session

Full document read, modify, write has a race window and moves the whole document over the wire. Subdocument ops mutate paths server side:

```python
import couchbase.subdocument as SD

collection.mutate_in("session::u42::2026-07-05", (
    SD.array_append("turns", {"role": "user", "content": "hi", "ts": now_iso()}),
    SD.upsert("last_active", now_iso()),
))
```

This one call is the heart of the chat history store in Chapter 9.

### Durability

For state you must not lose, such as long-term memories and eval baselines, request server-side durable writes:

```python
from couchbase.durability import DurabilityLevel, ServerDurability

collection.upsert(key, doc,
    UpsertOptions(durability=ServerDurability(level=DurabilityLevel.MAJORITY)))
```

---

## 2.4 SQL++ Queries

SQL++ is SQL over JSON. The tool for anything that isn't a point lookup: analytics over agent logs, metadata filtering, joins between operational data and AI state.

```python
from couchbase.options import QueryOptions

result = cluster.query(
    """
    SELECT m.text, m.importance, m.created_at
    FROM ai.agent.memories AS m
    WHERE m.user_id = $user_id AND m.importance >= $min_importance
    ORDER BY m.created_at DESC
    LIMIT 20
    """,
    QueryOptions(named_parameters={"user_id": "u42", "min_importance": 0.5}),
)
for row in result:
    print(row["text"])
```

Always use **parameters** (`$name` / positional `$1`). Agent-generated strings interpolated into queries are an injection risk, exactly like classic SQL.

Scope level queries save you the keyspace prefix and are what query-shaped agent tools should use (`scope.query` resolves unqualified names within that scope):

```python
scope = bucket.scope("agent")
scope.query("SELECT COUNT(*) AS n FROM memories WHERE user_id = $u",
            QueryOptions(named_parameters={"u": "u42"}))
```

Consistency: by default queries are `NOT_BOUNDED` (fast, may trail recent writes).

When an agent writes a memory and immediately queries for it, use `QueryOptions(scan_consistency=QueryScanConsistency.REQUEST_PLUS)` or `consistent_with=MutationState(...)` for read-your-own-writes.

Secondary indexes make these queries fast:

```sql
CREATE INDEX idx_memories_user ON ai.agent.memories(user_id, importance DESC, created_at DESC);
```

---

## 2.5 Search From the SDK
Chapter 5 covers vector search fully; the shape to recognize now is the **`SearchRequest`** API (the modern one, prefer it over the legacy `cluster.search_query` for anything vector-related):

```python
import couchbase.search as search
from couchbase.options import SearchOptions
from couchbase.vector_search import VectorQuery, VectorSearch

req = search.SearchRequest.create(
    VectorSearch.from_vector_query(
        VectorQuery("embedding", query_vector, num_candidates=5)))
result = bucket.scope("docs").search("chunks-vector-index", req, SearchOptions(limit=5))
for row in result.rows():
    print(row.id, row.score)
```

---

## 2.6 Async With `acouchbase`

Agent servers (FastAPI, LangGraph deployments) are async; the SDK mirrors its full API under `acouchbase`:

```python
from acouchbase.cluster import AsyncCluster

cluster = await AsyncCluster.connect(conn_str, ClusterOptions(auth))
bucket = cluster.bucket("ai")
await bucket.on_connect()                      # required for async buckets
coll = bucket.scope("agent").collection("sessions")
await coll.upsert(key, doc)
result = cluster.search(index, req, SearchOptions(limit=5))
async for row in result.rows():
    ...
```

The `apps/rag-api` service uses the sync SDK from FastAPI worker threads for simplicity; a high-concurrency production service should use `acouchbase` end to end.

---

## 2.7 Transactions and Data Structures
**Distributed ACID Transactions** group mutations across multiple documents (and collections) so they commit or roll back together.

This is useful whenever an agent must update more than one keyspace consistently, e.g. "write the long-term memory AND mark the session summarized":

```python
from couchbase.exceptions import TransactionFailed

sessions = bucket.scope("agent").collection("sessions")
memories = bucket.scope("agent").collection("memories")

def txn_logic(ctx):
    session_doc = ctx.get(sessions, session_key)
    session = session_doc.content_as[dict]
    session["summarized"] = True
    ctx.replace(session_doc, session)

    ctx.insert(memories, memory_key, {
        "user_id": "u42",
        "text": "Prefers Python examples",
        "importance": 0.8,
    })

try:
    cluster.transactions.run(txn_logic)
    print("transaction committed")
except TransactionFailed as e:
    # Every mutation ctx made is rolled back. The session update AND the memory
    # insert are both undone, never left half-applied.
    print(f"transaction rolled back: {e}")
```

Inside `txn_logic`, mutate only through `ctx` (`ctx.get`, `ctx.insert`, `ctx.replace`, `ctx.remove`), never through the collection directly. That's what makes the attempt transactional.

If any operation raises, or the function itself raises, the whole attempt rolls back; `cluster.transactions.run` also retries the callback on conflicting concurrent transactions, so `txn_logic` should be safe to re-run.

**Data structures**: `collection.couchbase_list(key)`, `couchbase_map`, `couchbase_set`, `couchbase_queue` expose a document as a Python-style container backed by subdocument ops. A `CouchbaseQueue` makes a serviceable lightweight task queue for agent work items.

---

## 2.8 Recap
You now have the six verbs the rest of the book conjugates: `connect`, `upsert/get` (KV), `mutate_in` (subdoc), `query` (SQL++), `search` (vector/FTS), and `create_collection` (provisioning).

Notebook: [`notebooks/01_python_sdk_quickstart.ipynb`](../notebooks/01_python_sdk_quickstart.ipynb).

Running into errors? See [Troubleshooting](troubleshooting.md).

Next: [Chapter 3: Data Processing for AI](03-data-processing.md).
