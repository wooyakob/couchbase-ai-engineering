# %% [markdown]
# # 01: Couchbase Python SDK Quickstart for AI Engineering
#
# Companion to [Chapter 2](../docs/02-python-sdk-foundations.md).
#
# This notebook sets up everything the rest of the series uses:
#
# 1. Connect to a Couchbase cluster (local or Capella)
# 2. Provision the `ai` bucket's scopes and collections
# 3. Key-value operations, TTL, and subdocument mutations
# 4. SQL++ queries with parameters
#
# **Prerequisites:** a Couchbase cluster (Capella or `docker run couchbase:enterprise`)
# with a bucket named `ai`, and a `.env` file (see `.env.server.example` / `.env.capella.example` in the repo root).

# %%
%pip install -q couchbase python-dotenv

# %%
import os
from datetime import timedelta

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(os.getenv("ENV_FILE", ".env"), usecwd=True))  # ENV_FILE lets .env.server / .env.capella run side by side

CB_CONN_STRING = os.getenv("CB_CONN_STRING")
CB_USERNAME = os.getenv("CB_USERNAME")
CB_PASSWORD = os.getenv("CB_PASSWORD")
CB_BUCKET = os.getenv("CB_BUCKET")

# Fail here with a clear message, not later with a cryptic AttributeError on
# CB_CONN_STRING.startswith(...), this almost always means .env/.env.server/.env.capella
# wasn't loaded, or ENV_FILE points at a file that doesn't exist (docs/troubleshooting.md).
_missing = [name for name, val in
           [("CB_CONN_STRING", CB_CONN_STRING), ("CB_USERNAME", CB_USERNAME),
            ("CB_PASSWORD", CB_PASSWORD), ("CB_BUCKET", CB_BUCKET)] if not val]
if _missing:
    raise RuntimeError(
        f"Missing required env var(s): {', '.join(_missing)}. Currently "
        f"ENV_FILE={os.getenv('ENV_FILE', '.env')!r}: confirm that file exists and "
        "has these set (see .env.server.example / .env.capella.example). "
        "See docs/troubleshooting.md."
    )

# %% [markdown]
# ## 1. Connect
#
# `couchbase://` for local/plain, `couchbases://` for Capella (TLS). For Capella we apply
# the `wan_development` profile, which relaxes timeouts for clusters that aren't on your LAN.

# %%
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.exceptions import AuthenticationException, CouchbaseException
from couchbase.options import ClusterOptions, KnownConfigProfiles

opts = ClusterOptions(PasswordAuthenticator(CB_USERNAME, CB_PASSWORD))
if CB_CONN_STRING.startswith("couchbases://"):
    opts.apply_profile(KnownConfigProfiles.WanDevelopment)

try:
    cluster = Cluster.connect(CB_CONN_STRING, opts)
    cluster.wait_until_ready(timedelta(seconds=10))
except AuthenticationException as e:
    raise RuntimeError(
        f"Couchbase rejected CB_USERNAME={CB_USERNAME!r} for {CB_CONN_STRING!r}, "
        "check CB_USERNAME/CB_PASSWORD. See docs/troubleshooting.md."
    ) from e
except CouchbaseException as e:
    raise RuntimeError(
        f"Couldn't connect to Couchbase at {CB_CONN_STRING!r}: {e}. Check the "
        "cluster is running/reachable (docker ps, or the Capella console) and that "
        "CB_CONN_STRING is correct: couchbases:// for Capella, couchbase:// for "
        "local. See docs/troubleshooting.md."
    ) from e
print("connected:", CB_CONN_STRING)

# %% [markdown]
# ## 2. Provision scopes and collections
#
# The whole book uses one bucket `ai` with scopes per concern. Provisioning is idempotent.
# Note the collection-level TTL on `agent.sessions`: short-term memory that expires by design.

# %%
from couchbase.exceptions import (
    BucketNotFoundException,
    CollectionAlreadyExistsException,
    ScopeAlreadyExistsException,
)
from couchbase.management.collections import CreateCollectionSettings

bucket = cluster.bucket(CB_BUCKET)
cm = bucket.collections()

LAYOUT = {
    "docs": ["chunks", "llm_cache", "semantic_cache"],
    "agent": ["sessions", "memories", "chat_history", "checkpoints", "checkpoint_writes"],
    "evals": ["runs", "samples"],
}
TTL = {("agent", "sessions"): timedelta(days=7)}

for scope_name, collections in LAYOUT.items():
    try:
        cm.create_scope(scope_name)
        print("created scope", scope_name)
    except ScopeAlreadyExistsException:
        pass
    except (BucketNotFoundException, CouchbaseException) as e:
        raise RuntimeError(
            f"Couldn't provision scope {scope_name!r} in bucket {CB_BUCKET!r}: {e}. "
            f"This notebook provisions scopes/collections INSIDE the bucket, it "
            f"does not create the bucket itself. Create {CB_BUCKET!r} in the "
            "Couchbase Server or Capella UI first, then re-run this cell. "
            "See docs/troubleshooting.md."
        ) from e
    for coll_name in collections:
        try:
            expiry = TTL.get((scope_name, coll_name))
            settings = CreateCollectionSettings(max_expiry=expiry) if expiry else None
            cm.create_collection(scope_name, coll_name, settings)
            print(f"created {scope_name}.{coll_name}")
        except CollectionAlreadyExistsException:
            pass

# %% [markdown]
# ## 3. Key-value operations
#
# KV is the fastest path to a document, since no query engine is involved. Keys follow
# `type::natural-id` so they're sortable and debuggable.

# %%
sessions = bucket.scope("agent").collection("sessions")

session_key = "session::demo-user::quickstart"
sessions.upsert(session_key, {
    "user_id": "demo-user",
    "turns": [],
    "created_at": "2026-07-05T10:00:00Z",
    "last_active": "2026-07-05T10:00:00Z",
})

doc = sessions.get(session_key).content_as[dict]
doc

# %% [markdown]
# ### TTL on a single document
#
# Session documents get a sliding 24h expiry: set it on write, extend it with `touch()`
# on every interaction.

# %%
from couchbase.options import UpsertOptions

sessions.upsert(session_key, doc, UpsertOptions(expiry=timedelta(hours=24)))
sessions.touch(session_key, timedelta(hours=24))

res = sessions.get(session_key, with_expiry=True)
print("expires at:", res.expiry_time)

# %% [markdown]
# ### Subdocument mutations
#
# Append a conversation turn *server-side*: no read-modify-write race, no shipping the
# whole transcript over the network. This one call is the heart of Chapter 9's session store.

# %%
import couchbase.subdocument as SD

sessions.mutate_in(session_key, (
    SD.array_append("turns", {"role": "user", "content": "What is vector search?",
                              "ts": "2026-07-05T10:01:00Z"}),
    SD.upsert("last_active", "2026-07-05T10:01:00Z"),
))
sessions.mutate_in(session_key, (
    SD.array_append("turns", {"role": "assistant",
                              "content": "Vector search finds documents by semantic similarity...",
                              "ts": "2026-07-05T10:01:02Z"}),
))

# Read back just the turns (a projection, not the whole doc)
result = sessions.lookup_in(session_key, (SD.get("turns"),))
turns = result.content_as[list](0)
for t in turns:
    print(f"{t['role']:>10}: {t['content'][:60]}")

# %% [markdown]
# ### Durability
#
# For state you must not lose (long-term memories, eval baselines), request durable writes.
#
# `DurabilityLevel.MAJORITY` requires acknowledgment from a majority of copies
# (active + replicas) before the write returns. With this cluster's 1 replica
# (2 copies total), majority is 2, so it needs both the active node and the one
# replica to ack. On a single-node dev cluster there are 0 replicas, so no
# majority is possible and Couchbase raises `DurabilityImpossibleException`;
# hence the try/except here, falling back to a plain (non-durable) write.
#
# A successful majority write means: if the active node dies immediately after
# the write returns, the data survives on the replica and a failover won't lose
# it. Without durability, an upsert can return success and still vanish if the
# active node crashes before replicating. Durability trades a little latency
# for that guarantee.

# %%
from couchbase.durability import DurabilityLevel, ServerDurability
from couchbase.exceptions import DurabilityImpossibleException

memories = bucket.scope("agent").collection("memories")
try:
    memories.upsert(
        "memory::demo-user::0001",
        {"user_id": "demo-user", "text": "Prefers Python examples", "importance": 0.8},
        UpsertOptions(durability=ServerDurability(level=DurabilityLevel.MAJORITY)),
    )
    print("durable write OK")
except DurabilityImpossibleException:
    memories.upsert("memory::demo-user::0001",
                    {"user_id": "demo-user", "text": "Prefers Python examples",
                     "importance": 0.8})
    print("single-node cluster: wrote without durability")

# %% [markdown]
# ### Transactions
#
# Distributed ACID transactions group mutations across documents (and collections)
# so they commit or roll back together. Mutate only through `ctx` inside the
# callback, never through the collection directly; that's what makes the
# attempt transactional.

# %%
from couchbase.exceptions import TransactionFailed

def txn_logic(ctx):
    session_doc = ctx.get(sessions, session_key)
    session = session_doc.content_as[dict]
    session["summarized"] = True
    ctx.replace(session_doc, session)

    ctx.insert(memories, "memory::demo-user::0002", {
        "user_id": "demo-user",
        "text": "Asked about vector search",
        "importance": 0.6,
    })

try:
    cluster.transactions.run(txn_logic)
    print("transaction committed")
except TransactionFailed as e:
    # Every mutation ctx made is rolled back: the session update AND the memory
    # insert are both undone, never left half-applied.
    print(f"transaction rolled back: {e}")

# %% [markdown]
# ## 4. SQL++ queries
#
# SQL over JSON. Always parameterize: agent-generated strings interpolated into queries
# are an injection risk, exactly like classic SQL.

# %%
from couchbase.options import QueryOptions

# A primary index makes ad-hoc queries possible on small dev datasets.
# In production you create targeted secondary indexes instead.
cluster.query(
    f"CREATE PRIMARY INDEX IF NOT EXISTS ON `{CB_BUCKET}`.agent.memories"
).execute()

rows = cluster.query(
    f"""
    SELECT m.text, m.importance
    FROM `{CB_BUCKET}`.agent.memories AS m
    WHERE m.user_id = $user_id AND m.importance >= $min_importance
    ORDER BY m.importance DESC
    """,
    QueryOptions(named_parameters={"user_id": "demo-user", "min_importance": 0.5}),
)
for row in rows:
    print(row)

# %% [markdown]
# Scope-level queries resolve unqualified collection names within the scope, handy for
# agent tools:

# %%
agent_scope = bucket.scope("agent")
result = agent_scope.query(
    "SELECT COUNT(*) AS n FROM memories WHERE user_id = $u",
    QueryOptions(named_parameters={"u": "demo-user"}),
)
print(list(result)[0])

# %% [markdown]
# ## Cleanup (optional)

# %%
sessions.remove(session_key)
print("done, the `ai` bucket is provisioned for the rest of the series")

# %% [markdown]
# **Next:** [02: Vector Search Fundamentals](02_vector_search_fundamentals.ipynb)
