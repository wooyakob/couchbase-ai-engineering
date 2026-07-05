# %% [markdown]
# # 01 — Couchbase Python SDK Quickstart for AI Engineering
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
# **Prerequisites:** a Couchbase cluster (Capella free tier or `docker run couchbase:enterprise`)
# with a bucket named `ai`, and a `.env` file (see `.env.example` in the repo root).

# %%
# %pip install -q couchbase python-dotenv

# %%
import os
from datetime import timedelta

from dotenv import load_dotenv

load_dotenv()

CB_CONN_STRING = os.getenv("CB_CONN_STRING", "couchbase://localhost")
CB_USERNAME = os.getenv("CB_USERNAME", "Administrator")
CB_PASSWORD = os.getenv("CB_PASSWORD", "password")
CB_BUCKET = os.getenv("CB_BUCKET", "ai")

# %% [markdown]
# ## 1. Connect
#
# `couchbase://` for local/plain, `couchbases://` for Capella (TLS). For Capella we apply
# the `wan_development` profile, which relaxes timeouts for clusters that aren't on your LAN.

# %%
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions, KnownConfigProfiles

opts = ClusterOptions(PasswordAuthenticator(CB_USERNAME, CB_PASSWORD))
if CB_CONN_STRING.startswith("couchbases://"):
    opts.apply_profile(KnownConfigProfiles.WanDevelopment)

cluster = Cluster.connect(CB_CONN_STRING, opts)
cluster.wait_until_ready(timedelta(seconds=10))
print("connected:", CB_CONN_STRING)

# %% [markdown]
# ## 2. Provision scopes and collections
#
# The whole book uses one bucket `ai` with scopes per concern. Provisioning is idempotent.
# Note the collection-level TTL on `agent.sessions` — short-term memory that expires by design.

# %%
from couchbase.exceptions import (
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
# KV is the fastest path to a document — no query engine involved. Keys follow
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
# Append a conversation turn *server-side* — no read-modify-write race, no shipping the
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
# Requires a cluster with replicas (on single-node dev clusters this raises an error — hence
# the try/except here).

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
# ## 4. SQL++ queries
#
# SQL over JSON. Always parameterize — agent-generated strings interpolated into queries
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
# Scope-level queries resolve unqualified collection names within the scope — handy for
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
print("done — the `ai` bucket is provisioned for the rest of the series")

# %% [markdown]
# **Next:** [02 — Vector Search Fundamentals](02_vector_search_fundamentals.ipynb)
