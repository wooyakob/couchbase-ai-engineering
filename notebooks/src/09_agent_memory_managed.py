# %% [markdown]
# # 09: Managed Agent Memory (server + SDK)
#
# Companion to [Chapter 9 §9.7–§9.8](../docs/09-agent-memory.md). Where
# [notebook 06](06_agent_memory.ipynb) builds a memory subsystem from SDK primitives,
# this one uses the **managed Couchbase Agent Memory product**: the
# `couchbase-agent-memory` SDK talking to an **Agent Memory server** that does the
# embedding, fact extraction, semantic ranking, and TTL for you, on self-managed
# Couchbase *or* Capella.
#
# What we cover:
#
# 1. Connect to the server (and fail loudly, showing the `docker run` command, if it isn't up)
# 2. Users → sessions → memory blocks
# 3. Store **messages** and **facts**; semantic **recall** across all sessions
# 4. User isolation, TTL, and the right to be forgotten
# 5. How `apps/support-agent` maps onto this model
#
# **Prerequisites:** a running Agent Memory server (see §9.7; it's one `docker run`),
# reachable at `AGENTMEMORY_BASE_URL` (default `http://localhost:8080`), pointed at your
# Couchbase/Capella cluster. Requires **Python 3.12+**.
#
# Both pieces are GA, but distributed differently: the `couchbase-agent-memory` SDK used
# below is a normal, `pip`-installable package on PyPI. The **server** is GA too, but not
# on a public registry: you get the container image (a `.tar`) by signing up for the
# free trial, then `docker load -i agentmemory-server-<arch>-<version>.tar` before the
# `docker run` below.

# %%
%pip install -q couchbase-agent-memory python-dotenv

# %%
import os

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(os.getenv("ENV_FILE", ".env"), usecwd=True))  # ENV_FILE lets .env.server / .env.capella run side by side

from agentmemory import AgentMemoryClient, ChatMessage
from agentmemory.exceptions import AgentMemoryError, ConflictError, NotFoundError

BASE_URL = os.getenv("AGENTMEMORY_BASE_URL")
TOKEN = os.getenv("AGENTMEMORY_TOKEN") or None  # only if the server has OIDC enabled

client = AgentMemoryClient(base_url=BASE_URL, token=TOKEN)

# %% [markdown]
# ## 1. Is the server up?
#
# The client constructor doesn't connect; the first request does. `health_ping()`
# checks the server and its dependencies (Couchbase + the embedding/LLM models).

# %%
DOCKER_HINT = """
Agent Memory server not reachable at {url}. Start it (§9.7):

  docker run -d --name agentmemory-server --env-file .env \\
    -p 8080:8080 -p 9090:9090 -v agentmemory-logs:/app/logs \\
    --restart unless-stopped agentmemory-server:arm64   # :amd64 on Intel

The .env for the *server* points it at your cluster (AGENTMEMORY_CONN_STRING,
AGENTMEMORY_USERNAME/PASSWORD/BUCKET, OPENAI_API_KEY, AGENTMEMORY_EMBEDDING_MODEL).
""".format(url=BASE_URL)

try:
    health = client.health_ping()
    print("server:", health.overall_status)
    print("checked:", ", ".join(health.checked_entities))
except AgentMemoryError as e:
    raise SystemExit(f"{e}\n{DOCKER_HINT}")

# %% [markdown]
# ## 2. Users and sessions
#
# The hierarchy is **user → session → memory block**. One user per real end-user (the
# isolation boundary), one session per conversation. `create_user`/`create_session`
# raise `ConflictError` if the id already exists, so we get-or-create for clean re-runs.

# %%
def get_or_create_user(user_id: str, name: str):
    try:
        return client.get_user(user_id)
    except NotFoundError:
        try:
            return client.create_user(user_id, name)
        except ConflictError:
            return client.get_user(user_id)


def get_or_create_session(user, session_id: str):
    try:
        return user.get_session(session_id)
    except NotFoundError:
        try:
            return user.create_session(session_id)
        except ConflictError:
            return user.get_session(session_id)


ada = get_or_create_user("nb09-ada", "Ada")
convo = get_or_create_session(ada, "nb09-ada::2026-07-05")
print("user:", ada.user_id, "| session:", convo.session_id)

# %% [markdown]
# ## 3. Store memory: messages and facts
#
# Two kinds of block. A **message** is a user/assistant exchange: the server extracts
# and summarizes it. A **fact** is a discrete durable statement. You can't mix both in
# one call. `async_processing=False` blocks until the block is embedded and searchable
# (handy in a notebook); the default `True` returns immediately and the block becomes
# searchable within a second or two.

# %%
convo.add_memory(messages=[
    ChatMessage(user_content="I'm setting up vector search for our payments platform.",
                assistant_content="Great, which embedding model are you planning to use?"),
    ChatMessage(user_content="text-embedding-3-small. And please always show Python, never Java.",
                assistant_content="Noted, Python it is."),
], async_processing=False)

resp = convo.add_memory(
    facts=["Works on a payments platform.",
           "Prefers Python examples, never Java."],
    annotations={"kind": "preference"},
    async_processing=False,
)
print("stored", resp.accepted_count, "fact block(s):", resp.block_ids)
if resp.rejected_count:
    print("rejected:", resp.rejected_details)

# %% [markdown]
# ## 4. Semantic recall
#
# `search_memory` embeds the query server-side and returns blocks ranked by relevance
# (`rel_score`). `filters={"session_ids": "all"}` searches every session this user has:
# so saved facts *and* things said in past conversations both surface. `relevant_k`
# caps the count.

# %%
def show(blocks):
    for b in blocks:
        text = b.fact or b.summary or (
            f"{b.message.user_content} / {b.message.assistant_content}" if b.message else "")
        print(f"{(b.rel_score or 0):.3f}  [{b.status}] {text}")


results = convo.search_memory(
    query="what programming language should examples use?",
    filters={"session_ids": "all", "relevant_k": 5})
show(results.memory_blocks)

# %% [markdown]
# ## 5. Isolation: recall never crosses users
#
# A second user with the opposite preference: the server keeps memory partitioned by
# user, so Ada's recall never sees Grace's Java preference (the managed equivalent of
# the §9.3 `user_id` prefilter).

# %%
grace = get_or_create_user("nb09-grace", "Grace")
gsession = get_or_create_session(grace, "nb09-grace::2026-07-05")
gsession.add_memory(facts=["Prefers Java examples."], async_processing=False)

print("Ada recall:")
show(convo.search_memory(query="preferred programming language",
                         filters={"session_ids": "all"}).memory_blocks)
print("\nGrace recall:")
show(gsession.search_memory(query="preferred programming language",
                            filters={"session_ids": "all"}).memory_blocks)

# %% [markdown]
# ## 6. TTL: memory that forgets
#
# Blocks can expire automatically. Set a TTL per block at write time, per session at
# creation, or in bulk per user. `0` means never expire. (STM *should* forget; §9.2.)

# %%
# A block that lives for one hour, then vanishes:
convo.add_memory(facts=["Debugging a one-off timeout on 2026-07-05."],
                 memory_block_ttl=3600, async_processing=False)

# Bulk: expire everything in this session in 24h (leave facts permanent by omitting this).
# Per-session TTL can only be modified once the session is ended; no more blocks after this.
convo.end()
ada.modify_ttl(new_ttl=86_400, session_id=convo.session_id)
print("TTLs updated")

# %% [markdown]
# ## 7. The right to be forgotten
#
# GDPR deletion is one call: it cascades to every session and memory block for the
# user (the managed equivalent of the §9.6 `DELETE … WHERE user_id = $u`). We use it to
# clean up this notebook's demo users.

# %%
for uid in ("nb09-ada", "nb09-grace"):
    try:
        client.delete_user(uid)
        print("deleted", uid)
    except NotFoundError:
        pass

client.close()

# %% [markdown]
# ## How the support-agent uses this
#
# [`apps/support-agent/agent/memory.py`](../apps/support-agent/agent/memory.py) wraps
# exactly these calls:
#
# - Durable facts go to a stable per-user **`profile`** session; each conversation is
#   its own session, so `search_memory(session_ids="all")` recalls both.
# - The `save_memory` tool → `add_memory(facts=…)`; the graph's `load_context` node →
#   `search_memory(...)`; `delete_user` is the GDPR path.
# - The client is a lazily-opened singleton, and recall degrades to *no memories* if the
#   server is offline, so a memory outage never breaks a turn.
#
# **Compare with [notebook 06](06_agent_memory.ipynb)**: the same behavior, but there
# you own the embedding calls, the vector index, the dedup threshold, and the extraction
# prompt. Build-your-own for control; the managed server for production.
#
# **Next:** [07: Agent Catalog + LangGraph](07_agent_catalog_langgraph.ipynb), which
# assembles the full agent.
