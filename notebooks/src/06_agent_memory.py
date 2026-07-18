# %% [markdown]
# # 06: Agent Memory on Couchbase
#
# Companion to [Chapter 9](../docs/09-agent-memory.md). Builds the two memory stores every
# agent needs, from scratch, on SDK primitives:
#
# 1. **Short-term memory**: session documents, subdoc appends + sliding TTL
# 2. **Long-term memory**: embedded facts, vector recall with a user prefilter
# 3. **Extraction & summarization**: how memories get written
# 4. **Hygiene**: dedup, correction, GDPR deletion
#
# **Prerequisites:** notebook 01 (provisioning); `OPENAI_API_KEY` (or Capella, as before).

# %%
%pip install -q couchbase python-dotenv openai

# %%
import os
import time
import warnings
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from dotenv import find_dotenv, load_dotenv

# scope.search() reads its own deprecated scope_name option internally on every call
# (couchbase-python-client bug, harmless); silence it rather than the query it's not about.
warnings.filterwarnings("ignore", message="Option scope_name is deprecated")

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

CB_BUCKET = os.getenv("CB_BUCKET")
bucket = cluster.bucket(CB_BUCKET)
agent_scope = bucket.scope("agent")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# %%
# Embeddings + a small LLM for extraction (OpenAI default, Capella switch as usual)
from openai import OpenAI

if os.getenv("CAPELLA_AI_ENDPOINT"):
    import base64
    key = os.getenv("CAPELLA_AI_TOKEN") or base64.b64encode(
        f"{os.environ['CB_USERNAME']}:{os.environ['CB_PASSWORD']}".encode()).decode()
    ai = OpenAI(base_url=os.environ["CAPELLA_AI_ENDPOINT"], api_key=key)
    EMBEDDING_MODEL = os.getenv("CAPELLA_EMBEDDING_MODEL", "intfloat/e5-mistral-7b-instruct")
    LLM_MODEL = os.getenv("CAPELLA_LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
else:
    ai = OpenAI()
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


def embed_one(text: str) -> list[float]:
    return ai.embeddings.create(model=EMBEDDING_MODEL, input=[text]).data[0].embedding

EMBEDDING_DIM = len(embed_one("probe"))

# %% [markdown]
# ## 1. Short-term memory: the session store
#
# One document per session; turns appended server-side; TTL slides on every interaction.
# STM *should* forget: expiry is the feature, not a limitation.

# %%
import couchbase.subdocument as SD
from couchbase.exceptions import DocumentNotFoundException
from couchbase.options import UpsertOptions


class SessionStore:
    def __init__(self, scope, ttl=timedelta(hours=24)):
        self.coll = scope.collection("sessions")
        self.ttl = ttl

    def append_turn(self, session_id: str, role: str, content: str):
        key = f"session::{session_id}"
        turn = {"role": role, "content": content, "ts": now_iso()}
        try:
            self.coll.mutate_in(key, (
                SD.array_append("turns", turn),
                SD.upsert("last_active", turn["ts"]),
            ))
        except DocumentNotFoundException:
            self.coll.upsert(key, {"session_id": session_id, "turns": [turn],
                                   "last_active": turn["ts"]},
                             UpsertOptions(expiry=self.ttl))
        self.coll.touch(key, self.ttl)          # sliding expiry

    def recent_turns(self, session_id: str, n: int = 10) -> list[dict]:
        try:
            doc = self.coll.get(f"session::{session_id}").content_as[dict]
        except DocumentNotFoundException:
            return []
        return doc["turns"][-n:]


sessions = SessionStore(agent_scope)
sid = "u42::demo"
sessions.append_turn(sid, "user", "Hi! I'm setting up vector search for our payments platform.")
sessions.append_turn(sid, "assistant", "Great, what embedding model are you planning to use?")
sessions.append_turn(sid, "user", "text-embedding-3-small. Also, please always show me Python, never Java.")

for t in sessions.recent_turns(sid):
    print(f"{t['role']:>10}: {t['content']}")

# %% [markdown]
# ## 2. Long-term memory: embedded facts with semantic recall
#
# First, the vector index over `agent.memories` (Ch. 5 definition; `user_id` indexed as
# text for the prefilter):

# %%
from couchbase.management.search import SearchIndex

MEM_INDEX = "memories-vector-index"
mem_index_def = {
    "type": "fulltext-index", "name": MEM_INDEX,
    "sourceType": "gocbcore", "sourceName": CB_BUCKET,
    "planParams": {"maxPartitionsPerPIndex": 1024, "indexPartitions": 1},
    "params": {
        "doc_config": {"mode": "scope.collection.type_field", "type_field": "type"},
        "mapping": {
            "default_mapping": {"dynamic": False, "enabled": False},
            "types": {
                "agent.memories": {
                    "dynamic": False, "enabled": True,
                    "properties": {
                        "embedding": {"enabled": True, "dynamic": False,
                                      "fields": [{"name": "embedding", "type": "vector",
                                                  "index": True, "dims": EMBEDDING_DIM,
                                                  "similarity": "dot_product",
                                                  "vector_index_optimized_for": "recall"}]},
                        "text": {"enabled": True, "dynamic": False,
                                 "fields": [{"name": "text", "type": "text",
                                             "index": True, "store": True}]},
                        "user_id": {"enabled": True, "dynamic": False,
                                    "fields": [{"name": "user_id", "type": "text",
                                                "index": True}]},
                        "kind": {"enabled": True, "dynamic": False,
                                 "fields": [{"name": "kind", "type": "text",
                                             "index": True, "store": True}]},
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
    agent_scope.search_indexes().drop_index(MEM_INDEX)  # rerun-safe: replace any stale definition
except SearchIndexNotFoundException:
    pass
agent_scope.search_indexes().upsert_index(SearchIndex.from_json(mem_index_def))
print("memory index upserted")

# %%
import couchbase.search as search
from couchbase.options import QueryOptions, SearchOptions
from couchbase.vector_search import VectorQuery, VectorSearch


class MemoryStore:
    def __init__(self, scope, index_name=MEM_INDEX):
        self.scope = scope
        self.coll = scope.collection("memories")
        self.index = index_name

    def remember(self, user_id: str, text: str, kind: str = "fact",
                 importance: float = 0.5, dedup_threshold: float | None = 0.9) -> str:
        # dedup-on-write: don't fill memory with near-copies (Ch. 9 §9.4)
        if dedup_threshold:
            existing = self.recall(user_id, text, k=1)
            if existing and existing[0]["score"] >= dedup_threshold:
                return existing[0]["id"]
        key = f"memory::{user_id}::{uuid4().hex[:12]}"
        self.coll.upsert(key, {
            "type": "memory", "user_id": user_id, "text": text, "kind": kind,
            "importance": importance, "embedding": embed_one(text),
            "created_at": now_iso(), "access_count": 0,
        })
        return key

    def recall(self, user_id: str, query: str, k: int = 5,
               min_score: float | None = None) -> list[dict]:
        req = search.SearchRequest.create(VectorSearch.from_vector_query(VectorQuery(
            "embedding", embed_one(query), num_candidates=k * 3,
            prefilter=search.MatchQuery(user_id, field="user_id"),   # never cross users
        )))
        result = self.scope.search(self.index, req,
                                   SearchOptions(limit=k, fields=["text", "kind"]))
        hits = [{"id": r.id, "score": round(r.score, 4), **(r.fields or {})}
                for r in result.rows()]
        if min_score is not None:
            # top-k always fills k slots from whatever candidates exist, with a small
            # memory store that means a weak, off-topic match can round out the list.
            # A score floor turns "ranked tail" into "actually relevant" (§9.5).
            hits = [h for h in hits if h["score"] >= min_score]
        for h in hits:  # track usage for aging (§9.6)
            self.coll.mutate_in(h["id"], (SD.counter("access_count", 1),
                                          SD.upsert("last_accessed", now_iso())))
        return hits

    def forget_user(self, user_id: str) -> int:
        rows = list(cluster.query(
            f"DELETE FROM `{CB_BUCKET}`.agent.memories m "
            f"WHERE m.user_id = $u RETURNING META(m).id",
            QueryOptions(named_parameters={"u": user_id})))
        return len(rows)


memories = MemoryStore(agent_scope)
memories.remember("u42", "Works on a payments platform", kind="context", importance=0.7)
memories.remember("u42", "Prefers Python examples, never Java", kind="preference", importance=0.9)
memories.remember("u42", "Uses text-embedding-3-small for embeddings", kind="context", importance=0.6)
memories.remember("u99", "Prefers Java examples", kind="preference", importance=0.9)  # different user!
time.sleep(3)  # let the index ingest

# %%
# Semantic recall, and the prefilter keeps u99's Java preference out
for m in memories.recall("u42", "what programming language should I use in examples?", k=3):
    print(f"{m['score']:>8}  [{m['kind']}] {m['text']}")

# %% [markdown]
# Notice the score drop-off: the two Python/Java preference memories land around
# `~0.58`, then the third result trails off near `~0.3`. `recall(..., k=3)` asks
# Couchbase's FTS vector search for the 3 closest matches, and `limit=k` always fills
# all `k` slots from whatever candidates pass the prefilter, even if the 3rd-closest is
# a weak, semantically unrelated match. With only a handful of memories for `u42`, that
# tail can be a different `kind` entirely (e.g. an embedding-model note showing up for a
# programming-language question), and the *closest remaining* candidate gets returned
# even though it isn't really about "what language should I use." That's expected
# nearest-neighbor behavior, not a storage bug: a low score paired with a sharp drop
# from the top hits means "weak match, included only to round out top-k," not "the
# query matched this memory well," and the fact is still filed under the right user and
# the right `kind`.
#
# (If you re-run this cell later, after §5's correction below replaces the
# `text-embedding-3-small` context memory, the tail match will differ, but the same
# drop-off pattern holds.)
#
# The fix is a **similarity-score floor**, applied on top of k: keep the ranked order,
# but only pull in memories confident enough to actually be relevant. Pass `min_score=`
# (added to `recall` above, §9.5) to compare the same query with a score floor set, and
# watch the weak tail drop out:

# %%
# Same query, with a score floor: weak/off-topic tail matches are excluded rather
# than silently padded in to fill k
for m in memories.recall("u42", "what programming language should I use in examples?",
                          k=3, min_score=0.4):
    print(f"{m['score']:>8}  [{m['kind']}] {m['text']}")

# %% [markdown]
# Dedup in action: remembering the same fact again returns the existing memory:

# %%
k1 = memories.remember("u42", "Prefers Python examples over Java")
print("deduped to:", k1)

# %% [markdown]
# ## 3. Extraction: turning conversation into memory
#
# The background-extraction pattern (§9.4): an LLM pass over the transcript pulls out
# *stable* facts with importance scores.

# %%
import json

from openai import BadRequestError

transcript = "\n".join(f"{t['role']}: {t['content']}" for t in sessions.recent_turns(sid, 100))

messages = [
    {"role": "system", "content":
        'Extract durable facts about the user from this conversation, stable '
        'preferences, constraints, and context only; nothing transient. Respond with '
        'ONLY JSON, no prose: {"facts": [{"text": ..., "kind": "preference|context|fact", '
        '"importance": 0.0-1.0}]}'},
    {"role": "user", "content": transcript},
]
try:
    # Not every OpenAI-compatible endpoint honors response_format (Ch. 8),
    # fall back to plain prompting if it's rejected outright.
    resp = ai.chat.completions.create(model=LLM_MODEL, temperature=0,
                                      response_format={"type": "json_object"}, messages=messages)
except BadRequestError:
    resp = ai.chat.completions.create(model=LLM_MODEL, temperature=0, messages=messages)
text = resp.choices[0].message.content
extracted = json.loads(text[text.find("{"):text.rfind("}") + 1])["facts"]
for f in extracted:
    # smaller Capella-hosted models don't always fill every field the prompt asked for
    kind = f.get("kind", "fact")
    importance = f.get("importance", 0.5)
    key = memories.remember("u42", f["text"], kind=kind, importance=importance)
    print(f"[{importance}] {f['text']}  →  {key}")

# %% [markdown]
# ## 4. Assembling the prompt: three queries, one context
#
# This is the context-assembly pattern every agent turn runs (§9.5), and it's just
# database reads:

# %%
current_message = "Can you remind me which embedding model we settled on?"

# min_score keeps a weak/irrelevant top-k tail match out of the prompt (see §2 above)
relevant = memories.recall("u42", current_message, k=3, min_score=0.4)
recent = sessions.recent_turns(sid, n=6)

system_prompt = (
    "You are a helpful engineering assistant.\n\n"
    "Relevant memories about this user:\n"
    + "\n".join(f"- {m['text']}" for m in relevant)
)
turns = [{"role": t["role"], "content": t["content"]} for t in recent]
turns.append({"role": "user", "content": current_message})

# recent's last stored turn and current_message can both be "user" (the demo transcript
# above ends mid-exchange, before the assistant's reply), which OpenAI's API tolerates
# but a strict Capella-hosted model rejects with "conversation roles must alternate
# user/assistant/user/assistant/...". Merge any consecutive same-role turns so the
# sequence always alternates, regardless of where in the exchange it was captured.
merged = []
for t in turns:
    if merged and merged[-1]["role"] == t["role"]:
        merged[-1]["content"] += "\n" + t["content"]
    else:
        merged.append(dict(t))

msgs = [{"role": "system", "content": system_prompt}] + merged

print(ai.chat.completions.create(model=LLM_MODEL, temperature=0,
                                 messages=msgs).choices[0].message.content)

# %% [markdown]
# ## 5. Hygiene: correction and the right to be forgotten

# %%
# Correction: replace, don't accumulate (§9.6)
old = memories.recall("u42", "embedding model", k=1)
if old:
    memories.coll.remove(old[0]["id"])
memories.remember("u42", "Switched to intfloat/e5-mistral-7b-instruct for embeddings",
                  kind="context", importance=0.7)
print("corrected")

# %%
# GDPR deletion is one query: auditable, verifiable
deleted = memories.forget_user("u99")
print(f"deleted {deleted} memories for u99")

# %% [markdown]
# Everything here becomes agent tooling in the next notebook: `remember` as a cataloged
# tool, recall as context assembly, and LangGraph checkpoints as working memory.
#
# **Next:** [07: Agent Catalog + LangGraph](07_agent_catalog_langgraph.ipynb)
