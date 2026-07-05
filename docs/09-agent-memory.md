# Chapter 9 — Agent Memory

> *An LLM is stateless and amnesiac: every request starts from nothing. Memory is what turns a chat completion into an agent — and memory is not a model feature, it's a database design. This chapter builds a complete memory subsystem on Couchbase.*

## 9.1 A taxonomy that maps to collections

Agent memory splits into four kinds, each with different access patterns — and each maps onto a Couchbase capability almost embarrassingly well:

| Memory kind | Contents | Access pattern | Couchbase feature |
|---|---|---|---|
| **Short-term (STM)** | current conversation turns | read/write by session key, every turn | KV + subdocument, collection TTL |
| **Long-term (LTM)** | durable facts, preferences, learned knowledge | semantic recall ("what do I know about X?") | documents + vector search |
| **Working / checkpoint** | the agent graph's in-flight state | save/restore by thread | LangGraph checkpointer collections (Ch. 11) |
| **Episodic** | full traces of past runs | append-heavy writes, analytical reads | activity logs (Agent Catalog, Ch. 10) |

We implement STM and LTM here; checkpoints and episodes get their own chapters.

## 9.2 Short-term memory: sessions

A session is one document; turns are appended with subdocument ops (no read-modify-write race, no shipping the whole transcript over the wire each turn):

```python
import couchbase.subdocument as SD
from couchbase.exceptions import DocumentNotFoundException
from couchbase.options import UpsertOptions

class SessionStore:
    def __init__(self, cluster, ttl=timedelta(hours=24)):
        self.coll = cluster.bucket("ai").scope("agent").collection("sessions")
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
        self.coll.touch(key, self.ttl)          # sliding expiry: active sessions live on

    def recent_turns(self, session_id: str, n: int = 10) -> list[dict]:
        try:
            doc = self.coll.get(f"session::{session_id}").content_as[dict]
        except DocumentNotFoundException:
            return []
        return doc["turns"][-n:]
```

Design notes:

- **TTL is the feature.** STM *should* forget; the collection was created with `max_expiry=timedelta(days=7)` (§2.2) as a backstop, and `touch()` implements sliding expiration. No cron jobs, no cleanup code.
- **Windowing at read time** (`turns[-n:]`) keeps prompts bounded. When a session outgrows the window, summarize (§9.5).
- If you'd rather not own this class, `CouchbaseChatMessageHistory` (§6.4) is the LangChain-shaped equivalent. Build your own when you need control over windowing, TTL, and summarization; use the integration when you don't.

## 9.3 Long-term memory: semantic recall

LTM stores *facts extracted from conversations*, embedded for semantic retrieval. A memory document:

```json
{
  "type": "memory",
  "user_id": "u42",
  "text": "Prefers Python examples over Java; works on a payments platform",
  "kind": "preference",
  "importance": 0.8,
  "embedding": [ ... ],
  "created_at": "2026-07-05T10:12:00Z",
  "source_session": "u42::2026-07-05",
  "access_count": 3
}
```

The store — write path embeds, read path is a vector search with a **user prefilter** (memory recall must never cross users):

```python
import couchbase.search as search
from couchbase.options import SearchOptions
from couchbase.vector_search import VectorQuery, VectorSearch

class MemoryStore:
    def __init__(self, cluster, embed_fn):
        self.scope = cluster.bucket("ai").scope("agent")
        self.coll = self.scope.collection("memories")
        self.embed = embed_fn

    def remember(self, user_id: str, text: str, kind: str = "fact",
                 importance: float = 0.5):
        key = f"memory::{user_id}::{uuid4().hex[:12]}"
        self.coll.upsert(key, {
            "type": "memory", "user_id": user_id, "text": text,
            "kind": kind, "importance": importance,
            "embedding": self.embed(text),
            "created_at": now_iso(), "access_count": 0,
        })
        return key

    def recall(self, user_id: str, query: str, k: int = 5) -> list[dict]:
        req = search.SearchRequest.create(
            VectorSearch.from_vector_query(VectorQuery(
                "embedding", self.embed(query), num_candidates=k * 3,
                prefilter=search.MatchQuery(user_id, field="user_id"),
            )))
        result = self.scope.search("memories-vector-index", req,
                                   SearchOptions(limit=k, fields=["text", "kind", "importance"]))
        return [{"id": r.id, "score": r.score, **r.fields} for r in result.rows()]
```

(The index is the Chapter 5 definition pointed at `agent.memories`, with `user_id` indexed as a text field for the prefilter.)

## 9.4 Writing memories: extraction

Who decides what's worth remembering? Two production patterns:

**Explicit tool** — give the agent a `remember` tool and let it decide (Chapter 10 catalogs it, Chapter 11 wires it):

```python
@agentc.catalog.tool
def save_memory(user_id: str, fact: str, kind: str) -> str:
    """Save a durable fact about the user for future conversations.
    Use when the user states a lasting preference, constraint, or correction."""
    return memory_store.remember(user_id, fact, kind)
```

**Background extraction** — after a session ends, an LLM pass over the transcript extracts memory-worthy facts. Cheap trigger: run it when the session document's TTL is near, or via Eventing on a `closed` flag (§3.6). Extraction prompt essentials: extract *stable* facts only, one fact per item, include a confidence score → `importance`.

Dedup on write, or memory fills with near-copies of "user prefers Python": before upserting, `recall(user_id, new_fact, k=1)` and skip/merge if similarity exceeds a threshold.

## 9.5 Summarization: compressing STM into LTM

When a session exceeds its window, fold it into a summary memory:

```python
def summarize_session(session_id: str, user_id: str):
    turns = session_store.recent_turns(session_id, n=10_000)
    transcript = "\n".join(f"{t['role']}: {t['content']}" for t in turns)
    summary = llm.invoke(
        f"Summarize durable facts and open tasks from this conversation, "
        f"3 bullets max:\n\n{transcript}").content
    memory_store.remember(user_id, summary, kind="episode_summary", importance=0.6)
```

The agent's prompt then assembles memory tiers explicitly — recent turns verbatim, older context as summaries, LTM by relevance:

```
System: ...instructions...
Relevant memories:      ← memory_store.recall(user_id, current_message, k=5)
Conversation summary:   ← episode summaries
Recent turns:           ← session_store.recent_turns(session_id, 10)
User: <current message>
```

That assembly *is* context engineering: three queries against one database.

## 9.6 Forgetting, aging, and hygiene

- **Relevance decay**: rank recall by `score × importance × recency`. Store `access_count`/`last_accessed` (a `mutate_in` counter on read) — memories never recalled are candidates for expiry.
- **Corrections**: when the user contradicts a memory ("actually I moved teams"), *replace*, don't accumulate: recall nearest, remove it, remember the correction.
- **Right to be forgotten**: memory-per-document makes GDPR deletion a query — `DELETE FROM ai.agent.memories m WHERE m.user_id = $u` — plus session removal. Auditable with SQL++. This is dramatically harder when memory lives inside opaque framework state.
- **Never store secrets** in memory text: it gets injected into future prompts.

## 9.7 Recap

Memory = four collections and two access patterns you already know (KV/subdoc from Ch. 2, vector search from Ch. 5). LangGraph will consume all of this in Chapter 11: `SessionStore`+`MemoryStore` as tools/context, checkpoints via `CouchbaseSaver`.

Notebook: [`notebooks/06_agent_memory.ipynb`](../notebooks/06_agent_memory.ipynb).

Next: [Chapter 10 — Agent Catalog](10-agent-catalog.md).
