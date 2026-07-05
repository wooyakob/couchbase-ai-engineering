"""Memory subsystem (Chapter 9): session store (STM) + memory store (LTM)."""

import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import couchbase.search as search
import couchbase.subdocument as SD
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.exceptions import DocumentNotFoundException
from couchbase.options import (ClusterOptions, KnownConfigProfiles,
                               SearchOptions, UpsertOptions)
from couchbase.vector_search import VectorQuery, VectorSearch
from openai import OpenAI

CB_BUCKET = os.getenv("CB_BUCKET", "ai")
MEM_INDEX = "memories-vector-index"

_cluster: Cluster | None = None


def cluster() -> Cluster:
    global _cluster
    if _cluster is None:
        conn = os.getenv("CB_CONN_STRING", "couchbase://localhost")
        opts = ClusterOptions(PasswordAuthenticator(
            os.getenv("CB_USERNAME", "Administrator"),
            os.getenv("CB_PASSWORD", "password")))
        if conn.startswith("couchbases://"):
            opts.apply_profile(KnownConfigProfiles.WanDevelopment)
        _cluster = Cluster.connect(conn, opts)
        _cluster.wait_until_ready(timedelta(seconds=10))
    return _cluster


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_ai = OpenAI()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def embed_one(text: str) -> list[float]:
    return _ai.embeddings.create(model=EMBEDDING_MODEL, input=[text]).data[0].embedding


class SessionStore:
    """STM: one document per session, subdoc appends, sliding TTL (Ch. 9 §9.2)."""

    def __init__(self, ttl: timedelta = timedelta(hours=24)):
        self.coll = cluster().bucket(CB_BUCKET).scope("agent").collection("sessions")
        self.ttl = ttl

    def append_turn(self, session_id: str, role: str, content: str) -> None:
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
        self.coll.touch(key, self.ttl)

    def recent_turns(self, session_id: str, n: int = 10) -> list[dict]:
        try:
            return self.coll.get(f"session::{session_id}").content_as[dict]["turns"][-n:]
        except DocumentNotFoundException:
            return []


class MemoryStore:
    """LTM: embedded facts, vector recall with a user prefilter (Ch. 9 §9.3)."""

    def __init__(self):
        self.scope = cluster().bucket(CB_BUCKET).scope("agent")
        self.coll = self.scope.collection("memories")

    def remember(self, user_id: str, text: str, kind: str = "fact",
                 importance: float = 0.5) -> str:
        existing = self.recall(user_id, text, k=1)
        if existing and existing[0]["score"] >= 0.9:  # dedup-on-write (§9.4)
            return existing[0]["id"]
        key = f"memory::{user_id}::{uuid4().hex[:12]}"
        self.coll.upsert(key, {
            "type": "memory", "user_id": user_id, "text": text, "kind": kind,
            "importance": importance, "embedding": embed_one(text),
            "created_at": now_iso(), "access_count": 0,
        })
        return key

    def recall(self, user_id: str, query: str, k: int = 5) -> list[dict]:
        req = search.SearchRequest.create(VectorSearch.from_vector_query(VectorQuery(
            "embedding", embed_one(query), num_candidates=k * 3,
            prefilter=search.MatchQuery(user_id, field="user_id"),
        )))
        try:
            result = self.scope.search(MEM_INDEX, req,
                                       SearchOptions(limit=k, fields=["text", "kind"]))
        except Exception:  # index not created yet -> no memories, not a crash
            return []
        return [{"id": r.id, "score": round(r.score, 4), **r.fields}
                for r in result.rows()]
