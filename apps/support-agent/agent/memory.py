"""Memory subsystem (Chapter 9) — the managed Couchbase Agent Memory product.

The support agent reaches user memory through the `couchbase-agent-memory` SDK,
which talks to an **Agent Memory server** (a Docker container) that persists into
Couchbase/Capella for us: users -> sessions -> memory blocks, with LLM-extracted
facts, embeddings, semantic search, and TTL all handled server-side (Ch. 9 §9.8).

Mapping the app onto the server's user/session model:
  - Durable facts about a user live in a stable per-user ``profile`` session.
  - Each conversation is its own session, so past dialogs stay recallable.
  - Recall spans *all* of a user's sessions (``session_ids="all"``).

``cluster()``, ``CB_BUCKET``, and ``embed_one()`` remain here because the cataloged
tools in ``tools/support_tools.py`` use them for order lookups and doc search — the
support agent still talks to Couchbase directly for everything that isn't memory.
"""

import base64
import logging
import os
import threading
from datetime import timedelta

import httpx
from agentmemory import AgentMemoryClient, ChatMessage
from agentmemory.exceptions import AgentMemoryError, ConflictError, NotFoundError
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.exceptions import AuthenticationException, CouchbaseException
from couchbase.options import ClusterOptions, KnownConfigProfiles
from openai import OpenAI

logger = logging.getLogger(__name__)

CB_BUCKET = os.getenv("CB_BUCKET", "ai")
PROFILE_SESSION = "profile"  # stable per-user session that holds durable facts


# ── Couchbase cluster (shared by the cataloged tools) ──────────────────────
_cluster: Cluster | None = None
_cluster_lock = threading.Lock()


def cluster() -> Cluster:
    global _cluster
    if _cluster is None:
        with _cluster_lock:  # concurrent first calls must not open duplicate connections
            if _cluster is None:
                conn = os.getenv("CB_CONN_STRING", "couchbase://localhost")
                opts = ClusterOptions(PasswordAuthenticator(
                    os.getenv("CB_USERNAME", "Administrator"),
                    os.getenv("CB_PASSWORD", "password")))
                if conn.startswith("couchbases://"):
                    opts.apply_profile(KnownConfigProfiles.WanDevelopment)
                try:
                    _cluster = Cluster.connect(conn, opts)
                    _cluster.wait_until_ready(timedelta(seconds=10))
                except AuthenticationException as e:
                    raise RuntimeError(
                        f"Couchbase rejected the configured credentials for "
                        f"{conn!r} — check CB_USERNAME/CB_PASSWORD in your .env. "
                        "See docs/troubleshooting.md."
                    ) from e
                except CouchbaseException as e:
                    raise RuntimeError(
                        f"Couldn't connect to Couchbase at {conn!r}: {e}. Check the "
                        "cluster is running/reachable and CB_CONN_STRING is correct. "
                        "See docs/troubleshooting.md."
                    ) from e
    return _cluster


# ── Embeddings (used by the search_docs tool) ──────────────────────────────
# OpenAI by default, Capella Model Service switch (Ch. 8) — same pattern as every
# other notebook/app here. search_docs queries notebook 02's corpus, so this MUST
# match whatever embedding model actually populated that corpus's vectors.
_CAPELLA_AI_ENDPOINT = os.getenv("CAPELLA_AI_ENDPOINT")
EMBEDDING_MODEL = (os.getenv("CAPELLA_EMBEDDING_MODEL", "intfloat/e5-mistral-7b-instruct")
                   if _CAPELLA_AI_ENDPOINT
                   else os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))

_ai: OpenAI | None = None


def embed_one(text: str) -> list[float]:
    # lazy init: OpenAI() raises without an API key, which would crash plain imports
    # of this module (e.g. pytest collection) before the env is configured
    global _ai
    if _ai is None:
        if _CAPELLA_AI_ENDPOINT:
            key = os.getenv("CAPELLA_AI_TOKEN") or base64.b64encode(
                f"{os.getenv('CB_USERNAME', '')}:{os.getenv('CB_PASSWORD', '')}".encode()).decode()
            _ai = OpenAI(base_url=_CAPELLA_AI_ENDPOINT, api_key=key)
        else:
            _ai = OpenAI()
    return _ai.embeddings.create(model=EMBEDDING_MODEL, input=[text]).data[0].embedding


# ── Managed Agent Memory (server + SDK) ────────────────────────────────────
_mem_client: AgentMemoryClient | None = None
_mem_lock = threading.Lock()


def memory_client() -> AgentMemoryClient:
    """Lazily-opened, reused client for the Agent Memory server. Construction is
    cheap and does not connect — the first request is what reaches the server."""
    global _mem_client
    if _mem_client is None:
        with _mem_lock:
            if _mem_client is None:
                _mem_client = AgentMemoryClient(
                    base_url=os.getenv("AGENTMEMORY_BASE_URL", "http://localhost:8080"),
                    token=os.getenv("AGENTMEMORY_TOKEN") or None,  # only if OIDC enabled
                )
    return _mem_client


def _get_or_create_user(user_id: str, name: str | None = None):
    client = memory_client()
    try:
        return client.get_user(user_id)
    except NotFoundError:
        try:
            return client.create_user(user_id, name or user_id)
        except ConflictError:  # created concurrently between get and create
            return client.get_user(user_id)


def _get_or_create_session(user, session_id: str):
    try:
        return user.get_session(session_id)
    except NotFoundError:
        try:
            return user.create_session(session_id)
        except ConflictError:
            return user.get_session(session_id)


def _block_to_dict(block) -> dict:
    """Flatten a MemoryBlock (fact or message) into the {text, kind, score} shape
    the graph's context assembly expects."""
    text = block.summary or block.fact or ""
    if not text and block.message:
        text = " / ".join(p for p in (block.message.user_content,
                                      block.message.assistant_content) if p)
    return {"id": block.block_id, "text": text,
            "kind": (block.annotations or {}).get("kind", "fact"),
            "score": round(block.rel_score or 0.0, 4)}


class AgentMemory:
    """A user's memory, reached through the Agent Memory server (Ch. 9 §9.8)."""

    def __init__(self, user_id: str, conversation_id: str | None = None,
                 name: str | None = None):
        self.user_id = user_id
        self.user = _get_or_create_user(user_id, name)
        self.profile = _get_or_create_session(self.user, PROFILE_SESSION)
        self.conversation = (_get_or_create_session(self.user, conversation_id)
                             if conversation_id else self.profile)

    def recall(self, query: str, k: int = 5) -> list[dict]:
        """Semantic recall across ALL of this user's sessions. The server embeds
        the query and ranks blocks by relevance — no index to manage (§9.8)."""
        res = self.profile.search_memory(
            query=query, filters={"session_ids": "all", "relevant_k": k})
        return [_block_to_dict(b) for b in res.memory_blocks]

    def remember(self, fact: str, kind: str = "fact") -> str:
        """Persist a durable fact to the user's profile session. ``async_processing
        =False`` blocks until it is embedded and immediately searchable."""
        resp = self.profile.add_memory(
            facts=[fact], annotations={"kind": kind}, async_processing=False)
        return resp.block_ids[0] if resp.block_ids else ""

    def add_exchange(self, user_content: str, assistant_content: str) -> None:
        """Record a completed turn as a message block on the conversation session,
        so future conversations can recall what was said in this one."""
        self.conversation.add_memory(
            messages=[ChatMessage(user_content=user_content,
                                  assistant_content=assistant_content)])

    def forget(self) -> None:
        """Right to be forgotten: delete the user and every associated session and
        memory block in one call (§9.8)."""
        memory_client().delete_user(self.user_id)


def recall(user_id: str, query: str, k: int = 5) -> list[dict]:
    """Module-level, defensive recall for context assembly: if the Agent Memory
    server is unreachable it yields no memories rather than breaking the turn
    (mirrors how the graph tolerates a not-yet-created vector index).

    Catches httpx.HTTPError alongside AgentMemoryError: a server that's down or
    crash-looping (e.g. mid-restart) drops the connection at the transport layer
    (httpx.RemoteProtocolError, ConnectError, ...) before the agentmemory client
    gets a chance to wrap it — AgentMemoryError alone doesn't cover that."""
    try:
        return AgentMemory(user_id).recall(query, k=k)
    except (AgentMemoryError, httpx.HTTPError) as e:  # server offline/unreachable -> no memories, not a crash
        # Logged, not silent: this degrades the agent (no personalization this turn)
        # without surfacing as an error, so it needs to be visible somewhere.
        logger.warning(
            "Agent Memory server unreachable for user_id=%r (%s) — recall degraded "
            "to no memories this turn. Check AGENTMEMORY_BASE_URL and that the "
            "server container is running (docs/troubleshooting.md).",
            user_id, e,
        )
        return []
