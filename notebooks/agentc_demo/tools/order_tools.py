import os

import agentc
import couchbase.auth
import couchbase.cluster
import couchbase.options
import dotenv

dotenv.load_dotenv(dotenv.find_dotenv(os.getenv("ENV_FILE", ".env"), usecwd=True))  # ENV_FILE lets .env.server / .env.capella run side by side

_cluster = couchbase.cluster.Cluster(
    os.getenv("CB_CONN_STRING"),
    couchbase.options.ClusterOptions(
        couchbase.auth.PasswordAuthenticator(
            os.getenv("CB_USERNAME"),
            os.getenv("CB_PASSWORD"))))


@agentc.catalog.tool
def lookup_order(order_id: str) -> dict:
    """Fetch a customer order by its numeric ID.
    Use when the user asks about order status, contents, or delivery."""
    bucket = _cluster.bucket(os.getenv("CB_BUCKET"))
    return bucket.scope("shop").collection("orders").get(f"order::{order_id}").content_as[dict]


@agentc.catalog.tool
def save_memory(user_id: str, fact: str) -> str:
    """Save a durable fact about the user for future conversations.
    Use when the user states a lasting preference, constraint, or correction."""
    # in a real app this calls MemoryStore.remember (notebook 06)
    from uuid import uuid4
    key = f"memory::{user_id}::{uuid4().hex[:12]}"
    bucket = _cluster.bucket(os.getenv("CB_BUCKET"))
    bucket.scope("agent").collection("memories").upsert(
        key, {"type": "memory", "user_id": user_id, "text": fact, "kind": "fact"})
    return key