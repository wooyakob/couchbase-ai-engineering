"""Cataloged agent tools (Chapter 10 §10.2).

Indexed by `agentc index tools/` — docstrings are the searchable contracts.
Import-time side effects are kept lazy so indexing stays safe.
"""

import os
import sys

import agentc
import dotenv

dotenv.load_dotenv()

# make the agent package importable when agentc code-generates/executes tools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@agentc.catalog.tool
def lookup_order(order_id: str) -> dict:
    """Fetch a customer order by its numeric ID, including status, items, and ETA.
    Use when the user asks about order status, contents, or delivery."""
    from agent.memory import CB_BUCKET, cluster
    from couchbase.exceptions import DocumentNotFoundException

    try:
        return (cluster().bucket(CB_BUCKET).scope("shop").collection("orders")
                .get(f"order::{order_id}").content_as[dict])
    except DocumentNotFoundException:
        return {"error": f"no order with id {order_id}"}


@agentc.catalog.tool
def search_docs(question: str) -> list[dict]:
    """Search product documentation for passages relevant to a customer question.
    Use for how-to and product-behavior questions."""
    import couchbase.search as search
    from couchbase.options import SearchOptions
    from couchbase.vector_search import VectorQuery, VectorSearch

    from agent.memory import CB_BUCKET, cluster, embed_one

    scope = cluster().bucket(CB_BUCKET).scope("docs")
    req = search.SearchRequest.create(VectorSearch.from_vector_query(
        VectorQuery("embedding", embed_one(question), num_candidates=5)))
    result = scope.search("chunks-vector-index", req,
                          SearchOptions(limit=5, fields=["text", "metadata.source"]))
    return [{"text": (r.fields or {}).get("text", ""),
             "source": (r.fields or {}).get("metadata.source"),
             "score": r.score} for r in result.rows()]


@agentc.catalog.tool
def save_memory(user_id: str, fact: str, kind: str = "fact") -> str:
    """Save a durable fact about the user for future conversations.
    Use when the user states a lasting preference, constraint, or correction."""
    from agent.memory import MemoryStore

    return MemoryStore().remember(user_id, fact, kind=kind)
