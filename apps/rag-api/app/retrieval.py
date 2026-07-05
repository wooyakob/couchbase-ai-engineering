"""Hybrid retrieval with the raw SDK (Chapter 5 §5.4).

We drop below LangChain's retriever here on purpose: hybrid (vector + keyword) queries
with tenant prefilters need the SearchRequest API.
"""

import couchbase.search as search
from couchbase.options import SearchOptions
from couchbase.vector_search import VectorQuery, VectorSearch

from . import config, db
from .ingest import embeddings
from .models import embed_query


def hybrid_search(question: str, k: int | None = None,
                  tenant: str | None = None) -> list[dict]:
    k = k or config.RETRIEVAL_K
    qvec = embed_query(embeddings(), question)

    # Tenancy belongs in a prefilter: the ANN search itself is restricted,
    # so K isn't wasted on documents the caller may not see.
    prefilter = search.MatchQuery(tenant, field="metadata.tenant") if tenant else None

    req = search.SearchRequest.create(
        search.MatchQuery(question, field="text")  # keyword side
    ).with_vector_search(
        VectorSearch.from_vector_query(
            VectorQuery("embedding", qvec, num_candidates=k * 2,
                        boost=1.5, prefilter=prefilter))
    )
    result = db.bucket().scope(config.DOCS_SCOPE).search(
        config.CHUNKS_INDEX, req,
        SearchOptions(limit=k, fields=["text", "metadata.source", "lineage.doc_id"]),
    )
    return [
        {
            "id": row.id,
            "score": row.score,
            "text": row.fields.get("text", ""),
            "source": row.fields.get("metadata.source"),
            "doc_id": row.fields.get("lineage.doc_id"),
        }
        for row in result.rows()
    ]
