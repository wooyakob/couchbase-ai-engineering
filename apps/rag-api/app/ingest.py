"""Ingestion: chunk -> embed -> upsert with lineage (Chapters 3-4, DIY path).

On Capella you can replace this whole module with a Vectorization workflow (Ch. 4) —
the retrieval side doesn't care who produced the vectors.
"""

import hashlib
import re

from couchbase.exceptions import DocumentNotFoundException

from . import config, db
from .models import make_embeddings

_embeddings = None


def embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = make_embeddings()
    return _embeddings


def chunk_markdown(text: str) -> list[str]:
    """Heading-aware, size-capped chunking (Ch. 3 §3.3)."""
    pieces: list[str] = []
    for section in re.split(r"(?m)^(?=#{1,3} )", text):
        section = section.strip()
        if not section:
            continue
        start = 0
        while start < len(section):
            pieces.append(section[start:start + config.CHUNK_MAX_CHARS])
            start += config.CHUNK_MAX_CHARS - config.CHUNK_OVERLAP
    return pieces


def ingest_document(doc_id: str, text: str, metadata: dict) -> dict:
    coll = db.bucket().scope(config.DOCS_SCOPE).collection(config.CHUNKS_COLLECTION)
    pieces = chunk_markdown(text)
    vectors = embeddings().embed_documents(pieces)  # batched

    written = skipped = 0
    for i, (piece, vec) in enumerate(zip(pieces, vectors)):
        key = f"chunk::{doc_id}::{i:04d}"
        content_hash = hashlib.sha256(piece.encode()).hexdigest()[:16]
        try:
            existing = coll.get(key).content_as[dict]
            if existing.get("lineage", {}).get("content_hash") == content_hash:
                skipped += 1
                continue
        except DocumentNotFoundException:
            pass
        coll.upsert(key, {
            "type": "chunk",
            "text": piece,
            "embedding": vec,
            "embedding_model": config.EMBEDDING_MODEL,
            "metadata": metadata,
            "lineage": {"doc_id": doc_id, "chunk_index": i,
                        "content_hash": content_hash,
                        "pipeline_version": config.PIPELINE_VERSION},
        })
        written += 1
    return {"doc_id": doc_id, "chunks_written": written, "chunks_unchanged": skipped}


def delete_document(doc_id: str) -> int:
    """Remove a source's chunks so retrieval never serves ghosts (Ch. 3 §3.4)."""
    from couchbase.options import QueryOptions

    rows = db.cluster().query(
        f"DELETE FROM `{config.CB_BUCKET}`.{config.DOCS_SCOPE}.{config.CHUNKS_COLLECTION} c "
        "WHERE c.lineage.doc_id = $doc_id RETURNING META(c).id",
        QueryOptions(named_parameters={"doc_id": doc_id}),
    )
    return len(list(rows))
