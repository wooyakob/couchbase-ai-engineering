# Chapter 3 — Data Processing for AI

> *Models don't eat databases; they eat well-shaped context. Data processing is the unglamorous work of turning raw documents into retrieval-ready units — cleaning, structuring, chunking, and keeping it all fresh as sources change. Do it badly and no amount of model quality saves you.*

## 3.1 The AI data pipeline

Every retrieval system runs some version of this pipeline:

```
raw sources ──► extract ──► clean/structure ──► chunk ──► embed ──► index
 (PDF, HTML,     (text)      (JSON + metadata)   (units)   (Ch.4)   (Ch.5)
  tickets, S3)
```

On Couchbase you have three ways to run it, and mature systems use all three:

1. **Capella's managed pipeline** — the AI Services **Data Processing + Vectorization workflows** handle extract → chunk → embed → index for you (Chapter 4 covers configuration; this chapter covers the *decisions* — especially chunking — that you still own).
2. **Your own batch pipeline** — Python + the SDK. Full control; the pattern below.
3. **Eventing** — in-database triggers that keep derived data fresh on every mutation.

## 3.2 Documents in, JSON out

Whatever the source, the target shape is a JSON document per retrieval unit with three parts: **content** (what gets embedded), **metadata** (what gets filtered), and **lineage** (where it came from):

```json
{
  "type": "chunk",
  "text": "To rotate credentials in Capella, open Settings → ...",
  "metadata": {
    "source": "docs-manual",
    "title": "Credential rotation",
    "url": "https://docs.example.com/security/rotation",
    "product": "capella",
    "updated_at": "2026-06-30"
  },
  "lineage": {
    "doc_id": "docs-manual::security-rotation",
    "chunk_index": 3,
    "pipeline_version": "2026-07-01",
    "content_hash": "sha256:9f2c..."
  },
  "embedding": [0.013, -0.082, ...],
  "embedding_model": "text-embedding-3-small"
}
```

Design rules that pay for themselves:

- **`content_hash`** makes the pipeline idempotent: re-running on unchanged input produces the same key and skips the (expensive) embedding call.
- **`metadata` fields are your retrieval filters** (Chapter 5's prefilters and Chapter 6's tenant scoping). If you'll ever need "only product = capella", capture it now.
- **`pipeline_version` + `embedding_model`** make re-processing auditable — you can SQL++ your way to "which chunks were built by the old chunker?"

## 3.3 Chunking: the decision that matters most

Chunking sets the resolution of retrieval. Too big → chunks match everything vaguely and blow the context budget; too small → answers get shredded across chunks. The strategies (these are also exactly the options Capella's vectorization workflows offer — see [the docs](https://docs.couchbase.com/ai/build/vectorization-service/data-processing.html)):

| Strategy | How it splits | Use when |
|---|---|---|
| **Fixed-size** | Every N tokens/chars (+ overlap) | Uniform processing of heterogeneous corpora; simple and predictable, but breaks semantic units |
| **Sentence** | Sentence boundaries | Short factual content; preserves complete thoughts |
| **Paragraph** | Paragraph boundaries | Well-authored docs where paragraphs are self-contained |
| **Semantic** | Embedding-similarity breakpoints — split where topic shifts | Default for quality (and Capella's default); costs embedding calls at processing time |

A practical structure-aware chunker (markdown-heading first, size-capped second — often beats pure semantic chunking on technical docs):

```python
import hashlib, re

def chunk_markdown(doc_id: str, text: str, max_chars: int = 2000, overlap: int = 200):
    sections = re.split(r"(?m)^(?=#{1,3} )", text)          # split at headings
    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        start = 0
        while start < len(section):                          # size-cap within a section
            piece = section[start:start + max_chars]
            chunks.append(piece)
            start += max_chars - overlap
    return [
        {
            "type": "chunk",
            "text": piece,
            "lineage": {
                "doc_id": doc_id,
                "chunk_index": i,
                "content_hash": hashlib.sha256(piece.encode()).hexdigest()[:16],
            },
        }
        for i, piece in enumerate(chunks)
    ]
```

Overlap (10–20%) hedges against answers straddling a boundary. Whatever you choose, **store the chunker's parameters in `pipeline_version`** and re-evaluate with Ragas when you change them (Chapter 13 — `context_precision` and `context_recall` are chunking-quality meters).

## 3.4 The batch pipeline with the SDK

Idempotent upsert loop — the skeleton every custom pipeline shares:

```python
from couchbase.options import UpsertOptions

chunks_coll = cluster.bucket("ai").scope("docs").collection("chunks")

def ingest(doc_id: str, text: str, metadata: dict):
    for chunk in chunk_markdown(doc_id, text):
        key = f"chunk::{doc_id}::{chunk['lineage']['chunk_index']:04d}"
        try:
            existing = chunks_coll.get(key).content_as[dict]
            if existing["lineage"]["content_hash"] == chunk["lineage"]["content_hash"]:
                continue                       # unchanged — skip re-embedding
        except DocumentNotFoundException:
            pass
        chunk["metadata"] = metadata
        chunk["embedding"] = embed(chunk["text"])     # Ch. 4: or let Capella do this
        chunk["embedding_model"] = EMBEDDING_MODEL
        chunks_coll.upsert(key, chunk)
```

Deletion is the half everyone forgets: when a source document is removed, its chunks must go too, or retrieval serves ghosts. Lineage makes it a query:

```sql
DELETE FROM ai.docs.chunks c WHERE c.lineage.doc_id = $doc_id;
```

## 3.5 Enrichment with AI Functions

Structuring is also an AI task now. Chapter 7's AI Functions run enrichment *inside* the database — classify, extract, summarize as a SQL++ `UPDATE`:

```sql
UPDATE ai.support.tickets AS t
SET t.entities = default:ai_extraction({
        "text": t.body, "labels": ["product", "version", "error_code"]
    }),
    t.language = default:ai_translation({"text": SUBSTR(t.body, 0, 200),
                                         "detect_only": true})
WHERE t.entities IS MISSING
LIMIT 500;
```

This "SELECT-batch, enrich, UPDATE-back" loop (run from a scheduler until no rows remain) is the in-database twin of the SDK pipeline above. Extracted entities land in `metadata`, where they become retrieval filters — enrichment directly buys retrieval precision.

## 3.6 Keeping data fresh: Eventing

Batch pipelines drift; **Eventing** doesn't. The Eventing service runs JavaScript functions on document mutations — the database's own change-triggered compute. The canonical AI use: mark chunks stale the moment their source changes, and let a small worker re-embed only what's flagged.

```javascript
// Eventing function on ai.docs.sources: flag downstream chunks on change
function OnUpdate(doc, meta) {
    var q = N1QL("UPDATE ai.docs.chunks SET needs_reembed = true " +
                 "WHERE lineage.doc_id = $1", [meta.id]);
    q.execQuery();
}
```

```python
# tiny worker: re-embed only stale chunks
rows = cluster.query("""
    SELECT META(c).id AS k, c.text FROM ai.docs.chunks c
    WHERE c.needs_reembed = true LIMIT 100""")
for row in rows:
    chunks_coll.mutate_in(row["k"], (
        SD.upsert("embedding", embed(row["text"])),
        SD.remove("needs_reembed"),
    ))
```

The same trigger pattern powers: TTL-on-inactivity for sessions, auto-summarization queues (write a summarize task when a session closes — Chapter 9), and change-data-capture into audit collections. If you're on Capella with vectorization workflows, freshness is handled for you (Chapter 4); Eventing is how you get the same property for pipelines you own.

## 3.7 Unstructured sources

For PDFs/HTML/office docs, extraction happens before everything above. Capella's Data Processing service handles common formats natively within a workflow (S3 or Couchbase sources). In your own pipelines, pair an extractor library (e.g. `unstructured`, `pypdf`) with §3.4 — extraction quality is corpus-specific, so spot-check extracted text *before* debating chunk sizes; garbage text with beautiful chunking is still garbage.

## 3.8 Recap

- Target shape: content + metadata + lineage per chunk, idempotent by content hash.
- Chunking is a measured decision — version it, evaluate it (Ch. 13).
- Enrich with AI Functions in-database (Ch. 7); keep derived data fresh with Eventing or Capella workflows (Ch. 4).

Notebook: the ingestion half of [`notebooks/02_vector_search_fundamentals.ipynb`](../notebooks/02_vector_search_fundamentals.ipynb) implements §3.3–3.4.

Next: [Chapter 4 — The Vectorization Service](04-vectorization-service.md).
