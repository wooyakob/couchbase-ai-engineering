# rag-api — a RAG service on Couchbase

The assembled system from [Chapter 6](../../docs/06-rag.md): FastAPI + Couchbase for
retrieval (hybrid vector + keyword with tenant prefilters), conversation history, and
evaluation logging — with OpenAI or the Capella Model Service behind one config switch.

```
POST   /ingest              chunk + embed + upsert (idempotent by content hash)
DELETE /documents/{doc_id}  remove a source's chunks
POST   /ask                 condense → hybrid retrieve → generate → cite
GET    /sessions/{id}       conversation transcript
GET    /healthz
```

## Setup

1. Provision the `ai` bucket and the `chunks-vector-index` — run
   [`notebooks/01`](../../notebooks/01_python_sdk_quickstart.ipynb) and
   [`notebooks/02`](../../notebooks/02_vector_search_fundamentals.ipynb), or create them
   in the Capella UI (index JSON in Chapter 5 §5.2).
2. Configure — copy the repo-root `.env.example` to `.env` here and fill in
   `CB_*` plus either `OPENAI_API_KEY` or `CAPELLA_AI_ENDPOINT`.
3. Run:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Try it

```bash
curl -s localhost:8000/ingest -H 'content-type: application/json' -d '{
  "doc_id": "guide::rotation",
  "text": "# Credential rotation\nOpen Settings → Database Access, create the new credential, deploy, then revoke the old one.",
  "metadata": {"source": "guide", "tenant": "acme"}
}'

curl -s localhost:8000/ask -H 'content-type: application/json' -d '{
  "question": "how do I rotate credentials?",
  "session_id": "demo::1",
  "tenant": "acme"
}'

curl -s localhost:8000/ask -H 'content-type: application/json' -d '{
  "question": "and what do I do right after?",
  "session_id": "demo::1",
  "tenant": "acme"
}'
```

The second call exercises query condensation — check `standalone_query` in the response.

## Where things are

| Concern | Module | Book chapter |
|---|---|---|
| Connection + provisioning | `app/db.py` | 2 |
| Chunking + lineage + idempotent ingest | `app/ingest.py` | 3–4 |
| Hybrid retrieval + prefilters | `app/retrieval.py` | 5 |
| Condense → generate → history → eval logging | `app/rag.py` | 6, 13 |
| OpenAI ⇄ Capella switch | `app/models.py`, `app/config.py` | 8 |

Every `/ask` is logged to `ai.evals.samples` — promote real traffic into eval cases and
score it with [`notebooks/08`](../../notebooks/08_ragas_evaluation.ipynb).
