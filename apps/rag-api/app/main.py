"""FastAPI RAG service on Couchbase — the assembled Chapter 6 system.

Run:  uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import db, ingest, rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.ensure_collections()
    yield


app = FastAPI(title="rag-api", description="RAG on Couchbase", lifespan=lifespan)


class IngestRequest(BaseModel):
    doc_id: str = Field(examples=["docs-manual::credential-rotation"])
    text: str
    metadata: dict = Field(default_factory=dict)


class AskRequest(BaseModel):
    question: str
    session_id: str = Field(examples=["u42::2026-07-05"])
    tenant: str | None = None


@app.post("/ingest")
def ingest_endpoint(req: IngestRequest):
    """Chunk, embed, and upsert a document (idempotent by content hash)."""
    return ingest.ingest_document(req.doc_id, req.text, req.metadata)


@app.delete("/documents/{doc_id}")
def delete_endpoint(doc_id: str):
    return {"doc_id": doc_id, "chunks_deleted": ingest.delete_document(doc_id)}


@app.post("/ask")
def ask_endpoint(req: AskRequest):
    """Conversational RAG: condense -> hybrid retrieve -> generate -> cite."""
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="empty question")
    return rag.answer(req.question, req.session_id, req.tenant)


@app.get("/sessions/{session_id}")
def session_endpoint(session_id: str):
    """The conversation transcript, straight from Couchbase."""
    history = rag.history_for(session_id)
    return {"session_id": session_id,
            "messages": [{"role": m.type, "content": m.content}
                         for m in history.messages]}


@app.get("/healthz")
def health():
    db.cluster().ping()
    return {"status": "ok"}
