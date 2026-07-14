"""End-to-end smoke tests against a live Couchbase cluster + LLM backend — the
same ingest -> ask -> sources flow validated manually during development.

Skips itself (not fails) when no live backend is configured, same convention as
the repo-root tests/test_notebooks.py. Run with:

    ENV_FILE=.env.server pytest tests/test_api_smoke.py    # or .env.capella
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from tests.conftest import HAS_LIVE_CB, HAS_LIVE_LLM

pytestmark = pytest.mark.skipif(
    not (HAS_LIVE_CB and HAS_LIVE_LLM),
    reason="no live Couchbase + LLM backend configured "
           "(CB_CONN_STRING/CB_USERNAME/CB_PASSWORD/CB_BUCKET and "
           "OPENAI_API_KEY or CAPELLA_AI_ENDPOINT)",
)


@pytest.fixture(scope="module")
def client():
    from app.main import app  # imported here, not at module scope, so the
                              # skipif above can act before app.config's
                              # import-time validation ever runs
    with TestClient(app) as c:  # triggers the lifespan -> db.ensure_collections()
        yield c


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ask_rejects_empty_question(client):
    resp = client.post("/ask", json={"question": "   ", "session_id": "smoke::empty"})
    assert resp.status_code == 422


def test_ingest_ask_delete_roundtrip(client):
    doc_id = f"test::smoke::{uuid.uuid4().hex[:8]}"

    ingest_resp = client.post("/ingest", json={
        "doc_id": doc_id,
        "text": ("# Widget returns\nYou can return a Widget within 30 days of "
                 "purchase for a full refund."),
        "metadata": {"source": "smoke-test"},
    })
    assert ingest_resp.status_code == 200, ingest_resp.text
    ingest_body = ingest_resp.json()
    assert ingest_body["doc_id"] == doc_id
    assert ingest_body["chunks_written"] >= 1
    assert ingest_body["chunks_unchanged"] == 0

    # Re-ingesting identical content should be a no-op (Ch. 3-4 idempotent lineage)
    reingest_resp = client.post("/ingest", json={
        "doc_id": doc_id,
        "text": ("# Widget returns\nYou can return a Widget within 30 days of "
                 "purchase for a full refund."),
        "metadata": {"source": "smoke-test"},
    })
    assert reingest_resp.json()["chunks_written"] == 0
    assert reingest_resp.json()["chunks_unchanged"] >= 1

    session_id = f"smoke::{uuid.uuid4().hex[:8]}"
    ask_resp = client.post("/ask", json={
        "question": "How long do I have to return a widget?",
        "session_id": session_id,
    })
    assert ask_resp.status_code == 200, ask_resp.text
    ask_body = ask_resp.json()
    assert "30" in ask_body["answer"]
    assert ask_body["sources"], "expected at least one cited source"
    assert ask_body["standalone_query"]
    assert set(ask_body["latency_ms"]) == {"condense", "retrieve", "generate"}

    history_resp = client.get(f"/sessions/{session_id}")
    assert history_resp.status_code == 200
    messages = history_resp.json()["messages"]
    assert len(messages) == 2  # the question and the answer, persisted
    assert messages[0]["role"] == "human"
    assert messages[1]["role"] == "ai"

    delete_resp = client.delete(f"/documents/{doc_id}")
    assert delete_resp.status_code == 200
    assert delete_resp.json()["chunks_deleted"] >= 1

    # Deleting again should find nothing left to remove
    second_delete = client.delete(f"/documents/{doc_id}")
    assert second_delete.json()["chunks_deleted"] == 0


def test_ask_grounds_refusal_on_unanswerable_question(client):
    """Grounding check mirrored from notebook 03: a question the corpus can't
    answer should get a refusal, not an invented answer."""
    resp = client.post("/ask", json={
        "question": "What is the airspeed velocity of an unladen swallow?",
        "session_id": f"smoke::unanswerable::{uuid.uuid4().hex[:8]}",
    })
    assert resp.status_code == 200
    answer = resp.json()["answer"].lower()
    # Wording varies by model run (contractions, phrasing) — check for the
    # *concept* of a refusal rather than one exact phrase.
    assert any(phrase in answer for phrase in
              ("don't know", "do not know", "doesn't contain", "does not contain",
               "no information", "not contain")), \
        f"expected a grounded refusal, got: {answer!r}"
