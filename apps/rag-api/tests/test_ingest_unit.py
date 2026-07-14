"""Unit tests for app/ingest.py — chunking (pure) and idempotent-upsert logic
(mocked Couchbase collection + embeddings, no live cluster or LLM call needed).
"""

from unittest.mock import MagicMock

import pytest
from couchbase.exceptions import DocumentNotFoundException

from app import config, ingest


# ── chunk_markdown: pure function, no mocks needed ──────────────────────────

def test_chunk_markdown_splits_on_headings():
    text = ("# Title\nintro text\n"
            "## Section one\nfirst section body\n"
            "### Subsection\nnested body\n"
            "## Section two\nsecond section body")
    pieces = ingest.chunk_markdown(text)
    # every heading level 1-3 starts a new section (Ch. 3 §3.3)
    assert len(pieces) == 4
    assert pieces[0].startswith("# Title")
    assert pieces[1].startswith("## Section one")
    assert pieces[2].startswith("### Subsection")
    assert pieces[3].startswith("## Section two")


def test_chunk_markdown_no_headings_is_one_section():
    text = "just plain text with no markdown headings at all"
    pieces = ingest.chunk_markdown(text)
    assert pieces == [text]


def test_chunk_markdown_empty_text():
    assert ingest.chunk_markdown("") == []


def test_chunk_markdown_respects_max_chars_and_overlap(monkeypatch):
    monkeypatch.setattr(config, "CHUNK_MAX_CHARS", 10)
    monkeypatch.setattr(config, "CHUNK_OVERLAP", 2)
    text = "# H\n" + "abcdefghij" * 3  # 4 + 30 chars, forces multiple windows
    pieces = ingest.chunk_markdown(text)
    assert len(pieces) > 1
    for piece in pieces:
        assert len(piece) <= 10
    # consecutive pieces overlap by CHUNK_OVERLAP characters
    assert pieces[0][-2:] == pieces[1][:2]


def test_chunk_markdown_stays_stable_with_real_config():
    """Sanity check against the real repo config, not just a monkeypatched one."""
    text = "# Heading\n" + ("word " * 1000)
    pieces = ingest.chunk_markdown(text)
    assert all(len(p) <= config.CHUNK_MAX_CHARS for p in pieces)
    assert len(pieces) >= 1


# ── ingest_document: idempotent upsert, mocked collection + embeddings ──────

class FakeCollection:
    """Minimal stand-in for a couchbase Collection: a plain dict-backed store."""

    def __init__(self):
        self.docs = {}

    def get(self, key):
        if key not in self.docs:
            raise DocumentNotFoundException(f"no such doc: {key}")
        result = MagicMock()
        result.content_as = {dict: self.docs[key]}
        return result

    def upsert(self, key, value):
        self.docs[key] = value


@pytest.fixture()
def fake_collection(monkeypatch):
    coll = FakeCollection()
    fake_bucket = MagicMock()
    fake_bucket.scope.return_value.collection.return_value = coll
    monkeypatch.setattr(ingest.db, "bucket", lambda: fake_bucket)
    return coll


@pytest.fixture()
def fake_embeddings(monkeypatch):
    fake = MagicMock()
    fake.embed_documents.side_effect = lambda pieces: [[0.1, 0.2] for _ in pieces]
    monkeypatch.setattr(ingest, "_embeddings", fake)
    return fake


def test_ingest_document_writes_all_chunks_first_time(fake_collection, fake_embeddings):
    text = "# A\nfirst\n# B\nsecond"
    result = ingest.ingest_document("doc::1", text, {"source": "test"})
    assert result == {"doc_id": "doc::1", "chunks_written": 2, "chunks_unchanged": 0}
    assert len(fake_collection.docs) == 2
    stored = next(iter(fake_collection.docs.values()))
    assert stored["type"] == "chunk"
    assert stored["metadata"] == {"source": "test"}
    assert stored["lineage"]["doc_id"] == "doc::1"


def test_ingest_document_is_idempotent_on_unchanged_content(fake_collection, fake_embeddings):
    text = "# A\nfirst\n# B\nsecond"
    ingest.ingest_document("doc::1", text, {})
    fake_embeddings.embed_documents.reset_mock()

    result = ingest.ingest_document("doc::1", text, {})
    assert result == {"doc_id": "doc::1", "chunks_written": 0, "chunks_unchanged": 2}
    # the whole point of content-hash lineage: unchanged content never re-embeds
    fake_embeddings.embed_documents.assert_called_once()


def test_ingest_document_rewrites_changed_chunk(fake_collection, fake_embeddings):
    ingest.ingest_document("doc::1", "# A\noriginal text", {})
    result = ingest.ingest_document("doc::1", "# A\nchanged text", {})
    assert result == {"doc_id": "doc::1", "chunks_written": 1, "chunks_unchanged": 0}
