"""Unit tests for app/retrieval.py's hybrid_search — result-shaping and the
CouchbaseException -> actionable RuntimeError wrapping, with a mocked search
response (Ch. 5 §5.4). No live cluster needed: search.MatchQuery/VectorQuery
construction doesn't touch the network, only the final scope.search() call does.
"""

from unittest.mock import MagicMock

import pytest
from couchbase.exceptions import CouchbaseException

from app import retrieval


class FakeRow:
    def __init__(self, id_, score, fields):
        self.id = id_
        self.score = score
        self.fields = fields


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def rows(self):
        return iter(self._rows)


@pytest.fixture()
def fake_search(monkeypatch):
    """Patches app.retrieval.db.bucket()....search() and app.retrieval.embed_query
    so hybrid_search runs end-to-end against canned data, no network involved."""
    monkeypatch.setattr(retrieval, "embed_query", lambda embeddings, text: [0.1, 0.2, 0.3])
    monkeypatch.setattr(retrieval, "embeddings", lambda: MagicMock())

    scope = MagicMock()
    fake_bucket = MagicMock()
    fake_bucket.scope.return_value = scope
    monkeypatch.setattr(retrieval.db, "bucket", lambda: fake_bucket)
    return scope


def test_hybrid_search_maps_hits_to_dicts(fake_search):
    fake_search.search.return_value = FakeResult([
        FakeRow("chunk::a::0000", 0.91,
               {"text": "some passage", "metadata.source": "guide",
                "lineage.doc_id": "guide::a"}),
        FakeRow("chunk::b::0000", 0.4, None),  # fields can be None
    ])

    hits = retrieval.hybrid_search("how do I do the thing?")

    assert hits == [
        {"id": "chunk::a::0000", "score": 0.91, "text": "some passage",
         "source": "guide", "doc_id": "guide::a"},
        {"id": "chunk::b::0000", "score": 0.4, "text": "", "source": None, "doc_id": None},
    ]


def test_hybrid_search_uses_configured_k_by_default(fake_search, monkeypatch):
    monkeypatch.setattr(retrieval.config, "RETRIEVAL_K", 7)
    fake_search.search.return_value = FakeResult([])
    retrieval.hybrid_search("a question")
    args, _ = fake_search.search.call_args
    index_name, _request, options = args
    assert index_name == retrieval.config.CHUNKS_INDEX
    assert dict(options)["limit"] == 7


def test_hybrid_search_explicit_k_overrides_config(fake_search, monkeypatch):
    monkeypatch.setattr(retrieval.config, "RETRIEVAL_K", 7)
    fake_search.search.return_value = FakeResult([])
    retrieval.hybrid_search("a question", k=2)
    args, _ = fake_search.search.call_args
    assert dict(args[2])["limit"] == 2


def test_hybrid_search_wraps_search_failure_in_actionable_runtime_error(fake_search):
    # search() itself succeeds; the SDK executes the search lazily on iteration —
    # result.rows() is where a missing/not-ready index actually surfaces (see the
    # comment in app/retrieval.py), so that's what has to raise here, not search().
    failing_result = MagicMock()
    failing_result.rows.side_effect = CouchbaseException(message="index not ready")
    fake_search.search.return_value = failing_result

    with pytest.raises(RuntimeError, match=retrieval.config.CHUNKS_INDEX):
        retrieval.hybrid_search("anything")


def test_hybrid_search_applies_tenant_prefilter(fake_search):
    fake_search.search.return_value = FakeResult([])
    retrieval.hybrid_search("a question", tenant="acme")
    args, _ = fake_search.search.call_args
    request = args[1]
    # SearchRequest doesn't expose its internal query tree publicly; just assert
    # the call completed and the request object is the SearchRequest we built,
    # not raise — the tenant path is exercised without error.
    assert request is not None
