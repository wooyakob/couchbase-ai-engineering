"""Unit tests for tools/support_tools.py — the cataloged agent tools — with
mocked Couchbase/Agent Memory collaborators, no live services needed.

Each tool does its imports lazily inside the function body (agentc indexes this
file directly, so import-time side effects are kept out — see the module
docstring), which means monkeypatching agent.memory's module-level names before
calling a tool is what actually takes effect, since the local `from agent.memory
import ...` re-reads the current binding at call time.
"""

from unittest.mock import MagicMock

import agent.memory as memory_module
from agentmemory.exceptions import AgentMemoryError
from couchbase.exceptions import DocumentNotFoundException

from tools.support_tools import lookup_order, save_memory, search_docs


# ── lookup_order ─────────────────────────────────────────────────────────────

def test_lookup_order_found(monkeypatch):
    fake_doc = {"id": 1042, "status": "shipped", "eta": "2026-07-08"}
    fake_result = MagicMock()
    fake_result.content_as = {dict: fake_doc}

    fake_collection = MagicMock()
    fake_collection.get.return_value = fake_result
    fake_cluster = MagicMock()
    fake_cluster.bucket.return_value.scope.return_value.collection.return_value = fake_collection
    monkeypatch.setattr(memory_module, "cluster", lambda: fake_cluster)

    result = lookup_order("1042")

    assert result == fake_doc
    fake_collection.get.assert_called_once_with("order::1042")


def test_lookup_order_not_found(monkeypatch):
    fake_collection = MagicMock()
    fake_collection.get.side_effect = DocumentNotFoundException("no such order")
    fake_cluster = MagicMock()
    fake_cluster.bucket.return_value.scope.return_value.collection.return_value = fake_collection
    monkeypatch.setattr(memory_module, "cluster", lambda: fake_cluster)

    result = lookup_order("999999")

    assert result == {"error": "no order with id 999999"}


# ── search_docs ──────────────────────────────────────────────────────────────

class FakeSearchRow:
    def __init__(self, fields, score):
        self.fields = fields
        self.score = score


class FakeSearchResult:
    def __init__(self, rows):
        self._rows = rows

    def rows(self):
        return iter(self._rows)


def test_search_docs_maps_results(monkeypatch):
    monkeypatch.setattr(memory_module, "embed_one", lambda text: [0.1, 0.2, 0.3])

    fake_scope = MagicMock()
    fake_scope.search.return_value = FakeSearchResult([
        FakeSearchRow({"text": "rotate creds via Settings", "metadata.source": "guide"}, 0.91),
        FakeSearchRow(None, 0.2),
    ])
    fake_cluster = MagicMock()
    fake_cluster.bucket.return_value.scope.return_value = fake_scope
    monkeypatch.setattr(memory_module, "cluster", lambda: fake_cluster)

    result = search_docs("how do I rotate credentials?")

    assert result == [
        {"text": "rotate creds via Settings", "source": "guide", "score": 0.91},
        {"text": "", "source": None, "score": 0.2},
    ]


# ── save_memory ──────────────────────────────────────────────────────────────

def test_save_memory_success(monkeypatch):
    class FakeAgentMemory:
        def __init__(self, user_id):
            self.user_id = user_id

        def remember(self, fact, kind="fact"):
            return "mem::new-id"

    monkeypatch.setattr(memory_module, "AgentMemory", FakeAgentMemory)

    result = save_memory("u42", "prefers email updates", kind="preference")
    assert result == "mem::new-id"


def test_save_memory_degrades_when_server_offline(monkeypatch):
    class FailingAgentMemory:
        def __init__(self, user_id):
            pass

        def remember(self, fact, kind="fact"):
            raise AgentMemoryError("server offline")

    monkeypatch.setattr(memory_module, "AgentMemory", FailingAgentMemory)

    result = save_memory("u42", "prefers email updates")
    assert result.startswith("could not save memory:")
