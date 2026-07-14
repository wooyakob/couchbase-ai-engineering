"""Unit tests for agent/memory.py — pure logic (_block_to_dict) and the OpenAI
<-> Capella switch for embed_one (Ch. 8), with mocked collaborators so nothing
here needs a live cluster, Agent Memory server, or real API key.
"""

import importlib

import httpx
import pytest
from agentmemory.exceptions import AgentMemoryError
from agentmemory.models import ChatMessage, MemoryBlock, MemoryBlockStatus

import agent.memory as memory_module
from agent.memory import _block_to_dict


# ── _block_to_dict ───────────────────────────────────────────────────────────

def _block(**overrides) -> MemoryBlock:
    defaults = dict(block_id="mem::1", user_id="u42", session_id="profile",
                    ingested_at="2026-01-01T00:00:00Z", rel_score=0.83,
                    annotations={"kind": "preference"})
    defaults.update(overrides)
    return MemoryBlock(**defaults)


def test_block_to_dict_prefers_summary():
    block = _block(summary="user prefers Python", fact="raw extracted fact")
    result = _block_to_dict(block)
    assert result == {"id": "mem::1", "text": "user prefers Python",
                      "kind": "preference", "score": 0.83}


def test_block_to_dict_falls_back_to_fact_without_summary():
    block = _block(fact="user works on payments")
    assert _block_to_dict(block)["text"] == "user works on payments"


def test_block_to_dict_falls_back_to_message_content():
    block = _block(message=ChatMessage(user_content="how do I rotate creds?",
                                       assistant_content="open settings..."))
    result = _block_to_dict(block)
    assert result["text"] == "how do I rotate creds? / open settings..."


def test_block_to_dict_defaults_kind_to_fact_without_annotations():
    block = _block(summary="something", annotations=None)
    assert _block_to_dict(block)["kind"] == "fact"


def test_block_to_dict_rounds_score_and_defaults_to_zero():
    block = _block(summary="x", rel_score=None)
    assert _block_to_dict(block)["score"] == 0.0

    block = _block(summary="x", rel_score=0.123456)
    assert _block_to_dict(block)["score"] == 0.1235


# ── EMBEDDING_MODEL / embed_one: OpenAI <-> Capella switch ──────────────────

@pytest.fixture(autouse=True)
def _isolated_memory_env(monkeypatch):
    """Same rationale as apps/rag-api/tests/test_config_unit.py: nothing in
    agent/memory.py calls load_dotenv() itself (only the entry points do), so no
    re-sourcing hazard here — but we still isolate the relevant env vars per test
    and always leave the module reloaded to a real, importable state afterward so
    other test files (and evals/test_agent.py, if run in the same session) don't
    see a stale reload from whatever the last test in this file set up."""
    for var in ("CAPELLA_AI_ENDPOINT", "CAPELLA_AI_TOKEN", "CAPELLA_EMBEDDING_MODEL",
               "EMBEDDING_MODEL"):
        monkeypatch.delenv(var, raising=False)
    yield
    monkeypatch.delenv("CAPELLA_AI_ENDPOINT", raising=False)
    importlib.reload(memory_module)


def test_embedding_model_defaults_to_openai(monkeypatch):
    mod = importlib.reload(memory_module)
    assert mod.EMBEDDING_MODEL == "text-embedding-3-small"


def test_embedding_model_switches_to_capella(monkeypatch):
    monkeypatch.setenv("CAPELLA_AI_ENDPOINT", "https://example.ai.cloud.couchbase.com/v1")
    mod = importlib.reload(memory_module)
    assert mod.EMBEDDING_MODEL == "intfloat/e5-mistral-7b-instruct"


def test_embedding_model_capella_override(monkeypatch):
    monkeypatch.setenv("CAPELLA_AI_ENDPOINT", "https://example.ai.cloud.couchbase.com/v1")
    monkeypatch.setenv("CAPELLA_EMBEDDING_MODEL", "nvidia/llama-3.2-nv-embedqa-1b-v2")
    mod = importlib.reload(memory_module)
    assert mod.EMBEDDING_MODEL == "nvidia/llama-3.2-nv-embedqa-1b-v2"


def test_embed_one_constructs_capella_client_with_base_url(monkeypatch):
    monkeypatch.setenv("CAPELLA_AI_ENDPOINT", "https://example.ai.cloud.couchbase.com/v1")
    monkeypatch.setenv("CAPELLA_AI_TOKEN", "cbsk-test-token")
    mod = importlib.reload(memory_module)

    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.embeddings = self

        def create(self, model, input):
            class Resp:
                data = [type("D", (), {"embedding": [0.1, 0.2]})()]
            return Resp()

    monkeypatch.setattr(mod, "OpenAI", FakeOpenAI)
    mod.embed_one("probe text")
    assert captured["base_url"] == "https://example.ai.cloud.couchbase.com/v1"
    assert captured["api_key"] == "cbsk-test-token"


# ── recall(): graceful degradation on a down/unreachable Agent Memory server ─

def test_recall_degrades_on_agent_memory_error(monkeypatch):
    class FailingAgentMemory:
        def __init__(self, user_id):
            raise AgentMemoryError("server offline")

    monkeypatch.setattr(memory_module, "AgentMemory", FailingAgentMemory)
    assert memory_module.recall("u42", "any query") == []


def test_recall_degrades_on_httpx_error(monkeypatch):
    class FailingAgentMemory:
        def __init__(self, user_id):
            raise httpx.RemoteProtocolError("server disconnected")

    monkeypatch.setattr(memory_module, "AgentMemory", FailingAgentMemory)
    assert memory_module.recall("u42", "any query") == []


def test_recall_returns_blocks_on_success(monkeypatch):
    class FakeAgentMemory:
        """recall() (the module function) just delegates to AgentMemory.recall(),
        which already returns _block_to_dict-converted dicts, not raw MemoryBlocks
        — mirror that here rather than the module function doing the conversion."""

        def __init__(self, user_id):
            self.user_id = user_id

        def recall(self, query, k=5):
            return [_block_to_dict(_block(summary="prefers Python"))]

    monkeypatch.setattr(memory_module, "AgentMemory", FakeAgentMemory)
    result = memory_module.recall("u42", "what language?")
    assert result == [{"id": "mem::1", "text": "prefers Python",
                       "kind": "preference", "score": 0.83}]
