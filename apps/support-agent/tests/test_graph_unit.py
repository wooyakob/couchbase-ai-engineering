"""Unit tests for agent/graph.py — pure logic + mocked collaborators, no live
catalog/cluster/LLM needed. Two things this file exists specifically to pin down
as regressions, both root-caused earlier this session:

  1. _make_chat_model() must NEVER switch to Capella, even if CAPELLA_AI_ENDPOINT
     is set — the models available there don't support tool calling, which this
     ReAct agent requires.
  2. fold_memory_into_last_message() must fold recalled memories into the human
     turn's content, never append a second system message (breaks the Capella
     backend's strict role-alternation check) — and must never mutate the
     caller's original state (what gets checkpointed).
"""

from langchain_core.messages import HumanMessage

from agent.graph import (SupportState, _make_chat_model,
                         fold_memory_into_last_message, human_review)


def _base_state(**overrides) -> SupportState:
    state: SupportState = {
        "messages": [HumanMessage(content="where is order 1042?")],
        "user_id": "u42", "needs_human": False, "is_last_step": False,
        "previous_node": None, "memory_context": "",
    }
    state.update(overrides)
    return state


# ── _make_chat_model: always OpenAI, regardless of Capella config ───────────

def test_chat_model_ignores_capella_endpoint(monkeypatch):
    monkeypatch.setenv("CAPELLA_AI_ENDPOINT", "https://example.ai.cloud.couchbase.com/v1")
    monkeypatch.setenv("CAPELLA_AI_TOKEN", "cbsk-test-token")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    model = _make_chat_model()
    assert model.openai_api_base is None
    assert model.model_name == "gpt-4o-mini"


def test_chat_model_respects_llm_model_override(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")
    model = _make_chat_model()
    assert model.model_name == "gpt-4o"
    assert model.openai_api_base is None


# ── fold_memory_into_last_message ───────────────────────────────────────────

def test_fold_memory_no_context_is_noop():
    state = _base_state(memory_context="")
    result = fold_memory_into_last_message(state)
    assert result is state  # true no-op, not just an equal copy


def test_fold_memory_folds_into_last_human_message():
    state = _base_state(memory_context="- prefers email updates\n- works on payments")
    result = fold_memory_into_last_message(state)

    # the persisted state (what gets checkpointed) is untouched
    assert state["messages"][0].content == "where is order 1042?"

    folded = result["messages"][-1]
    assert isinstance(folded, HumanMessage)
    assert "prefers email updates" in folded.content
    assert "where is order 1042?" in folded.content
    # memory context comes first, then the real question
    assert folded.content.index("prefers email") < folded.content.index("where is order")


def test_fold_memory_does_not_mutate_original_messages_list():
    original_messages = [HumanMessage(content="hello")]
    state = _base_state(messages=original_messages, memory_context="some fact")
    fold_memory_into_last_message(state)
    assert len(original_messages) == 1
    assert original_messages[0].content == "hello"


def test_fold_memory_noop_with_no_messages():
    state = _base_state(messages=[], memory_context="some fact")
    result = fold_memory_into_last_message(state)
    assert result is state


# ── human_review ─────────────────────────────────────────────────────────

def test_human_review_appends_escalation_message():
    result = human_review(_base_state())
    assert len(result["messages"]) == 1
    role, content = result["messages"][0]
    assert role == "assistant"
    assert "escalated" in content.lower()
