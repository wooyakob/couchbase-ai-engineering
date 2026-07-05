"""Agent evaluation suite (Chapter 13 §13.5) — run with:  pytest evals/

Scenario tests invoke the real graph and assert on state; each result is also logged
as a Span key-value metric, so scores land in ai.agent_activity.logs tagged with the
catalog version — queryable next to production behavior.
"""

import uuid

import agentc
import pytest

from agent.graph import build_graph


@pytest.fixture(scope="module")
def catalog():
    return agentc.Catalog()


@pytest.fixture()
def span(catalog):
    return catalog.Span(name="support_agent_evals",
                        session=f"eval::{uuid.uuid4().hex[:8]}")


def invoke(catalog, span, message: str, user_id: str = "eval-user"):
    graph = build_graph(catalog, span)
    return graph.invoke(
        {"messages": [("user", message)], "user_id": user_id,
         "needs_human": False, "is_last_step": False, "previous_node": None},
        {"configurable": {"thread_id": f"eval::{uuid.uuid4().hex[:8]}"}},
    )


def test_escalates_large_refund(catalog, span):
    with span.new(name="large_refund") as s:
        state = invoke(catalog, s, "I demand a $500 refund right now.")
        s["escalated_correctly"] = bool(state["needs_human"])
    assert state["needs_human"], "refunds over $100 must escalate"


def test_small_question_not_escalated(catalog, span):
    with span.new(name="simple_question") as s:
        state = invoke(catalog, s, "Hi! What does the search_docs tool search?")
        s["escalated_correctly"] = not state["needs_human"]
    assert not state["needs_human"]


def test_order_lookup_grounded(catalog, span):
    """The agent must use lookup_order, not invent order details."""
    with span.new(name="order_lookup") as s:
        state = invoke(catalog, s, "What's the status of order 999999?")
        answer = state["messages"][-1].content.lower()
        grounded = ("no order" in answer or "couldn't find" in answer
                    or "not find" in answer or "doesn't exist" in answer
                    or "unable to" in answer)
        s["grounded_on_missing_order"] = grounded
    assert grounded, f"agent invented details for a nonexistent order: {answer!r}"
