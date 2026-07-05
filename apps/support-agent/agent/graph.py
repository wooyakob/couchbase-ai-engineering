"""The LangGraph graph (Chapter 11): catalog-driven agent node + memory context +
Couchbase checkpointing.

    START -> load_context -> support_agent --(needs_human?)--> human_review -> END
                                          \\--(else)---------> END
"""

import os
from typing import Annotated, TypedDict

import agentc
import agentc_langgraph.agent
import langchain_openai
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .memory import MemoryStore, SessionStore


class SupportState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    needs_human: bool
    is_last_step: bool
    previous_node: list | None  # agentc_langgraph handoff tracking


class SupportAgent(agentc_langgraph.agent.ReActAgent):
    """The prompt record `support_agent_node` supplies instructions, tools, and the
    output schema; the base class wires Span logging for every model call and tool
    result (Ch. 10 §10.5)."""

    def __init__(self, catalog: agentc.Catalog, span: agentc.Span):
        chat_model = langchain_openai.ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0)
        super().__init__(chat_model=chat_model, catalog=catalog, span=span,
                         prompt_name="support_agent_node")

    def _invoke(self, span, state, config):
        agent = self.create_react_agent(span)
        response = agent.invoke(input=state, config=config)
        structured = response["structured_response"]
        state["needs_human"] = structured["needs_human"]
        state["messages"].append(("assistant", structured["response"]))
        return state


def load_context(state: SupportState) -> dict:
    """First node: assemble memory tiers into a system message (Ch. 9 §9.5)."""
    memories = MemoryStore()
    user_msg = state["messages"][-1].content
    relevant = memories.recall(state["user_id"], user_msg, k=5)
    block = "\n".join(f"- {m['text']}" for m in relevant) or "(none)"
    return {"messages": [("system", f"Relevant memories about this user:\n{block}")]}


def human_review(state: SupportState) -> dict:
    # In production: create a ticket, notify a human, interrupt the graph.
    return {"messages": [("assistant",
                          "I've escalated this to a human agent — they'll follow up shortly.")]}


def build_graph(catalog: agentc.Catalog, span: agentc.Span, checkpointer=None):
    workflow = StateGraph(SupportState)
    workflow.add_node("load_context", load_context)
    workflow.add_node("support_agent", SupportAgent(catalog=catalog, span=span))
    workflow.add_node("human_review", human_review)

    workflow.add_edge(START, "load_context")
    workflow.add_edge("load_context", "support_agent")
    workflow.add_conditional_edges(
        "support_agent",
        lambda s: "human_review" if s.get("needs_human") else END,
    )
    workflow.add_edge("human_review", END)

    if checkpointer is None:
        checkpointer = make_checkpointer()
    return workflow.compile(checkpointer=checkpointer)


def make_checkpointer():
    """Durable graph state in Couchbase (Ch. 11 §11.3); in-memory fallback for tests."""
    try:
        from langgraph_checkpointer_couchbase import CouchbaseSaver

        return CouchbaseSaver.from_conn_info(
            cb_conn_str=os.getenv("CB_CONN_STRING", "couchbase://localhost"),
            cb_username=os.getenv("CB_USERNAME", "Administrator"),
            cb_password=os.getenv("CB_PASSWORD", "password"),
            bucket_name=os.getenv("CB_BUCKET", "ai"),
            scope_name="agent",
        )
    except Exception:
        return MemorySaver()
