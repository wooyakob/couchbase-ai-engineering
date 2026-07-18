"""The LangGraph graph (Chapter 11): catalog-driven agent node + memory context +
Couchbase checkpointing.

    START -> load_context -> support_agent --(needs_human?)--> human_review -> END
                                          \\--(else)---------> END
"""

import logging
import os
import warnings
from typing import Annotated, TypedDict

import agentc
import agentc_langgraph.agent
import langchain_openai
from couchbase.logic.supportability import CouchbaseDeprecationWarning
from langchain_core._api.deprecation import LangChainDeprecationWarning
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.warnings import LangGraphDeprecatedSinceV10

from .memory import recall

logger = logging.getLogger(__name__)

# All three fire from library internals we don't control, on every turn — narrowly
# silenced here (the module both main.py and evals/test_agent.py import, so the
# filters apply regardless of entry point) rather than fixed upstream:
#  - agentc_langchain's chat callback calls the now-deprecated AIMessage.text()
#    method instead of the .text property (its code, not ours).
#  - agentc_langgraph.agent.ReActAgent.create_react_agent still calls LangGraph's
#    now-relocated langgraph.prebuilt.create_react_agent (moved to
#    langchain.agents.create_agent in LangGraph v1.0) — agentc_langgraph's code,
#    not ours to change without forking it.
#  - the Couchbase SDK's scope.search() request builder unconditionally *reads*
#    the deprecated SearchOptions.scope_name property internally, regardless of
#    caller code (search_docs never sets it) — same root cause as notebooks 02/03.
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning,
                        message=".*\\.text\\(\\).*")
warnings.filterwarnings("ignore", category=LangGraphDeprecatedSinceV10,
                        message=".*create_react_agent.*")
warnings.filterwarnings("ignore", category=CouchbaseDeprecationWarning,
                        message=".*scope_name.*")


def _make_chat_model() -> langchain_openai.ChatOpenAI:
    """Always OpenAI — deliberately NOT the Capella Model Service switch every other
    notebook/app here has. This agent's ReAct loop needs real tool-calling support
    (lookup_order / search_docs / save_memory) plus strict structured-output
    compliance together; the models available on this Capella deployment don't
    support tool calling at all, so there's no working Capella option for chat here.
    Embeddings (agent/memory.py's embed_one, used by search_docs) still switch to
    Capella when CAPELLA_AI_ENDPOINT is set — that only needs embedding calls, no
    tool calling, and must match notebook 02's corpus dimensionality regardless."""
    return langchain_openai.ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0)


class SupportState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    needs_human: bool
    is_last_step: bool
    previous_node: list | None  # agentc_langgraph handoff tracking
    memory_context: str  # recalled facts, folded into the turn at generation time —
                         # NOT a messages entry (see load_context/_invoke below)


def fold_memory_into_last_message(state: "SupportState") -> dict:
    """Fold `memory_context` into the last human message's content for a single
    model call, rather than appending it as a second system message.

    create_react_agent's own `prompt` (the catalog record's instructions) is
    injected as a single LEADING system message. The Capella Model Service backend
    enforces strict role alternation after that one system message (400:
    "conversation roles must alternate") — appending memory_context as a trailing
    system message breaks that. The returned dict is a *new* state for this one
    `agent.invoke()` call; the caller's original `state["messages"]` (what gets
    checkpointed) is left untouched, so the persisted conversation stays exactly
    what the user said.

    A no-op (returns `state` itself) if there's no memory_context or no messages
    to fold into.
    """
    if not state.get("memory_context") or not state.get("messages"):
        return state
    messages = list(state["messages"])
    last = messages[-1]
    messages[-1] = HumanMessage(
        content=f"Relevant memories about this user:\n{state['memory_context']}"
                f"\n\n{last.content}")
    return {**state, "messages": messages}


class SupportAgent(agentc_langgraph.agent.ReActAgent):
    """The prompt record `support_agent_node` supplies instructions, tools, and the
    output schema; the base class wires Span logging for every model call and tool
    result (Ch. 10 §10.5)."""

    def __init__(self, catalog: agentc.Catalog, span: agentc.Span):
        chat_model = _make_chat_model()
        super().__init__(chat_model=chat_model, catalog=catalog, span=span,
                         prompt_name="support_agent_node")

    def _invoke(self, span, state, config):
        agent = self.create_react_agent(span)
        invoke_state = fold_memory_into_last_message(state)
        response = agent.invoke(input=invoke_state, config=config)
        structured = response["structured_response"]
        state["needs_human"] = structured["needs_human"]
        state["messages"].append(("assistant", structured["response"]))
        return state


def load_context(state: SupportState) -> dict:
    """First node: recall relevant memories from the Agent Memory server, held as
    plain state (not a messages entry — see SupportAgent._invoke for why) so the
    persisted conversation stays exactly what the user said (Ch. 9 §9.8). Degrades
    to no memories if the server is down."""
    user_msg = state["messages"][-1].content
    relevant = recall(state["user_id"], user_msg, k=5)
    block = "\n".join(f"- {m['text']}" for m in relevant) or "(none)"
    return {"memory_context": block}


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


# from_conn_info is a context manager (closes its own cluster connection on
# __exit__), not a saver itself — the CM object must stay referenced for the life
# of the process, or GC runs its `finally: cluster.close()` and leaves the
# checkpointer holding a closed cluster (Ch. 11 §11.3; same pattern as notebook 07).
_checkpointer_cm = None


def make_checkpointer():
    """Durable graph state in Couchbase (Ch. 11 §11.3); in-memory fallback for tests."""
    global _checkpointer_cm
    try:
        from langgraph_checkpointer_couchbase import CouchbaseSaver

        _checkpointer_cm = CouchbaseSaver.from_conn_info(
            cb_conn_str=os.getenv("CB_CONN_STRING", "couchbase://localhost"),
            cb_username=os.getenv("CB_USERNAME", "Administrator"),
            cb_password=os.getenv("CB_PASSWORD", "password"),
            bucket_name=os.getenv("CB_BUCKET", "ai"),
            scope_name="agent",
        )
        checkpointer = _checkpointer_cm.__enter__()
        # CouchbaseSaver queries both collections via SQL++ (get_tuple/put_writes) but
        # doesn't create any index itself — notebook 01 only indexes agent.memories, not
        # these, so a fresh bucket has no index to plan the query against at all.
        for coll in (checkpointer.checkpoints_collection_name,
                    checkpointer.checkpoint_writes_collection_name):
            checkpointer.cluster.query(
                f"CREATE PRIMARY INDEX IF NOT EXISTS ON "
                f"`{checkpointer.bucket_name}`.`{checkpointer.scope_name}`.`{coll}`"
            ).execute()
        return checkpointer
    except Exception as e:
        # Falling back here means graph state is NOT durable — it's lost on process
        # restart. Fine for evals/tests without a live cluster; a silent surprise in
        # production. Logged, not swallowed, so it's visible which mode you're in.
        logger.warning(
            "Couldn't set up the Couchbase checkpointer (%s) — falling back to "
            "in-memory graph state (NOT durable across restarts). Check "
            "CB_CONN_STRING/CB_USERNAME/CB_PASSWORD/CB_BUCKET if this is "
            "unexpected. See docs/troubleshooting.md.", e,
        )
        return MemorySaver()
