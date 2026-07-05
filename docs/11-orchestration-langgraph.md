# Chapter 11 — Orchestrating Agents with LangGraph

> *Couchbase deliberately doesn't ship an orchestration framework — orchestration is application logic, and the ecosystem already has a strong answer. This book uses LangGraph. The division of labor is clean: LangGraph decides* what happens next*; Couchbase remembers* everything*.*

## 11.1 Why LangGraph

LangGraph models an agent as a **state machine**: typed state flowing through nodes (LLM calls, tools, routers) connected by edges (including conditional ones and cycles — the "loop" in *LLM in a loop*). That explicitness is what production agents need: you can draw the graph, test nodes in isolation, and — crucially for us — **persist the state**.

Everything Couchbase-side is already built: retrieval (Ch. 5–6), memory (Ch. 9), cataloged tools/prompts and audit (Ch. 10). This chapter is the assembly manual.

## 11.2 A tool-using agent in one call

`create_react_agent` gives you the standard tool-calling loop. Tools are plain decorated functions — ours wrap the SDK:

```python
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def search_docs(question: str) -> list[dict]:
    """Search product documentation for relevant passages."""
    return semantic_search(question, k=5)               # Chapter 5

@tool
def lookup_order(order_id: str) -> dict:
    """Fetch an order by its ID."""
    return orders_coll.get(f"order::{order_id}").content_as[dict]   # Chapter 2

agent = create_react_agent(
    "openai:gpt-4o-mini",              # or Capella Model Service via ChatOpenAI (Ch. 8)
    tools=[search_docs, lookup_order],
)
agent.invoke({"messages": [("user", "Where is order 1042?")]})
```

## 11.3 Durable state: the Couchbase checkpointer

By default that agent forgets everything between invocations and dies with the process mid-run. A **checkpointer** persists graph state after every step, keyed by `thread_id`. Couchbase has a dedicated one:

```bash
pip install langgraph-checkpointer-couchbase
```

```python
from langgraph_checkpointer_couchbase import CouchbaseSaver

checkpointer = CouchbaseSaver.from_conn_info(
    cb_conn_str=CB_CONN_STRING, cb_username=CB_USERNAME, cb_password=CB_PASSWORD,
    bucket_name="ai", scope_name="agent",     # uses collections: checkpoints, checkpoint_writes
)

agent = create_react_agent(model, tools=tools, checkpointer=checkpointer)

config = {"configurable": {"thread_id": "u42::support::2026-07-05"}}
agent.invoke({"messages": [("user", "Where is order 1042?")]}, config)
agent.invoke({"messages": [("user", "And when will it arrive?")]}, config)  # remembers
```

(`AsyncCouchbaseSaver` mirrors it for async apps; both also construct `from_cluster(cluster=..., ...)` to reuse your SDK connection.)

What this buys, beyond multi-turn memory: **crash recovery** (re-invoke the thread; execution resumes from the last checkpoint), **human-in-the-loop** (interrupt the graph, resume after approval — state waits in Couchbase), and **time travel** (checkpoint history per thread is queryable). Working memory has become a database concern, which is exactly where you want it.

## 11.4 Wiring in the memory subsystem

Checkpoints persist *graph* state; they are not user-level memory. The Chapter 9 stores plug in as context assembly + tools:

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END, add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str

def load_context(state: AgentState) -> dict:
    """First node: assemble memory tiers into a system message (§9.5)."""
    user_msg = state["messages"][-1].content
    memories = memory_store.recall(state["user_id"], user_msg, k=5)
    memory_block = "\n".join(f"- {m['text']}" for m in memories) or "none"
    return {"messages": [("system", f"Relevant memories about this user:\n{memory_block}")]}

@tool
def save_memory(user_id: str, fact: str) -> str:
    """Save a durable fact about the user for future conversations."""
    return memory_store.remember(user_id, fact)
```

The full graph in `apps/support-agent`:

```
START → load_context → agent (ReAct: search_docs, lookup_order, save_memory)
             │                    │
             └── CouchbaseSaver checkpoints every step ──► ai.agent.checkpoints
                                  │
                        escalate? ├──→ human_review node
                                  └──→ END
```

## 11.5 Catalog-driven nodes

Chapter 10's prompt records specify a node completely (instructions + tools + output schema). The `agentc_langgraph` integration turns the record into the node:

```python
import agentc
import agentc_langgraph.agent
import langchain_openai

class SupportAgent(agentc_langgraph.agent.ReActAgent):
    def __init__(self, catalog: agentc.Catalog, span: agentc.Span):
        chat_model = langchain_openai.ChatOpenAI(model="gpt-4o-mini", temperature=0)
        super().__init__(chat_model=chat_model, catalog=catalog, span=span,
                         prompt_name="support_agent_node")     # ← the YAML record

    def _invoke(self, span, state, config):
        agent = self.create_react_agent(span)                  # tools & schema from the prompt
        response = agent.invoke(input=state, config=config)
        structured = response["structured_response"]           # matches the prompt's output schema
        state["needs_human"] = structured["needs_human"]
        state["messages"].append(("assistant", structured["response"]))
        return state
```

What you get for free: the prompt and tools come from the *published, Git-versioned* catalog; every model call, tool result, and node transition is logged to `ai.agent_activity.logs` via the built-in Spans/callbacks. Change the YAML, commit, republish — the agent picks up the new version and the logs record which version answered which user.

Multi-agent graphs follow the same pattern — the agent-catalog travel example composes three `ReActAgent` nodes (front desk → endpoint finder → route finder) with conditional edges, each node defined by its own prompt record, handoffs logged as `EdgeContent`.

## 11.6 The state architecture, complete

One way to see the whole book at once — where a production agent's state lives:

| State | Collection | Written by | Chapter |
|---|---|---|---|
| Conversation turns | `ai.agent.sessions` / checkpointer | SessionStore / `CouchbaseSaver` | 9, 11 |
| Graph execution state | `ai.agent.checkpoints{,_writes}` | `CouchbaseSaver` | 11 |
| Long-term memories | `ai.agent.memories` | `save_memory` tool, extraction | 9 |
| Knowledge (RAG) | `ai.docs.chunks` | ingestion / vectorization | 3–5 |
| Tools & prompts | `ai.agent_catalog.*` | `agentc publish` | 10 |
| Activity / audit | `ai.agent_activity.logs` | Spans (automatic) | 10 |
| Eval results | `ai.evals.*` | Ragas harness | 13 |

The LLM and the graph are stateless and disposable; every durable byte is in Couchbase, queryable with SQL++, secured by RBAC, and versioned where it matters. That's the thesis of this book, made concrete.

## 11.7 Production notes

- **Thread ID design**: `user::purpose::date` gives natural session boundaries and makes checkpoint cleanup a ranged operation. Put a TTL policy on checkpoint collections — in-flight graphs don't need immortal history.
- **Idempotent tools**: LangGraph retries nodes; tools that mutate (Ch. 2 upserts with deterministic keys) should tolerate replay.
- **Interrupts for dangerous tools**: gate refunds/writes behind `interrupt_before=["dangerous_tool_node"]` + human approval; the checkpoint holds state while a human decides.
- **Model routing**: planner nodes earn a frontier model; extraction/summarization nodes run fine on a small Capella-hosted model (Ch. 8). It's one `ChatOpenAI(base_url=...)` per node.

Notebook: [`notebooks/07_agent_catalog_langgraph.ipynb`](../notebooks/07_agent_catalog_langgraph.ipynb). App: [`apps/support-agent`](../apps/support-agent/).

Next: [Chapter 12 — The Couchbase MCP Server](12-mcp-server.md).
