# %% [markdown]
# # 07 — Agent Catalog + LangGraph: a Governed, Durable Agent
#
# Companion to [Chapters 10–11](../docs/10-agent-catalog.md). Builds a support agent where:
#
# - tools and prompts live in **Agent Catalog** (`agentc`) — versioned, searchable, audited
# - orchestration is **LangGraph** (Couchbase has no orchestration framework; this is the pairing)
# - graph state persists in Couchbase via the **checkpointer**
# - long-term memory is notebook 06's `MemoryStore` as a cataloged tool
#
# **Prerequisites:** notebooks 01 & 06; Python 3.11+; a Git repo (agentc snapshots are
# keyed to commits); `OPENAI_API_KEY`. `agentc` is pre-GA (0.2.x) — pin versions.

# %%
# %pip install -q agentc "agentc[langgraph]" langgraph langgraph-checkpointer-couchbase langchain-openai couchbase python-dotenv

# %% [markdown]
# ## 1. Project layout: tools and prompts as files
#
# Agent Catalog indexes *files*. A real project keeps them in `tools/` and `prompts/`
# (see `apps/support-agent/`); here we write a minimal set to disk so the notebook is
# self-contained.

# %%
import os
import pathlib

from dotenv import load_dotenv

load_dotenv()

root = pathlib.Path("agentc_demo")
(root / "tools").mkdir(parents=True, exist_ok=True)
(root / "prompts").mkdir(exist_ok=True)

# %%
# A Python tool: docstring is the searchable contract (Ch. 10 §10.2)
(root / "tools" / "order_tools.py").write_text('''
import os

import agentc
import couchbase.auth
import couchbase.cluster
import couchbase.options
import dotenv

dotenv.load_dotenv()

_cluster = couchbase.cluster.Cluster(
    os.getenv("CB_CONN_STRING", "couchbase://localhost"),
    couchbase.options.ClusterOptions(
        couchbase.auth.PasswordAuthenticator(
            os.getenv("CB_USERNAME", "Administrator"),
            os.getenv("CB_PASSWORD", "password"))))


@agentc.catalog.tool
def lookup_order(order_id: str) -> dict:
    """Fetch a customer order by its numeric ID.
    Use when the user asks about order status, contents, or delivery."""
    bucket = _cluster.bucket(os.getenv("CB_BUCKET", "ai"))
    return bucket.scope("shop").collection("orders").get(f"order::{order_id}").content_as[dict]


@agentc.catalog.tool
def save_memory(user_id: str, fact: str) -> str:
    """Save a durable fact about the user for future conversations.
    Use when the user states a lasting preference, constraint, or correction."""
    # in a real app this calls MemoryStore.remember (notebook 06)
    from uuid import uuid4
    key = f"memory::{user_id}::{uuid4().hex[:12]}"
    bucket = _cluster.bucket(os.getenv("CB_BUCKET", "ai"))
    bucket.scope("agent").collection("memories").upsert(
        key, {"type": "memory", "user_id": user_id, "text": fact, "kind": "fact"})
    return key
'''.strip())

# %%
# A prompt record: instructions + required tools + output schema, in one versioned file
(root / "prompts" / "support_agent.yaml").write_text('''
record_kind: prompt
name: support_agent_node
description: >
  Instructions for the customer-support agent that answers order questions
  and remembers user preferences.
annotations:
  framework: "langgraph"
tools:
  - name: "lookup_order"
  - name: "save_memory"
content:
  agent_instructions:
    - >
      You are a customer-support agent. Ground every answer in tool results;
      never invent order details. If the user states a lasting preference,
      save it with the save_memory tool.
  output_format_instructions: >
    Be concise. Refer to orders by their ID.
'''.strip())

print("wrote", *[str(p) for p in root.rglob("*.*")], sep="\n  ")

# %% [markdown]
# ## 2. Index and publish
#
# In a terminal (agentc is a CLI-first workflow — snapshots are keyed to Git commits):
#
# ```bash
# cd agentc_demo
# git init && git add -A && git commit -m "Support agent v1"
# agentc init            # local catalog + Couchbase collections (agent_catalog / agent_activity)
# agentc index .         # scan tools/ and prompts/ into the local catalog
# agentc publish         # push the Git-versioned snapshot to Couchbase
#
# agentc find tools --query "anything for checking order status"
# ```
#
# The cell below runs the same steps programmatically so the notebook is reproducible
# end-to-end (requires `AGENT_CATALOG_*` env vars — see `.env.example`).

# %%
import subprocess


def run(cmd, cwd=root):
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    print(f"$ {cmd}\n{(r.stdout or r.stderr)[-600:]}")
    return r.returncode


if run("git rev-parse --git-dir 2>/dev/null || git init -q") == 0:
    run("git add -A && git -c user.email=nb@example.com -c user.name=nb commit -qm 'v1' || true")
run("agentc init")
run("agentc index .")
run("agentc publish || true")   # requires a clean repo + reachable cluster

# %% [markdown]
# ## 3. Consume the catalog from Python

# %%
os.chdir(root)  # Catalog() discovers .agent-catalog by walking up from CWD

import agentc

catalog = agentc.Catalog()

# by meaning — retrieval-augmented tool selection
tools = catalog.find("tool", query="checking on a customer's order", limit=1)
print("found tool:", tools[0].meta.name)

# prompts arrive with their declared tools resolved
prompt = catalog.find("prompt", name="support_agent_node")
print("prompt tools:", [t.meta.name for t in prompt.tools])
print("instructions:", prompt.content["agent_instructions"][0][:80], "…")

# %% [markdown]
# ## 4. The LangGraph agent, catalog-driven and durable
#
# Three integrations meet here:
# - prompt + tools from the **catalog** (versioned)
# - a **Span** logging every step to `ai.agent_activity.logs`
# - the **Couchbase checkpointer** making graph state durable per thread

# %%
import langchain_core.tools
import langchain_openai
from langgraph.prebuilt import create_react_agent
from langgraph_checkpointer_couchbase import CouchbaseSaver

llm = langchain_openai.ChatOpenAI(model="gpt-4o-mini", temperature=0)

# catalog tools -> LangChain tools
lc_tools = [
    langchain_core.tools.StructuredTool.from_function(
        t.func, name=t.meta.name, description=t.meta.description)
    for t in prompt.tools
]

instructions = "\n".join(prompt.content["agent_instructions"]) + \
               "\n" + prompt.content.get("output_format_instructions", "")

checkpointer = CouchbaseSaver.from_conn_info(
    cb_conn_str=os.getenv("CB_CONN_STRING", "couchbase://localhost"),
    cb_username=os.getenv("CB_USERNAME", "Administrator"),
    cb_password=os.getenv("CB_PASSWORD", "password"),
    bucket_name=os.getenv("CB_BUCKET", "ai"),
    scope_name="agent",           # uses collections: checkpoints, checkpoint_writes
)

agent = create_react_agent(llm, tools=lc_tools, prompt=instructions,
                           checkpointer=checkpointer)

# %%
# Seed an order for the tool to find
import couchbase.auth
import couchbase.cluster
import couchbase.options
from couchbase.exceptions import (CollectionAlreadyExistsException,
                                  ScopeAlreadyExistsException)

cluster = couchbase.cluster.Cluster(
    os.getenv("CB_CONN_STRING", "couchbase://localhost"),
    couchbase.options.ClusterOptions(couchbase.auth.PasswordAuthenticator(
        os.getenv("CB_USERNAME", "Administrator"), os.getenv("CB_PASSWORD", "password"))))
bucket = cluster.bucket(os.getenv("CB_BUCKET", "ai"))
try:
    bucket.collections().create_scope("shop")
except ScopeAlreadyExistsException:
    pass
try:
    bucket.collections().create_collection("shop", "orders")
except CollectionAlreadyExistsException:
    pass
bucket.scope("shop").collection("orders").upsert("order::1042", {
    "id": 1042, "status": "shipped", "eta": "2026-07-08",
    "items": [{"sku": "CB-TSHIRT-L", "qty": 2}],
})

# %%
# Run it — with a Span so the whole exchange is audited
span = catalog.Span(name="support_agent", session="notebook-07-demo")

config = {"configurable": {"thread_id": "u42::support::demo"}}
with span.new(name="turn_1") as s:
    s.log(content=agentc.span.UserContent(value="Where is order 1042?"))
    result = agent.invoke({"messages": [("user", "Where is order 1042?")]}, config)
    answer = result["messages"][-1].content
    s.log(content=agentc.span.AssistantContent(value=answer))
print(answer)

# %%
# Same thread_id -> the checkpointer restores state; the agent remembers the context
result = agent.invoke({"messages": [("user", "And when will it arrive? Btw I prefer "
                                             "email updates, not SMS.")]}, config)
print(result["messages"][-1].content)

# %% [markdown]
# ## 5. What just got written to Couchbase
#
# - `ai.agent.checkpoints` / `checkpoint_writes` — every step of graph state (crash
#   recovery, human-in-the-loop, time travel)
# - `ai.agent.memories` — the email preference, if the agent chose to `save_memory`
# - `ai.agent_activity.logs` — the Span tree: user/assistant content, and (with the
#   `agentc_langgraph.agent.ReActAgent` wrapper used in `apps/support-agent`) every model
#   call and tool result, all tagged with the Git catalog version
#
# Analysis is SQL++ (views installed by `agentc init`):
#
# ```sql
# SELECT * FROM ai.agent_activity.Sessions() s
# WHERE s.sid = ai.agent_activity.LastSession();
# ```
#
# The observability loop: ship → query what the agent did → fix the prompt YAML →
# commit (new snapshot) → measure again. The measuring is the next notebook.
#
# **Next:** [08 — Evaluating with Ragas](08_ragas_evaluation.ipynb)
