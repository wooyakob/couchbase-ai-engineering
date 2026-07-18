# %% [markdown]
# # 07: Agent Catalog + LangGraph, a Governed, Durable Agent
#
# Companion to [Chapters 10–11](../docs/10-agent-catalog.md). Builds a support agent where:
#
# - tools and prompts live in **Agent Catalog** (`agentc`): versioned, searchable, audited
# - orchestration is **LangGraph** (Couchbase has no orchestration framework; this is the pairing)
# - graph state persists in Couchbase via the **checkpointer**
# - long-term memory is notebook 06's `MemoryStore` as a cataloged tool
#
# **Prerequisites:** notebooks 01 & 06; Python 3.11+; a Git repo (agentc snapshots are
# keyed to commits); `OPENAI_API_KEY`. `agentc` is pre-GA (0.2.x); pin versions.

# %%
%pip install -q agentc "agentc[langgraph]" langgraph langgraph-checkpointer-couchbase langchain langchain-openai couchbase python-dotenv

# %% [markdown]
# ## 1. Project layout: tools and prompts as files
#
# Agent Catalog indexes *files*. A real project keeps them in `tools/` and `prompts/`
# (see `apps/support-agent/`); here we write a minimal set to disk so the notebook is
# self-contained.

# %%
import os
import pathlib

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(os.getenv("ENV_FILE", ".env"), usecwd=True))  # ENV_FILE lets .env.server / .env.capella run side by side

# AGENTC_DEMO_DIR gives each backend its own folder + own nested git repo, so a
# server-pointed kernel and a Capella-pointed kernel can run this notebook at the
# same time without racing on the same files/git repo. Set both env vars in a cell
# BEFORE this one, e.g.:
#   os.environ["ENV_FILE"] = ".env.capella"
#   os.environ["AGENTC_DEMO_DIR"] = "agentc_demo_capella"
root = pathlib.Path(os.getenv("AGENTC_DEMO_DIR", "agentc_demo"))
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

dotenv.load_dotenv(dotenv.find_dotenv(os.getenv("ENV_FILE", ".env"), usecwd=True))  # ENV_FILE lets .env.server / .env.capella run side by side

_cluster = couchbase.cluster.Cluster(
    os.getenv("CB_CONN_STRING"),
    couchbase.options.ClusterOptions(
        couchbase.auth.PasswordAuthenticator(
            os.getenv("CB_USERNAME"),
            os.getenv("CB_PASSWORD"))))


@agentc.catalog.tool
def lookup_order(order_id: str) -> dict:
    """Fetch a customer order by its numeric ID.
    Use when the user asks about order status, contents, or delivery."""
    bucket = _cluster.bucket(os.getenv("CB_BUCKET"))
    return bucket.scope("shop").collection("orders").get(f"order::{order_id}").content_as[dict]


@agentc.catalog.tool
def save_memory(user_id: str, fact: str) -> str:
    """Save a durable fact about the user for future conversations.
    Use when the user states a lasting preference, constraint, or correction."""
    # in a real app this calls MemoryStore.remember (notebook 06)
    from uuid import uuid4
    key = f"memory::{user_id}::{uuid4().hex[:12]}"
    bucket = _cluster.bucket(os.getenv("CB_BUCKET"))
    bucket.scope("agent").collection("memories").upsert(
        key, {"type": "memory", "user_id": user_id, "text": fact, "kind": "fact"})
    return key
'''.strip())

# %% [markdown]
# **About the `1397` shown after this cell:** `pathlib.Path.write_text()` returns the
# number of characters it wrote, and since that call is the last expression in the
# cell, Jupyter auto-displays it as `Out[]`. `1397` is just the length of the
# `order_tools.py` source string above; it has nothing to do with Analytics, catalog
# size, or a document count. (The "Now creating the analytics views for the auditor…"
# message you may see is unrelated and comes from a later cell; see the note after
# §2's `agentc init` below.)

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
# Everything below runs from notebook cells — no terminal required. `agentc` is a
# CLI-first tool, so the cell after this one drives it with `subprocess`:
#
# ```bash
# git init && git add -A && git commit -m "Support agent v1"
# agentc init            # local catalog + Couchbase collections (agent_catalog / agent_activity)
# agentc index .         # scan tools/ and prompts/ into the local catalog
# agentc publish         # push the Git-versioned snapshot to Couchbase
#
# agentc find tools --query "anything for checking order status"
# ```
#
# **Running server and Capella at the same time:** each backend needs its own
# `AGENTC_DEMO_DIR` (own folder, own nested git repo) so two kernels don't race on the
# same files. In a cell *before* the one above, per kernel:
#
# ```python
# import os
# os.environ["ENV_FILE"] = ".env.capella"          # or ".env.server" in the other kernel
# os.environ["AGENTC_DEMO_DIR"] = "agentc_demo_capella"  # or "agentc_demo_server"
# ```
# then run this notebook from the top in that kernel. Both publish independently —
# `CB_CONN_STRING`/`AGENT_CATALOG_CONN_STRING` point at different clusters, so there's
# no collision even though both use bucket `ai`.
#
# Requires `AGENT_CATALOG_*` env vars; see `.env.server.example` / `.env.capella.example`.

# %%
import subprocess


def run(cmd, cwd=root):
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    # print full stdout/stderr (labeled) rather than an arbitrary last-N-chars
    # slice, which can cut mid-line and leave a bare trailing number with no
    # label (e.g. a "files changed, N insertions" count, or an exit code, with
    # the word in front of it sliced off).
    out, err = r.stdout.strip(), r.stderr.strip()
    print(f"$ {cmd}")
    if out:
        print(out)
    if err:
        print(err)
    return r.returncode


# `git rev-parse --git-dir` walks up parent directories, so it finds this repo's
# own .git even though agentc_demo has none of its own — since this notebook lives
# inside a larger git repo, that check always "succeeds" and never inits a repo
# scoped to agentc_demo. `git add -A` then stages/commits the WHOLE outer repo.
# Testing for a literal .git in this exact directory avoids that.
run("[ -d .git ] || git init -q")
run("git add -A && git -c user.email=nb@example.com -c user.name=nb commit -qm 'v1' || true")
run("agentc init")
run("agentc index .")
run("agentc publish || true")   # requires a clean repo + reachable cluster

# %% [markdown]
# **About the "Now creating the analytics views for the auditor…" messages:** those
# lines (and "Now creating the analytics collections for our catalog…") are printed by
# the `agentc init` CLI itself, not by anything in this notebook, from
# `agentc_cli/cmds/init.py`. Every `agentc init` run provisions a fixed set of Couchbase
# artifacts, whether or not you've touched the Analytics service before:
#
# - a metadata + catalog collection for the indexed tools/prompts (`ai.agent_catalog…`)
# - a `logs` collection + primary index for the **auditor** (`ai.agent_activity.logs`)
# - a couple of query UDFs used by the auditor
# - if the cluster's Analytics service is enabled, three Analytics **views** over that
#   activity data
#
# `agentc init` wraps the Analytics-view creation in a try/except: if the Analytics
# service isn't available on your cluster, it catches `ServiceUnavailableException` and
# prints a yellow "service not available" warning instead of failing, so seeing the
# green "successfully created" message means your cluster genuinely has the Analytics
# service enabled, not that `agentc` silently skipped or faked it.
#
# ## 3. Consume the catalog from Python

# %%
os.chdir(root)  # Catalog() discovers .agent-catalog by walking up from CWD

import agentc

catalog = agentc.Catalog()

# by meaning: retrieval-augmented tool selection
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
from langchain.agents import create_agent
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

# from_conn_info is a context manager (closes its own cluster connection on exit);
# entered manually so `checkpointer` stays live across the rest of this notebook.
# The context-manager object itself must stay referenced too, otherwise it gets
# garbage-collected, which runs its `finally: cluster.close()` immediately and
# leaves `checkpointer` holding a closed cluster.
_checkpointer_cm = CouchbaseSaver.from_conn_info(
    cb_conn_str=os.getenv("CB_CONN_STRING"),
    cb_username=os.getenv("CB_USERNAME"),
    cb_password=os.getenv("CB_PASSWORD"),
    bucket_name=os.getenv("CB_BUCKET"),
    scope_name="agent",           # uses collections: checkpoints, checkpoint_writes
)
checkpointer = _checkpointer_cm.__enter__()

agent = create_agent(llm, tools=lc_tools, system_prompt=instructions,
                     checkpointer=checkpointer)

# %%
# Seed an order for the tool to find
import couchbase.auth
import couchbase.cluster
import couchbase.options
from couchbase.exceptions import (CollectionAlreadyExistsException,
                                  ScopeAlreadyExistsException)

cluster = couchbase.cluster.Cluster(
    os.getenv("CB_CONN_STRING"),
    couchbase.options.ClusterOptions(couchbase.auth.PasswordAuthenticator(
        os.getenv("CB_USERNAME"), os.getenv("CB_PASSWORD"))))
bucket = cluster.bucket(os.getenv("CB_BUCKET"))
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
# Run it, with a Span so the whole exchange is audited
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
# - `ai.agent.checkpoints` / `checkpoint_writes`: every step of graph state (crash
#   recovery, human-in-the-loop, time travel)
# - `ai.agent.memories`: the email preference, if the agent chose to `save_memory`
# - `ai.agent_activity.logs`: the Span tree, user/assistant content, and (with the
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
# **Next:** [08: Evaluating with Ragas](08_ragas_evaluation.ipynb)
