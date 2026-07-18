# %% [markdown]
# # 12: The Couchbase MCP Server
#
# Companion to [Chapter 12](../docs/12-mcp-server.md). Every other notebook talks to Couchbase
# through the Python SDK, in-process. This one instead spawns the **Couchbase MCP server**
# (`couchbase-mcp-server`, via `uvx`) as a subprocess and talks to it over the Model Context
# Protocol, the same way Claude Desktop, Claude Code, or any other MCP client would.
#
# 1. Connect and discover tools (`langchain-mcp-adapters` turns MCP tools into LangChain tools)
# 2. Call a tool directly, no LLM involved
# 3. Wire the tools into a LangGraph agent (Ch. 11) and ask it a natural-language question
# 4. Security notes: read-only by default, least-privilege credentials
#
# **Prerequisites:** notebook 01 (provisioning); `OPENAI_API_KEY` (or Capella, as before);
# [`uv`](https://docs.astral.sh/uv/) installed (`uvx` ships with it): the MCP server itself
# is *not* a Python dependency you `pip install`, `uvx` fetches and runs it on demand.

# %%
%pip install -q langchain-mcp-adapters langchain langchain-openai langgraph python-dotenv

# %%
import os

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(os.getenv("ENV_FILE", ".env"), usecwd=True))  # ENV_FILE lets .env.server / .env.capella run side by side

CB_CONN_STRING = os.environ["CB_CONN_STRING"]
CB_BUCKET = os.getenv("CB_BUCKET", "ai")
# A dedicated least-privilege user is the production move (§4 below); this notebook
# falls back to the app credentials only so it runs with zero extra setup.
MCP_USERNAME = os.getenv("MCP_CB_USERNAME") or os.environ["CB_USERNAME"]
MCP_PASSWORD = os.getenv("MCP_CB_PASSWORD") or os.environ["CB_PASSWORD"]

# %% [markdown]
# ## 1. Connect and discover tools
#
# `MultiServerMCPClient` spawns `uvx couchbase-mcp-server` over `stdio` and speaks MCP to it.
# Credentials go as CLI flags here: the server also accepts them as `CB_CONNECTION_STRING`/
# `CB_USERNAME`/`CB_PASSWORD` env vars (both work; flags are more explicit for a notebook).
# There's no bucket-scoping flag; each tool takes `bucket_name` as an explicit parameter,
# so the server discovers tools against the whole cluster, not one bucket.

# %%
from langchain_mcp_adapters.client import MultiServerMCPClient

mcp_client = MultiServerMCPClient({
    "couchbase": {
        "transport": "stdio",
        "command": "uvx",
        "args": [
            "couchbase-mcp-server",
            "--connection-string", CB_CONN_STRING,
            "--username", MCP_USERNAME,
            "--password", MCP_PASSWORD,
            # --read-only-mode defaults to True: no KV/query writes exposed, read tools only
        ],
    }
})

tools = await mcp_client.get_tools()
print(f"{len(tools)} tools discovered:")
for t in tools:
    print(f"  {t.name:40s} {t.description.splitlines()[0][:70]}")

# %% [markdown]
# ## 2. Call a tool directly
#
# No LLM needed for this: each MCP tool is just a LangChain `StructuredTool` once adapted,
# callable like any other:

# %%
by_name = {t.name: t for t in tools}

result = await by_name["get_scopes_and_collections_in_bucket"].ainvoke(
    {"bucket_name": CB_BUCKET})
print(result)

# %% [markdown]
# ## 3. Wire into a LangGraph agent
#
# Same `create_agent` used in notebook 07 (Ch. 11): MCP tools bind in exactly like the
# hand-written catalog tools there. The agent decides which tool(s) to call from the
# question alone; nothing here names a scope or collection.

# %%
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0)
agent = create_agent(llm, tools=tools,
                     system_prompt="You are a Couchbase data assistant. Use the tools to "
                                   "answer questions about the cluster's data.")

response = await agent.ainvoke({"messages": [(
    "user",
    f"Which scopes exist in the {CB_BUCKET} bucket, and how many documents are in "
    "the docs.chunks collection, if it exists?")]})
print(response["messages"][-1].content)

# %% [markdown]
# The model wrote and executed its own SQL++ (via `run_sql_plus_plus_query`) to answer the
# count, that's the whole pattern from Ch. 12 §12.4: *model writes SQL++, server executes,
# model interprets*. It only worked because the schema-discovery tools ran first, in the
# same turn, so the model knew `docs.chunks` existed before querying it.
#
# ## 4. Security notes
#
# - **Read-only by default.** `--read-only-mode` defaults to `True`: no KV/query write
#   tools are even registered unless you explicitly opt in. Nothing above could have
#   mutated data.
# - **Dedicated, least-privilege credentials.** This notebook fell back to the app's own
#   `CB_USERNAME`/`CB_PASSWORD` for zero-setup convenience, but production should use
#   `MCP_CB_USERNAME`/`MCP_CB_PASSWORD`: a separate database user scoped to only what
#   the MCP server needs (see `.env.server.example`/`.env.capella.example`, and the
#   `agent/mcp_tools.py` module in `apps/support-agent`, which reads the same two vars).
# - **Untrusted context.** Query results returned into the model's context are exactly
#   that: context. If your documents can contain adversarial text, treat tool results
#   the same as any other untrusted input before acting on them further.
# - **Nothing to tear down.** This client spawns the server fresh per call rather than
#   holding a long-lived connection; there's no explicit disconnect step in this notebook.

# %% [markdown]
# Next: [Chapter 13: Evaluating with Ragas](../docs/13-evaluation-ragas.md).
