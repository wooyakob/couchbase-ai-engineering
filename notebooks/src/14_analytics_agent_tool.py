# %% [markdown]
# # 14: Analytics: An Agent Reporting Tool
#
# Companion to [Chapter 1 §1.2](../docs/01-ai-engineering-on-couchbase.md)'s stack table and
# [Chapter 10](../docs/10-agent-catalog.md)/[11](../docs/11-orchestration-langgraph.md)'s agent
# tooling. Notebook 07's `agentc init` silently created three **Analytics** views over
# `ai.agent_activity.logs` *if* the Analytics service happened to be available on your
# cluster (its own §"About the analytics views..." note explains that fallback). This
# notebook shows what's behind that: how to turn any collection into an Analytics-queryable
# dataset yourself, and how to hand the resulting aggregate query to an agent as a tool.
# **Works unchanged on self-managed Couchbase Server and Capella** (Analytics is a cluster
# service on both, queried through the same SDK call either way).
#
# Why a second query engine at all: the Query service (KV + SQL++, notebooks 01-02) is
# tuned for point lookups and small scans: an agent's own operational reads/writes. The
# **Analytics service (CBAS)** is a separate MPP query engine tuned for large scans and
# aggregations, ingesting a live shadow copy of a collection so those heavy reporting
# queries never compete with the agent's operational traffic for Query-service resources.
#
# 1. Check the Analytics service is available on this cluster (Server or Capella)
# 2. Seed a collection of synthetic agent tool-invocation telemetry
# 3. Turn it into an Analytics-queryable dataset with one DDL statement
# 4. Run an aggregate usage report across it
# 5. Wrap that report as a tool a LangChain agent can call on its own
#
# **Prerequisites:** notebook 01 (the `ai` bucket exists); `OPENAI_API_KEY`. A cluster with
# the **Analytics service** enabled (see §1 below if the check fails).

# %%
%pip install -q couchbase langchain-openai langchain python-dotenv

# %%
import os
import random
import time
from datetime import timedelta

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(os.getenv("ENV_FILE", ".env"), usecwd=True))  # ENV_FILE lets .env.server / .env.capella run side by side

from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.exceptions import (AuthenticationException,
                                  CollectionAlreadyExistsException,
                                  CouchbaseException, DatasetAlreadyExistsException,
                                  ScopeAlreadyExistsException, ServiceUnavailableException)
from couchbase.options import ClusterOptions, KnownConfigProfiles

conn = os.getenv("CB_CONN_STRING")
CB_USERNAME, CB_PASSWORD, CB_BUCKET = (os.getenv("CB_USERNAME"), os.getenv("CB_PASSWORD"),
                                       os.getenv("CB_BUCKET"))
_missing = [n for n, v in [("CB_CONN_STRING", conn), ("CB_USERNAME", CB_USERNAME),
                          ("CB_PASSWORD", CB_PASSWORD), ("CB_BUCKET", CB_BUCKET)] if not v]
if _missing:
    raise RuntimeError(
        f"Missing required env var(s): {', '.join(_missing)}. Check ENV_FILE="
        f"{os.getenv('ENV_FILE', '.env')!r} is set and that file has these. "
        "See docs/troubleshooting.md."
    )

opts = ClusterOptions(PasswordAuthenticator(CB_USERNAME, CB_PASSWORD))
if conn.startswith("couchbases://"):
    opts.apply_profile(KnownConfigProfiles.WanDevelopment)
try:
    cluster = Cluster.connect(conn, opts)
    cluster.wait_until_ready(timedelta(seconds=10))
except AuthenticationException as e:
    raise RuntimeError(
        f"Couchbase rejected CB_USERNAME={CB_USERNAME!r} for {conn!r}: check "
        "CB_USERNAME/CB_PASSWORD. See docs/troubleshooting.md."
    ) from e
except CouchbaseException as e:
    raise RuntimeError(
        f"Couldn't connect to Couchbase at {conn!r}: {e}. See docs/troubleshooting.md."
    ) from e
bucket = cluster.bucket(CB_BUCKET)

# %% [markdown]
# ## 1. Is the Analytics service available on this cluster?
#
# Analytics is an optional cluster service, same as Eventing (notebook 13). Check for it
# explicitly rather than let §3's `ALTER COLLECTION` fail with a generic error three cells
# from now.

# %%
def run_analytics(statement: str, **kwargs) -> list:
    """Runs a SQL++ statement against the Analytics service and forces execution by
    consuming the (lazily streamed) result: translating the two failure modes worth
    a clear message: the service isn't there at all, or the statement itself is bad."""
    try:
        return list(cluster.analytics_query(statement, **kwargs))
    except DatasetAlreadyExistsException:
        raise
    except ServiceUnavailableException as e:
        raise RuntimeError(
            "The Analytics service isn't available on this cluster. On Capella: open "
            "your cluster's **Data Tools → Analytics** tab; if it's greyed out, add "
            "**Analytics** to a service group under **Settings → Service Groups** and "
            "wait for the cluster to finish scaling. On self-managed Couchbase Server: "
            "add the **Analytics** service to a node, at cluster-init time, or via "
            "Servers → Add Server on an existing cluster. Re-run this cell once the "
            "service shows as healthy. See docs/troubleshooting.md."
        ) from e
    except CouchbaseException as e:
        raise RuntimeError(
            f"Analytics statement failed: {statement!r}: {e}. See docs/troubleshooting.md."
        ) from e


run_analytics("SELECT 1")
print("Analytics service is available")

# %% [markdown]
# ## 2. Seed synthetic agent tool-invocation telemetry
#
# Stand-in for what Ch. 10's Agent Catalog auditor (or your own logging) would accumulate
# in production: one document per tool call, with the fields a usage report needs.

# %%
SCOPE = "analytics_demo"
COLL = "agent_events"

cm = bucket.collections()
try:
    cm.create_scope(SCOPE)
except ScopeAlreadyExistsException:
    pass
try:
    cm.create_collection(SCOPE, COLL)
    print(f"created {SCOPE}.{COLL}")
except CollectionAlreadyExistsException:
    pass

events_coll = bucket.scope(SCOPE).collection(COLL)
cluster.query(f"CREATE PRIMARY INDEX IF NOT EXISTS ON `{CB_BUCKET}`.`{SCOPE}`.`{COLL}`").execute()

TOOLS = ["lookup_order", "save_memory", "search_docs"]
random.seed(7)  # rerun-stable synthetic telemetry

# Analytics isn't enabled on this collection until §3, so count via the Query service here
existing = cluster.query(
    f"SELECT RAW COUNT(*) FROM `{CB_BUCKET}`.`{SCOPE}`.`{COLL}`"
).execute()[0]

N_EVENTS = 300
if existing < N_EVENTS:
    for i in range(N_EVENTS):
        tool = TOOLS[i % len(TOOLS)]
        # give each tool a distinct latency/cost/success profile so the aggregate below
        # actually differentiates them, instead of three near-identical rows
        base_latency = {"lookup_order": 120, "save_memory": 60, "search_docs": 350}[tool]
        fail_rate = {"lookup_order": 0.02, "save_memory": 0.01, "search_docs": 0.08}[tool]
        events_coll.upsert(f"event::{i:05d}", {
            "type": "agent_event",
            "tool_name": tool,
            "latency_ms": max(1, round(random.gauss(base_latency, base_latency * 0.2))),
            "cost_usd": round(random.uniform(0.0001, 0.002), 6),
            "success": random.random() > fail_rate,
            "ts": f"2026-07-{(i % 28) + 1:02d}T{(i % 24):02d}:00:00Z",
        })
    print(f"upserted {N_EVENTS} synthetic tool-invocation events")
else:
    print(f"reusing {existing} previously-upserted events")

# %% [markdown]
# ## 3. Enable Analytics on the collection
#
# `ALTER COLLECTION ... ENABLE ANALYTICS` is the modern (collection-aware) path: one
# statement, no separate dataverse/dataset/link management. Analytics begins shadowing the
# collection immediately: a live, continuously-updated copy on the Analytics service's own
# storage, so the aggregate query in §4 never touches the Query/KV path at all.

# %%
# Not idempotent like the CREATE ... IF NOT EXISTS calls elsewhere in this notebook:
# ENABLE ANALYTICS has no IF NOT EXISTS form, so re-running this cell (or the whole
# notebook) against a collection already shadowed raises DatasetAlreadyExistsException.
try:
    run_analytics(f"ALTER COLLECTION `{CB_BUCKET}`.`{SCOPE}`.`{COLL}` ENABLE ANALYTICS")
except DatasetAlreadyExistsException:
    print(f"{SCOPE}.{COLL} is already shadowed by Analytics")


def wait_ingested(expected_count: int, timeout_s: int = 60):
    for _ in range(timeout_s):
        rows = run_analytics(f"SELECT RAW COUNT(*) FROM `{CB_BUCKET}`.`{SCOPE}`.`{COLL}`")
        if rows and rows[0] >= expected_count:
            return
        time.sleep(1)
    raise TimeoutError(
        f"Analytics hadn't ingested all {expected_count} documents within {timeout_s}s. "
        "Shadowing is asynchronous; a large backlog on a small dev cluster can take "
        "longer; re-run this cell with a longer timeout_s. See docs/troubleshooting.md."
    )


wait_ingested(N_EVENTS)
print("Analytics has ingested every event document")

# %% [markdown]
# ## 4. An aggregate usage report
#
# The query every tool-usage dashboard needs: count, latency, cost, success rate, grouped
# by tool. Run once as a sanity check before it becomes a callable tool in §5.

# %%
REPORT_QUERY = f"""
    SELECT e.tool_name,
           COUNT(*) AS invocations,
           ROUND(AVG(e.latency_ms), 1) AS avg_latency_ms,
           ROUND(SUM(e.cost_usd), 4) AS total_cost_usd,
           ROUND(100.0 * SUM(CASE WHEN e.success THEN 1 ELSE 0 END) / COUNT(*), 1)
               AS success_rate_pct
    FROM `{CB_BUCKET}`.`{SCOPE}`.`{COLL}` e
    WHERE ($tool_name IS NULL OR e.tool_name = $tool_name)
    GROUP BY e.tool_name
    ORDER BY invocations DESC
"""

for row in run_analytics(REPORT_QUERY, named_parameters={"tool_name": None}):
    print(row)

# %% [markdown]
# ## 5. Wrap it as a tool, hand it to an agent
#
# One `@tool`-decorated function running `REPORT_QUERY`: the same pattern as notebook 07's
# cataloged tools, minus Agent Catalog's versioning/audit layer, to keep this notebook
# focused on the Analytics half. `create_agent` (Ch. 11) picks it up like any other tool;
# the LLM decides when a question calls for it and which `tool_name` (if any) to pass.

# %%
from langchain_core.tools import tool


@tool
def tool_usage_report(tool_name: str | None = None) -> list[dict]:
    """Get invocation count, average latency, total cost, and success rate for the
    agent's own tools, optionally filtered to one tool by name. Use this whenever the
    user asks how a tool is performing, how much it costs, or how reliable it is."""
    rows = run_analytics(REPORT_QUERY, named_parameters={"tool_name": tool_name})
    return list(rows)


# %%
import langchain_openai
from langchain.agents import create_agent

llm = langchain_openai.ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = create_agent(
    llm, tools=[tool_usage_report],
    system_prompt="You are an ops assistant for an AI agent's tools. Ground every "
                 "answer in the tool_usage_report tool, never guess at numbers.",
)

result = agent.invoke({"messages": [(
    "user", "How is the search_docs tool doing: is it slower or less reliable than "
            "the others?")]})
print(result["messages"][-1].content)

# %% [markdown]
# `search_docs` was seeded with ~3x the latency and ~4-8x the failure rate of the other two
# tools (§2); the agent's answer above should reflect that, sourced from the Analytics
# aggregate, not from anything stated in this notebook's prose.

# %% [markdown]
# ## 6. Cleanup

# %%
run_analytics(f"ALTER COLLECTION `{CB_BUCKET}`.`{SCOPE}`.`{COLL}` DISABLE ANALYTICS")
print("Analytics disabled on the collection")

# %% [markdown]
# ## Recap
#
# `ALTER COLLECTION ... ENABLE ANALYTICS` turns any KV collection into a live-shadowed
# Analytics dataset with no dataverse/dataset/link bookkeeping to manage by hand; the
# tradeoff for that convenience is eventual (not transactional) consistency between the two
# copies, which is exactly the right tradeoff for a reporting/aggregation workload. Wrapping
# the resulting query as a plain LangChain tool (§5) is all "an agent using an analytics
# tool" is: the same tool-calling mechanics as notebook 07's Agent Catalog tools, aimed at
# a query engine chosen for scan-heavy aggregation instead of point lookups.
#
# Back to: [Chapter 1: AI Engineering on Couchbase](../docs/01-ai-engineering-on-couchbase.md).
# See also: [13: Eventing](13_eventing.ipynb) for the reactive (not aggregate) half of
# keeping derived AI data useful.
