# %% [markdown]
# # 13: Eventing: Reactive Data Processing for AI Pipelines
#
# Companion to [Chapter 3 §3.6](../docs/03-data-processing.md#36-keeping-data-fresh-eventing).
# That section sketched the idea in prose; this notebook deploys a real Eventing function
# with the Python SDK's management API and watches it react. **Works unchanged on
# self-managed Couchbase Server and Capella** (Eventing is a cluster service on both, and
# the Python SDK talks to it the same way regardless of which one is running underneath).
#
# The problem Eventing solves: a source document changes, and something derived from it
# (an embedding, a summary, a cache entry, an audit trail) is now stale. Polling for that is
# wasteful; a batch job means staleness lasts until the next run. Eventing runs a JavaScript
# handler **in the database, on the mutation itself**: no external CDC pipeline, no cron.
#
# 1. Check the Eventing service is available on this cluster (Server or Capella)
# 2. Provision a source collection and a derived collection
# 3. Write and deploy an Eventing function: flag derived docs stale on source mutation
# 4. Mutate a source document and watch the derived flag flip, with no code in this
#    notebook polling for it
# 5. Handle deletes the same way, then clean up
#
# **Prerequisites:** notebook 01 (the `ai` bucket exists). A cluster with the **Eventing
# service** enabled (see §1 below if the check fails). `agentc init` (notebook 07) also
# touches Eventing-adjacent Analytics views, but this notebook is independent of that one.

# %%
%pip install -q couchbase python-dotenv

# %%
import os
import time
from datetime import timedelta

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(os.getenv("ENV_FILE", ".env"), usecwd=True))  # ENV_FILE lets .env.server / .env.capella run side by side

from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.exceptions import (AuthenticationException,
                                  CollectionAlreadyExistsException,
                                  CouchbaseException, DocumentNotFoundException,
                                  ScopeAlreadyExistsException,
                                  ServiceUnavailableException)
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
# ## 1. Is the Eventing service available on this cluster?
#
# Eventing is an optional cluster service: not every cluster has it, and the failure mode
# if it's missing is a generic `ServiceUnavailableException`, not a helpful message. This
# notebook checks for it explicitly and, if it's missing, tells you exactly what to add,
# rather than letting a cryptic error surface three cells from now.

# %%
eventing_mgr = cluster.eventing_functions()

try:
    eventing_mgr.get_all_functions()
except ServiceUnavailableException as e:
    raise RuntimeError(
        "The Eventing service isn't available on this cluster. On Capella: open your "
        "cluster's **Settings → Service Groups** and add a service group that includes "
        "**Eventing** (or add it to an existing service group), then wait for the cluster "
        "to finish scaling. On self-managed Couchbase Server: add the **Eventing** service "
        "to a node, either at cluster-init time, or via Servers → Add Server on an "
        "existing cluster. Re-run this cell once the service shows as healthy. "
        "See docs/troubleshooting.md."
    ) from e
print("Eventing service is available")

# %% [markdown]
# ## 2. Source and derived collections
#
# `source_docs` holds documents an app owns and edits; `derived_docs` holds something
# computed from them (stand-in for an embedding, a summary, a rendered cache entry: Ch. 3's
# examples). The Eventing function keeps `derived_docs` in sync with `source_docs`, so an
# app never has to remember to invalidate it by hand.

# %%
SCOPE = "eventing_demo"
SOURCE_COLL = "source_docs"
DERIVED_COLL = "derived_docs"
METADATA_COLL = "eventing_metadata"  # Eventing's own checkpoint bookkeeping: must differ
                                     # from both the source and any bound collection

cm = bucket.collections()
try:
    cm.create_scope(SCOPE)
except ScopeAlreadyExistsException:
    pass
for coll in (SOURCE_COLL, DERIVED_COLL, METADATA_COLL):
    try:
        cm.create_collection(SCOPE, coll)
        print(f"created {SCOPE}.{coll}")
    except CollectionAlreadyExistsException:
        pass

source_coll = bucket.scope(SCOPE).collection(SOURCE_COLL)
derived_coll = bucket.scope(SCOPE).collection(DERIVED_COLL)

# %% [markdown]
# ## 3. Write and deploy the Eventing function
#
# `OnUpdate` fires on every mutation of `source_docs`. It writes a stale-flagged derived
# document keyed off the same ID, through a **bucket binding**: Eventing's way of granting
# a handler read/write access to a specific bucket/scope/collection by alias, so the handler
# code never touches raw credentials. `OnDelete` mirrors it on removal.
#
# This binds the alias directly to one collection, so the handler can mutate it with plain
# JavaScript object syntax (`derivedColl[key] = {...}`), no query language involved, which
# is why this works identically on Server and Capella: it's pure KV, no Query-service
# privileges to configure. Chapter 3 §3.6's SQL++ variant (`UPDATE ... WHERE lineage.doc_id`)
# is the right choice instead when one source fans out to *many* derived documents; here
# each source maps to exactly one derived doc, so a direct keyed write is simpler and cheaper.

# %%
FUNCTION_NAME = "ai_engineering_flag_stale"

HANDLER_CODE = f"""
function OnUpdate(doc, meta) {{
    var derivedKey = meta.id;
    derivedColl[derivedKey] = {{
        source_id: meta.id,
        needs_refresh: true,
        source_rev: meta.cas.toString()
    }};
    log("flagged stale:", derivedKey);
}}

function OnDelete(meta, options) {{
    delete derivedColl[meta.id];
    log("removed derived doc for deleted source:", meta.id);
}}
""".strip()

print(HANDLER_CODE)

# %% [markdown]
# Deploying is two management-API calls: `upsert_function` registers the handler's code and
# bindings; `deploy_function` starts it consuming the DCP mutation stream. Re-running this
# cell is safe: `upsert_function` overwrites the existing definition, and deploying an
# already-deployed function is a no-op the SDK tolerates below.

# %%
from couchbase.management.eventing import (EventingFunction,
                                            EventingFunctionBucketAccess,
                                            EventingFunctionBucketBinding,
                                            EventingFunctionDcpBoundary,
                                            EventingFunctionKeyspace,
                                            EventingFunctionLanguageCompatibility,
                                            EventingFunctionLogLevel, EventingFunctionSettings,
                                            EventingFunctionState)
from couchbase.exceptions import EventingFunctionAlreadyDeployedException

function = EventingFunction(
    FUNCTION_NAME,
    HANDLER_CODE,
    metadata_keyspace=EventingFunctionKeyspace(CB_BUCKET, SCOPE, METADATA_COLL),
    source_keyspace=EventingFunctionKeyspace(CB_BUCKET, SCOPE, SOURCE_COLL),
    bucket_bindings=[
        EventingFunctionBucketBinding(
            alias="derivedColl",
            name=EventingFunctionKeyspace(CB_BUCKET, SCOPE, DERIVED_COLL),
            access=EventingFunctionBucketAccess.ReadWrite,
        ),
    ],
    settings=EventingFunctionSettings.new_settings(
        dcp_stream_boundary=EventingFunctionDcpBoundary.FromNow,
        language_compatibility=EventingFunctionLanguageCompatibility.Version_7_2_0,
        log_level=EventingFunctionLogLevel.Info,
    ),
)

eventing_mgr.upsert_function(function)
try:
    eventing_mgr.deploy_function(FUNCTION_NAME)
except EventingFunctionAlreadyDeployedException:
    pass


def wait_deployed(name: str, timeout_s: int = 60):
    for _ in range(timeout_s):
        statuses = eventing_mgr.functions_status().functions
        match = next((f for f in statuses if f.name == name), None)
        if match and match.state == EventingFunctionState.Deployed:
            return
        time.sleep(1)
    raise TimeoutError(
        f"{name!r} did not reach the deployed state within {timeout_s}s: check "
        "eventing_mgr.functions_status() for its actual state, or the Eventing tab in "
        "the UI for a compilation/bootstrap error. See docs/troubleshooting.md."
    )


wait_deployed(FUNCTION_NAME)
print(f"{FUNCTION_NAME} deployed")

# %% [markdown]
# `dcp_stream_boundary=FromNow` means the function only reacts to mutations from this point
# forward: it ignores whatever was already in `source_docs` before deployment. Use
# `Everything` instead if a newly-deployed function should also backfill derived docs for
# existing source data.
#
# ## 4. Mutate a source doc, watch the derived flag appear
#
# Nothing in this notebook polls or triggers the derived write: it's the deployed function
# reacting to the KV mutation stream. The `time.sleep` below is only so the demo waits long
# enough to observe it; a real app has no equivalent wait, because it never depended on the
# derived doc being synchronously fresh in the first place.

# %%
DOC_ID = "source::rag-config-42"

source_coll.upsert(DOC_ID, {
    "type": "rag_config", "chunk_size": 400, "chunk_overlap": 50,
    "updated_at": "2026-07-15T10:00:00Z",
})


def wait_for_derived(doc_id: str, timeout_s: int = 30) -> dict:
    for _ in range(timeout_s):
        try:
            return derived_coll.get(doc_id).content_as[dict]
        except DocumentNotFoundException:
            time.sleep(1)
    raise TimeoutError(
        f"No derived doc appeared for {doc_id!r} within {timeout_s}s. Confirm the "
        f"function shows as Deployed ({FUNCTION_NAME!r}) and check its logs via "
        "eventing_mgr.functions_status() or the Eventing UI tab for a runtime error "
        "(a bad bucket binding alias is the most common one). See docs/troubleshooting.md."
    )


derived = wait_for_derived(DOC_ID)
print("derived doc:", derived)
assert derived["needs_refresh"] is True

# %% [markdown]
# A worker consuming this signal (Ch. 3's re-embed loop is the template) would now query
# `derived_docs` for `needs_refresh = true`, do the real work (re-embed, re-summarize),
# then clear the flag, the same stale→refresh handoff, whatever the derived artifact is.

# %%
# simulate that worker clearing the flag once it's done the real work
import couchbase.subdocument as SD

derived_coll.mutate_in(DOC_ID, (SD.upsert("needs_refresh", False),))
print("worker cleared the stale flag")

# %% [markdown]
# ## 5. Deletes propagate too

# %%
source_coll.remove(DOC_ID)


def wait_removed(doc_id: str, timeout_s: int = 30):
    for _ in range(timeout_s):
        try:
            derived_coll.get(doc_id)
            time.sleep(1)
        except DocumentNotFoundException:
            return
    raise TimeoutError(f"Derived doc for {doc_id!r} was not removed within {timeout_s}s.")


wait_removed(DOC_ID)
print("derived doc removed alongside its source")

# %% [markdown]
# ## 6. Cleanup
#
# Undeploy before dropping: dropping a still-deployed (or still-*un*deploying) function
# raises `EventingFunctionNotUnDeployedException`. Undeploy is asynchronous, same as
# deploy in §3, so wait for it to actually reach `Undeployed` before dropping.

# %%
from couchbase.exceptions import EventingFunctionNotDeployedException


def wait_undeployed(name: str, timeout_s: int = 60):
    for _ in range(timeout_s):
        statuses = eventing_mgr.functions_status().functions
        match = next((f for f in statuses if f.name == name), None)
        if match is None or match.state == EventingFunctionState.Undeployed:
            return
        time.sleep(1)
    raise TimeoutError(
        f"{name!r} did not reach the undeployed state within {timeout_s}s: check "
        "eventing_mgr.functions_status() for its actual state. See docs/troubleshooting.md."
    )


try:
    eventing_mgr.undeploy_function(FUNCTION_NAME)
except EventingFunctionNotDeployedException:
    pass
wait_undeployed(FUNCTION_NAME)
eventing_mgr.drop_function(FUNCTION_NAME)
print(f"{FUNCTION_NAME} undeployed and dropped")

# %% [markdown]
# ## Recap
#
# Eventing turns "keep derived AI data fresh" from an app-level responsibility (remember to
# invalidate, run a cron job, wire a CDC pipeline) into a database-level guarantee: a
# handler that reacts to the mutation stream directly, with its own checkpointing so it
# survives node failures without replaying already-processed mutations. The same shape
# (bucket-bound handler, `OnUpdate`/`OnDelete`) covers Ch. 3's re-embed-on-change, TTL-driven
# session cleanup, auto-summarization queues (Ch. 9), and change-data-capture into audit
# collections, only the handler body and the target collection change.
#
# Back to: [Chapter 3: Data Processing for AI](../docs/03-data-processing.md). Next:
# [14: Analytics: an Agent Reporting Tool](14_analytics_agent_tool.ipynb).
