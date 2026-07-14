# %% [markdown]
# # 04 — AI Functions: LLM Tasks in SQL++
#
# Companion to [Chapter 7](../docs/07-ai-functions.md).
#
# Capella AI Functions (`ai_sentiment`, `ai_summary`, `ai_classification`, …) run LLM tasks
# inside SQL++ queries. They require a Capella cluster (Server 8.0+) with AI Functions
# configured against a model integration (Capella Model Service, OpenAI, or Bedrock).
#
# **This notebook runs in two modes:**
# - **Capella mode** — executes real AI Functions via SQL++ (set `AI_FUNCTIONS_ENABLED=true`).
# - **Portable mode** (default) — implements the same enrichment pattern with an SDK worker
#   + any OpenAI-compatible model, so the pattern works on self-managed clusters too.

# %%
%pip install -q couchbase python-dotenv openai

# %%
import json
import os
from datetime import timedelta

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(os.getenv("ENV_FILE", ".env"), usecwd=True))  # ENV_FILE lets .env.server / .env.capella run side by side

from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions, KnownConfigProfiles, QueryOptions

opts = ClusterOptions(PasswordAuthenticator(os.getenv("CB_USERNAME"),
                                            os.getenv("CB_PASSWORD")))
conn = os.getenv("CB_CONN_STRING")
if conn.startswith("couchbases://"):
    opts.apply_profile(KnownConfigProfiles.WanDevelopment)
cluster = Cluster.connect(conn, opts)
cluster.wait_until_ready(timedelta(seconds=10))

CB_BUCKET = os.getenv("CB_BUCKET")
AI_FUNCTIONS_ENABLED = os.getenv("AI_FUNCTIONS_ENABLED", "false").lower() == "true"
print("mode:", "Capella AI Functions" if AI_FUNCTIONS_ENABLED else "portable SDK worker")

# %% [markdown]
# ## Seed data: support tickets to enrich

# %%
from couchbase.exceptions import (CollectionAlreadyExistsException,
                                  ScopeAlreadyExistsException)

bucket = cluster.bucket(CB_BUCKET)
try:
    bucket.collections().create_scope("support")
except ScopeAlreadyExistsException:
    pass
try:
    bucket.collections().create_collection("support", "tickets")
except CollectionAlreadyExistsException:
    pass

tickets = bucket.scope("support").collection("tickets")

SEED = [
    ("ticket::1001", "I was charged twice for my subscription this month. Please refund "
                     "the duplicate charge, this is really frustrating!"),
    ("ticket::1002", "The vector search index build fails with error CB-4012 whenever I "
                     "include a prefilter. Cluster version 8.0.1, Python SDK 4.4."),
    ("ticket::1003", "Loving the new semantic caching feature — would be great if the "
                     "score threshold could be set per-request instead of per-cache."),
]
for key, body in SEED:
    tickets.upsert(key, {"id": key, "body": body})

cluster.query(f"CREATE PRIMARY INDEX IF NOT EXISTS ON `{CB_BUCKET}`.support.tickets").execute()
print(f"seeded {len(SEED)} tickets")

# %% [markdown]
# ## Path A — Capella AI Functions (SQL++)
#
# Query-time enrichment: sentiment + summary per ticket in a single query.
# (Function namespace may differ per deployment — `default:` here.)

# %%
if AI_FUNCTIONS_ENABLED:
    rows = cluster.query(f"""
        SELECT t.id,
               default:ai_sentiment({{"text": t.body, "temperature": 0.0,
                                      "max_tokens": 100}}) AS sentiment,
               default:ai_summary({{"text": t.body, "max_tokens": 60}}) AS summary
        FROM `{CB_BUCKET}`.support.tickets AS t
    """)
    for row in rows:
        print(json.dumps(row, indent=2))
else:
    print("skipped — set AI_FUNCTIONS_ENABLED=true on a configured Capella cluster")

# %% [markdown]
# Enrichment **at rest** — materialize once with `UPDATE`, batched and idempotent
# (`IS MISSING` predicate makes re-runs cheap):

# %%
if AI_FUNCTIONS_ENABLED:
    cluster.query(f"""
        UPDATE `{CB_BUCKET}`.support.tickets AS t
        SET t.category = default:ai_classification({{
                "text": t.body,
                "labels": ["billing", "bug", "feature-request", "account"],
                "temperature": 0.0}}),
            t.enriched_at = NOW_STR()
        WHERE t.category IS MISSING
        LIMIT 100
    """).execute()
    print("materialized classifications")

# %% [markdown]
# ## Path B — the same pattern, portable
#
# `SELECT` the unenriched batch → call the model → `UPDATE` back (well, `mutate_in`).
# This is exactly what AI Functions automate; knowing the manual form means the pattern
# works anywhere — and you understand what you're paying for.

# %%
import couchbase.subdocument as SD
from openai import BadRequestError, OpenAI

if not AI_FUNCTIONS_ENABLED:
    if os.getenv("CAPELLA_AI_ENDPOINT"):
        import base64
        key = os.getenv("CAPELLA_AI_TOKEN") or base64.b64encode(
            f"{os.environ['CB_USERNAME']}:{os.environ['CB_PASSWORD']}".encode()).decode()
        client = OpenAI(base_url=os.environ["CAPELLA_AI_ENDPOINT"], api_key=key)
        MODEL = os.getenv("CAPELLA_LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    else:
        client = OpenAI()
        MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

    LABELS = ["billing", "bug", "feature-request", "account"]

    def enrich(body: str) -> dict:
        messages = [
            {"role": "system", "content":
                "You enrich support tickets. Respond with ONLY JSON, no prose: "
                '{"sentiment": "positive|negative|neutral|mixed", '
                f'"category": one of {LABELS}, '
                '"summary": "<= 20 words"}'},
            {"role": "user", "content": body},
        ]
        try:
            # Not every OpenAI-compatible endpoint honors response_format (Ch. 8) —
            # fall back to plain prompting if it's rejected outright.
            resp = client.chat.completions.create(
                model=MODEL, temperature=0,
                response_format={"type": "json_object"}, messages=messages)
        except BadRequestError:
            resp = client.chat.completions.create(model=MODEL, temperature=0, messages=messages)
        text = resp.choices[0].message.content
        return json.loads(text[text.find("{"):text.rfind("}") + 1])

    # SELECT the unenriched batch...
    batch = list(cluster.query(f"""
        SELECT META(t).id AS k, t.body FROM `{CB_BUCKET}`.support.tickets t
        WHERE t.category IS MISSING LIMIT 50"""))

    # ...enrich, and write back with subdoc (validating the label — LLM output is still LLM output)
    for row in batch:
        result = enrich(row["body"])
        category = result["category"] if result["category"] in LABELS else "unclassified"
        tickets.mutate_in(row["k"], (
            SD.upsert("sentiment", result["sentiment"]),
            SD.upsert("category", category),
            SD.upsert("summary", result["summary"]),
            SD.upsert("enriched_by", MODEL),
        ))
        print(f"{row['k']}: {category} / {result['sentiment']} — {result['summary']}")

# %% [markdown]
# ## Either path: enriched fields are now just... fields
#
# The payoff — downstream, AI-derived data is queryable like any other data:

# %%
for row in cluster.query(f"""
    SELECT t.category, COUNT(*) AS n,
           ARRAY_AGG(t.summary)[0] AS example
    FROM `{CB_BUCKET}`.support.tickets t
    WHERE t.category IS NOT MISSING
    GROUP BY t.category ORDER BY n DESC"""):
    print(row)

# %% [markdown]
# Engineering guidance (Ch. 7 §7.6): temperature 0 for transforms; enrich **at rest** for
# anything read twice; validate labels before trusting them; and never put a per-row LLM
# call in a hot-path query without a cache.
#
# **Next:** [05 — The Capella Model Service](05_model_service.ipynb)
