# %% [markdown]
# # 08 — Evaluating RAG with Ragas (and Storing Results in Couchbase)
#
# Companion to [Chapter 13](../docs/13-evaluation-ragas.md). Couchbase has no built-in
# eval framework — we use **Ragas**, and store every run in Couchbase so quality becomes
# a queryable time series.
#
# 1. Build an eval set and run the notebook-03 pipeline over it
# 2. Score with the four core RAG metrics
# 3. Persist runs + per-sample scores to `ai.evals.*`
# 4. Regression analysis with SQL++
#
# **Prerequisites:** notebooks 01–03 (corpus, index, RAG chain); `OPENAI_API_KEY`.

# %%
# %pip install -q couchbase python-dotenv ragas langchain-couchbase langchain-openai

# %%
import os
import subprocess
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv

load_dotenv()

from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions, KnownConfigProfiles, QueryOptions

opts = ClusterOptions(PasswordAuthenticator(os.getenv("CB_USERNAME", "Administrator"),
                                            os.getenv("CB_PASSWORD", "password")))
conn = os.getenv("CB_CONN_STRING", "couchbase://localhost")
if conn.startswith("couchbases://"):
    opts.apply_profile(KnownConfigProfiles.WanDevelopment)
cluster = Cluster.connect(conn, opts)
cluster.wait_until_ready(timedelta(seconds=10))

CB_BUCKET = os.getenv("CB_BUCKET", "ai")
bucket = cluster.bucket(CB_BUCKET)

# %% [markdown]
# ## 1. The system under test — notebook 03's RAG pipeline, reassembled

# %%
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_couchbase.vectorstores import CouchbaseSearchVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

vector_store = CouchbaseSearchVectorStore(
    cluster=cluster, bucket_name=CB_BUCKET, scope_name="docs",
    collection_name="chunks", embedding=embeddings, index_name="chunks-vector-index",
)

answer_prompt = ChatPromptTemplate.from_template(
    "Answer using ONLY this context. If it's not in the context, say you don't know."
    "\n\nContext:\n{context}\n\nQuestion: {question}")
gen_chain = answer_prompt | llm | StrOutputParser()


def run_pipeline(question: str, k: int = 3) -> dict:
    """Run retrieval + generation, returning everything Ragas needs."""
    docs = vector_store.similarity_search(question, k=k)
    contexts = [d.page_content for d in docs]
    answer = gen_chain.invoke({"context": "\n\n---\n\n".join(contexts),
                               "question": question})
    return {"question": question, "contexts": contexts, "answer": answer}

# %% [markdown]
# ## 2. The eval set
#
# Question + reference pairs. Start with 20–50 written from *real user questions*; grow it
# every time you find a failure. It's data — in a real project it lives in `ai.evals.cases`,
# versioned like code. (Ours is small so the notebook runs cheaply.)

# %%
EVAL_CASES = [
    {"question": "How do I rotate database credentials in Capella?",
     "reference": "Open Settings, choose Database Access, create a new credential, deploy "
                  "it to applications, then revoke the old one. Rotate at least quarterly."},
    {"question": "What settings does a vector field need in a Couchbase search index?",
     "reference": "dims matching the embedding model, similarity (dot_product or l2_norm), "
                  "and vector_index_optimized_for (recall or latency)."},
    {"question": "How should agent short-term memory expire?",
     "reference": "Session documents carry a TTL that slides on each interaction, so "
                  "inactive sessions expire automatically."},
    {"question": "How do you delete all memories for a user under GDPR?",
     "reference": "Run a SQL++ DELETE on the memories collection filtered by user_id."},
]

# %%
from ragas import EvaluationDataset, SingleTurnSample

samples = []
for case in EVAL_CASES:
    out = run_pipeline(case["question"])
    samples.append(SingleTurnSample(
        user_input=out["question"],
        retrieved_contexts=out["contexts"],
        response=out["answer"],
        reference=case["reference"],
    ))
print(f"collected {len(samples)} samples")
print("example answer:", samples[0].response[:120], "…")

# %% [markdown]
# ## 3. Score: the four metrics that triangulate failure
#
# - `faithfulness` low → generation invents (fix prompt/model)
# - `context_recall` low → retrieval misses (fix chunking/embeddings/hybrid)
# - `context_precision` low → retrieval is noisy (fix k/filters)
# - `answer_relevancy` low → answers dodge the question
#
# Judge rules: strong model, pinned version, temperature 0.

# %%
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (answer_relevancy, context_precision, context_recall,
                           faithfulness)

judge = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0))

result = evaluate(
    dataset=EvaluationDataset(samples=samples),
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=judge,
    embeddings=embeddings,
)
result

# %% [markdown]
# ## 4. Persist the run — quality as a time series

# %%
def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


runs_coll = bucket.scope("evals").collection("runs")
samples_coll = bucket.scope("evals").collection("samples")

df = result.to_pandas()
metric_cols = [c for c in df.columns
               if c not in ("user_input", "retrieved_contexts", "response", "reference")]

run_key = "evalrun::" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
runs_coll.upsert(run_key, {
    "type": "eval_run",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "git_commit": git_sha(),
    "pipeline": {"embedding_model": "text-embedding-3-small", "k": 3,
                 "chunking": "markdown-600-80", "llm": "gpt-4o-mini",
                 "judge": "gpt-4o"},
    "metrics": {m: float(df[m].mean()) for m in metric_cols},
    "n_samples": len(df),
})

for i, row in enumerate(df.to_dict("records")):
    samples_coll.upsert(f"{run_key}::sample::{i:04d}", {
        "type": "eval_sample", "run": run_key,
        "user_input": row["user_input"],
        "response": row["response"],
        "reference": row["reference"],
        **{m: (float(row[m]) if row[m] == row[m] else None) for m in metric_cols},  # NaN-safe
    })

print("stored", run_key, "with", len(df), "samples")

# %% [markdown]
# ## 5. Regression analysis in SQL++
#
# Aggregate scores say *that* something broke; per-sample storage says *what*.

# %%
cluster.query(f"CREATE PRIMARY INDEX IF NOT EXISTS ON `{CB_BUCKET}`.evals.runs").execute()
cluster.query(f"CREATE PRIMARY INDEX IF NOT EXISTS ON `{CB_BUCKET}`.evals.samples").execute()

print("run history:")
for row in cluster.query(f"""
    SELECT r.created_at, r.git_commit, r.pipeline.embedding_model,
           ROUND(r.metrics.faithfulness, 3) AS faithfulness,
           ROUND(r.metrics.context_recall, 3) AS recall
    FROM `{CB_BUCKET}`.evals.runs r
    ORDER BY r.created_at DESC LIMIT 5"""):
    print(" ", row)

# %%
# The debugging goldmine: which QUESTIONS scored worst this run?
for row in cluster.query(f"""
    SELECT s.user_input, ROUND(s.faithfulness, 3) AS faithfulness,
           ROUND(s.context_recall, 3) AS recall
    FROM `{CB_BUCKET}`.evals.samples s
    WHERE s.run = $run
    ORDER BY s.faithfulness ASC LIMIT 3""",
    QueryOptions(named_parameters={"run": run_key})):
    print(row)

# %% [markdown]
# Re-run this notebook after changing anything upstream — chunk size in 02, k or prompt in
# 03, the embedding model — and the runs table becomes your before/after evidence. Wire the
# same code into CI as a PR gate (Ch. 13 §13.6: smoke set per-PR, full set nightly).
#
# For **agent** evaluation (tool efficiency, response faithfulness from activity logs), see
# `apps/support-agent/evals/` and Ragas' Agent Catalog integration
# (`ragas.integrations.agentc`).
#
# *This closes the loop the book opened in Chapter 1: ship → observe → evaluate → fix →
# re-measure, with every byte of it — corpus, memory, catalogs, checkpoints, and now
# quality scores — in one database.*
