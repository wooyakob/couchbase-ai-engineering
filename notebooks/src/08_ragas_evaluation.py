# %% [markdown]
# # 08: Evaluating RAG with Ragas (and Storing Results in Couchbase)
#
# Companion to [Chapter 13](../docs/13-evaluation-ragas.md). Couchbase has no built-in
# eval framework: we use **Ragas**, and store every run in Couchbase so quality becomes
# a queryable time series.
#
# 1. Build an eval set and run the notebook-03 pipeline over it
# 2. Score with the four core RAG metrics
# 3. Persist runs + per-sample scores to `ai.evals.*`
# 4. Regression analysis with SQL++
#
# **Prerequisites:** notebooks 01–03 (corpus, index, RAG chain); `OPENAI_API_KEY`
# (or `CAPELLA_AI_ENDPOINT` for the all-Capella path).

# %%
%pip install -q couchbase python-dotenv ragas langchain-couchbase langchain-openai tqdm

# %%
import os
import subprocess
from datetime import datetime, timezone, timedelta

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(os.getenv("ENV_FILE", ".env"), usecwd=True))  # ENV_FILE lets .env.server / .env.capella run side by side

# Ragas phones home around every judge call (t.explodinggradients.com). That endpoint
# no longer resolves, and the silently-swallowed DNS failure blocks the event loop for
# ~30s per call — the difference between this notebook scoring in seconds vs. an hour.
# Must be set before the first ragas import.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

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
bucket = cluster.bucket(CB_BUCKET)

# %% [markdown]
# ## 1. The system under test: notebook 03's RAG pipeline, reassembled

# %%
import warnings

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_couchbase.vectorstores import CouchbaseSearchVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# langchain_couchbase's CouchbaseSearchVectorStore passes scope_name into the FTS
# SearchOptions on every similarity_search() call; the search API itself ignores it
# (scoping comes from the index), so the couchbase SDK warns on every query. The
# vector store constructor still requires scope_name, so there's no way to avoid
# triggering it from here — silence just this one deprecation message.
warnings.filterwarnings("ignore", message=r"Option scope_name is deprecated.*")

if os.getenv("CAPELLA_AI_ENDPOINT"):
    import base64
    key = os.getenv("CAPELLA_AI_TOKEN") or base64.b64encode(
        f"{os.environ['CB_USERNAME']}:{os.environ['CB_PASSWORD']}".encode()).decode()
    embeddings = OpenAIEmbeddings(openai_api_base=os.environ["CAPELLA_AI_ENDPOINT"],
                                  openai_api_key=key,
                                  model=os.getenv("CAPELLA_EMBEDDING_MODEL",
                                                  "intfloat/e5-mistral-7b-instruct"),
                                  check_embedding_ctx_length=False, tiktoken_enabled=False)
    llm = ChatOpenAI(openai_api_base=os.environ["CAPELLA_AI_ENDPOINT"], openai_api_key=key,
                     model=os.getenv("CAPELLA_LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
                     temperature=0)
else:
    embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0)

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
# every time you find a failure. It's data; in a real project it lives in `ai.evals.cases`,
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
from ragas import SingleTurnSample

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
# Judge rules: strong model, pinned version, temperature 0. With `CAPELLA_AI_ENDPOINT`
# set, this notebook judges with the same Capella-hosted model the pipeline uses, so the
# whole notebook runs on one backend. Know the compromise you're making: Ch. 13's rule
# for real projects is a *strong* judge that is *not* the system under test.

# %%
import asyncio

import instructor
import pandas as pd
from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.llms.base import InstructorLLM, InstructorModelArgs
from ragas.metrics.collections import (AnswerRelevancy, ContextPrecisionWithReference,
                                       ContextRecall, Faithfulness)
from tqdm.auto import tqdm

if os.getenv("CAPELLA_AI_ENDPOINT"):
    judge_client = AsyncOpenAI(base_url=os.environ["CAPELLA_AI_ENDPOINT"], api_key=key)
    judge_model = os.getenv("CAPELLA_LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    judge_embed_model = os.getenv("CAPELLA_EMBEDDING_MODEL", "intfloat/e5-mistral-7b-instruct")
    # llm_factory's default Mode.JSON sends a response_format param Capella AI Services
    # rejects ("unsupported response format type"); MD_JSON sends none.
    judge_llm = InstructorLLM(
        client=instructor.from_openai(judge_client, mode=instructor.Mode.MD_JSON),
        model=judge_model, provider="openai", model_args=InstructorModelArgs())
else:
    judge_client = AsyncOpenAI()
    judge_model = "gpt-4o"
    judge_embed_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    judge_llm = llm_factory(judge_model, client=judge_client)

judge_embeddings = RagasOpenAIEmbeddings(client=judge_client, model=judge_embed_model)

faithfulness_metric = Faithfulness(llm=judge_llm)
answer_relevancy_metric = AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings)
context_precision_metric = ContextPrecisionWithReference(llm=judge_llm)
context_recall_metric = ContextRecall(llm=judge_llm)

max_workers, timeout = 8, 300
semaphore = asyncio.Semaphore(max_workers)
METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
print(f"scoring {len(samples)} samples × {len(METRIC_NAMES)} metrics...")
progress = tqdm(total=len(samples) * len(METRIC_NAMES), desc="scoring", unit="metric")


async def score_sample(sample):
    async def bounded(coro):
        async with semaphore:
            try:
                return await asyncio.wait_for(coro, timeout=timeout)
            finally:
                progress.update(1)  # count failures too — NaNs in the table flag them

    faithfulness, answer_relevancy, context_precision, context_recall = await asyncio.gather(
        bounded(faithfulness_metric.ascore(
            user_input=sample.user_input, response=sample.response,
            retrieved_contexts=sample.retrieved_contexts)),
        bounded(answer_relevancy_metric.ascore(
            user_input=sample.user_input, response=sample.response)),
        bounded(context_precision_metric.ascore(
            user_input=sample.user_input, reference=sample.reference,
            retrieved_contexts=sample.retrieved_contexts)),
        bounded(context_recall_metric.ascore(
            user_input=sample.user_input, retrieved_contexts=sample.retrieved_contexts,
            reference=sample.reference)),
        return_exceptions=True,
    )

    def val(r):
        return r.value if not isinstance(r, Exception) else float("nan")

    return {
        "user_input": sample.user_input,
        "retrieved_contexts": sample.retrieved_contexts,
        "response": sample.response,
        "reference": sample.reference,
        "faithfulness": val(faithfulness),
        "answer_relevancy": val(answer_relevancy),
        "context_precision": val(context_precision),
        "context_recall": val(context_recall),
    }


try:
    rows = await asyncio.gather(*(score_sample(s) for s in samples))
finally:
    progress.close()
result = pd.DataFrame(rows)
result

# %% [markdown]
# ## 4. Persist the run: quality as a time series

# %%
def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


runs_coll = bucket.scope("evals").collection("runs")
samples_coll = bucket.scope("evals").collection("samples")

df = result  # already a DataFrame — the collections API scores per-sample, not via evaluate()
metric_cols = [c for c in df.columns
               if c not in ("user_input", "retrieved_contexts", "response", "reference")]

run_key = "evalrun::" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
runs_coll.upsert(run_key, {
    "type": "eval_run",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "git_commit": git_sha(),
    "pipeline": {"embedding_model": embeddings.model, "k": 3,
                 "chunking": "markdown-600-80", "llm": llm.model_name,
                 "judge": judge_model},
    "metrics": {m: (None if pd.isna(v := df[m].mean()) else float(v)) for m in metric_cols},
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
# Re-run this notebook after changing anything upstream: chunk size in 02, k or prompt in
# 03, the embedding model, and the runs table becomes your before/after evidence. Wire the
# same code into CI as a PR gate (Ch. 13 §13.6: smoke set per-PR, full set nightly).
#
# For **agent** evaluation (tool efficiency, response faithfulness from activity logs), see
# `apps/support-agent/evals/` and Ragas' Agent Catalog integration
# (`ragas.integrations.agentc`).
#
# *This closes the loop the book opened in Chapter 1: ship → observe → evaluate → fix →
# re-measure, with every byte of it (corpus, memory, catalogs, checkpoints, and now
# quality scores) in one database.*
