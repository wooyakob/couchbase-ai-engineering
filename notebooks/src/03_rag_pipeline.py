# %% [markdown]
# # 03: A Complete RAG Pipeline
#
# Companion to [Chapter 6](../docs/06-rag.md). Builds on notebook 02's corpus and index.
#
# 1. RAG chain with `langchain-couchbase`
# 2. LLM caching: exact-match and semantic, both on Couchbase
# 3. Multi-turn RAG with Couchbase-backed chat history + query condensation
#
# **Prerequisites:** notebooks 01–02 have been run; `OPENAI_API_KEY` in `.env`
# (or Capella Model Service, same switch as notebook 02).

# %%
%pip install -q couchbase python-dotenv langchain-couchbase langchain-openai

# %%
import os
from datetime import timedelta

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(os.getenv("ENV_FILE", ".env"), usecwd=True))  # ENV_FILE lets .env.server / .env.capella run side by side

from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions, KnownConfigProfiles

opts = ClusterOptions(PasswordAuthenticator(os.getenv("CB_USERNAME"),
                                            os.getenv("CB_PASSWORD")))
conn = os.getenv("CB_CONN_STRING")
if conn.startswith("couchbases://"):
    opts.apply_profile(KnownConfigProfiles.WanDevelopment)
cluster = Cluster.connect(conn, opts)
cluster.wait_until_ready(timedelta(seconds=10))

CB_BUCKET = os.getenv("CB_BUCKET")

# %% [markdown]
# ## Models: OpenAI default, Capella Model Service switch (Ch. 8)

# %%
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

if os.getenv("CAPELLA_AI_ENDPOINT"):
    import base64
    key = os.getenv("CAPELLA_AI_TOKEN") or base64.b64encode(
        f"{os.environ['CB_USERNAME']}:{os.environ['CB_PASSWORD']}".encode()).decode()
    llm = ChatOpenAI(openai_api_base=os.environ["CAPELLA_AI_ENDPOINT"], openai_api_key=key,
                     model=os.getenv("CAPELLA_LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
                     temperature=0)
    embeddings = OpenAIEmbeddings(openai_api_base=os.environ["CAPELLA_AI_ENDPOINT"],
                                  openai_api_key=key,
                                  model=os.getenv("CAPELLA_EMBEDDING_MODEL",
                                                  "intfloat/e5-mistral-7b-instruct"),
                                  check_embedding_ctx_length=False, tiktoken_enabled=False)
else:
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0)
    embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))

# %% [markdown]
# ## 1. The RAG chain
#
# `CouchbaseSearchVectorStore` reads the corpus notebook 02 built (`text` / `embedding` /
# `metadata` fields, exactly what the index maps).

# %%
import warnings

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_couchbase.vectorstores import CouchbaseSearchVectorStore
from couchbase.logic.supportability import CouchbaseDeprecationWarning

# The SDK's scope.search() request builder unconditionally *reads* the deprecated
# SearchOptions.scope_name property internally (to decide which scope to run
# against) even though neither this notebook nor langchain-couchbase ever sets
# scope_name, the warning fires on every scoped search (here, and later via the
# LLM caches) regardless of caller code. No real migration is available on our
# side, so we silence this specific, known-noisy warning once, notebook-wide.
warnings.filterwarnings("ignore", category=CouchbaseDeprecationWarning,
                        message=".*scope_name.*")

vector_store = CouchbaseSearchVectorStore(
    cluster=cluster,
    bucket_name=CB_BUCKET, scope_name="docs", collection_name="chunks",
    embedding=embeddings, index_name="chunks-vector-index",
)

prompt = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the context below.
If the context doesn't contain the answer, say you don't know.

Context:
{context}

Question: {question}"""
)


def format_docs(docs):
    return "\n\n---\n\n".join(d.page_content for d in docs)


rag_chain = (
    {"context": vector_store.as_retriever(search_kwargs={"k": 3}) | format_docs,
     "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

print(rag_chain.invoke("How do I rotate database credentials in Capella?"))

# %% [markdown]
# Grounding check: a question the corpus can't answer should get a refusal, not a guess:

# %%
print(rag_chain.invoke("What is the capital of France?"))

# %% [markdown]
# ## 2a. Exact-match LLM cache on Couchbase

# %%
import time

from langchain_core.globals import set_llm_cache
from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
from langchain_couchbase.cache import CouchbaseCache

# CouchbaseCache deserializes cached generations via langchain_core's `loads()`,
# which as of langchain-core 1.3.3 warns unless the caller passes an explicit
# `allowed_objects` allowlist, but that parameter isn't threaded through
# CouchbaseCache's constructor (it only accepts cluster/bucket/scope/collection/
# ttl), so there's nothing in this notebook to configure. We only ever cache our
# own LLM `Generation` objects here (nothing untrusted), so the risk this warning
# guards against doesn't apply; silence it rather than fork/monkeypatch the cache.
warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning,
                        message=".*allowed_objects.*")

set_llm_cache(CouchbaseCache(
    cluster=cluster, bucket_name=CB_BUCKET,
    scope_name="docs", collection_name="llm_cache",
))

q = "How do I rotate database credentials in Capella?"
t0 = time.perf_counter(); rag_chain.invoke(q); cold = time.perf_counter() - t0
t0 = time.perf_counter(); rag_chain.invoke(q); warm = time.perf_counter() - t0
print(f"cold: {cold:.2f}s   warm (cached): {warm:.2f}s")

# %% [markdown]
# `cold` is the first call: a cache miss, so the full chain runs end-to-end: retrieval,
# then a real LLM generation. `warm (cached)` re-runs the *identical* prompt:
# `CouchbaseCache` hashes the prompt + model params, finds a stored `Generation` for that
# exact hash, and returns it directly; no LLM call at all. That's why `warm` is faster,
# but retrieval still runs every time (only the LLM step is cached), and the *hit* itself
# is a Couchbase KV read over the network, not free. So the gap between the two numbers
# is roughly the LLM generation time you skipped; the remaining ~0.5s is retrieval + the
# cache round-trip, not zero.
#
# ## 2b. Semantic cache
#
# Real users never phrase things identically; semantic caching is where hit-rates come from.
# It needs a vector index over the cache collection; we create a minimal one here.
# **Tune `score_threshold` empirically** (Ch. 6 §6.3): too loose serves wrong answers,
# too strict never hits.

# %%
from couchbase.management.search import SearchIndex

EMBEDDING_DIM = len(embeddings.embed_query("probe"))
SEM_CACHE_INDEX = "semantic-cache-index"

cache_index_def = {
    "type": "fulltext-index", "name": SEM_CACHE_INDEX,
    "sourceType": "gocbcore", "sourceName": CB_BUCKET,
    "planParams": {"maxPartitionsPerPIndex": 1024, "indexPartitions": 1},
    "params": {
        "doc_config": {"mode": "scope.collection.type_field", "type_field": "type"},
        "mapping": {
            "default_mapping": {"dynamic": False, "enabled": False},
            "types": {
                "docs.semantic_cache": {
                    "dynamic": False, "enabled": True,
                    "properties": {
                        "text": {"enabled": True, "dynamic": False,
                                 "fields": [{"name": "text", "type": "text", "index": True,
                                             "store": True, "analyzer": "en"}]},
                        "embedding": {"enabled": True, "dynamic": False,
                                      "fields": [{"name": "embedding", "type": "vector",
                                                  "index": True, "dims": EMBEDDING_DIM,
                                                  "similarity": "dot_product",
                                                  "vector_index_optimized_for": "recall"}]},
                        # langchain_couchbase reads these back via the search API's
                        # `fields=["*"]`, which only returns *stored* fields, dynamic
                        # mapping alone doesn't store metadata subfields, so both are
                        # declared explicitly (Ch. 6 §6.3).
                        "metadata": {"enabled": True, "dynamic": False, "properties": {
                            "llm_string": {"enabled": True, "dynamic": False,
                                          "fields": [{"name": "llm_string", "type": "text",
                                                      "index": True, "store": True,
                                                      "analyzer": "keyword"}]},
                            "return_val": {"enabled": True, "dynamic": False,
                                          "fields": [{"name": "return_val", "type": "text",
                                                      "index": True, "store": True,
                                                      "analyzer": "keyword"}]},
                        }},
                    },
                }
            },
        },
        "store": {"indexType": "scorch", "segmentVersion": 16},
    },
    "sourceParams": {},
}
from couchbase.exceptions import SearchIndexNotFoundException

sem_cache_index_mgr = cluster.bucket(CB_BUCKET).scope("docs").search_indexes()
try:
    sem_cache_index_mgr.drop_index(SEM_CACHE_INDEX)  # rerun-safe: replace any stale definition
except SearchIndexNotFoundException:
    pass
sem_cache_index_mgr.upsert_index(SearchIndex.from_json(cache_index_def))

# %%
from langchain_couchbase.cache import CouchbaseSemanticCache

set_llm_cache(CouchbaseSemanticCache(
    cluster=cluster, embedding=embeddings,
    bucket_name=CB_BUCKET, scope_name="docs", collection_name="semantic_cache",
    index_name=SEM_CACHE_INDEX, score_threshold=0.8,
))

rag_chain.invoke("How do I rotate database credentials in Capella?")   # seeds the cache
time.sleep(2)                                                          # let the index ingest
t0 = time.perf_counter()
out = rag_chain.invoke("What are the steps to change my Capella DB credentials?")  # paraphrase!
print(f"paraphrase answered in {time.perf_counter() - t0:.2f}s")
print(out[:200])

# %% [markdown]
# Note this is a *different* question string from the seed call: "What are the steps to
# change my Capella DB credentials?" vs. "How do I rotate database credentials in
# Capella?" There's no exact-match hit possible. `CouchbaseSemanticCache` instead embeds
# the new question, does a vector search against previously-cached questions, and returns
# the stored answer if similarity clears `score_threshold` (0.8 here). The ~1.85s you see
# is still slower than an exact-match hit (cell above) because it pays for an embedding
# call *plus* a vector search, not just a KV lookup, but it's serving the seeded answer
# from cache, not generating a fresh one, which is why it's meaningfully faster than a
# full `cold` RAG call once you factor out embedding overhead.
#
# ## 3. Multi-turn RAG: chat history + query condensation
#
# "and how do I undo that?" only means something given history. Two pieces:
# history stored in Couchbase, and a condensation step that rewrites follow-ups into
# standalone queries *before retrieval* (Ch. 6 §6.4).

# %%
set_llm_cache(None)  # don't cache conversational answers

from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core._api.deprecation import LangChainDeprecationWarning
from langchain_couchbase.chat_message_histories import CouchbaseChatMessageHistory

condense_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the user's message as a standalone search query, "
               "using the conversation history for context. Return ONLY the query."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])
condense = condense_prompt | llm | StrOutputParser()

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using ONLY this context:\n\n{context}"),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])


def retrieve_condensed(x):
    standalone = condense.invoke(x) if x["history"] else x["question"]
    # k=1, not the k=5 used elsewhere in this notebook: this demo corpus is only
    # 3 short docs (~5 chunks total, notebook 02 §1), so k=3 was pulling in a
    # chunk from every doc regardless of how specific the condensed query was,
    # e.g. "how often" also nearest-neighbor-matches the index-configuration and
    # memory-hygiene chunks and the model would blend all three into its answer.
    docs = vector_store.similarity_search(standalone, k=1)
    return format_docs(docs)


# RunnableWithMessageHistory is deprecated in favor of LangGraph's built-in
# persistence: a real migration means rebuilding this chain as a graph, which
# is out of scope for this notebook. Scope the suppression tightly to
# construction/invocation of this one deprecated API rather than hiding
# unrelated warnings.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning,
                            message=".*RunnableWithMessageHistory.*")

    conversational_chain = RunnableWithMessageHistory(
        (RunnablePassthrough.assign(context=retrieve_condensed)
         | answer_prompt | llm | StrOutputParser()),
        lambda session_id: CouchbaseChatMessageHistory(
            cluster=cluster, bucket_name=CB_BUCKET,
            scope_name="agent", collection_name="chat_history",
            session_id=session_id),
        input_messages_key="question",
        history_messages_key="history",
    )

    cfg = {"configurable": {"session_id": "demo::notebook-03"}}
    print(conversational_chain.invoke({"question": "How do I rotate credentials in Capella?"}, cfg))

# %%
# The follow-up only resolves to "credentials" because condensation turns it into a
# standalone query first; without condensation the raw question "How often should I
# do that?" carries no topic for retrieval to match against at all.
print(conversational_chain.invoke({"question": "How often should I do that?"}, cfg))

# %% [markdown]
# The transcript is just documents in `ai.agent.chat_history`: inspect it with SQL++,
# expire it with TTL, delete it per-user for GDPR. Storage you can reason about.
#
# **Next:** [04: AI Functions](04_ai_functions.ipynb)
