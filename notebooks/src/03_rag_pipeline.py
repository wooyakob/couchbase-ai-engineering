# %% [markdown]
# # 03 — A Complete RAG Pipeline
#
# Companion to [Chapter 6](../docs/06-rag.md). Builds on notebook 02's corpus and index.
#
# 1. RAG chain with `langchain-couchbase`
# 2. LLM caching: exact-match and semantic, both on Couchbase
# 3. Multi-turn RAG with Couchbase-backed chat history + query condensation
#
# **Prerequisites:** notebooks 01–02 have been run; `OPENAI_API_KEY` in `.env`
# (or Capella Model Service — same switch as notebook 02).

# %%
# %pip install -q couchbase python-dotenv langchain-couchbase langchain-openai

# %%
import os
from datetime import timedelta

from dotenv import load_dotenv

load_dotenv()

from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions, KnownConfigProfiles

opts = ClusterOptions(PasswordAuthenticator(os.getenv("CB_USERNAME", "Administrator"),
                                            os.getenv("CB_PASSWORD", "password")))
conn = os.getenv("CB_CONN_STRING", "couchbase://localhost")
if conn.startswith("couchbases://"):
    opts.apply_profile(KnownConfigProfiles.WanDevelopment)
cluster = Cluster.connect(conn, opts)
cluster.wait_until_ready(timedelta(seconds=10))

CB_BUCKET = os.getenv("CB_BUCKET", "ai")

# %% [markdown]
# ## Models — OpenAI default, Capella Model Service switch (Ch. 8)

# %%
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

if os.getenv("CAPELLA_AI_ENDPOINT"):
    import base64
    key = base64.b64encode(
        f"{os.environ['CB_USERNAME']}:{os.environ['CB_PASSWORD']}".encode()).decode()
    llm = ChatOpenAI(openai_api_base=os.environ["CAPELLA_AI_ENDPOINT"], openai_api_key=key,
                     model="meta-llama/Llama-3.1-8B-Instruct", temperature=0)
    embeddings = OpenAIEmbeddings(openai_api_base=os.environ["CAPELLA_AI_ENDPOINT"],
                                  openai_api_key=key,
                                  model="intfloat/e5-mistral-7b-instruct",
                                  check_embedding_ctx_length=False, tiktoken_enabled=False)
else:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# %% [markdown]
# ## 1. The RAG chain
#
# `CouchbaseSearchVectorStore` reads the corpus notebook 02 built (`text` / `embedding` /
# `metadata` fields — exactly what the index maps).

# %%
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_couchbase.vectorstores import CouchbaseSearchVectorStore

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
# Grounding check — a question the corpus can't answer should get a refusal, not a guess:

# %%
print(rag_chain.invoke("What is the capital of France?"))

# %% [markdown]
# ## 2a. Exact-match LLM cache on Couchbase

# %%
import time

from langchain_core.globals import set_llm_cache
from langchain_couchbase.cache import CouchbaseCache

set_llm_cache(CouchbaseCache(
    cluster=cluster, bucket_name=CB_BUCKET,
    scope_name="docs", collection_name="llm_cache",
))

q = "How do I rotate database credentials in Capella?"
t0 = time.perf_counter(); rag_chain.invoke(q); cold = time.perf_counter() - t0
t0 = time.perf_counter(); rag_chain.invoke(q); warm = time.perf_counter() - t0
print(f"cold: {cold:.2f}s   warm (cached): {warm:.2f}s")

# %% [markdown]
# ## 2b. Semantic cache
#
# Real users never phrase things identically — semantic caching is where hit-rates come from.
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
                    "dynamic": True, "enabled": True,
                    "properties": {
                        "embedding": {"enabled": True, "dynamic": False,
                                      "fields": [{"name": "embedding", "type": "vector",
                                                  "index": True, "dims": EMBEDDING_DIM,
                                                  "similarity": "dot_product",
                                                  "vector_index_optimized_for": "recall"}]},
                    },
                }
            },
        },
        "store": {"indexType": "scorch", "segmentVersion": 16},
    },
    "sourceParams": {},
}
cluster.bucket(CB_BUCKET).scope("docs").search_indexes().upsert_index(
    SearchIndex.from_json(cache_index_def))

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
# ## 3. Multi-turn RAG: chat history + query condensation
#
# "and how do I undo that?" only means something given history. Two pieces:
# history stored in Couchbase, and a condensation step that rewrites follow-ups into
# standalone queries *before retrieval* (Ch. 6 §6.4).

# %%
set_llm_cache(None)  # don't cache conversational answers

from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
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
    docs = vector_store.similarity_search(standalone, k=3)
    return format_docs(docs)


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
# The follow-up only works because condensation turns it into a standalone query first
print(conversational_chain.invoke({"question": "How often should I do that?"}, cfg))

# %% [markdown]
# The transcript is just documents in `ai.agent.chat_history` — inspect it with SQL++,
# expire it with TTL, delete it per-user for GDPR. Storage you can reason about.
#
# **Next:** [04 — AI Functions](04_ai_functions.ipynb)
