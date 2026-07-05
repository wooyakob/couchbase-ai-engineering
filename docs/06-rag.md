# Chapter 6 — Retrieval-Augmented Generation

> *RAG is the pattern that made foundation models useful for private data: retrieve what's relevant, stuff it in the prompt, generate a grounded answer. Chapters 3–5 built every ingredient; this chapter assembles them — and then makes the result fast and stateful with caching and chat history, all on Couchbase.*

## 6.1 The minimal RAG chain

With the `ai.docs.chunks` corpus and index from Chapter 5, a complete RAG pipeline in LangChain:

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_couchbase.vectorstores import CouchbaseSearchVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# Capella Model Service instead? Swap per Chapter 8 — nothing else changes.

vector_store = CouchbaseSearchVectorStore(
    cluster=cluster, bucket_name="ai", scope_name="docs",
    collection_name="chunks", embedding=embeddings,
    index_name="chunks-vector-index",
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
    {"context": vector_store.as_retriever(search_kwargs={"k": 5}) | format_docs,
     "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

rag_chain.invoke("How do I rotate database credentials?")
```

That's a demo. The rest of the chapter is what separates a demo from a system.

## 6.2 Grounding discipline

- **Instruct refusal** ("say you don't know") *and* verify it: Ragas `faithfulness` (Ch. 13) measures whether answers actually stick to retrieved context.
- **Cite sources.** Return chunk lineage with the answer — retrieval results carry `metadata` — so users (and evaluators) can check. The `rag-api` app returns `sources: [...]` on every answer.
- **Bound the context.** k=5 × 2000-char chunks ≈ manageable; k=20 stuffs noise and cost into every call. Retrieve wide, *select* narrow (score cutoffs or a reranker between retrieval and prompt).
- **Filter at retrieval, not in the prompt.** Tenancy and permissions belong in prefilters (§5.4) — never rely on "please only use documents belonging to tenant X."

## 6.3 Caching: the same question twice

LLM calls are the slowest, priciest thing you run. Two Couchbase-backed caches from `langchain-couchbase`, both drop-in:

**Exact-match cache** — same prompt string → stored response:

```python
from langchain_core.globals import set_llm_cache
from langchain_couchbase.cache import CouchbaseCache

set_llm_cache(CouchbaseCache(
    cluster=cluster, bucket_name="ai",
    scope_name="docs", collection_name="llm_cache",
))
```

**Semantic cache** — a new question *similar enough* to a cached one returns the cached answer. This is where real hit-rates come from (users never phrase things identically), and where the risk is — set the threshold empirically (§5.6):

```python
from langchain_couchbase.cache import CouchbaseSemanticCache

set_llm_cache(CouchbaseSemanticCache(
    cluster=cluster, embedding=embeddings,
    bucket_name="ai", scope_name="docs", collection_name="semantic_cache",
    index_name="semantic-cache-index",          # a vector index on this collection
    score_threshold=0.8,
))
```

Cache-safety rules: only cache **user-independent** answers (a cached answer computed from tenant A's context must never serve tenant B — if context differs per user, key the cache accordingly or don't cache); give cache collections a TTL (`max_expiry` at collection creation) so stale answers age out. Note Capella's Model Service can also do both kinds of caching *at the serving layer* (§8.3) — pick one layer, not both.

## 6.4 Conversation: multi-turn RAG

Single-shot RAG can't answer "and how do I undo that?" — the question only means something given history. Store history in Couchbase:

```python
from langchain_couchbase.chat_message_histories import CouchbaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

def history_for(session_id: str):
    return CouchbaseChatMessageHistory(
        cluster=cluster, bucket_name="ai",
        scope_name="agent", collection_name="chat_history",
        session_id=session_id,
    )

conversational_chain = RunnableWithMessageHistory(
    rag_chain_with_history_slot,        # a chain whose prompt includes MessagesPlaceholder("history")
    history_for,
    input_messages_key="question",
    history_messages_key="history",
)

conversational_chain.invoke(
    {"question": "and how do I undo that?"},
    config={"configurable": {"session_id": "u42::2026-07-05"}},
)
```

Two patterns for using history well:

- **Query condensation**: before retrieval, have the LLM rewrite the user's message into a standalone query given the history ("and how do I undo that?" → "how do I undo a credential rotation in Capella?"). Retrieval quality on follow-ups depends on this.
- **Windowing**: prompt with the last N turns, not all of them. For durable cross-session memory, that's Chapter 9's job, not chat history's.

## 6.5 The assembled service

`apps/rag-api` is this chapter as a running FastAPI service:

```
POST /ingest     → chunk + embed + upsert (Ch. 3–4)
POST /ask        → condense → hybrid retrieve (Ch. 5) → generate → cite
                   with semantic cache in front and chat history per session
GET  /sessions/{id} → conversation transcript from Couchbase
```

Notable engineering details in the app:

- Retrieval uses the **raw SDK hybrid query** (§5.4) rather than the vanilla retriever — keyword + vector with a tenant prefilter.
- Every answer document (question, answer, sources, latencies, cache-hit flag) is **logged to `ai.evals.samples`** — Chapter 13 evaluates straight from production traffic.
- Config is env-driven; switching OpenAI → Capella Model Service is a 3-variable change (Ch. 8).

## 6.6 Failure modes and their meters

| Symptom | Usual cause | Meter (Ch. 13) |
|---|---|---|
| Confident wrong answers | retrieval missed; model free-styled | `faithfulness` ↓ |
| Right doc exists, not retrieved | chunking too coarse/fine; embedding mismatch; missing hybrid | `context_recall` ↓ |
| Answers drown in noise | k too high; no score cutoff | `context_precision` ↓ |
| Great first answer, lost follow-ups | no query condensation | manual/eval set |
| Fast then suddenly slow | cache misses (threshold too strict) | cache-hit rate in app logs |

RAG quality work is iterating on this table with an eval set — which is exactly why eval results live in Couchbase where you can query them across versions.

Notebook: [`notebooks/03_rag_pipeline.ipynb`](../notebooks/03_rag_pipeline.ipynb). App: [`apps/rag-api`](../apps/rag-api/).

Next: [Chapter 7 — AI Functions](07-ai-functions.md).
