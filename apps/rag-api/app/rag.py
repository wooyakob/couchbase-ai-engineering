"""The RAG core: condense -> retrieve -> generate -> cite -> log (Chapter 6)."""

import time
from datetime import datetime, timezone
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_couchbase.chat_message_histories import CouchbaseChatMessageHistory

from . import config, db
from .models import make_llm
from .retrieval import hybrid_search

_llm = None


def llm():
    global _llm
    if _llm is None:
        _llm = make_llm()
    return _llm


CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the user's LAST message as a single standalone search query for "
               "a document retrieval system, using the conversation history only for "
               "context. Output exactly one line: a question or search phrase — never a "
               "list, never an answer, never wrapped in quotes.\n\n"
               "Example:\n"
               "history: user: How do I rotate credentials in Capella?\n"
               "         assistant: Open Settings, create a new credential, deploy it, "
               "revoke the old one.\n"
               "follow-up: and what do I do right after?\n"
               "standalone query: what to do immediately after creating a new Capella "
               "credential"),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using ONLY the context below. If the context does "
               "not contain the answer, say you don't know.\n\nContext:\n{context}"),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])


def _clean_standalone(raw: str, fallback: str) -> str:
    """Defends against weaker instruction-following models (observed with Capella's
    default Llama-3.1-8B) ignoring "one line, never an answer" — wrapping the query
    in quotes, or drifting into a multi-line answer instead of condensing. Strip the
    former; fall back to the raw question for the latter rather than sending a
    multi-line non-query into retrieval."""
    cleaned = raw.strip().strip('"').strip("'").strip()
    lines = [line for line in cleaned.splitlines() if line.strip()]
    if len(lines) != 1:
        return fallback
    return lines[0].strip()


def history_for(session_id: str) -> CouchbaseChatMessageHistory:
    return CouchbaseChatMessageHistory(
        cluster=db.cluster(), bucket_name=config.CB_BUCKET,
        scope_name=config.CHAT_SCOPE, collection_name=config.CHAT_COLLECTION,
        session_id=session_id,
    )


def answer(question: str, session_id: str, tenant: str | None = None) -> dict:
    t0 = time.perf_counter()
    history = history_for(session_id)
    past = history.messages[-10:]  # window (Ch. 6 §6.4)

    # 1. Condense follow-ups into a standalone query before retrieval
    if past:
        raw_standalone = (CONDENSE_PROMPT | llm() | StrOutputParser()).invoke(
            {"history": past, "question": question})
        standalone = _clean_standalone(raw_standalone, question)
    else:
        standalone = question
    t_condense = time.perf_counter()

    # 2. Hybrid retrieval with tenant prefilter (Ch. 5)
    hits = hybrid_search(standalone, tenant=tenant)
    context = "\n\n---\n\n".join(h["text"] for h in hits)
    t_retrieve = time.perf_counter()

    # 3. Grounded generation
    response = (ANSWER_PROMPT | llm() | StrOutputParser()).invoke(
        {"context": context, "history": past, "question": question})
    t_generate = time.perf_counter()

    # 4. Persist the turn
    history.add_message(HumanMessage(content=question))
    history.add_message(AIMessage(content=response))

    result = {
        "answer": response,
        "sources": [{"doc_id": h["doc_id"], "source": h["source"],
                     "score": h["score"]} for h in hits],
        "standalone_query": standalone,
        "latency_ms": {
            "condense": int((t_condense - t0) * 1000),
            "retrieve": int((t_retrieve - t_condense) * 1000),
            "generate": int((t_generate - t_retrieve) * 1000),
        },
    }

    # 5. Log for evaluation (Ch. 13): production traffic becomes eval cases
    db.bucket().scope(config.EVALS_SCOPE).collection(config.SAMPLES_COLLECTION).upsert(
        f"traffic::{uuid4().hex[:16]}",
        {
            "type": "traffic_sample",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "user_input": question,
            "standalone_query": standalone,
            "retrieved_contexts": [h["text"] for h in hits],
            "response": response,
            "pipeline_version": config.PIPELINE_VERSION,
            "latency_ms": result["latency_ms"],
        },
    )
    return result
