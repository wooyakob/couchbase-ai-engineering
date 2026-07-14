# Chapter 1 — AI Engineering on Couchbase

> *Building applications with foundation models requires more than calling a model API. It requires a data platform that can store knowledge, retrieve context, remember conversations, run models close to data, and observe what your agents actually did. This book shows how to do all of that on Couchbase.*

## 1.1 What is AI engineering?

AI engineering is the discipline of building production applications on top of foundation models. Unlike classic machine-learning engineering, you usually do not train the model — you *adapt* it with context. The core loop of nearly every AI application looks like this:

1. **Ingest** raw data (documents, tickets, product data, chat logs).
2. **Process** it into model-friendly units (chunks, structured fields, metadata).
3. **Vectorize** it with an embedding model.
4. **Retrieve** relevant context at query time (vector, full-text, and SQL-style retrieval).
5. **Generate** an answer or an action with an LLM.
6. **Remember** what happened (conversation state, long-term memory).
7. **Evaluate** the quality of the whole system, continuously.

Each of those steps needs a data system. Most teams end up with a sprawl: a document store, a vector database, a cache, a message-history store, a metadata catalog, and an analytics system — all glued together with brittle pipelines. Every extra system is another consistency boundary, another security review, and another operational burden.

Couchbase's proposition for AI engineering is simple: **one data platform for all of an AI application's state** — operational documents, vectors, agent memory, tool catalogs, caches, and evaluation logs — with in-database AI services that move the AI work close to the data. Couchbase packages those AI services as the **AI Data Plane (AIDP)**: AI Functions, MCP Server, Agent Memory, Data Processing Service, Vectorization Service, Agent Catalog, and Model Service, all sitting on top of Couchbase's AI-native data platform and available on **either** deployment model — self-managed (Enterprise Edition) or Capella (as-a-service). §1.5 maps AIDP's components to book chapters and to which of the two deployment models this book's examples actually exercise.

## 1.2 The Couchbase AI stack

This book covers the full stack, bottom-up. Rows marked **AIDP** are components of the Couchbase AI Data Plane (§1.5):

| Layer | Couchbase capability | Chapter |
|---|---|---|
| Storage & query | JSON documents, KV API, SQL++ | 2 |
| Data processing | Eventing, SQL++ transforms, ingest pipelines — **AIDP** Data Processing Service | 3 |
| Embeddings | **AIDP** Vectorization Service (auto-vectorization workflows) | 4 |
| Retrieval | **Vector Search** (Search service), hybrid & full-text search | 5 |
| RAG | Retrieval-augmented generation patterns, semantic caching | 6 |
| In-database AI | **AIDP** AI Functions in SQL++ (summarize, classify, mask, sentiment…) | 7 |
| Model hosting | **AIDP** Model Service (OpenAI-compatible endpoints) | 8 |
| Agent state | **AIDP** Agent Memory: short-term, long-term, episodic — from primitives *or* the managed Agent Memory server + SDK | 9 |
| Agent assets | **AIDP** Agent Catalog (`agentc`): tools, prompts, activity/audit | 10 |
| Orchestration | **LangGraph** (Couchbase has no orchestration framework — we use LangGraph) | 11 |
| Tool access | **AIDP** MCP Server (Model Context Protocol) | 12 |
| Evaluation | **Ragas** (Couchbase has no built-in evals — we use Ragas), results stored in Couchbase | 13 |

Two honest notes about scope, because good engineering starts with knowing what your platform does *not* do:

- **Couchbase does not ship an agent orchestration framework.** You bring your own; this book uses [LangGraph](https://langchain-ai.github.io/langgraph/) throughout, with Couchbase providing the persistence layer (checkpoints, memory, tools, and retrieval).
- **Couchbase does not ship an evaluation framework.** We use [Ragas](https://docs.ragas.io/) to evaluate RAG pipelines and agents, and we store evaluation runs *in* Couchbase so quality becomes a queryable time series.

## 1.3 Why one platform matters for AI workloads

### Context is a data problem

The quality ceiling of a RAG system or an agent is set by the quality of the context you assemble. Assembling context is a *query* problem:

- "Find the 5 most semantically similar chunks" → vector search.
- "…but only for this tenant, product line, and date range" → metadata filtering (SQL++ / search filters).
- "…and include the customer's open tickets" → operational KV/query lookups.
- "…and the last 10 turns of this conversation" → agent memory.

When those live in one system, one query (or one small set of queries against the same cluster) assembles the whole context — with the same consistency, the same access control, and no synchronization pipelines between a "vector DB" and the source of truth. Your vectors are stored *next to* the JSON they describe, so a document update and its re-vectorization are a single system's concern.

### Memory is a database problem

An "agent" is an LLM in a loop with tools and state. The state is the hard part:

- **Short-term memory** — the conversation so far. Needs fast reads/writes keyed by session, and TTL expiry.
- **Long-term memory** — durable facts about users and the world. Needs upserts, dedup, and *semantic* recall (vector search over memories).
- **Episodic traces** — what the agent did, step by step. Needs append-heavy writes and analytical queries later.

These are classic database workloads — KV with TTL, documents with vector indexes, append + query. Chapter 9 builds a complete memory subsystem two ways — from SDK primitives, and with Couchbase's **managed Agent Memory** product (a server + the `couchbase-agent-memory` SDK, on self-managed or Capella) that provides it as a service; Chapter 11 plugs memory into LangGraph.

### Governance is a catalog problem

Once you have more than one agent and more than one developer, you need to answer: *Which tools exist? Which prompt version was this agent running? What exactly did the agent do at 3 a.m.?* Couchbase **Agent Catalog** (`agentc`) versions tools and prompts in Git + Couchbase and logs agent activity into Couchbase collections you can query with SQL++. That's Chapter 10.

## 1.4 The reference architecture

Every example in this book fits this architecture:

```
                        ┌─────────────────────────────────────────┐
                        │              Application                │
                        │  (FastAPI service / LangGraph agent)    │
                        └───────┬──────────────┬──────────────────┘
                                │              │
                     Python SDK │              │ OpenAI-compatible API
                                │              │
        ┌───────────────────────▼──────┐  ┌────▼─────────────────────┐
        │       Couchbase Capella      │  │  Models                  │
        │                              │  │  • Capella Model Service │
        │  Data service (KV, JSON)     │  │  • or OpenAI/Anthropic/… │
        │  Query service (SQL++,       │  └──────────────────────────┘
        │    AI Functions)             │
        │  Search service (FTS +       │
        │    Vector Search)            │
        │  Eventing (data processing)  │
        │  AI Data Plane (AIDP):       │
        │    • Data Processing Service │
        │    • Vectorization Service   │
        │    • AI Functions            │
        │    • Model Service           │
        │    • Agent Memory            │
        │    • Agent Catalog           │
        │    • MCP Server              │
        │                              │
        │  Collections used in this    │
        │  book (bucket `ai`):         │
        │    docs.chunks      (RAG)    │
        │    agent.sessions   (STM)    │
        │    agent.memories   (LTM)    │
        │    agent.checkpoints         │
        │    agent.catalog/activity    │
        │    evals.runs / evals.samples│
        └──────────────────────────────┘
```

A note on naming: Couchbase organizes data as **bucket → scope → collection**. We use one bucket (`ai`) with scopes per concern (`docs`, `agent`, `evals`). In your own systems you might use one scope per application or per tenant; the SDK code is identical.

## 1.5 Self-managed vs. Capella

Everything in Chapters 2, 5, 6, 9, 10, 11, 12 and 13 works on **any** Couchbase Server 7.6+ / 8.0+ deployment (self-managed or Capella), because it only uses the SDK, Search, and SQL++.

The **AI Data Plane (AIDP)** is Couchbase's name for the seven AI services layered on the AI-native data platform — AI Functions, MCP Server, Agent Memory, Data Processing Service, Vectorization Service, Agent Catalog, and Model Service — and, as a product, it ships on **both** deployment models: Enterprise Edition (self-managed) and Capella (as-a-service). It is **not** Capella-exclusive; don't call it "Capella AI Data Plane."

That said, *this book's* examples for a few AIDP components currently exercise only the Capella-hosted implementation, because that's what's available today for hands-on use:

- Vectorization Service (Ch. 4, managed auto-vectorization workflows)
- AI Functions (Ch. 7, SQL++ `ai_*()` functions)
- Model Service (Ch. 8, OpenAI-compatible hosted endpoints)
- The managed Data Processing Service workflows (Ch. 3; the DIY Eventing/SQL++ path in the same chapter runs on either deployment)

If you run self-managed, those chapters show the equivalent "bring your own model" pattern (e.g., calling OpenAI for embeddings and doing the same upserts yourself), so the rest of the book still applies unchanged. Agent Memory (Ch. 9), Agent Catalog (Ch. 10, except its optional web UI), and the MCP Server (Ch. 12) are also AIDP components, but this book's examples for them already run on either deployment model — see the note in §1.7.

## 1.6 How to read this book

- **Chapters** (`docs/`) explain concepts and show the idiomatic code, in depth.
- **Notebooks** (`notebooks/`) are runnable end-to-end walkthroughs of each chapter's material.
- **Apps** (`apps/`) are two small but complete applications:
  - `apps/rag-api` — a FastAPI RAG service with hybrid vector retrieval, a semantic cache, and conversation history, all on Couchbase.
  - `apps/support-agent` — a LangGraph customer-support agent with Couchbase-backed short/long-term memory, Agent Catalog tools and prompts, and a Ragas evaluation harness.

If you read nothing else, read Chapters 5 (Vector Search), 9 (Agent Memory), and 11 (LangGraph) — they are the load-bearing walls of AI engineering on Couchbase.

## 1.7 Prerequisites

- Python 3.10–3.12.
- A Couchbase cluster:
  - Easiest: a [Capella](https://cloud.couchbase.com/) cluster (includes Search service).
  - Or self-managed Couchbase Server 8.0+ with Data, Query, Index, and Search services.
- An LLM + embeddings provider: Capella Model Service, or any OpenAI-compatible endpoint (we default to OpenAI and show the Capella switch everywhere).
- `pip install -r requirements.txt` from the repo root.
- Copy `.env.server.example` to `.env.server` (local Couchbase) and/or
  `.env.capella.example` to `.env.capella` (Capella), fill in your connection string and
  credentials, then set `ENV_FILE=.env.server` or `ENV_FILE=.env.capella` when launching
  Jupyter/apps (`load_dotenv()` reads `os.getenv("ENV_FILE", ".env")` everywhere in this
  repo). Since `ENV_FILE` is a per-process env var, running both backends at once is just
  two processes started with different values — nothing gets overwritten. Most notebooks
  run against either backend; **Capella is required** (no local equivalent) for the Model
  Service (Ch. 8), the Agent Catalog UI
  (Ch. 10-11), Couchbase AI Functions (Ch. 7), and the managed Data Processing /
  Vectorization workflows (Ch. 3-4). See the [README Quickstart](../README.md#quickstart)
  for the full breakdown.

Running into errors? See [Troubleshooting](troubleshooting.md).

Next: [Chapter 2 — Python SDK Foundations](02-python-sdk-foundations.md).
