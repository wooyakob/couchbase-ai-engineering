# AI Engineering on Couchbase

**Building applications with foundation models — with Couchbase as the single data
platform for knowledge, vectors, memory, agent assets, and evaluation.**

This repo is a book-in-progress with runnable companions: 13 detailed chapters
(`docs/`), 8 end-to-end notebooks (`notebooks/`), and 2 complete sample applications
(`apps/`), all built on the [Couchbase Python SDK](https://docs.couchbase.com/python-sdk/current/hello-world/overview.html).

Two deliberate scope choices, stated up front:

- **Orchestration**: Couchbase does not ship an agent orchestration framework — we use
  **[LangGraph](https://langchain-ai.github.io/langgraph/)**, with Couchbase as its
  persistence layer (checkpoints, memory, tools, retrieval).
- **Evaluation**: Couchbase does not ship an eval framework — we use
  **[Ragas](https://docs.ragas.io/)**, and store every eval run in Couchbase so quality
  becomes a queryable time series.

## The book

| # | Chapter | Covers |
|---|---|---|
| 1 | [AI Engineering on Couchbase](docs/01-ai-engineering-on-couchbase.md) | The stack, the reference architecture, why one platform |
| 2 | [Python SDK Foundations](docs/02-python-sdk-foundations.md) | Connect (local/Capella), KV, subdocument, TTL, SQL++, provisioning |
| 3 | [Data Processing for AI](docs/03-data-processing.md) | Chunking strategies, idempotent pipelines, lineage, Eventing, AI-function enrichment |
| 4 | [Embeddings & the Vectorization Service](docs/04-vectorization-service.md) | Embedding mechanics, Capella auto-vectorization workflows vs. DIY, model migrations |
| 5 | [Vector Search](docs/05-vector-search.md) | Index definitions, `SearchRequest`/`VectorQuery`, hybrid search, prefilters, tuning |
| 6 | [Retrieval-Augmented Generation](docs/06-rag.md) | RAG chains, grounding, exact + semantic caching, multi-turn with chat history |
| 7 | [AI Functions](docs/07-ai-functions.md) | LLM tasks in SQL++: `ai_sentiment`, `ai_summary`, `ai_classification`, enrichment at rest |
| 8 | [The Capella Model Service](docs/08-model-service.md) | OpenAI-compatible hosted models, serving-layer cache, guardrails, model routing |
| 9 | [Agent Memory](docs/09-agent-memory.md) | Short-term (TTL sessions), long-term (vector recall), extraction, summarization, forgetting |
| 10 | [Agent Catalog](docs/10-agent-catalog.md) | `agentc`: versioned tools & prompts, semantic tool discovery, activity auditing |
| 11 | [Orchestrating with LangGraph](docs/11-orchestration-langgraph.md) | Graphs, the Couchbase checkpointer, catalog-driven nodes, the full state architecture |
| 12 | [The Couchbase MCP Server](docs/12-mcp-server.md) | Model Context Protocol, cluster access for AI tools, security posture |
| 13 | [Evaluating with Ragas](docs/13-evaluation-ragas.md) | RAG metrics, agent evals from activity logs, eval results as data, CI gates |

## The notebooks

Each notebook is the runnable form of one or more chapters, in sequence — run 01 first
(it provisions the `ai` bucket everything else uses):

1. [`01_python_sdk_quickstart`](notebooks/01_python_sdk_quickstart.ipynb) — connect, provision, KV/subdoc/TTL, SQL++
2. [`02_vector_search_fundamentals`](notebooks/02_vector_search_fundamentals.ipynb) — chunk → embed → index → semantic/hybrid/prefiltered search
3. [`03_rag_pipeline`](notebooks/03_rag_pipeline.ipynb) — RAG chain, exact + semantic caching, conversational RAG
4. [`04_ai_functions`](notebooks/04_ai_functions.ipynb) — SQL++ AI functions (Capella) + the portable equivalent
5. [`05_model_service`](notebooks/05_model_service.ipynb) — Capella Model Service: chat, embeddings, cache, guardrails
6. [`06_agent_memory`](notebooks/06_agent_memory.ipynb) — session store, memory store, extraction, hygiene
7. [`07_agent_catalog_langgraph`](notebooks/07_agent_catalog_langgraph.ipynb) — cataloged tools/prompts + durable LangGraph agent
8. [`08_ragas_evaluation`](notebooks/08_ragas_evaluation.ipynb) — score the RAG pipeline, store runs, regression queries

Notebooks are generated from `notebooks/src/*.py` (jupytext percent format) via
`python scripts/build_notebooks.py` — edit the sources, not the `.ipynb`.

## The apps

- **[`apps/rag-api`](apps/rag-api/)** — a FastAPI RAG service: hybrid retrieval with
  tenant prefilters, query condensation, Couchbase-backed chat history, and every
  interaction logged for evaluation.
- **[`apps/support-agent`](apps/support-agent/)** — a LangGraph support agent with
  Agent Catalog tools/prompts, short/long-term memory, Couchbase checkpoints, optional
  MCP tools, and a pytest eval suite.

## Quickstart

```bash
git clone <this repo> && cd ai-engineering
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env    # point at your cluster + model provider

jupyter lab notebooks/  # start with 01
```

You need a Couchbase cluster with Data + Query + Index + Search services and a bucket
named `ai`: a free [Capella](https://cloud.couchbase.com/) trial, or locally

```bash
docker run -d --name cb -p 8091-8097:8091-8097 -p 11210:11210 couchbase:enterprise
```

Models default to OpenAI (`OPENAI_API_KEY`); set `CAPELLA_AI_ENDPOINT` to switch every
notebook and app to Capella-hosted models (Chapter 8).

## Repo map

```
docs/                the book (13 chapters)
notebooks/           runnable companions (.ipynb generated from src/*.py)
apps/rag-api/        FastAPI RAG service
apps/support-agent/  LangGraph + Agent Catalog agent
indexes/             reusable Search index definitions
scripts/             notebook build tooling
```
