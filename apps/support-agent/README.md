# support-agent — a governed, durable LangGraph agent on Couchbase

The assembled system from [Chapters 9–13](../../docs/): a customer-support agent whose
*every* piece of state lives in Couchbase.

| Piece | Where | Chapter |
|---|---|---|
| Tools (order lookup, doc search, save_memory) | `tools/support_tools.py` → Agent Catalog | 10 |
| Prompt + output schema + tool bindings | `prompts/support_agent.yaml` → Agent Catalog | 10 |
| Graph (context → agent → escalation) | `agent/graph.py` (LangGraph) | 11 |
| User memory (facts + past dialogs) | `agent/memory.py` → managed **Agent Memory** server + SDK | 9 |
| Durable graph state | `CouchbaseSaver` → `CB_BUCKET`.agent.checkpoints (`ai` by default) | 11 |
| Activity/audit trace | Spans → `AGENT_CATALOG_BUCKET`.agent_activity.logs (`ai-support-agent` by convention — see "Why a separate file" below) | 10 |
| MCP tools (optional) | `agent/mcp_tools.py` | 12 |
| Evals | `evals/test_agent.py` (pytest + span metrics) | 13 |

## Setup

Requires **Python 3.12+** (the `couchbase-agent-memory` SDK), a Couchbase cluster with
the `ai` bucket provisioned (run [`notebooks/01`](../../notebooks/01_python_sdk_quickstart.ipynb)),
and a running **Agent Memory server** pointed at that cluster — the managed product that
stores the agent's user memory (see [Chapter 9 §9.8](../../docs/09-agent-memory.md) and
[`notebooks/09`](../../notebooks/09_agent_memory_managed.ipynb) for the `docker run`).

This app keeps its **own** `.env.server` / `.env.capella` in this directory, separate
from the repo-root ones the notebooks use — `find_dotenv(ENV_FILE, usecwd=True)` walks
up from the current directory and returns the first match, so running from here always
picks up this local file first.

**Why a separate file, and specifically why a separate `AGENT_CATALOG_BUCKET`:** agentc
has no per-project namespacing. "Latest catalog snapshot" is resolved **bucket-wide**
(`WHERE t.kind = $kind ORDER BY version.timestamp DESC LIMIT 1`, across *every* project
ever published into that bucket) — not scoped to this app's own commits. If this app
shared `AGENT_CATALOG_BUCKET=ai` with [notebook 07](../../notebooks/07_agent_catalog_langgraph.ipynb)'s
`agentc_demo` catalog, whichever of the two you ran `agentc publish` on **most recently**
would silently become "the" catalog for *both* — the other's prompt/tool lookups would
start returning empty with no obvious error. This app's env files point
`AGENT_CATALOG_BUCKET` at a dedicated bucket (`ai-support-agent` by convention) instead,
so the two never compete for "latest." Create that bucket once in the Capella Console
(or on your local cluster) — it only needs to hold catalog + activity/checkpoint data.

```bash
pip install -r requirements.txt
# .env.server / .env.capella already exist in this directory — edit them in place
# rather than copying the repo-root .example files over them (that would lose the
# separate AGENT_CATALOG_BUCKET). Fill in CB_*, OPENAI_API_KEY or CAPELLA_*, and
# AGENTMEMORY_BASE_URL for your setup; AGENT_CATALOG_BUCKET is already set.
export ENV_FILE=.env.server                 # or .env.capella — main.py's load_dotenv() reads this

# `agentc` is a separate CLI process — it reads plain shell env vars, not ENV_FILE/
# .env files, so export the file's contents into this shell first:
set -a && source "$ENV_FILE" && set +a

agentc init
# Commit BEFORE indexing, not after: `agentc index` records whether the tree was
# dirty at index time and bakes that into the snapshot; `agentc publish` refuses
# a snapshot indexed while dirty even if you commit afterward.
git add -A && git commit -m "support agent v1"
agentc index .
agentc publish
```

(`agentc init` may print `ERROR: Git repository not found!` from an unrelated bonus
step — it tries to install a git post-commit hook by checking for `.git/` relative to
the *current* directory, which doesn't exist since this app isn't its own git repo, only
nested inside one. Harmless: the actual catalog/activity collections it created before
that point are unaffected.)

The agent degrades gracefully if the Agent Memory server is offline: recall returns
nothing and writes are skipped, so `pytest evals/` runs without it.

**`ModuleNotFoundError: No module named 'agentc_langchain'`** — `agentc_langgraph.agent`
imports `agentc_langchain` at load time without declaring it as a dependency. It's listed
explicitly in `requirements.txt` for exactly this reason; if you installed some other way
(e.g. an editable install of just `agentc-langgraph`), run
`pip install agentc-langchain` directly.

Or just run `./start.sh` (see below) — it installs `requirements.txt` into the repo's
shared `.venv` for you and tells you if the one-time catalog setup is still needed.

Seed a demo order. This needs `ENV_FILE` set and the app's own dependencies importable,
so launch the REPL from this directory with those in place — reusing the same shell
session where you already exported `ENV_FILE` and ran the `agentc` setup above works:

```bash
cd apps/support-agent          # agent.memory must be importable from cwd
export ENV_FILE=.env.capella    # or .env.server — whichever backend you're using
../../.venv/bin/python          # the repo's shared venv, not a bare `python`
```

Then, at the `>>>` prompt:

```python
from agent.memory import cluster, CB_BUCKET
bucket = cluster().bucket(CB_BUCKET)
bucket.collections().create_scope("shop")            # ignore AlreadyExists
bucket.collections().create_collection("shop", "orders")
bucket.scope("shop").collection("orders").upsert("order::1042",
    {"id": 1042, "status": "shipped", "eta": "2026-07-08",
     "items": [{"sku": "CB-TSHIRT-L", "qty": 2}]})
```

## Run

```bash
./start.sh u42                       # local Couchbase + OpenAI (.env.server)
ENV_FILE=.env.capella ./start.sh u42 # Capella Model Service
```

`start.sh` creates/reuses the repo's shared `.venv`, installs this app's
`requirements.txt` into it, and warns (without auto-committing anything) if the
one-time Agent Catalog setup above hasn't been run yet. Equivalent to running it by
hand:

```bash
python main.py u42
```

```
you> where is order 1042?
agent> Order 1042 shipped and is expected on 2026-07-08. ...
you> great — btw I prefer email updates, never SMS
agent> Noted — I'll remember you prefer email updates. ...   (save_memory fired)
you> I want a $500 refund
agent> I've escalated this to a human agent — they'll follow up shortly.
```

Inspect what happened with SQL++ (views installed by `agentc init`). Note the bucket
here is `AGENT_CATALOG_BUCKET` (`ai-support-agent` by convention — see "Why a separate
file" above), **not** `CB_BUCKET` (`ai`) — `agentc init` provisions the `agent_activity`
scope into whichever bucket the catalog itself lives in:

```sql
SELECT * FROM `ai-support-agent`.agent_activity.Sessions() s
WHERE s.sid = `ai-support-agent`.agent_activity.LastSession();
```

## Test

```bash
pytest tests/                        # unit tests only (no live services needed)
ENV_FILE=.env.capella pytest tests/ evals/   # + real scenario evals against a live setup
```

- `tests/test_graph_unit.py`, `tests/test_memory_unit.py`, `tests/test_tools_unit.py` —
  pure logic and mocked collaborators (the always-OpenAI chat model decision, folding
  memory into the human turn instead of a second system message, the `_block_to_dict`
  shaping, the embedding-model Capella switch, and each cataloged tool's error handling).
  Always run, no config, no live cluster/catalog/memory-server needed.
- `evals/test_agent.py` (pre-existing) — real scenario evals against a live catalog +
  cluster + LLM (does the agent actually escalate large refunds? stay grounded on a
  missing order?). Loads config via `dotenv.load_dotenv(... os.getenv("ENV_FILE", ".env"))`
  — since this directory has no plain `.env`, only `.env.server` / `.env.capella`, you must
  set `ENV_FILE` or the OpenAI client fails with a missing-credentials error before any
  test runs. If you already `export ENV_FILE=...` in this shell (per the Setup section
  above), plain `pytest evals/` picks it up too. Each scenario logs a key-value metric to
  the activity store, so eval scores are queryable next to production traces — the
  Chapter 13 loop.

Running into errors? See [`docs/troubleshooting.md`](../../docs/troubleshooting.md) —
covers `agentc` git/publish failures, Agent Memory server issues (image tags, ports,
crash-looping containers), and connection/auth failures, across every notebook and app.
