# support-agent — a governed, durable LangGraph agent on Couchbase

The assembled system from [Chapters 9–13](../../docs/): a customer-support agent whose
*every* piece of state lives in Couchbase.

| Piece | Where | Chapter |
|---|---|---|
| Tools (order lookup, doc search, save_memory) | `tools/support_tools.py` → Agent Catalog | 10 |
| Prompt + output schema + tool bindings | `prompts/support_agent.yaml` → Agent Catalog | 10 |
| Graph (context → agent → escalation) | `agent/graph.py` (LangGraph) | 11 |
| Short/long-term memory | `agent/memory.py` → `ai.agent.sessions` / `.memories` | 9 |
| Durable graph state | `CouchbaseSaver` → `ai.agent.checkpoints` | 11 |
| Activity/audit trace | Spans → `ai.agent_activity.logs` | 10 |
| MCP tools (optional) | `agent/mcp_tools.py` | 12 |
| Evals | `evals/test_agent.py` (pytest + span metrics) | 13 |

## Setup

Requires Python 3.11+ (agentc), a Couchbase cluster with the `ai` bucket provisioned
(run [`notebooks/01`](../../notebooks/01_python_sdk_quickstart.ipynb)), and the memory
vector index (run [`notebooks/06`](../../notebooks/06_agent_memory.ipynb)).

```bash
pip install -r requirements.txt
cp ../../.env.example .env          # fill in CB_* , AGENT_CATALOG_* , OPENAI_API_KEY

# publish tools + prompts to the catalog (snapshots are keyed to git commits)
agentc init
agentc index .
git add -A && git commit -m "support agent v1"
agentc publish
```

Seed a demo order:

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

Inspect what happened with SQL++ (views installed by `agentc init`):

```sql
SELECT * FROM ai.agent_activity.Sessions() s
WHERE s.sid = ai.agent_activity.LastSession();
```

## Evaluate

```bash
pytest evals/
```

Each scenario logs a key-value metric to the activity store, so eval scores are
queryable next to production traces — the Chapter 13 loop.
