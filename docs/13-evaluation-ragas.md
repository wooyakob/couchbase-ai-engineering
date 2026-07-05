# Chapter 13 — Evaluating with Ragas

> *"It seems better" is not an engineering discipline. Couchbase doesn't ship an evaluation framework — evaluation is methodology, not storage — so we use Ragas. But eval* results *are absolutely data: we store every run in Couchbase, where quality becomes a queryable time series next to the system it measures.*

## 13.1 Why evals are non-negotiable

Every chapter so far ended with a knob: chunk size (3), embedding model (4), k and hybrid boosts (5), prompt wording (6), memory recall depth (9), agent instructions (10–11). None of those knobs comes with a gauge. LLM systems fail *quietly* — a chunking change that tanks recall produces no exception, just worse answers. Evaluation is the gauge, and it has to run continuously, because your corpus, your users, and your models all drift.

[Ragas](https://docs.ragas.io/) evaluates LLM applications using LLMs as judges, with well-studied metrics for exactly the pipelines this book builds.

## 13.2 The core RAG metrics

For a RAG interaction — question, retrieved contexts, answer, (optionally) reference answer — four metrics triangulate where quality is lost:

| Metric | Question it answers | Diagnoses |
|---|---|---|
| **Faithfulness** | Is every claim in the answer supported by the retrieved context? | Hallucination (generation problem) |
| **Answer relevancy** | Does the answer actually address the question? | Evasion, rambling |
| **Context precision** | Are the retrieved chunks relevant (and ranked well)? | Noisy retrieval — k too high, bad chunking |
| **Context recall** | Did retrieval find everything the reference answer needs? | Missed retrieval — embedding/chunking/hybrid gaps (needs a reference) |

The split is the point: **faithfulness low → fix the prompt/model; recall low → fix retrieval.** Without the split you tweak randomly.

## 13.3 Running an evaluation

```python
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (answer_relevancy, context_precision,
                           context_recall, faithfulness)
from langchain_openai import ChatOpenAI

# 1. Run YOUR pipeline over an eval set, capturing retrieval + answer
samples = []
for case in eval_cases:                      # [{question, reference}, ...]
    hits = semantic_search(case["question"], k=5)          # Chapter 5
    answer = rag_chain.invoke(case["question"])            # Chapter 6
    samples.append(SingleTurnSample(
        user_input=case["question"],
        retrieved_contexts=[h["text"] for h in hits],
        response=answer,
        reference=case["reference"],
    ))

# 2. Judge
judge = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0))
result = evaluate(
    dataset=EvaluationDataset(samples=samples),
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=judge,
)
print(result)   # {'faithfulness': 0.94, 'answer_relevancy': 0.89, ...}
```

Judge-model rules: use a *strong* model (judging is harder than generating), pin its version (an unpinned judge makes scores incomparable across runs), and never let the system under test judge itself if you can avoid it.

**The eval set** is the real asset. Start with 20–50 question/reference pairs written from real user questions (the `rag-api` app logs every production Q&A to `ai.evals.samples` precisely so you can promote real traffic into eval cases). Grow it every time you find a failure. It's data — keep it in Couchbase, version cases like code.

## 13.4 Storing results: quality as a time series

A run is a document; per-sample scores are documents. Then regressions are a query:

```python
run_key = f"evalrun::{now_compact()}"
evals_runs.upsert(run_key, {
    "type": "eval_run",
    "created_at": now_iso(),
    "git_commit": git_sha(),                 # ties scores to code AND catalog version (Ch. 10)
    "pipeline": {"embedding_model": EMBEDDING_MODEL, "k": 5,
                 "chunking": "markdown-2000-200", "llm": "gpt-4o-mini"},
    "metrics": {k: float(v) for k, v in result._repr_dict.items()},
    "n_samples": len(samples),
})
for i, row in enumerate(result.to_pandas().to_dict("records")):
    evals_samples.upsert(f"{run_key}::sample::{i:04d}", {**row, "run": run_key})
```

```sql
-- did the last change regress faithfulness?
SELECT r.git_commit, r.pipeline.embedding_model, r.metrics.faithfulness
FROM ai.evals.runs r ORDER BY r.created_at DESC LIMIT 5;

-- which QUESTIONS got worse? (the debugging goldmine)
SELECT cur.user_input, prev.faithfulness AS before, cur.faithfulness AS after
FROM ai.evals.samples cur JOIN ai.evals.samples prev
  ON cur.user_input = prev.user_input
WHERE cur.run = $current_run AND prev.run = $baseline_run
  AND cur.faithfulness < prev.faithfulness - 0.1;
```

That second query is the workflow: metric drops → *these five questions* broke → look at their retrieved contexts → root cause. Aggregate scores tell you *that* something broke; per-sample storage tells you *what*.

## 13.5 Evaluating agents

Agents need more than RAG metrics: did it call the right tools, take a sane path, achieve the goal? Two complementary approaches:

**From activity logs.** Chapter 10's Spans already captured every tool call and message. Ragas ships an Agent Catalog integration that parses those logs directly:

```python
from ragas.integrations.agentc import AgentCTraceParser, AgentCEvaluator
from ragas.metrics import ToolStepEfficiency, AgentResponseFaithfulness

sessions = AgentCTraceParser().parse_logs(activity_logs)     # from ai.agent_activity.logs
scores = AgentCEvaluator(metrics=[ToolStepEfficiency(), AgentResponseFaithfulness()],
                         llm=judge).evaluate_sessions(sessions)
```

**As pytest.** Scenario tests that run the graph and assert on state + span metrics, run with `pytest evals/` in CI (the agent-catalog travel example and `apps/support-agent/evals/` both follow this shape):

```python
def test_escalates_large_refund(agent, span):
    state = agent.invoke({"messages": [("user", "I want a $500 refund")], "user_id": "test"})
    span["escalated_correctly"] = state["needs_human"] is True     # logged as KeyValueContent
    assert state["needs_human"]
```

Those `span[...] = value` metrics land in `ai.agent_activity.logs` with the catalog version attached — eval results and production behavior share one queryable store.

## 13.6 The evaluation loop in CI

1. **PR gate**: run the eval set on every change to prompts/retrieval config; fail if `faithfulness` or `context_recall` drops >0.05 vs. the stored baseline (a SQL++ lookup).
2. **Nightly**: full eval + store the run; alert on trend, not single runs (LLM-judged metrics have variance — judge each sample once, compare *distributions*).
3. **Continuous**: sample production traffic from `ai.evals.samples`, run reference-free metrics (faithfulness and answer relevancy don't need references), chart weekly.

Cost control: evals are LLM calls (≈ 2–4 judge calls per sample per metric). A 50-case set with 4 metrics ≈ hundreds of calls — fine nightly, wasteful per-commit; use a 10-case smoke set for PRs.

## 13.7 Recap — and the end of the loop

The book's architecture closes here. Data flows in (3–4), gets retrieved (5–6), enriched (7), generated over (8), remembered (9), governed (10), orchestrated (11), exposed (12) — and measured (13), with the measurements stored where everything else lives, joined to the exact code and catalog versions that produced them. That loop — *ship, observe, evaluate, fix, re-measure* — is AI engineering.

Notebook: [`notebooks/08_ragas_evaluation.ipynb`](../notebooks/08_ragas_evaluation.ipynb). Harness: [`apps/support-agent/evals/`](../apps/support-agent/evals/).
