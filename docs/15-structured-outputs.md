# Chapter 15 — Structured Outputs: Pydantic-Verified JSON Generation

> *Chapter 8 covered the Model Service as a chat/embeddings/cache endpoint. This is the
> narrower, very common job that sits on top of it: using an LLM as a synthetic-data
> generator that must produce JSON matching an exact schema, every single time — not
> "usually," not "if the backend supports it."*

## 15.1 Schema mode isn't guaranteed everywhere

`response_format={"type": "json_schema", ...}` (OpenAI's `strict` schema mode) is the most
reliable way to get well-formed JSON out of a chat completion — the decoder is constrained to
only emit tokens that keep the output on-schema. OpenAI's own models guarantee it. Capella
Model Service models are served through NVIDIA NIM / vLLM, where JSON-schema-constrained
decoding support differs by model and version — some deployments honor it, some reject the
request outright with a `BadRequestError`.

The fix isn't to hope every backend supports schema mode; it's to treat schema mode as a
**speed-up**, not a **substitute**, for validating the response yourself. Pydantic is the part
that guarantees correctness regardless of which mode the endpoint actually honored:

```python
from openai import APIError, BadRequestError
from pydantic import ValidationError

def generate_structured(prompt: str, schema: type[BaseModel], max_repairs: int = 2):
    messages = [
        {"role": "system", "content": (
            "You generate synthetic data. Respond with ONLY a single JSON object "
            f"matching this schema, no prose:\n{json.dumps(schema.model_json_schema())}"
        )},
        {"role": "user", "content": prompt},
    ]
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL, temperature=0.9, messages=messages,
            response_format={"type": "json_schema", "json_schema": {
                "name": schema.__name__, "schema": schema.model_json_schema(), "strict": True}},
        )
        mode = "json_schema"
    except (BadRequestError, APIError):
        resp = client.chat.completions.create(
            model=LLM_MODEL, temperature=0.9, messages=messages,
            response_format={"type": "json_object"},
        )
        mode = "json_object (schema mode unsupported by this endpoint)"

    raw = resp.choices[0].message.content
    for attempt in range(max_repairs + 1):
        try:
            return schema.model_validate_json(raw), mode
        except ValidationError as e:
            if attempt == max_repairs:
                raise
            messages += [{"role": "assistant", "content": raw},
                        {"role": "user", "content": f"That failed validation: {e}. "
                                                     "Return corrected JSON only."}]
            resp = client.chat.completions.create(
                model=LLM_MODEL, temperature=0.9, messages=messages,
                response_format={"type": "json_object"})
            raw = resp.choices[0].message.content
            mode += " +repair"
```

Three layers, in order of cost: try schema mode; fall back to unconstrained JSON mode plus a
schema spelled out in the prompt if the endpoint refuses schema mode; validate through
Pydantic either way, feeding a `ValidationError` back to the model for one corrective turn
before giving up. A batch job built on this fails loudly on a genuinely bad generation instead
of silently writing malformed records downstream.

## 15.2 The schema, as a Pydantic model

The target shape is just a `BaseModel` — the same one you'd use to validate an API request:

```python
from enum import Enum
from pydantic import BaseModel, Field

class Sentiment(str, Enum):
    positive = "positive"
    negative = "negative"
    neutral = "neutral"
    mixed = "mixed"

class ProductReview(BaseModel):
    product: str
    rating: int = Field(ge=1, le=5)
    sentiment: Sentiment
    review_text: str = Field(max_length=280)
    tags: list[str] = Field(default_factory=list, max_length=5)
```

`model_json_schema()` is what both request the constrained decoding (§15.1) and gets embedded
in the prompt for the unconstrained fallback — one definition, two uses. Field constraints
(`ge`, `le`, `max_length`) do double duty too: they shape the JSON Schema sent to the model
*and* they're what actually catches a plausible-looking-but-out-of-range generation (a rating
of `7`, a 900-character review) that schema mode alone wouldn't reject.

## 15.3 Batch generation

The payoff of validating locally: a batch runs unattended, and each record is either a valid
`ProductReview` or an exception — no silent corruption partway through a run of hundreds:

```python
reviews: list[ProductReview] = []
for product in PRODUCTS:
    for _ in range(2):
        review, mode = generate_structured(
            f"A realistic customer review for: {product}.", ProductReview)
        reviews.append(review)
```

## 15.4 Storing and querying structured records

Once validated, a record is just a document — `model_dump(mode="json")` upserts directly (the
`mode="json"` argument matters: it serializes the `Sentiment` enum to its string value instead
of a Python `Enum` object):

```python
reviews_col.upsert(f"review::{uuid.uuid4()}",
                   {**review.model_dump(mode="json"), "generated_by": LLM_MODEL})
```

The return on validating up front shows up here: every field is reliably queryable with plain
SQL++, no defensive `TRY_CAST` or null-checks for malformed generations —

```sql
SELECT r.sentiment, COUNT(*) AS n, ROUND(AVG(r.rating), 2) AS avg_rating
FROM `ai`.data_gen.reviews r
GROUP BY r.sentiment ORDER BY n DESC;
```

This is the same principle as Ch. 7's AI Functions (schema-enforced enrichment at rest) and
Ch. 13's eval records (typed, queryable results) — an LLM's output is only as useful as the
database's ability to trust its shape without re-checking it on every read.

## 15.5 Picking a Capella model for structured output

From the current Model Service catalog, the LLMs explicitly documented for synthetic data
generation:

| Model | Notes |
|---|---|
| `meta/llama-3.3-70b-instruct` | 70B, multilingual, coding + synthetic data generation — best quality |
| `meta/llama-3.1-8b-instruct` | 8B, same focus, cheaper/faster — good for large batches |
| `nvidia/llama-3.3-nemotron-super-49b-v1.5` | reasoning-focused — the better pick if generated records need multi-step reasoning, not just formatting |

All are served through NVIDIA NIM behind the same OpenAI-compatible API (Ch. 8 §8.1), so
switching backends is a `CAPELLA_LLM_MODEL` config change, not a code change — the entire
point of §15.1's validate-and-repair fallback is that the pipeline keeps working even when a
given deployment's JSON-schema support differs from another's.

## 15.6 Recap

Schema mode is an optimization; Pydantic validation is the guarantee. Design the target shape
once as a `BaseModel`, request constrained decoding where the backend supports it, fall back
to prompted JSON plus local validation where it doesn't, and feed validation failures back to
the model for a bounded number of repairs before failing the record outright. What lands in
Couchbase is then reliably typed data, not "JSON that's probably fine" — queryable with plain
SQL++, the same discipline Ch. 7 and Ch. 13 depend on elsewhere in this book.

Notebook: [`notebooks/10_structured_outputs.ipynb`](../notebooks/10_structured_outputs.ipynb).
Running into errors? See [Troubleshooting](troubleshooting.md).

Back to: [Chapter 8 — The Capella Model Service](08-model-service.md).
