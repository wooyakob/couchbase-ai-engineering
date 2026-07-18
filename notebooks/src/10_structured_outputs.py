# %% [markdown]
# # 10: Structured Outputs: Pydantic-Verified JSON Generation
#
# Extends [Chapter 8](../docs/08-model-service.md). Where notebook 05 exercised chat,
# embeddings, and caching against the Model Service, this one covers a narrower but very
# common job: using an LLM as a **synthetic data generator** that must produce JSON
# matching an exact schema, every time.
#
# 1. Define the target shape as a `pydantic` model
# 2. Ask the model for JSON: native `response_format` structured-output mode where the
#    endpoint supports it, with a **validate-and-repair fallback** where it doesn't
# 3. Generate a batch of synthetic records
# 4. Store them in Couchbase and query the structure back with SQL++
#
# **Why the fallback matters:** `response_format={"type": "json_schema", ...}` (or
# OpenAI's `strict` mode) is the most reliable way to get well-formed JSON, but support
# varies by backend: OpenAI's own models guarantee it; Capella Model Service models are
# served through NVIDIA NIM / vLLM, where JSON-schema-constrained decoding support
# differs by model and version. Pydantic validation is the part that actually
# guarantees correctness regardless of which mode the endpoint honored: treat schema
# mode as a **speed-up**, not a **substitute** for validating the response.
#
# **Prerequisites:** notebook 01 (bucket provisioned). Runs on OpenAI by default; set
# `CAPELLA_AI_ENDPOINT` to use the Capella Model Service instead (Ch. 8); see §5 below
# for which deployed models are worth pointing this at.

# %%
%pip install -q couchbase python-dotenv openai "pydantic>=2"

# %%
import base64
import json
import os
import uuid
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
# ## Model: OpenAI default, Capella Model Service switch (Ch. 8)
#
# `meta/llama-3.3-70b-instruct` and `meta/llama-3.1-8b-instruct` are the two models in
# the current Capella catalog explicitly called out for **synthetic data generation**
# (see §5); that's the default when `CAPELLA_AI_ENDPOINT` is set. Override with
# `CAPELLA_LLM_MODEL` for any other deployed model.

# %%
from openai import OpenAI

CAPELLA = bool(os.getenv("CAPELLA_AI_ENDPOINT"))

if CAPELLA:
    BASE_URL = os.environ["CAPELLA_AI_ENDPOINT"]  # must end with /v1
    API_KEY = os.getenv("CAPELLA_AI_TOKEN") or base64.b64encode(
        f"{os.environ['CB_USERNAME']}:{os.environ['CB_PASSWORD']}".encode()).decode()
    LLM_MODEL = os.getenv("CAPELLA_LLM_MODEL", "meta/llama-3.3-70b-instruct")
else:
    BASE_URL, API_KEY = None, os.environ["OPENAI_API_KEY"]
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
print(f"backend: {'Capella Model Service' if CAPELLA else 'OpenAI'}   model: {LLM_MODEL}")

# %% [markdown]
# ## 1. The schema, as a Pydantic model
#
# The scenario: generating synthetic product review data (test fixtures, demo seed data,
# eval datasets, any case where you need realistic-but-fake structured records).

# %%
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


SCHEMA = ProductReview.model_json_schema()
print(json.dumps(SCHEMA, indent=2))

# %% [markdown]
# ## 2. Generate: schema mode first, validate-and-repair always
#
# `generate_structured` tries `response_format={"type": "json_schema", ...}` first. If
# the endpoint rejects it (older/unsupported backends raise `BadRequestError`), it falls
# back to `{"type": "json_object"}` with the schema spelled out in the system prompt.
# Either way, the response is parsed through the Pydantic model; a `ValidationError`
# feeds the error back to the model for one corrective turn instead of failing outright.

# %%
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


review, mode = generate_structured("A review for a pair of noise-cancelling headphones.",
                                   ProductReview)
print(f"mode: {mode}")
print(review.model_dump_json(indent=2))

# %% [markdown]
# ## 3. Batch generation
#
# The whole point of validating locally: a batch can run unattended, and anything that
# doesn't parse fails loudly instead of poisoning downstream data.

# %%
PRODUCTS = ["wireless earbuds", "standing desk", "espresso machine",
           "mechanical keyboard", "running shoes"]

reviews: list[ProductReview] = []
for product in PRODUCTS:
    for _ in range(2):
        review, mode = generate_structured(
            f"A realistic customer review for: {product}.", ProductReview)
        reviews.append(review)
        print(f"[{mode}] {product}: {review.rating}/5 {review.sentiment.value}")

print(f"\ngenerated {len(reviews)} validated records")

# %% [markdown]
# ## 4. Store in Couchbase
#
# Structured records are just documents once validated: `model_dump()` (with the enum
# as its string value) upserts directly.

# %%
from couchbase.exceptions import (CollectionAlreadyExistsException,
                                  ScopeAlreadyExistsException)

bucket = cluster.bucket(CB_BUCKET)
try:
    bucket.collections().create_scope("data_gen")
except ScopeAlreadyExistsException:
    pass
try:
    bucket.collections().create_collection("data_gen", "reviews")
except CollectionAlreadyExistsException:
    pass

reviews_col = bucket.scope("data_gen").collection("reviews")
for review in reviews:
    key = f"review::{uuid.uuid4()}"
    reviews_col.upsert(key, {**review.model_dump(mode="json"), "id": key,
                            "generated_by": LLM_MODEL})

cluster.query(f"CREATE PRIMARY INDEX IF NOT EXISTS ON `{CB_BUCKET}`.data_gen.reviews").execute()
print(f"stored {len(reviews)} reviews in `{CB_BUCKET}`.data_gen.reviews")

# %% [markdown]
# ## 5. Query it back
#
# The payoff of validating structure up front: every field is reliably queryable,
# no defensive `TRY_CAST` or null-checks for malformed generations.

# %%
for row in cluster.query(f"""
    SELECT r.sentiment, COUNT(*) AS n, ROUND(AVG(r.rating), 2) AS avg_rating
    FROM `{CB_BUCKET}`.data_gen.reviews r
    GROUP BY r.sentiment ORDER BY n DESC"""):
    print(row)

print()
for row in cluster.query(f"""
    SELECT r.product, r.rating, r.review_text
    FROM `{CB_BUCKET}`.data_gen.reviews r
    WHERE r.rating <= 2"""):
    print(row)

# %% [markdown]
# ## 6. Picking a Capella model for structured output
#
# From the current Model Service catalog, the LLMs explicitly documented for synthetic
# data generation are:
#
# | Model | Notes |
# |---|---|
# | `meta/llama-3.3-70b-instruct` | 70B, multilingual, coding + **synthetic data generation**; best quality, this notebook's Capella default |
# | `meta/llama-3.1-8b-instruct` | 8B, same synthetic-data-generation focus, cheaper/faster; good for large batches |
# | `nvidia/llama-3.3-nemotron-super-49b-v1.5` | reasoning-focused, strong at agentic/RAG tasks; the better pick if generated records need multi-step reasoning, not just formatting |
#
# All are served through NVIDIA NIM behind the same OpenAI-compatible API (Ch. 8 §8.1),
# so switching is a `CAPELLA_LLM_MODEL` change, not a code change; the point of §2's
# validate-and-repair fallback is that this notebook keeps working even if a given
# deployment's JSON-schema support differs from another's.
#
# **Next:** back to [Chapter 8: The Capella Model Service](../docs/08-model-service.md)
# for caching and guardrails, or [notebook 08](08_ragas_evaluation.ipynb) if these
# synthetic records are destined for an eval dataset.
