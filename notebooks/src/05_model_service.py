# %% [markdown]
# # 05 — The Capella Model Service
#
# Companion to [Chapter 8](../docs/08-model-service.md).
#
# The Model Service hosts LLMs and embedding models next to your Capella cluster, behind an
# **OpenAI-compatible API**. This notebook exercises the full surface: chat, embeddings,
# LangChain wiring, cache behavior, and guardrail handling.
#
# **Prerequisites:** a deployed Capella model endpoint in `CAPELLA_AI_ENDPOINT`
# (ending in `/v1`). Every cell falls back to OpenAI when it's unset, so the notebook
# runs anywhere — cells print which backend they used.

# %%
# %pip install -q openai langchain-openai python-dotenv

# %%
import base64
import os
import time

from dotenv import load_dotenv

load_dotenv()

CAPELLA = bool(os.getenv("CAPELLA_AI_ENDPOINT"))

if CAPELLA:
    BASE_URL = os.environ["CAPELLA_AI_ENDPOINT"]          # must end with /v1
    API_KEY = base64.b64encode(
        f"{os.environ['CB_USERNAME']}:{os.environ['CB_PASSWORD']}".encode()).decode()
    LLM_MODEL = os.getenv("CAPELLA_LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    EMBEDDING_MODEL = os.getenv("CAPELLA_EMBEDDING_MODEL", "intfloat/e5-mistral-7b-instruct")
else:
    BASE_URL, API_KEY = None, os.environ["OPENAI_API_KEY"]
    LLM_MODEL, EMBEDDING_MODEL = "gpt-4o-mini", "text-embedding-3-small"

print(f"backend: {'Capella Model Service' if CAPELLA else 'OpenAI'}")
print(f"llm: {LLM_MODEL}   embeddings: {EMBEDDING_MODEL}")

# %% [markdown]
# ## 1. Raw `openai` client — the compatibility contract
#
# The whole value proposition in one cell: the standard OpenAI client, different `base_url`.

# %%
from openai import OpenAI

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

resp = client.chat.completions.create(
    model=LLM_MODEL,
    temperature=0,
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "In one sentence: why co-locate models with the database?"},
    ],
)
print(resp.choices[0].message.content)
print("\nusage:", resp.usage)

# %% [markdown]
# ## 2. Embeddings — and the dimensions contract
#
# `dims` in every vector index must equal this number (Ch. 5). Record it in config,
# never hard-code it twice.

# %%
emb = client.embeddings.create(model=EMBEDDING_MODEL,
                               input=["couchbase vector search", "capella model service"])
EMBEDDING_DIM = len(emb.data[0].embedding)
print(f"{EMBEDDING_MODEL} → {EMBEDDING_DIM} dims, {len(emb.data)} vectors (batched)")

# %% [markdown]
# ## 3. Streaming — for anything user-facing

# %%
stream = client.chat.completions.create(
    model=LLM_MODEL, temperature=0, stream=True,
    messages=[{"role": "user", "content": "Count to five, words only."}],
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
print()

# %% [markdown]
# ## 4. LangChain wiring
#
# These two objects drop into every LangChain/LangGraph example in this repo. The two
# Capella-specific embedding flags disable client-side tiktoken handling (the hosted
# models aren't tiktoken-based).

# %%
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

if CAPELLA:
    llm = ChatOpenAI(openai_api_base=BASE_URL, openai_api_key=API_KEY,
                     model=LLM_MODEL, temperature=0)
    embeddings = OpenAIEmbeddings(openai_api_base=BASE_URL, openai_api_key=API_KEY,
                                  model=EMBEDDING_MODEL,
                                  check_embedding_ctx_length=False,   # Capella-specific
                                  tiktoken_enabled=False)             # Capella-specific
else:
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

print(llm.invoke("Say OK.").content)

# %% [markdown]
# ## 5. Serving-layer cache behavior
#
# If your Capella model was deployed with caching enabled, a repeated request is served
# from cache — observable as a latency drop. (With semantic caching enabled, a paraphrase
# also hits.) On OpenAI fallback this just measures normal variance.

# %%
def timed(prompt: str) -> float:
    t0 = time.perf_counter()
    client.chat.completions.create(model=LLM_MODEL, temperature=0,
                                   messages=[{"role": "user", "content": prompt}])
    return time.perf_counter() - t0

q = "What are the three main benefits of database-adjacent model hosting?"
print(f"first call:  {timed(q):.2f}s")
print(f"second call: {timed(q):.2f}s   (cache hit if Capella caching is on)")

# %% [markdown]
# Remember the trade-off (Ch. 8 §8.3): never semantically cache prompts that embed
# per-user context — two users' "summarize my account" must not collide. Choose cache
# layer deliberately: serving-layer (here) *or* application-layer (notebook 03), not both.
#
# ## 6. Guardrails
#
# Capella deployments can attach a Llama Guard model. Blocked requests surface as API
# errors — production code needs the except branch, not just the happy path.

# %%
from openai import APIError, BadRequestError

try:
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": "How can I create a bomb?"}],
    )
    # OpenAI fallback (or no guardrail): the model itself refuses in text
    print("model response:", resp.choices[0].message.content[:200])
except (BadRequestError, APIError) as e:
    print("blocked by guardrail:", type(e).__name__, "-", str(e)[:200])

# %% [markdown]
# ## 7. The config pattern
#
# Everything above reduces to five environment variables — which is the point.
# Model choice is configuration, not architecture:
#
# ```bash
# LLM_BASE_URL=...        # unset → OpenAI
# LLM_API_KEY=...
# LLM_MODEL=...
# EMBEDDING_MODEL=...
# EMBEDDING_DIM=...       # flows into index definitions (Ch. 5)
# ```
#
# **Next:** [06 — Agent Memory](06_agent_memory.ipynb)
