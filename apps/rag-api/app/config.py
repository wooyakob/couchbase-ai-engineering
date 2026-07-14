"""Environment-driven configuration.

Model choice is configuration, not architecture (Chapter 8): set CAPELLA_AI_ENDPOINT
to run against the Capella Model Service, leave it unset for OpenAI.
"""

import base64
import os

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(os.getenv("ENV_FILE", ".env"), usecwd=True))  # ENV_FILE lets .env.server / .env.capella run side by side

# --- Couchbase ---
CB_CONN_STRING = os.getenv("CB_CONN_STRING", "couchbase://localhost")
CB_USERNAME = os.getenv("CB_USERNAME", "Administrator")
CB_PASSWORD = os.getenv("CB_PASSWORD", "password")
CB_BUCKET = os.getenv("CB_BUCKET", "ai")

DOCS_SCOPE = "docs"
CHUNKS_COLLECTION = "chunks"
CHUNKS_INDEX = "chunks-vector-index"
CHAT_SCOPE = "agent"
CHAT_COLLECTION = "chat_history"
EVALS_SCOPE = "evals"
SAMPLES_COLLECTION = "samples"

# --- Models ---
CAPELLA_AI_ENDPOINT = os.getenv("CAPELLA_AI_ENDPOINT")  # ends with /v1 when set

if CAPELLA_AI_ENDPOINT:
    if not CAPELLA_AI_ENDPOINT.rstrip("/").endswith("/v1"):
        raise RuntimeError(
            f"CAPELLA_AI_ENDPOINT={CAPELLA_AI_ENDPOINT!r} doesn't end with /v1 — the "
            "Capella Model Service's OpenAI-compatible API requires it (Ch. 8 §8.1). "
            "Fix the URL in your .env, or unset CAPELLA_AI_ENDPOINT to use OpenAI "
            "instead. See docs/troubleshooting.md."
        )
    LLM_BASE_URL = CAPELLA_AI_ENDPOINT
    # Prefer a Capella AI API key/token if the deployment issued one; otherwise
    # fall back to database credentials encoded as base64 (username:password).
    LLM_API_KEY = os.getenv("CAPELLA_AI_TOKEN") or base64.b64encode(
        f"{CB_USERNAME}:{CB_PASSWORD}".encode()).decode()
    LLM_MODEL = os.getenv("CAPELLA_LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    EMBEDDING_MODEL = os.getenv("CAPELLA_EMBEDDING_MODEL", "intfloat/e5-mistral-7b-instruct")
else:
    LLM_BASE_URL = None
    LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")
    if not LLM_API_KEY:
        raise RuntimeError(
            "No model backend configured: set OPENAI_API_KEY, or set "
            "CAPELLA_AI_ENDPOINT (+ CAPELLA_AI_TOKEN) to use the Capella Model "
            "Service instead (Ch. 8). Check which .env file ENV_FILE points at — "
            f"currently ENV_FILE={os.getenv('ENV_FILE', '.env')!r}. "
            "See docs/troubleshooting.md."
        )
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Chunking (Chapter 3) — versioned so re-processing is auditable
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "2000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
PIPELINE_VERSION = "rag-api-v1"

RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))
