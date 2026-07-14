"""Shared fixtures/config for rag-api's test suite.

Two tiers, same convention as the repo-root tests/test_notebooks.py:
  - test_*_unit.py: pure logic, no network/live services. app/config.py validates
    a model backend is configured at *import* time though, so if nothing real is
    configured we supply a harmless placeholder OPENAI_API_KEY here — just enough
    to satisfy that check, never used to make a real call.
  - test_api_smoke.py: the real end-to-end flow against a live Couchbase cluster +
    LLM backend. Skips itself (not fails) when HAS_LIVE_CB / HAS_LIVE_LLM are False.
"""

import os

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(os.getenv("ENV_FILE", ".env"), usecwd=True))

_PLACEHOLDER_KEY = "sk-test-placeholder-not-a-real-key"

HAS_LIVE_LLM = bool(os.getenv("OPENAI_API_KEY") or os.getenv("CAPELLA_AI_ENDPOINT"))
HAS_LIVE_CB = bool(os.getenv("CB_CONN_STRING") and os.getenv("CB_USERNAME")
                   and os.getenv("CB_PASSWORD") and os.getenv("CB_BUCKET"))

# Only fill in the placeholder if nothing real is configured — never clobber a
# genuine key/endpoint a developer has set for running the smoke tests too.
if not HAS_LIVE_LLM:
    os.environ["OPENAI_API_KEY"] = _PLACEHOLDER_KEY
