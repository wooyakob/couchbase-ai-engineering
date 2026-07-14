"""Shared fixtures/config for support-agent's test suite.

Same two-tier convention as apps/rag-api/tests/ and the repo-root
tests/test_notebooks.py:
  - test_*_unit.py: pure logic + mocked collaborators, no live services needed.
  - evals/test_agent.py (pre-existing): real scenario evals against a live
    catalog + cluster + LLM — see that file's own docstring.

A placeholder OPENAI_API_KEY is supplied only if nothing real is configured, just
enough for any module-level construction that expects a key to be present without
making it a real network call.
"""

import os
import socket
import urllib.request
from urllib.parse import urlparse

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(os.getenv("ENV_FILE", ".env"), usecwd=True))

_PLACEHOLDER_KEY = "sk-test-placeholder-not-a-real-key"
if not (os.getenv("OPENAI_API_KEY") or os.getenv("CAPELLA_AI_ENDPOINT")):
    os.environ["OPENAI_API_KEY"] = _PLACEHOLDER_KEY


def has_cb() -> bool:
    return all(os.getenv(n) for n in
              ("CB_CONN_STRING", "CB_USERNAME", "CB_PASSWORD", "CB_BUCKET"))


def has_llm() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") not in (None, "", _PLACEHOLDER_KEY)
               or os.getenv("CAPELLA_AI_ENDPOINT"))


def _host_resolves(conn_string: str) -> bool:
    host = urlparse(conn_string).hostname
    if not host:
        return False
    try:
        socket.getaddrinfo(host, None)
        return True
    except socket.gaierror:
        return False


def has_agent_catalog() -> bool:
    if not all(os.getenv(n) for n in
              ("AGENT_CATALOG_CONN_STRING", "AGENT_CATALOG_USERNAME",
               "AGENT_CATALOG_PASSWORD")):
        return False
    return _host_resolves(os.environ["AGENT_CATALOG_CONN_STRING"])


def has_agent_memory_server() -> bool:
    base_url = os.getenv("AGENTMEMORY_BASE_URL")
    if not base_url:
        return False
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=2):
            return True
    except (urllib.error.URLError, socket.timeout, ConnectionError):
        return False
