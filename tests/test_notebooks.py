"""Executes every notebook end-to-end and fails on any cell error.

These are integration smoke tests, not unit tests — each notebook talks to a real
Couchbase cluster and a real model API, so there is nothing meaningful to mock. Run
with:

    pytest tests/test_notebooks.py

A notebook whose prerequisites aren't met (no cluster configured, no LLM key, a
sidecar service that isn't running) is **skipped**, not failed — see the `NEEDS_*`
markers below. Set `ENV_FILE` beforehand exactly as you would to open the notebooks in
Jupyter (`.env.server` / `.env.capella`); this suite reads the same environment.

Each notebook can take minutes (batch LLM calls, index builds) — this is not a fast
suite. Run a single notebook while iterating:

    pytest tests/test_notebooks.py -k 04_ai_functions
"""

import os
import socket
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import nbformat
import pytest
from dotenv import load_dotenv
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from nbclient.exceptions import CellTimeoutError

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"

# Every notebook's own load_dotenv() reads a path relative to the *kernel's* cwd
# (the notebook's directory), not this process's cwd — and not necessarily where this
# repo's .env.server/.env.capella actually live (repo root). Resolve ENV_FILE to an
# absolute path once, here, so both our own skip checks below and every notebook
# kernel (which inherits this process's environment) resolve the same file regardless
# of cwd.
os.environ["ENV_FILE"] = str((ROOT / os.getenv("ENV_FILE", ".env")).resolve())
load_dotenv(os.environ["ENV_FILE"])
DEFAULT_TIMEOUT = int(os.getenv("NOTEBOOK_TEST_TIMEOUT", "1800"))  # seconds, per cell


def _env(*names: str) -> bool:
    return all(os.getenv(n) for n in names)


def _has_cb() -> bool:
    return _env("CB_CONN_STRING", "CB_USERNAME", "CB_PASSWORD", "CB_BUCKET")


def _has_llm() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("CAPELLA_AI_ENDPOINT"))


def _has_ai_functions() -> bool:
    return os.getenv("AI_FUNCTIONS_ENABLED", "").lower() == "true"


def _has_vector_ddl() -> bool:
    """Hyperscale/Composite vector indexes need Server/Capella 8.0+ (notebook 11)."""
    return os.getenv("CB_SUPPORTS_VECTOR_DDL", "").lower() == "true"


def _host_resolves(conn_string: str) -> bool:
    host = urlparse(conn_string).hostname
    if not host:
        return False
    try:
        socket.getaddrinfo(host, None)
        return True
    except socket.gaierror:
        return False


def _has_agent_catalog() -> bool:
    if not _env("AGENT_CATALOG_CONN_STRING", "AGENT_CATALOG_USERNAME", "AGENT_CATALOG_PASSWORD"):
        return False
    return _host_resolves(os.environ["AGENT_CATALOG_CONN_STRING"])


def _has_agent_memory_server() -> bool:
    base_url = os.getenv("AGENTMEMORY_BASE_URL")
    if not base_url:
        return False
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=2):
            return True
    except (urllib.error.URLError, socket.timeout, ConnectionError):
        return False


NEEDS_CB = pytest.mark.skipif(not _has_cb(), reason="no Couchbase cluster configured "
                              "(CB_CONN_STRING/CB_USERNAME/CB_PASSWORD/CB_BUCKET)")
NEEDS_LLM = pytest.mark.skipif(not _has_llm(), reason="no LLM configured "
                               "(OPENAI_API_KEY or CAPELLA_AI_ENDPOINT)")
NEEDS_AI_FUNCTIONS = pytest.mark.skipif(not _has_ai_functions(), reason="AI_FUNCTIONS_ENABLED "
                                        "is not 'true' (Capella cluster with AI Functions only)")
NEEDS_VECTOR_DDL = pytest.mark.skipif(not _has_vector_ddl(), reason="CB_SUPPORTS_VECTOR_DDL "
                                      "is not 'true' (Couchbase Server/Capella 8.0+ only)")
NEEDS_AGENT_CATALOG = pytest.mark.skipif(not _has_agent_catalog(), reason="no Agent Catalog "
                                         "cluster configured (AGENT_CATALOG_CONN_STRING/...)")
NEEDS_AGENT_MEMORY_SERVER = pytest.mark.skipif(not _has_agent_memory_server(),
                                               reason="no Agent Memory server reachable at "
                                               "AGENTMEMORY_BASE_URL")


def run_notebook(name: str, timeout: int = DEFAULT_TIMEOUT) -> None:
    path = NOTEBOOKS / name
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(NOTEBOOKS)}})
    try:
        client.execute()
    except (CellExecutionError, CellTimeoutError) as e:
        pytest.fail(f"{name} raised during execution:\n{e}")


@NEEDS_CB
@NEEDS_LLM
def test_01_python_sdk_quickstart():
    run_notebook("01_python_sdk_quickstart.ipynb")


@NEEDS_CB
@NEEDS_LLM
def test_02_vector_search_fundamentals():
    run_notebook("02_vector_search_fundamentals.ipynb")


@NEEDS_CB
@NEEDS_LLM
def test_03_rag_pipeline():
    run_notebook("03_rag_pipeline.ipynb")


@NEEDS_CB
@NEEDS_LLM
def test_04_ai_functions():
    # Runs in portable-SDK mode unless AI_FUNCTIONS_ENABLED=true; either way is valid.
    run_notebook("04_ai_functions.ipynb")


@NEEDS_CB
@NEEDS_LLM
def test_05_model_service():
    run_notebook("05_model_service.ipynb")


@NEEDS_CB
@NEEDS_LLM
def test_06_agent_memory():
    run_notebook("06_agent_memory.ipynb")


@NEEDS_CB
@NEEDS_LLM
@NEEDS_AGENT_CATALOG
def test_07_agent_catalog_langgraph():
    run_notebook("07_agent_catalog_langgraph.ipynb")


@NEEDS_CB
@NEEDS_LLM
def test_08_ragas_evaluation():
    run_notebook("08_ragas_evaluation.ipynb")


@NEEDS_CB
@NEEDS_AGENT_MEMORY_SERVER
def test_09_agent_memory_managed():
    run_notebook("09_agent_memory_managed.ipynb")


@NEEDS_CB
@NEEDS_LLM
def test_10_structured_outputs():
    run_notebook("10_structured_outputs.ipynb")


@NEEDS_CB
@NEEDS_VECTOR_DDL
def test_11_vector_index_architectures():
    run_notebook("11_vector_index_architectures.ipynb")
