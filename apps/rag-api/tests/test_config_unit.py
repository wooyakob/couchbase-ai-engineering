"""Unit tests for the OpenAI <-> Capella Model Service switch in app/config.py
(Chapter 8). config.py resolves the backend at *import time*, so these tests
monkeypatch env vars and importlib.reload the module to exercise each branch.
"""

import importlib
import os

import pytest

import app.config as config_module

# A snapshot of the REAL values (or absence) of just the config-relevant vars this
# process was launched with, taken before any test in this module monkeypatches
# anything — used to restore app.config to a real, non-placeholder state once,
# after the whole file's tests finish (see _restore_real_config_after_module
# below). Without this, whatever the LAST test in this file leaves app.config's
# attributes as (e.g. a placeholder OpenAI key) would leak into any test file that
# runs after this one and does a plain `import app.config` — module objects are
# cached in sys.modules, and a plain import does not re-execute an already-imported
# module, so it would see stale attributes from this file's last reload() rather
# than the real config (this is exactly what broke tests/test_api_smoke.py the
# first time these ran together in one `pytest tests/` invocation).
#
# Deliberately NOT os.environ.clear() + restore-everything: that wipes vars pytest
# itself adds after this snapshot (e.g. PYTEST_CURRENT_TEST), breaking pytest's own
# teardown. Only the specific keys config.py's import-time logic reads are safe to
# touch here.
_CONFIG_VARS = ("CAPELLA_AI_ENDPOINT", "CAPELLA_AI_TOKEN", "CAPELLA_LLM_MODEL",
               "CAPELLA_EMBEDDING_MODEL", "OPENAI_API_KEY", "LLM_MODEL",
               "EMBEDDING_MODEL", "CB_USERNAME", "CB_PASSWORD")
_REAL_CONFIG_ENV = {var: os.environ[var] for var in _CONFIG_VARS if var in os.environ}


@pytest.fixture(scope="module", autouse=True)
def _restore_real_config_after_module():
    yield
    for var in _CONFIG_VARS:
        if var in _REAL_CONFIG_ENV:
            os.environ[var] = _REAL_CONFIG_ENV[var]
        else:
            os.environ.pop(var, None)
    importlib.reload(config_module)


@pytest.fixture(autouse=True)
def _isolated_config_env(monkeypatch):
    """Every test gets a clean slate: no backend env vars leaking in from the
    real .env this process loaded.

    config.py calls load_dotenv() itself at module scope — every reload() below
    would otherwise re-source the real .env.capella/.env.server this process was
    launched with, silently undoing the monkeypatch.delenv calls right below (that
    only happens when a real ENV_FILE is actually configured, e.g. running the
    full suite together with the smoke tests — it's real, not hypothetical).
    Patching dotenv.load_dotenv to a no-op — not config.load_dotenv, which reload()
    would just re-bind from dotenv's namespace anyway — keeps these tests in full
    control of the env regardless of what's really configured for the smoke tests.
    """
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    for var in ("CAPELLA_AI_ENDPOINT", "CAPELLA_AI_TOKEN", "CAPELLA_LLM_MODEL",
               "CAPELLA_EMBEDDING_MODEL", "OPENAI_API_KEY", "LLM_MODEL",
               "EMBEDDING_MODEL"):
        monkeypatch.delenv(var, raising=False)


def test_openai_default(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-real-enough")
    cfg = importlib.reload(config_module)
    assert cfg.LLM_BASE_URL is None
    assert cfg.LLM_API_KEY == "sk-test-real-enough"
    assert cfg.LLM_MODEL == "gpt-4o-mini"
    assert cfg.EMBEDDING_MODEL == "text-embedding-3-small"


def test_openai_model_overrides(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-real-enough")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-large")
    cfg = importlib.reload(config_module)
    assert cfg.LLM_MODEL == "gpt-4o"
    assert cfg.EMBEDDING_MODEL == "text-embedding-3-large"


def test_no_backend_configured_raises(monkeypatch):
    with pytest.raises(RuntimeError, match="No model backend configured"):
        importlib.reload(config_module)


def test_capella_switch_with_token(monkeypatch):
    monkeypatch.setenv("CAPELLA_AI_ENDPOINT", "https://example.ai.cloud.couchbase.com/v1")
    monkeypatch.setenv("CAPELLA_AI_TOKEN", "cbsk-test-token")
    cfg = importlib.reload(config_module)
    assert cfg.LLM_BASE_URL == "https://example.ai.cloud.couchbase.com/v1"
    assert cfg.LLM_API_KEY == "cbsk-test-token"
    assert cfg.LLM_MODEL == "meta-llama/Llama-3.1-8B-Instruct"
    assert cfg.EMBEDDING_MODEL == "intfloat/e5-mistral-7b-instruct"


def test_capella_switch_model_overrides(monkeypatch):
    monkeypatch.setenv("CAPELLA_AI_ENDPOINT", "https://example.ai.cloud.couchbase.com/v1")
    monkeypatch.setenv("CAPELLA_AI_TOKEN", "cbsk-test-token")
    monkeypatch.setenv("CAPELLA_LLM_MODEL", "mistralai/mistral-7b-instruct-v0.3")
    monkeypatch.setenv("CAPELLA_EMBEDDING_MODEL", "nvidia/llama-3.2-nv-embedqa-1b-v2")
    cfg = importlib.reload(config_module)
    assert cfg.LLM_MODEL == "mistralai/mistral-7b-instruct-v0.3"
    assert cfg.EMBEDDING_MODEL == "nvidia/llama-3.2-nv-embedqa-1b-v2"


def test_capella_falls_back_to_basic_auth_without_token(monkeypatch):
    monkeypatch.setenv("CAPELLA_AI_ENDPOINT", "https://example.ai.cloud.couchbase.com/v1")
    monkeypatch.setenv("CB_USERNAME", "alice")
    monkeypatch.setenv("CB_PASSWORD", "hunter2")
    cfg = importlib.reload(config_module)
    import base64
    assert cfg.LLM_API_KEY == base64.b64encode(b"alice:hunter2").decode()


def test_capella_endpoint_must_end_with_v1(monkeypatch):
    monkeypatch.setenv("CAPELLA_AI_ENDPOINT", "https://example.ai.cloud.couchbase.com")
    with pytest.raises(RuntimeError, match=r"/v1"):
        importlib.reload(config_module)


def test_capella_endpoint_trailing_slash_before_v1_is_ok(monkeypatch):
    monkeypatch.setenv("CAPELLA_AI_ENDPOINT", "https://example.ai.cloud.couchbase.com/v1/")
    monkeypatch.setenv("CAPELLA_AI_TOKEN", "cbsk-test-token")
    cfg = importlib.reload(config_module)
    assert cfg.LLM_BASE_URL == "https://example.ai.cloud.couchbase.com/v1/"
