# Contributing

Thanks for considering a contribution to **AI Engineering on Couchbase**. This is a personal project and not Couchbase-supported (book + notebooks + apps), so contributions are reviewed by me directly. Please follow the process below so review stays quick and predictable.

---

## Reporting Issues
Found a bug, a broken notebook cell, a doc error, or something that doesn't match the book's text? Open a GitHub issue rather than emailing for anything code-related. Issues are searchable and let others avoid hitting the same problem.

A good issue includes:
- **What you did**: the exact command, notebook cell, or app endpoint you ran.
- **What you expected** vs. **what happened**: include the full error/traceback, not a paraphrase.
- **Environment**: Couchbase Server or Capella, version, OS, Python version, and whether you're using OpenAI or the Capella Model Service.
- **Scope**: which notebook/app/chapter this affects, so it's easy to triage.

Vague reports ("it doesn't work") will get a request for more detail before anything else happens, so including the above up front saves a round trip.

---

## Contributing Changes
1. **Fork the repo**: don't request push access; all changes come in via fork.
2. **Branch off `main`**: in your fork (`git checkout -b fix/short-description`). Avoid working directly on your fork's `main`, since it makes it harder to keep in sync with upstream.
3. **Make your change**: matching the conventions of the surrounding code. See the repo's existing notebooks and apps for style (e.g. notebooks are generated from `notebooks/src/*.py` via `python scripts/build_notebooks.py`. Edit the sources, not the `.ipynb` files directly).
4. **Run the relevant test suite locally and make sure it passes**: before opening a PR, see below. PRs without passing tests won't be merged, and PRs that don't demonstrate the tests were run will be asked to do so.
5. **Open a pull request** from your fork's branch back to this repo's `main`, for review by me (@wooyakob). Describe what changed and why, and which test suite(s) you ran.

I'll review PRs as time allows. This is a side project, so please be patient. I may ask for changes before merging.

---

## Running The Test Suite
There are three independent suites, one per notebooks/app. Each has a **unit** tier (no live services, always runs) and a **live** tier (needs a real Couchbase cluster + LLM backend, and skips itself (not fails) when that's not configured). Run whichever tier(s) match what you touched; live-tier runs are strongly preferred for anything that touches the notebooks or app runtime code, not just docs/comments.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Notebooks** (repo root): executes every notebook end-to-end, cell by cell:

```bash
pytest tests/test_notebooks.py
```

**`apps/rag-api`**:

```bash
cd apps/rag-api
pip install -r requirements.txt
pytest
```

**`apps/support-agent`**:

```bash
cd apps/support-agent
pip install -r requirements.txt
pytest
```

For the live tier, copy the repo-root `.env.server.example` or `.env.capella.example` into a `.env` (or set `ENV_FILE`) with real Couchbase `OPENAI_API_KEY` or `CAPELLA_AI_ENDPOINT` values first.

See each app's own README for specifics (e.g. `apps/support-agent` also needs a running Agent Memory server and a dedicated `AGENT_CATALOG_BUCKET`).

Without those, the live-tier tests skip automatically and only the unit tier runs.

If a suite fails against `main` before your change, mention that in the PR rather than trying to fix unrelated failures as part of your change. Open a separate issue for it instead.

---

## Scope
This repo pairs a book with runnable code. Small fixes (typos, broken links, bug fixes, dependency bumps) are the easiest to review and merge quickly.

For larger changes (new chapters, new apps, architectural changes), please open an issue to discuss the idea first, so we don't diverge on direction before you've put in the work.
