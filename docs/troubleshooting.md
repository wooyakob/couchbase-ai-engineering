# Troubleshooting
Common failures across the notebooks and apps in this repo, grouped by where they show up.

Every chapter and app README links back here, this is the one place to check regardless of which notebook or app you're running.

First move: read the exception type and message. Every notebook and app in this repo raises Couchbase SDK / OpenAI SDK exceptions with a specific type: `AuthenticationException` and `BucketNotFoundException` are different bugs with different fixes, and the traceback tells you which one before you've changed anything.

---

## Connecting to Couchbase

| Error | Cause | Fix |
|---|---|---|
| `AttributeError: 'NoneType' object has no attribute 'startswith'` on `conn.startswith("couchbases://")`, or `InvalidArgumentException: The username must be a str` from `PasswordAuthenticator(...)` | `CB_CONN_STRING`/`CB_USERNAME`/`CB_PASSWORD` came back `None`. Every `load_dotenv()` call in this repo resolves `ENV_FILE` with `find_dotenv(..., usecwd=True)`, which walks up from the notebook's CWD (`notebooks/`, not the repo root) looking for that filename, so this fires when either `ENV_FILE` wasn't set before the Jupyter/Python process started, or it's set but that file genuinely doesn't exist anywhere above the CWD. | `ENV_FILE` is read once, at process start: setting it in a *different* terminal, or after Jupyter is already running, does nothing for existing kernels. Stop Jupyter, `ENV_FILE=.env.server jupyter lab` (or `.env.capella`) in the shell you're launching from, then re-run from cell 1. Confirm with a fresh cell: `import os; from dotenv import find_dotenv; print(os.getenv("ENV_FILE"), find_dotenv(os.getenv("ENV_FILE", ".env"), usecwd=True))`; the second value should be the absolute path to your env file, not an empty string. |
| `AuthenticationException` | Wrong `CB_USERNAME`/`CB_PASSWORD`, or a Capella database credential that hasn't been granted access to this bucket. | Verify credentials directly with `cbc` or the UI; on Capella, check the credential's bucket-level access in **Settings → Database Access**. |
| `UnAmbiguousTimeoutException` / connect hangs on `couchbases://...` | Missing the WAN profile: cloud (TLS) connections need different timeouts than local Docker. Every notebook/app in this repo already applies `opts.apply_profile(KnownConfigProfiles.WanDevelopment)` when `conn.startswith("couchbases://")`; this fires if that code path was skipped or edited out. | Confirm the profile line is still present and firing (add a `print` if unsure) before assuming it's a network/firewall issue. |
| `BucketNotFoundException` | The `ai` bucket doesn't exist yet. Notebook 01 provisions **scopes/collections inside** the bucket; it does not create the bucket itself (see README). | Create `ai` in the Couchbase Server or Capella UI first, *then* run notebook 01. |
| `ScopeNotFoundException` / `CollectionNotFoundException` | Notebook 01 (or the specific notebook that provisions that scope, e.g. notebook 04's `support` scope) hasn't been run against this cluster yet. | Run the prerequisite notebook listed at the top of the one that's failing: each notebook's intro cell states its prerequisites explicitly. |

---

## Search / Vector Indexes

| Error | Cause | Fix |
|---|---|---|
| `QueryIndexNotFoundException` right after creating an index | Search/GSI indexes build asynchronously: the index exists but hasn't finished ingesting/coming online yet. | Poll before querying: `get_indexed_documents_count()` for Search indexes (Ch. 5 §5.2), or `SELECT raw state FROM system:indexes WHERE name = $name` for GSI/vector indexes (Ch. 14), waiting for `"online"`. |
| Vector search returns plausible-looking but wrong results, no error at all | `dims` in the index definition doesn't match the embedding model's actual output length, or the `similarity` metric doesn't match what the model was trained for (Ch. 5 §5.1, Ch. 14 §14.2). This **never raises**; it just silently ranks badly. | Verify `ARRAY_LENGTH(embedding)` matches your index's `dims` via SQL++ (notebook 02's sanity-check cell does this), and confirm `similarity` matches your model (`dot_product`/`COSINE` for OpenAI-style normalized embeddings). |
| `CREATE VECTOR INDEX` / `APPROX_VECTOR_DISTANCE` parse error | Cluster predates Couchbase Server / Capella 8.0: Hyperscale/Composite Vector Indexes (Ch. 14) don't exist before that. | Use the Search Vector Index (Ch. 5) instead, or upgrade the cluster. |

---

## Eventing (Notebook 13) and Analytics (Notebook 14)

| Error | Cause | Fix |
|---|---|---|
| `ServiceUnavailableException` from `eventing_functions()` or `cluster.analytics_query()` | Eventing/Analytics are optional cluster services; neither is enabled on this cluster. | On Capella: **Settings → Service Groups**, add the missing service to a group (or add a new group), wait for scaling to finish. On self-managed Server: add the service to a node at cluster-init or via **Servers → Add Server**. Both notebooks check for this explicitly in their first cell and print the same guidance. |
| `EventingFunctionIdenticalKeyspaceException` | The Eventing function's metadata keyspace is the same collection as its source or a bucket binding. | Metadata must live in its own collection, distinct from every other keyspace the function touches (notebook 13 uses a dedicated `eventing_metadata` collection). |
| `EventingFunctionNotUnDeployedException` on `drop_function` | Dropping a function that's still deployed. | `undeploy_function(name)` first, wait for it to reach `Undeployed` (or catch `EventingFunctionNotDeployedException` if it already is), then `drop_function`. |
| Analytics aggregate query returns stale/partial counts right after `ALTER COLLECTION ... ENABLE ANALYTICS` | Shadowing is asynchronous: Analytics ingests a live copy of the collection, but "live" still means "eventually," not "synchronously." | Poll a `COUNT(*)` until it matches the expected document count before running the real report (notebook 14's `wait_ingested`), same pattern as waiting for a Search/GSI index to come online. |

---

## Models (OpenAI / Capella Model Service)

| Error | Cause | Fix |
|---|---|---|
| `openai.AuthenticationError` | Missing/invalid `OPENAI_API_KEY`, or (Capella path) a bad token/base64(username:password) combination. | Check which credential path is active, every notebook prints `backend: OpenAI` or `backend: Capella Model Service` on the connect cell; confirm the corresponding key/token is set. |
| `openai.NotFoundError` / 404 against a Capella endpoint | `CAPELLA_AI_ENDPOINT` is missing the trailing `/v1`, or the model name in `CAPELLA_LLM_MODEL`/`CAPELLA_EMBEDDING_MODEL` isn't actually deployed on that cluster. | Confirm the endpoint ends in `/v1` and the model name matches one listed in the Capella console's Model Service page. |
| `BadRequestError` when requesting `response_format={"type": "json_schema", ...}` | That backend/model doesn't support constrained JSON-schema decoding (varies by NIM/vLLM model version): expected, not a bug. | This is exactly what notebook 10 / Ch. 15's validate-and-repair fallback handles; if you're not using that pattern, catch the exception and fall back to `{"type": "json_object"}` plus Pydantic validation. |
| Every Ragas metric call stalls ~30s: notebook 08's scoring cell looks hung, with no error and no output, on OpenAI and Capella alike | Ragas sends a usage-analytics HTTP POST around every judge call. Its endpoint (`t.explodinggradients.com`) no longer resolves; the DNS failure takes ~30s (a `requests` timeout doesn't cover DNS resolution) and is silently swallowed — blocking the async event loop once per call, which also defeats any client-side concurrency. | Set `RAGAS_DO_NOT_TRACK=true` **before the first `ragas` import**. Notebook 08 does this in its env cell; set it in your own eval harnesses too. |

---

## AI Functions (Notebook 04)

| Error | Cause | Fix |
|---|---|---|
| `InternalServerFailureException` / `insufficient_credentials`, `first_error_message: "Error evaluating projection"`, `missing_role: query_external_access` | AI Functions call the model through the query service's `CURL()` mechanism internally; the database credential running the query lacks the cluster-level role that permits it. This is separate from bucket-level Read & Write access, which is all a "Basic" Capella credential's **Access** tab grants — it won't show `query_external_access` at all. | In Capella, **Settings → Database Access → \<credential\> → Roles tab → Add New Role**. Simplest fix: create a role with **all privileges** (not just `Query Curl Access`/`query_external_access`) and assign it to the credential — narrower roles are easy to under-grant and you'll just be back here for the next missing permission. Creating the role isn't enough by itself: confirm it's actually attached to the credential afterward. |
| Query runs (no permission error) but every row's `ai_sentiment`/`ai_summary`/`ai_classification` value is `{"errorType": "InternalServerError", "message": "Oops, something went wrong. Try again later."}` | Permissions are fine; the function's **associated model** integration itself is misconfigured or unreachable — a different failure mode than the row above, easy to mistake for the same problem. Check **AI Functions** tab → each function shows an **Associated Model**; click into it. | Confirm the associated model's endpoint/key/token are current — on Capella's own hosted "LLMModel", this means checking/re-adding its key and token if it shows stale, not the `CAPELLA_AI_TOKEN` in your `.env.*` (that var is only read by this repo's own client code in Path B, never by server-side AI Functions). Isolate the deployment itself first with a direct `curl` to `$CAPELLA_AI_ENDPOINT/chat/completions` using `CAPELLA_AI_TOKEN` — if that succeeds, the deployment is healthy and the problem is specifically in the AI Functions → model association, not the model. |

---

## Agent Catalog (`agentc`)

| Error | Cause | Fix |
|---|---|---|
| `agentc index` / `agentc publish` fails with a git-related error | `agentc` snapshots are keyed to git commits: it needs a real, committed git repo, not just a directory of files. | `git init && git add -A && git commit -m "..."` before indexing (notebook 07's `run()` helper does this for you); commit again after any tool/prompt change before re-publishing. |
| `agentc publish` fails to reach Couchbase | `AGENT_CATALOG_CONN_STRING`/`AGENT_CATALOG_USERNAME`/`AGENT_CATALOG_PASSWORD` unset or wrong; this is a **separate** credential namespace from `CB_*`, even if pointing at the same cluster. | Confirm all four `AGENT_CATALOG_*` vars are set (see `.env.server.example`/`.env.capella.example`); they don't fall back to `CB_*`. |

---

## Agent Memory Server (Docker)

Covered in detail, with a dedicated table for image-tag mismatches, port conflicts,
crash-looping containers, and TLS cert errors: [Chapter 9 §"Troubleshooting"](09-agent-memory.md#troubleshooting).

---

## apps/rag-api and apps/support-agent

| Error | Cause | Fix |
|---|---|---|
| `support-agent` recall silently returns nothing, no error | By design, the agent degrades gracefully if the Agent Memory server is unreachable (writes are skipped too). | Check `AGENTMEMORY_BASE_URL` is reachable (`curl $AGENTMEMORY_BASE_URL/docs`) if you expect memory to be active; the graceful-degrade means evals still pass without it, which can mask a real outage. |
| `rag-api` / `support-agent` picks the wrong backend (OpenAI vs. Capella) | `ENV_FILE` wasn't exported before starting the process (`uvicorn`, `pytest`, etc.), unlike Jupyter, a plain shell command needs `ENV_FILE` set in that same shell or passed inline. | `ENV_FILE=.env.capella uvicorn app.main:app --reload` (inline), or `export ENV_FILE=.env.capella` before starting, confirm with `echo $ENV_FILE` in that shell. |
| `500` from `/ingest` or `/ask` referencing a missing index/collection | The app assumes notebooks 01–02's provisioning already ran against this cluster; it doesn't provision anything itself. | Run notebooks 01–02 (or create the equivalent bucket/scope/collection/index by hand) against the same cluster the app's `.env` points at. |

---

## General

1. Read the exception **type**, not just the message: it's almost always a real Couchbase SDK or OpenAI SDK exception class, and the class name narrows the cause immediately.
2. Print/confirm the env vars actually in effect (`os.getenv(...)`, or `docker inspect <container> --format '{{range .Config.Env}}{{println .}}{{end}}'` for a running container); most failures in this repo trace back to a config value that's unset, stale, or pointed at the wrong cluster, not a code bug.
3. Check the specific notebook/chapter's **Prerequisites** line: most "not found" errors mean an earlier notebook in the sequence hasn't been run yet against this cluster.
