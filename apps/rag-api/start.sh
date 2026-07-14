#!/usr/bin/env bash
# Start rag-api locally. Backend switch matches every notebook/app in this repo:
#   ENV_FILE=.env.capella ./start.sh   # Capella Model Service
#   ./start.sh                        # local Couchbase + OpenAI (default)
set -euo pipefail
cd "$(dirname "$0")"

export ENV_FILE="${ENV_FILE:-.env.server}"
if [ ! -f "../../$ENV_FILE" ]; then
    echo "error: ../../$ENV_FILE not found — copy .env.server.example / .env.capella.example at the repo root and fill it in." >&2
    exit 1
fi

VENV="../../.venv"
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q -r requirements.txt

echo "rag-api starting on http://localhost:8000 (ENV_FILE=$ENV_FILE)"
exec "$VENV/bin/uvicorn" app.main:app --reload --host 0.0.0.0 --port 8000
