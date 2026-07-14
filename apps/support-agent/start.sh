#!/usr/bin/env bash
# Start the support-agent CLI. Backend switch matches every notebook/app in this repo:
#   ENV_FILE=.env.capella ./start.sh alice   # Capella Model Service, user "alice"
#   ./start.sh                               # local Couchbase + OpenAI, demo-user
set -euo pipefail
cd "$(dirname "$0")"

export ENV_FILE="${ENV_FILE:-.env.server}"
# find_dotenv(usecwd=True) checks this directory before climbing to the repo root —
# mirror that here so the error message points at whichever one will actually be used.
if [ -f "$ENV_FILE" ]; then
    :
elif [ -f "../../$ENV_FILE" ]; then
    :
else
    echo "error: no $ENV_FILE found in this directory or the repo root — see README's 'Why a separate file' note (this app needs its own copy, not the repo-root one, so its AGENT_CATALOG_BUCKET can differ)." >&2
    exit 1
fi

VENV="../../.venv"
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q -r requirements.txt

if [ ! -d ".agent-catalog" ]; then
    cat >&2 <<'EOF'
note: no local Agent Catalog found. One-time setup before this will work:
  agentc init
  agentc index .
  git add -A && git commit -m "support agent v1"
  agentc publish
(main.py will also tell you this if you skip ahead and run it anyway.)
EOF
fi

USER_ID="${1:-demo-user}"
echo "support-agent starting (ENV_FILE=$ENV_FILE, user=$USER_ID)"
exec "$VENV/bin/python" main.py "$USER_ID"
