#!/usr/bin/env bash
# Assembles docs/*.md (in chapter order) into a single downloadable PDF via pandoc.
# Run locally (requires pandoc + a LaTeX engine, e.g. `brew install pandoc basictex`)
# or via .github/workflows/build-book-pdf.yml on every push that touches docs/.
set -euo pipefail

cd "$(dirname "$0")/.."

OUT="AI-Engineering-on-Couchbase.pdf"
TITLE_PAGE="$(mktemp).md"

cat > "$TITLE_PAGE" <<'EOF'
---
title: "AI Engineering on Couchbase"
author: "Jake Wood, Solutions Engineer at Couchbase"
date: ""
---

![](images/book_logo.png)

© 2026 Jake Wood.
EOF

CHAPTERS=(docs/[0-9][0-9]-*.md)

pandoc \
  "$TITLE_PAGE" \
  "${CHAPTERS[@]}" \
  -o "$OUT" \
  --pdf-engine=xelatex \
  --toc \
  --toc-depth=2 \
  -V geometry:margin=1in \
  -V linkcolor:blue \
  --resource-path=.:docs:images

echo "Built $OUT"
