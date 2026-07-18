#!/usr/bin/env bash
# Assembles docs/*.md (in chapter order) into a single downloadable PDF via pandoc.
# Run locally (requires pandoc + a LaTeX engine, e.g. `brew install pandoc basictex`)
# or via .github/workflows/build-book-pdf.yml on every push that touches docs/.
set -euo pipefail

cd "$(dirname "$0")/.."

OUT="AI-Engineering-on-Couchbase.pdf"
TITLE_PAGE="$(mktemp).md"
HEADER_TEX="$(mktemp).tex"
CHAPTER_BREAK_LUA="$(mktemp).lua"

# No pandoc title/author metadata here — that would trigger \maketitle, which
# renders BEFORE any body content, so the logo could never appear above it.
# Instead the whole front matter (logo, title, author, copyright, TOC) is
# hand-built as body content so the logo is unambiguously the first thing in
# the PDF, followed immediately by the title.
#
# Title/byline are raw centered LaTeX text, not "# "/"### " pandoc headings:
# a heading here would (a) get auto-added to its own table of contents as a
# bogus top-level entry pointing at the title page itself, and (b) render at
# ordinary body-heading size instead of actual title-page size. Front matter
# (this page + the TOC) is numbered in roman numerals and the title page
# itself shows no visible number at all; page "1" starts at Chapter 1.
cat > "$TITLE_PAGE" <<'EOF'
```{=latex}
\pagenumbering{gobble}
\begin{center}
\vspace*{0.75in}
```

![](images/book_logo.png)

```{=latex}
\vspace{1in}
{\fontsize{30}{36}\selectfont\bfseries AI Engineering on Couchbase}\par
\vspace{0.75em}
{\Large Jake Wood, Solutions Engineer at Couchbase}\par
\vspace{2em}
{\normalsize © 2026 Jake Wood.}
\end{center}
\clearpage
\pagenumbering{roman}
\tableofcontents
\clearpage
\pagenumbering{arabic}
\setcounter{page}{1}
```
EOF

# Both mainfont (serif) and sansfont are set to Arial so it's used regardless
# of which family a given element resolves to; \familydefault forces plain
# body text specifically onto the sans family (fontspec's default otherwise
# leaves body text on the serif "mainfont", with sans only used for explicit
# \textsf/headings) — that's the actual "clear and readable" switch. Falls
# back to metric-compatible Liberation Sans if Arial itself isn't installed
# (true Arial is a licensed font, not open-source; the CI workflow installs it
# via ttf-mscorefonts-installer, but this keeps local builds working on
# machines without it).
cat > "$HEADER_TEX" <<'EOF'
\IfFontExistsTF{Arial}{\setmainfont{Arial}\setsansfont{Arial}}{\setmainfont{Liberation Sans}\setsansfont{Liberation Sans}}
\renewcommand{\familydefault}{\sfdefault}
\setcounter{tocdepth}{2}
% fvextra replaces fancyvrb's Verbatim (which backs pandoc's code blocks) with a
% version that can break mid-line — without this, a code line longer than the
% margin (e.g. a long SQL++/Python line) runs off the page instead of wrapping.
\usepackage{fvextra}
\fvset{breaklines=true,breakanywhere=true}
EOF

# Two fixes that both require seeing the whole merged document at once, so both
# live in a single Pandoc(doc) filter rather than separate per-element functions:
#
# 1. Chapter breaks. Each chapter is a level-1 heading ("# Chapter N: ..."), but
#    they're plain LaTeX \section commands (article class) with no page break
#    between them — Chapter 2 would otherwise start mid-page wherever Chapter
#    1's text happens to end. Insert a \clearpage before every level-1 heading
#    except the first.
#
# 2. Internal links. Every chapter's "Next:"/"Back to:" footer, and every
#    "Troubleshooting" reference, links to a sibling file by relative path
#    (e.g. "03-data-processing.md", "troubleshooting.md") — correct when
#    browsing docs/ on GitHub, but meaningless in a single merged PDF: chapter
#    order is already fixed by the binding, and troubleshooting.md/README.md
#    were never part of the compiled book. Rather than leave dead or
#    redundant footer text behind, those paragraphs are dropped from each
#    chapter's recap entirely.
cat > "$CHAPTER_BREAK_LUA" <<'EOF'
-- Mirrors GitHub's heading-slug algorithm (lowercase, strip punctuation,
-- spaces -> hyphens) so a hand-written "#148-through-langchain-instead"-style
-- fragment (written for GitHub's slugger) can be matched back to whichever
-- heading actually produces that slug, then translated to *pandoc's* own
-- auto-generated identifier for that heading (which strips leading numbers
-- entirely, e.g. "through-langchain-instead") for the fragment to resolve
-- inside the compiled PDF.
local function github_slug(text)
  text = text:lower()
  text = text:gsub("[^%w%s%-_]", "")
  text = text:gsub("%s+", "-")
  return text
end

function Pandoc(doc)
  local anchor_by_chapter = {}
  local anchor_by_github_slug = {}
  for _, block in ipairs(doc.blocks) do
    if block.t == "Header" then
      local text = pandoc.utils.stringify(block.content)
      anchor_by_github_slug[github_slug(text)] = block.identifier
      if block.level == 1 then
        local num = text:match("^Chapter%s+(%d+):")
        if num then
          anchor_by_chapter[tonumber(num)] = block.identifier
        end
      end
    end
  end

  local function is_dropped_footer(block)
    if block.t ~= "Para" then
      return false
    end
    local text = pandoc.utils.stringify(block.content)
    return text:match("^Next:") or text:match("^Back to:") or text:match("^Running into errors%?")
  end

  local seen_first = false
  local new_blocks = {}
  for _, block in ipairs(doc.blocks) do
    if block.t == "Header" and block.level == 1 then
      if seen_first then
        table.insert(new_blocks, pandoc.RawBlock("latex", "\\clearpage"))
      end
      seen_first = true
    end
    if not is_dropped_footer(block) then
      table.insert(new_blocks, block)
    end
  end
  doc.blocks = new_blocks

  return doc:walk({
    Link = function(el)
      if el.target:match("troubleshooting%.md") or el.target:match("README%.md") then
        return el.content
      end
      local fname, rest = el.target:match("^(%d%d%-[%w%-]+%.md)(.*)$")
      if fname then
        if rest:match("^#.+") then
          local anchor = anchor_by_github_slug[rest:sub(2)]
          if anchor then
            el.target = "#" .. anchor
            return el
          end
        end
        local anchor = anchor_by_chapter[tonumber(fname:sub(1, 2))]
        if anchor then
          el.target = "#" .. anchor
          return el
        end
      end
      return el
    end,
  })
end
EOF

CHAPTERS=(docs/[0-9][0-9]-*.md)

# -yaml_metadata_block: without this, pandoc treats the "---" section
# dividers used throughout docs/*.md as YAML frontmatter blocks, and chokes
# on any colon-terminated line inside one (e.g. "...looks like this:").
pandoc \
  -f markdown-yaml_metadata_block \
  "$TITLE_PAGE" \
  "${CHAPTERS[@]}" \
  -o "$OUT" \
  --pdf-engine=xelatex \
  -V geometry:margin=1in \
  -V linkcolor:blue \
  --include-in-header="$HEADER_TEX" \
  --lua-filter="$CHAPTER_BREAK_LUA" \
  --resource-path=.:docs:images

echo "Built $OUT"
