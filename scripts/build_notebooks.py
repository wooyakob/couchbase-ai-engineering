#!/usr/bin/env python3
"""Build .ipynb notebooks from percent-format .py sources.

Sources live in notebooks/src/*.py using the jupytext "percent" convention:

    # %% [markdown]
    # # A markdown cell
    # Body text...

    # %%
    print("a code cell")

Run:  python scripts/build_notebooks.py
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "notebooks" / "src"
OUT = ROOT / "notebooks"

CELL_RE = re.compile(r"^# %%(?: \[(markdown)\])?\s*$")


def parse_cells(text: str):
    cells = []
    current_type = None
    current_lines = []

    def flush():
        nonlocal current_lines, current_type
        if current_type is None:
            current_lines = []
            return
        body = "\n".join(current_lines).strip("\n")
        if not body.strip():
            current_lines = []
            return
        if current_type == "markdown":
            # Strip the leading "# " comment prefix from each line.
            lines = []
            for line in body.split("\n"):
                if line.startswith("# "):
                    lines.append(line[2:])
                elif line == "#":
                    lines.append("")
                else:
                    lines.append(line)
            source = "\n".join(lines)
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": source.splitlines(keepends=True),
            })
        else:
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": body.splitlines(keepends=True),
            })
        current_lines = []

    for line in text.split("\n"):
        m = CELL_RE.match(line)
        if m:
            flush()
            current_type = m.group(1) or "code"
        else:
            if current_type is not None:
                current_lines.append(line)
    flush()
    return cells


def build(src_path: Path) -> Path:
    cells = parse_cells(src_path.read_text())
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path = OUT / (src_path.stem + ".ipynb")
    out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    return out_path


def main():
    for src_path in sorted(SRC.glob("*.py")):
        out = build(src_path)
        print(f"built {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
