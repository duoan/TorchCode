#!/usr/bin/env python3
"""Add (or update) a "📖 Reference solution" link cell to every template notebook.

The link cell sits right after the first cell (the problem statement). It is a separate
cell on purpose: the web UI shows the first markdown cell as the problem description, and
a solution link there would be a spoiler and a broken link.

Two relative links, because templates and solutions live in different places:
- side by side in JupyterLab (Docker / `make setup-local` copy both into one folder)
- in sibling folders in the repo itself (GitHub, VS Code on a clone): ../solutions/

Generated prerequisite / bonus templates get the cell from scripts/prereqs/build.py; this
script handles the hand-written problem templates.

Usage:
    python scripts/sync_solution_links.py           update templates
    python scripts/sync_solution_links.py --check   exit 1 if any template is out of date
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MARKER = "📖 **Reference solution:**"


def link_text(stem: str) -> str:
    sol = f"{stem}_solution.ipynb"
    return (f"{MARKER} [`{sol}`]({sol}) (JupyterLab) · "
            f"[`solutions/{sol}`](../solutions/{sol}) (GitHub / VS Code). Try it yourself first!")


def link_cell(stem: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [link_text(stem)]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    stale = []
    for path in sorted((ROOT / "templates").glob("*.ipynb")):
        stem = path.stem
        if stem == "00_welcome" or stem.startswith(("00_p", "00_b")):   # generated ones: build.py
            continue
        if not (ROOT / "solutions" / f"{stem}_solution.ipynb").exists():
            raise SystemExit(f"{path.name}: no solutions/{stem}_solution.ipynb to link to")
        nb = json.loads(path.read_text(encoding="utf-8"))
        cells = nb["cells"]
        want = link_cell(stem)
        has = len(cells) > 1 and "".join(cells[1]["source"]).startswith(MARKER)
        if has and cells[1] == want:
            continue
        stale.append(path)
        if not args.check:
            if has:
                cells[1] = want
            else:
                cells.insert(1, want)
            path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    for p in stale:
        print(f"{'stale' if args.check else 'updated'}: {p.relative_to(ROOT)}")
    if not stale:
        print("solution links up to date")
    return 1 if (args.check and stale) else 0


if __name__ == "__main__":
    sys.exit(main())
