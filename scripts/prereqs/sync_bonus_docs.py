#!/usr/bin/env python3
"""Regenerate the bonus-track tables in README.md, CURRICULUM.md and templates/00_welcome.ipynb.

Rows come from scripts/prereqs/curriculum.py (BONUS) and the bonus content files, so the
three tables can't drift apart. Run via `make prereqs`.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import curriculum  # noqa: E402
from build import load_notebooks  # noqa: E402
from repo_config import colab_url  # noqa: E402

BADGE = "https://colab.research.google.com/assets/colab-badge.svg"


def exercise_names(nb):
    return ", ".join("`" + re.search(r"`([A-Za-z0-9_]+)", e["spec"].strip().split("\n")[0]).group(1) + "`" for e in nb["exercises"])


def replace_block(text: str, start: str, end: str, body: str, where: str) -> str:
    i, j = text.find(start), text.find(end)
    if i < 0 or j < 0 or j < i:
        raise SystemExit(f"{where}: markers {start!r} / {end!r} not found")
    return text[: i + len(start)] + "\n" + body + text[j:]


def main() -> int:
    built = {nb["prereq_id"]: nb for nb in load_notebooks() if nb["prereq_id"].startswith("B")}

    readme_rows, curr_rows, welcome_rows = [], [], []
    for bid, title in curriculum.BONUS:
        nb = built.get(bid)
        if not nb:
            curr_rows.append(f"| {bid} | {title} | | | | planned |")
            continue
        f = nb["file"]
        tries = sorted(n for n, _ in nb["unlocks"])
        readme_rows.append(
            f'| {bid} | <a href="templates/{f}.ipynb" target="_blank">{nb["title"]}</a> '
            f'<a href="{colab_url(f"templates/{f}.ipynb")}" target="_blank"><img src="{BADGE}" alt="Open In Colab" height="20"></a> | '
            f'{exercise_names(nb)} | {", ".join(f"#{n:02d}" for n in tries)} |')
        curr_rows.append(f"| {bid} | [{title}](templates/{f}.ipynb) | {nb.get('covers', '')} | {exercise_names(nb)} | "
                         f"{', '.join(f'{n:02d}' for n in tries)} | ✅ |")
        welcome_rows.append(f"| {bid} | {nb['title']} | [Open]({f}.ipynb) | [Open]({f}_solution.ipynb) |")

    marker = ("<!-- bonus-table:start -->", "<!-- bonus-table:end -->")
    readme = ROOT / "README.md"
    readme.write_text(replace_block(readme.read_text(encoding="utf-8"), *marker,
                                    "| # | Bonus | Exercises | Try it on |\n|:---:|---|---|---|\n" + "\n".join(readme_rows) + "\n",
                                    "README.md"), encoding="utf-8")
    curr = ROOT / "CURRICULUM.md"
    curr.write_text(replace_block(curr.read_text(encoding="utf-8"), *marker,
                                  "| # | Notebook | Covers | Exercises | Try it on | Status |\n"
                                  "|---|----------|--------|-----------|-----------|:------:|\n" + "\n".join(curr_rows) + "\n",
                                  "CURRICULUM.md"), encoding="utf-8")
    wpath = ROOT / "templates" / "00_welcome.ipynb"
    nb = json.loads(wpath.read_text(encoding="utf-8"))
    cell = next(c for c in nb["cells"] if "".join(c["source"]).startswith("## ⭐ Bonus"))
    cell["source"] = ("## ⭐ Bonus (optional)\n\nTechniques beyond the problems. No problem requires them.\n\n"
                      "| # | Bonus | Template | Solution |\n|:---:|---|:---:|:---:|\n" + "\n".join(welcome_rows)).splitlines(keepends=True)
    wpath.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"bonus tables: {len(welcome_rows)} built, {len(curriculum.BONUS) - len(welcome_rows)} planned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
