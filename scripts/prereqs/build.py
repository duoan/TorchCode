#!/usr/bin/env python3
"""Generate the prerequisite notebooks and LEARNING_PATH.md from Python definitions.

Sources (edit these):
    scripts/prereqs/content/pXX_<topic>.py   one NOTEBOOK dict per prerequisite
    scripts/prereqs/curriculum.py              prerequisite list and module order

Generated (never edit by hand):
    templates/00_pXX_<topic>.ipynb             blank template (stubs)
    solutions/00_pXX_<topic>_solution.ipynb    same notebook with reference solutions
    LEARNING_PATH.md                           recommended order with links

The judge tests in torch_judge/tasks/pXX_<topic>.py are hand-written, not generated.

Usage:
    python scripts/prereqs/build.py           write all outputs
    python scripts/prereqs/build.py --check   exit 1 if any output is stale (for CI)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent))

from add_colab_badges import badge_markdown  # noqa: E402
from torch_judge.tasks import TASKS  # noqa: E402

sys.path.insert(0, str(HERE))
import curriculum  # noqa: E402

REPO_URL = "https://github.com/duoan/TorchCode/blob/master"
COLAB_URL = "https://colab.research.google.com/github/duoan/TorchCode/blob/master"
DIFF_ICON = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}
# Template file stems whose judge task id differs (keep in sync with api/parser.py)
TASK_ID_ALIASES = {"multihead_attention": "mha"}

HOW_TO_USE = """
### How to use this notebook
1. **Run the lesson cells** (sections 1, 2, …) from top to bottom. They are demos: change them and re-run.
2. **Do the exercises in order.** Each one has its own `✏️ YOUR IMPLEMENTATION HERE` cell, followed by a debug cell for checking shapes before you submit.
3. **Submit at any time** with the `check("{task_id}")` cell at the bottom. It grades every exercise at once, and each test is named after its exercise, so a partial score shows which ones are left.
4. **Stuck?** `hint("{task_id}")` gives one hint per exercise; `{file}_solution.ipynb` has the reference code.
5. **Already comfortable with this skill?** Prerequisites are optional. Skip straight to the problems listed in **🔓 What this unlocks** at the bottom.
"""

COLAB_CELL = """# Install torch-judge in Colab (no-op in JupyterLab/Docker)
try:
    import google.colab
    get_ipython().run_line_magic('pip', 'install -q torch-judge')
except ImportError:
    pass
"""


# ── Inputs ──────────────────────────────────────────────────────────────────

def load_problems() -> dict[int, tuple[str, str, str]]:
    """Problem number -> (file stem, display name, difficulty), from the README tables."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    names = dict(re.findall(r'templates/(\d+_\w+)\.ipynb" target="_blank">([^<]+)</a>', readme))
    problems = {}
    for stem, name in names.items():
        num = int(stem.split("_", 1)[0])
        if num == 0:
            continue
        task_id = stem.split("_", 1)[1]
        task_id = TASK_ID_ALIASES.get(task_id, task_id)
        problems[num] = (stem, name, TASKS[task_id]["difficulty"])
    for stem in (p.stem for p in (ROOT / "templates").glob("[0-9]*.ipynb")):
        num = int(stem.split("_", 1)[0])
        if num and num not in problems:
            raise SystemExit(f"Problem {stem} is missing from the README problem tables")
    return problems


def load_notebooks() -> list[dict]:
    notebooks = []
    for path in sorted((HERE / "content").glob("p*.py")):
        spec = importlib.util.spec_from_file_location(f"prereq_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        nb = dict(mod.NOTEBOOK, source=f"scripts/prereqs/content/{path.name}")
        if nb["task_id"] not in TASKS:
            raise SystemExit(f"{path.name}: no judge task torch_judge/tasks/{nb['task_id']}.py")
        notebooks.append(nb)
    return notebooks


# ── Notebook rendering ──────────────────────────────────────────────────────

def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")}


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.strip("\n")}


def unlocks_cell(nb_def: dict, problems: dict) -> dict:
    rows = []
    for num, reuse in nb_def["unlocks"]:
        if "|" in reuse:
            raise SystemExit(f"{nb_def['file']}: '|' in unlock text for #{num} breaks the table")
        stem, name, diff = problems[num]
        links = (f"[Open]({stem}.ipynb) · <a href=\"{REPO_URL}/templates/{stem}.ipynb\" target=\"_blank\">GitHub</a>"
                 f" · <a href=\"{COLAB_URL}/templates/{stem}.ipynb\" target=\"_blank\">Colab</a>")
        rows.append(f"| #{num} | {name} | {DIFF_ICON[diff]} {diff} | {reuse} | {links} |")
    return md(
        "---\n## 🔓 What this unlocks\n\n"
        f"With **{nb_def['title']}** in your toolkit, you're ready for these problems:\n\n"
        "| # | Problem | Difficulty | What you'll reuse | Links |\n"
        "|:---:|---|:---:|---|---|\n" + "\n".join(rows) + "\n\n"
        f"🗺️ See the full recommended order in "
        f"<a href=\"{REPO_URL}/LEARNING_PATH.md\" target=\"_blank\">LEARNING_PATH.md</a>."
    )


def render_notebook(nb_def: dict, problems: dict, solution: bool) -> str:
    folder = "solutions" if solution else "templates"
    filename = f"{nb_def['file']}_solution.ipynb" if solution else f"{nb_def['file']}.ipynb"
    task_id = nb_def["task_id"]

    if solution:
        header = (f"# 📖 Solution: {nb_def['title']}\n\nReference solutions for every exercise in "
                  f"`{nb_def['file']}.ipynb`. Try the template first!")
    else:
        header = (nb_def["intro"].strip("\n") + "\n\n"
                  + HOW_TO_USE.strip("\n").format(task_id=task_id, file=nb_def["file"]))
    cells = [md(header), code(COLAB_CELL), code(nb_def.get("imports", "import torch"))]
    for kind, payload in nb_def["lesson"]:
        cells.append(md(payload) if kind == "md" else code(payload))
    cells.append(md(nb_def["exercises_intro"]))
    for ex in nb_def["exercises"]:
        cells.append(md(ex["spec"]))
        if solution:
            cells.append(code("# ✅ SOLUTION\n\n" + ex["solution"].strip("\n")))
        else:
            cells.append(code("# ✏️ YOUR IMPLEMENTATION HERE\n\n" + ex["stub"].strip("\n")))
        cells.append(code(ex["debug"]))
    # Hidden from learners (HTML comment), visible to anyone editing the notebook JSON.
    # Not in the first cell: the web UI shows that cell as the problem description.
    generated_note = (f"<!-- Generated from {nb_def['source']} by `make prereqs`. "
                      f"Edit that file instead of this notebook; manual edits are overwritten. -->\n")
    cells.append(md(generated_note + "## ✅ Submit\n\nRun the judge on **all** exercises in this notebook. "
                    "Each test is labelled with the exercise it checks."))
    cells.append(code(f'from torch_judge import check, hint\ncheck("{task_id}")'))
    if not solution:
        cells.append(code(f'# 💡 Stuck? Uncomment for a nudge\n# hint("{task_id}")'))
    cells.append(unlocks_cell(nb_def, problems))

    for c in cells:
        c["source"] = c["source"].splitlines(keepends=True)
    # Same layout as scripts/add_colab_badges.py, so running that script afterwards is a no-op
    cells[0]["source"].insert(0, badge_markdown(filename, folder) + "\n\n")
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
            "torchcode": {
                "generated_by": "scripts/prereqs/build.py (make prereqs)",
                "source": nb_def["source"],
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(nb, ensure_ascii=False, indent=1) + "\n"


# ── LEARNING_PATH.md ────────────────────────────────────────────────────────

def render_learning_path(notebooks: list[dict], problems: dict) -> str:
    by_id = {nb["prereq_id"]: nb for nb in notebooks}
    unknown = set(by_id) - {pid for pid, _, _ in curriculum.PREREQS}
    if unknown:
        raise SystemExit(f"Notebooks with prereq_id not listed in curriculum.PREREQS: {sorted(unknown)}")

    prereqs = []  # (id, title, notebook file stem or None, unlocked problem numbers)
    for pid, title, planned in curriculum.PREREQS:
        nb = by_id.get(pid)
        if nb:
            prereqs.append((pid, title, nb["file"], [n for n, _ in nb["unlocks"]]))
        elif planned is None:
            raise SystemExit(f"{pid} has no notebook and no planned unlock list")
        else:
            prereqs.append((pid, title, None, planned))

    covered = sorted(n for _, _, nums in curriculum.MODULES for n in nums)
    if covered != sorted(problems):
        raise SystemExit(f"curriculum.MODULES must cover every problem exactly once, got {covered}")

    prereqs_of: dict[int, list[tuple[str, str | None]]] = {}
    for pid, _title, stem, nums in prereqs:
        for n in nums:
            prereqs_of.setdefault(n, []).append((pid, stem))

    def prereq_link(pid, stem):
        return f"[{pid}](templates/{stem}.ipynb)" if stem else pid

    def problem_links(stem):
        return (f"[Template](templates/{stem}.ipynb) · [Solution](solutions/{stem}_solution.ipynb) · "
                f"[Colab]({COLAB_URL}/templates/{stem}.ipynb)")

    out = [
        "# 🧭 Learning Path",
        "",
        "A recommended order for working through TorchCode. Start with the **prerequisites** "
        "(PyTorch skills), then work through the modules from top to bottom. Within a module, each problem builds on the previous one.",
        "",
        "Problem numbers are unchanged; this is only an ordering. For the reasoning behind it and how "
        "prerequisite notebooks work, see [CURRICULUM.md](CURRICULUM.md).",
        "",
        *([
            "> 🔵 = prerequisite available · ⚪ = prerequisite planned",
            "",
        ] if any(stem is None for _, _, stem, _ in prereqs) else []),
        "## Track 0: Prerequisites",
        "",
        "| # | Prerequisite | Unlocks | Links |",
        "|:---:|---|---|---|",
    ]
    for pid, title, stem, nums in prereqs:
        unl = ", ".join(f"[#{n}](templates/{problems[n][0]}.ipynb)" for n in sorted(nums))
        if stem:
            out.append(f"| 🔵 {pid} | **{title}** | {unl} | {problem_links(stem)} |")
        else:
            out.append(f"| ⚪ {pid} | {title} *(planned)* | {unl} | — |")
    for mtitle, mdesc, nums in curriculum.MODULES:
        out += ["", f"## Module {mtitle}", "", mdesc, "",
                "| Step | # | Problem | Difficulty | Builds on | Links |",
                "|:---:|:---:|---|:---:|---|---|"]
        for step, n in enumerate(nums, 1):
            stem, name, diff = problems[n]
            pre = prereqs_of.get(n, [])
            pre_txt = ", ".join(("🔵 " if s else "⚪ ") + prereq_link(pid, s) for pid, s in pre) or "—"
            out.append(f"| {step} | {n} | **{name}** | {DIFF_ICON[diff]} {diff} | {pre_txt} | {problem_links(stem)} |")
    out.append("")
    return "\n".join(out)


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="only report stale outputs; exit 1 if any")
    args = parser.parse_args()

    problems = load_problems()
    notebooks = load_notebooks()

    outputs: dict[Path, str] = {}
    for nb in notebooks:
        outputs[ROOT / "templates" / f"{nb['file']}.ipynb"] = render_notebook(nb, problems, solution=False)
        outputs[ROOT / "solutions" / f"{nb['file']}_solution.ipynb"] = render_notebook(nb, problems, solution=True)
    outputs[ROOT / "LEARNING_PATH.md"] = render_learning_path(notebooks, problems)

    stale = [p for p, text in outputs.items() if not p.exists() or p.read_text(encoding="utf-8") != text]
    for path in stale:
        rel = path.relative_to(ROOT)
        if args.check:
            print(f"stale: {rel}")
        else:
            path.write_text(outputs[path], encoding="utf-8")
            print(f"wrote: {rel}")
    if not stale:
        print(f"up to date ({len(outputs)} files)")
    return 1 if (args.check and stale) else 0


if __name__ == "__main__":
    sys.exit(main())
