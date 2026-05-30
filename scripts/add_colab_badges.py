#!/usr/bin/env python3
"""Add 'Open in Colab' links to all template and solution notebooks."""

import json
from pathlib import Path

REPO = "duoan/TorchCode"
BRANCH = "master"
ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = ROOT / "templates"
SOLUTIONS_DIR = ROOT / "solutions"


def colab_url(filename: str, folder: str) -> str:
    return (
        f"https://colab.research.google.com/github/{REPO}"
        f"/blob/{BRANCH}/{folder}/{filename}"
    )


def colab_markdown(filename: str, folder: str) -> str:
    return f"[Open in Colab]({colab_url(filename, folder)})"


def process_notebook(path: Path, folder: str) -> bool:
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    if not cells or cells[0].get("cell_type") != "markdown":
        return False

    source_lines = cells[0]["source"]
    flat = "".join(source_lines) if isinstance(source_lines, list) else source_lines
    if "colab.research.google.com/github/" in flat:
        return False

    link = colab_markdown(path.name, folder)
    cells[0]["source"] = [link + "\n\n"] + (
        source_lines if isinstance(source_lines, list) else [source_lines]
    )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")

    return True


def main() -> None:
    updated = 0
    for nb_path in sorted(TEMPLATES_DIR.glob("*.ipynb")):
        if process_notebook(nb_path, "templates"):
            print(f"  ✅ templates/{nb_path.name}")
            updated += 1
        else:
            print(f"  ⏭️  templates/{nb_path.name} (already has badge or skipped)")
    for nb_path in sorted(SOLUTIONS_DIR.glob("*.ipynb")):
        if process_notebook(nb_path, "solutions"):
            print(f"  ✅ solutions/{nb_path.name}")
            updated += 1
        else:
            print(f"  ⏭️  solutions/{nb_path.name} (already has badge or skipped)")
    print(f"\nDone — updated {updated} notebooks.")


if __name__ == "__main__":
    main()
