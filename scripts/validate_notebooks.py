#!/usr/bin/env python3
"""Validate and optionally repair notebook cell schemas."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_GLOBS = ("templates/*.ipynb", "solutions/*.ipynb")
CODE_ONLY_FIELDS = ("outputs", "execution_count")
CELL_ID_RE = re.compile(r"^[A-Za-z0-9-_]+$")


def notebook_paths() -> list[Path]:
    paths: list[Path] = []
    for pattern in NOTEBOOK_GLOBS:
        paths.extend(ROOT.glob(pattern))
    return sorted(paths)


def load_notebook(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_notebook(path: Path, notebook: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
        f.write("\n")


def source_text(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(str(line) for line in source)
    return str(source)


def stable_cell_id(path: Path, index: int, cell: dict[str, Any], used: set[str]) -> str:
    seed = (
        f"{path.relative_to(ROOT)}:{index}:"
        f"{cell.get('cell_type', '')}:{source_text(cell)}"
    )
    base = f"cell-{hashlib.sha1(seed.encode('utf-8')).hexdigest()[:12]}"
    cell_id = base
    suffix = 1
    while cell_id in used:
        cell_id = f"{base}-{suffix}"
        suffix += 1
    return cell_id


def sanitize_notebook(path: Path, fix: bool) -> list[str]:
    notebook = load_notebook(path)
    errors: list[str] = []
    changed = False
    used_ids: set[str] = set()
    needs_cell_ids = False
    nbformat_minor = int(notebook.get("nbformat_minor", 0))
    version_error_added = False

    for index, cell in enumerate(notebook.get("cells", [])):
        cell_id = cell.get("id")
        if not isinstance(cell_id, str) or not cell_id:
            errors.append(f"{path.relative_to(ROOT)} cell {index}: missing cell id")
            needs_cell_ids = True
            if fix:
                cell["id"] = stable_cell_id(path, index, cell, used_ids)
                cell_id = cell["id"]
                changed = True
        elif not CELL_ID_RE.match(cell_id):
            errors.append(f"{path.relative_to(ROOT)} cell {index}: invalid cell id")
            needs_cell_ids = True
            if fix:
                cell["id"] = stable_cell_id(path, index, cell, used_ids)
                cell_id = cell["id"]
                changed = True

        if isinstance(cell_id, str):
            if cell_id in used_ids:
                errors.append(f"{path.relative_to(ROOT)} cell {index}: duplicate cell id")
                if fix:
                    cell["id"] = stable_cell_id(path, index, cell, used_ids)
                    cell_id = cell["id"]
                    changed = True
            used_ids.add(cell_id)

        if (
            isinstance(cell_id, str)
            and cell_id
            and nbformat_minor < 5
            and not version_error_added
        ):
            errors.append(
                f"{path.relative_to(ROOT)}: cell ids require nbformat_minor >= 5"
            )
            version_error_added = True
            if fix:
                needs_cell_ids = True
                changed = True

        if cell.get("cell_type") == "code":
            continue

        for field in CODE_ONLY_FIELDS:
            if field in cell:
                errors.append(
                    f"{path.relative_to(ROOT)} cell {index}: "
                    f"non-code cell contains '{field}'"
                )
                if fix:
                    del cell[field]
                    changed = True

    if changed:
        if needs_cell_ids:
            notebook["nbformat"] = 4
            notebook["nbformat_minor"] = max(int(notebook.get("nbformat_minor", 0)), 5)
        write_notebook(path, notebook)

    return errors


def validate_with_nbformat(path: Path) -> str | None:
    try:
        import nbformat
    except ImportError:
        return None

    try:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
    except Exception as exc:  # pragma: no cover - message is for CLI output
        return f"{path.relative_to(ROOT)}: {exc}"

    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fix",
        action="store_true",
        help="repair missing/invalid ids and remove code-only fields from non-code cells",
    )
    args = parser.parse_args()

    schema_errors: list[str] = []
    for path in notebook_paths():
        schema_errors.extend(sanitize_notebook(path, args.fix))

    if schema_errors and not args.fix:
        print("Notebook schema errors:")
        print("\n".join(schema_errors))
        print("\nRun scripts/validate_notebooks.py --fix to repair them.")
        return 1

    nbformat_errors: list[str] = []
    for path in notebook_paths():
        error = validate_with_nbformat(path)
        if error:
            nbformat_errors.append(error)

    if nbformat_errors:
        print("nbformat validation errors:")
        print("\n".join(nbformat_errors))
        return 1

    if schema_errors:
        print(f"Fixed {len(schema_errors)} notebook schema issue(s).")
    else:
        print("All notebooks passed validation.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
