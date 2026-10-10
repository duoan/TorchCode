"""Read reference implementations and the checked-in explanations beside them.

Notebook cells are parsed as data; none of their code is executed here.
"""

import ast
import json
from pathlib import Path
import re

from fastapi import HTTPException

from api.parser import TASK_ID_ALIASES
from torch_judge.tasks import get_task

ROOT = Path(__file__).resolve().parent.parent
SOLUTION_MARKER = "# ✅ SOLUTION"


def solution_files(root: Path = ROOT) -> dict[str, Path]:
    files = {}
    for path in sorted((root / "solutions").glob("*_solution.ipynb")):
        match = re.fullmatch(r"\d+_(.+)_solution\.ipynb", path.name)
        if match:
            stem = match.group(1)
            files[TASK_ID_ALIASES.get(stem, stem)] = path
    return files


def parse_solution_code(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    imports = []
    implementations = []
    found_solution = False
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if SOLUTION_MARKER in source:
            found_solution = True
            implementations.append(source.replace(SOLUTION_MARKER, "", 1).strip())
        elif not found_solution:
            # Only top-level imports before the implementation are dependencies.
            # Colab installation, demos, and judge calls do not belong in the editor.
            if "google.colab" in source or "get_ipython" in source:
                continue
            for statement in ast.parse(source).body:
                if isinstance(statement, (ast.Import, ast.ImportFrom)):
                    segment = ast.get_source_segment(source, statement)
                    if segment and segment not in imports:
                        imports.append(segment)
    if not implementations:
        raise ValueError("Reference notebook has no marked solution cell")
    code = "\n".join(imports) + "\n\n" + "\n\n".join(implementations) + "\n"
    compile(code, str(path), "exec")
    return code.lstrip()


def get_solution(task_id: str, root: Path = ROOT) -> dict:
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Problem not found")
    path = solution_files(root).get(task_id)
    if not path:
        raise HTTPException(status_code=404, detail="Reference solution not available")
    explanation_path = root / "explanations" / (path.stem.removesuffix("_solution") + ".md")
    try:
        code = parse_solution_code(path)
        explanation = explanation_path.read_text(encoding="utf-8").strip()
        if not explanation:
            raise ValueError("Explanation is empty")
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="This explanation is not available yet")
    except (OSError, ValueError, SyntaxError):
        raise HTTPException(status_code=503, detail="We couldn’t load this reference solution")
    return {
        "id": task_id,
        "title": task["title"],
        "difficulty": task.get("difficulty", "Unknown"),
        "code": code,
        "explanation": explanation,
        "solution_file": path.relative_to(root).as_posix(),
        "explanation_file": explanation_path.relative_to(root).as_posix(),
    }
