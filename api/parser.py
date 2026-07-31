import json
import os
import glob
import re

def parse_notebook_template(filepath: str) -> dict:
    """
    Parses a Jupyter Notebook template to extract the problem description and initial code.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            
        description = ""
        initial_code = ""
        
        for cell in nb.get("cells", []):
            if cell["cell_type"] == "markdown":
                # Only take the first markdown block (usually the problem statement)
                if not description:
                    source = cell.get("source", [])
                    # Filter out Colab badge lines
                    filtered_source = [
                        line for line in source 
                        if "![Open In Colab]" not in line
                    ]
                    description = "".join(filtered_source).strip()
                    
            elif cell["cell_type"] == "code":
                source = cell.get("source", [])
                source_str = "".join(source)
                
                # Check for the implementation placeholder
                if "# ✏️ YOUR IMPLEMENTATION HERE" in source_str:
                    initial_code = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nimport math\n\n" + source_str
                    
        return {
            "description": description,
            "initial_code": initial_code
        }
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return {
            "description": "Error loading description.",
            "initial_code": "# Error loading template code."
        }

# Template filenames whose stem doesn't match their torch_judge task module name
TASK_ID_ALIASES = {
    "multihead_attention": "mha",
}


_SOLUTION_MARKER_RE = re.compile(r"#\s*✅?\s*SOLUTION\b", re.IGNORECASE)


def _is_imports_only(source: str) -> bool:
    """True if every non-empty, non-comment line is an import."""
    saw_import = False
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("import ") or stripped.startswith("from "):
            saw_import = True
            continue
        return False
    return saw_import


def _is_solution_boilerplate(cell: dict) -> bool:
    """
    True for cells we omit from the Solution tab: Colab bootstrap, bare
    imports, verify/demo scaffolding, and judge check(...) calls.
    """
    if cell["type"] != "code":
        return False

    source = cell["source"]
    if "google.colab" in source:
        return True
    if "torch_judge" in source or re.search(r"\bcheck\s*\(", source):
        return True
    if _SOLUTION_MARKER_RE.search(source):
        return False

    first_line = source.lstrip().split("\n", 1)[0]
    if re.match(r"^#\s*(Verify|Demo|Run judge|No imports needed)\b", first_line, re.IGNORECASE):
        return True
    if _is_imports_only(source):
        return True

    # Bare demo cells (no SOLUTION marker, no def/class) — notebook scaffolding.
    if not re.search(r"^(def|class)\s", source, re.MULTILINE):
        return True

    return False


def parse_notebook_solution(filepath: str) -> dict:
    """
    Parses a solution Jupyter Notebook into an ordered list of renderable cells.

    Returns {"cells": [{"type": "markdown"|"code", "source": str}, ...]},
    skipping empty cells, Colab-install boilerplate, imports-only cells,
    verify/demo scaffolding, and judge check(...) cells.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        raw_cells = []
        for cell in nb.get("cells", []):
            cell_type = cell.get("cell_type")
            if cell_type not in ("markdown", "code"):
                continue

            source_lines = cell.get("source", [])
            if cell_type == "markdown":
                # Drop the Colab badge line, same as the template parser.
                source_lines = [
                    line for line in source_lines
                    if "![Open In Colab]" not in line
                ]

            source = "".join(source_lines).strip()
            if not source:
                continue

            raw_cells.append({"type": cell_type, "source": source})

        cells = [c for c in raw_cells if not _is_solution_boilerplate(c)]
        return {"cells": cells, "_raw_cells": raw_cells}
    except Exception as e:
        print(f"Error parsing solution {filepath}: {e}")
        return {"cells": [], "_raw_cells": []}


def _extract_check_id(cells):
    """Return the argument of the notebook's check("<id>") call, if present."""
    for cell in cells:
        if cell["type"] != "code":
            continue
        match = re.search(r"check\(\s*[\"']([^\"']+)[\"']", cell["source"])
        if match:
            return match.group(1)
    return None


def get_all_solutions(solutions_dir: str = "../solutions") -> dict:
    """
    Returns a dict mapping task_id -> {"cells": [...]} for every solution notebook.

    The task_id is the argument of the notebook's check("<id>") call (the
    authoritative judge id, matching torch_judge.tasks), which is more reliable
    than the filename slug (e.g. 06_multihead_attention_solution.ipynb -> "mha").
    Falls back to the filename slug if no check(...) call is found.
    """
    solutions = {}

    base_dir = os.path.dirname(os.path.abspath(__file__))
    solutions_path = os.path.join(base_dir, solutions_dir)

    for filepath in glob.glob(os.path.join(solutions_path, "*.ipynb")):
        parsed = parse_notebook_solution(filepath)
        # Resolve task id from the raw notebook (before boilerplate is dropped).
        task_id = _extract_check_id(parsed.get("_raw_cells", parsed["cells"]))

        if not task_id:
            filename = os.path.basename(filepath)
            match = re.match(r"^\d+_(.+)_solution\.ipynb$", filename)
            if not match:
                continue
            task_id = match.group(1)

        solutions[task_id] = {"cells": parsed["cells"]}

    return solutions



def get_all_templates(templates_dir: str = "../templates") -> dict:
    """
    Returns a dictionary mapping task_ids to their extracted template data.
    task_id is inferred from the filename (e.g., '01_relu.ipynb' -> 'relu').
    """
    templates = {}
    
    # Resolve absolute path based on current directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    templates_path = os.path.join(base_dir, templates_dir)
    
    for filepath in glob.glob(os.path.join(templates_path, "*.ipynb")):
        filename = os.path.basename(filepath)
        if filename == "00_welcome.ipynb":
            continue
            
        # Extract task_id (e.g. 01_relu.ipynb -> relu)
        match = re.match(r"^\d+_(.+)\.ipynb$", filename)
        if match:
            task_id = match.group(1)
            task_id = TASK_ID_ALIASES.get(task_id, task_id)
            templates[task_id] = parse_notebook_template(filepath)
            
    return templates

if __name__ == "__main__":
    # Test parser
    res = get_all_templates()
    if "relu" in res:
        print("Successfully parsed relu:")
        print("Description:", res["relu"]["description"][:100], "...")
        print("Code:", res["relu"]["initial_code"][:50], "...")
