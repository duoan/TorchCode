#!/usr/bin/env python3
"""Check every prerequisite notebook against its judge task (needs PyTorch).

For each notebook in scripts/prereqs/content/:
  - the reference solutions must pass every test
  - the blank template stubs must pass no tests (so no test passes for free)

Runs through torch_judge.web_engine, so no Jupyter kernel is needed.

Usage:
    python scripts/prereqs/verify.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from build import load_notebooks  # noqa: E402
from torch_judge.web_engine import execute_code  # noqa: E402

PRELUDE = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nimport math\n\n"


def run(task_id: str, code: str) -> dict:
    result = execute_code(task_id, code)
    if result.get("error"):
        print(f"    error: {result['error']}")
        if result.get("traceback"):
            print("    " + result["traceback"].replace("\n", "\n    "))
    return result


def main() -> int:
    ok = True
    for nb in load_notebooks():
        task_id = nb["task_id"]
        print(f"{task_id}")

        prelude = PRELUDE + nb.get("imports", "") + "\n\n"   # the notebook's own setup cell
        solution = prelude + "\n\n".join(ex["solution"].strip("\n") for ex in nb["exercises"])
        r = run(task_id, solution)
        good = r["total"] > 0 and r["passed"] == r["total"]
        print(f"  {'✅' if good else '❌'} solutions pass {r['passed']}/{r['total']}")
        for t in r["tests"]:
            if not t["passed"]:
                print(f"    failed: {t['name']}: {t['error_msg']}")
        ok &= good

        stubs = prelude + "\n\n".join(ex["stub"].strip("\n") for ex in nb["exercises"])
        r = execute_code(task_id, stubs)
        good = r["passed"] == 0
        print(f"  {'✅' if good else '❌'} blank stubs pass {r['passed']}/{r['total']}")
        for t in r["tests"]:
            if t["passed"]:
                print(f"    passes on an empty stub: {t['name']}")
        ok &= good
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
