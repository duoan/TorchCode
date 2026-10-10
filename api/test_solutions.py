"""Coverage for the notebook-to-reference reader and explanation inventory."""

import ast
import json
from pathlib import Path
import tempfile
import unittest

from fastapi import HTTPException

from api.solutions import ROOT, get_solution, parse_solution_code, solution_files
from torch_judge.tasks import TASKS


class SolutionTests(unittest.TestCase):
    def test_every_problem_has_its_own_solution_and_explanation(self):
        self.assertEqual(set(solution_files()), set(TASKS))
        for task_id, task in TASKS.items():
            with self.subTest(task_id=task_id):
                result = get_solution(task_id)
                self.assertEqual(result["id"], task_id)
                self.assertEqual(result["title"], task["title"])
                code = result["code"]
                tree = ast.parse(code)
                definitions = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
                self.assertIn(task["function_name"], definitions)
                self.assertNotIn("google.colab", code)
                self.assertNotIn("get_ipython", code)
                self.assertNotIn("from torch_judge import check", code)
                self.assertNotIn("# Demo", code)
                explanation = result["explanation"]
                self.assertTrue(explanation.startswith("# "))
                self.assertGreater(len(explanation.split()), 100)
                self.assertTrue((ROOT / result["explanation_file"]).is_file())

    def test_mha_uses_the_multihead_notebook_alias(self):
        self.assertEqual(get_solution("mha")["solution_file"], "solutions/06_multihead_attention_solution.ipynb")

    def test_parser_keeps_imports_and_multiple_solution_cells_only(self):
        cells = [
            "try:\n import google.colab\n get_ipython()\nexcept ImportError:\n pass",
            "import torch\nfrom math import (\n sqrt,\n pi,\n)\nprint('setup')",
            "# ✅ SOLUTION\ndef first(x):\n return x + 1",
            "print('demo')",
            "# ✅ SOLUTION\ndef second(x):\n return first(x)",
            "from torch_judge import check\ncheck('relu')",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "solution.ipynb"
            path.write_text(json.dumps({"cells": [{"cell_type": "code", "source": source} for source in cells]}))
            code = parse_solution_code(path)
            self.assertIn("from math import (", code)
            self.assertIn("def first", code)
            self.assertIn("def second", code)
            self.assertNotIn("print(", code)
            self.assertNotIn("torch_judge", code)

    def test_unknown_task_and_path_traversal_do_not_read_files(self):
        for task_id in ("missing", "../../README.md"):
            with self.subTest(task_id=task_id), self.assertRaises(HTTPException) as caught:
                get_solution(task_id)
            self.assertEqual(caught.exception.status_code, 404)

    def test_missing_explanation_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "solutions").mkdir()
            source = solution_files()["relu"].read_text()
            (root / "solutions/01_relu_solution.ipynb").write_text(source)
            with self.assertRaises(HTTPException) as caught:
                get_solution("relu", root)
            self.assertEqual(caught.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
