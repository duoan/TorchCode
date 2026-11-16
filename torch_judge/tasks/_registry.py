"""Auto-discovery registry for task definitions."""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Any

DIFFICULTY_ORDER = {"Prereq": -1, "Easy": 0, "Medium": 1, "Hard": 2, "Bonus": 3}

TASKS: dict[str, dict[str, Any]] = {}

_pkg_dir = str(Path(__file__).parent)
for _info in pkgutil.iter_modules([_pkg_dir]):
    if _info.name.startswith("_"):
        continue
    _mod = importlib.import_module(f"{__package__}.{_info.name}")
    if hasattr(_mod, "TASK"):
        TASKS[_info.name] = _mod.TASK


def get_task(task_id: str) -> dict[str, Any] | None:
    return TASKS.get(task_id)


def task_function_names(task: dict[str, Any]) -> list[str]:
    """Names the user must define for a task.

    Most tasks define a single ``function_name``. Prerequisite tasks bundle
    several small exercises and list them all in ``function_names``.
    """
    return list(task.get("function_names") or [task["function_name"]])


def list_tasks() -> list[tuple[str, dict[str, Any]]]:
    return sorted(
        TASKS.items(),
        key=lambda t: (DIFFICULTY_ORDER.get(t[1]["difficulty"], 9), t[1].get("order", 0)),
    )
