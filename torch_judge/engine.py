"""Core judge engine — grabs user functions from the Jupyter namespace and tests them."""

from __future__ import annotations

import time
import traceback
from typing import Any

from torch_judge.tasks import get_task, task_function_names, TASKS
from torch_judge.progress import mark_solved, mark_attempted

_RESET = "\033[0m"
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_DIM = "\033[90m"
_BOLD = "\033[1m"


def _get_user_namespace() -> dict[str, Any]:
    """Get the calling notebook's global namespace via IPython."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            return ip.user_ns
    except ImportError:
        pass
    # Fallback: try to grab caller's globals
    import inspect
    frame = inspect.currentframe()
    if frame and frame.f_back and frame.f_back.f_back:
        return frame.f_back.f_back.f_globals
    return {}


def check(task_id: str) -> None:
    """Run all tests for a task against the user's implementation.

    Usage in a notebook cell:
        from torch_judge import check
        check("relu")
    """
    task = get_task(task_id)
    if task is None:
        available = ", ".join(f"'{k}'" for k in TASKS)
        print(f"{_RED}Unknown task '{task_id}'. Available: {available}{_RESET}")
        return

    fn_name = task["function_name"]
    fn_names = task_function_names(task)
    user_ns = _get_user_namespace()

    missing = [n for n in fn_names if n not in user_ns]
    if missing:
        names = ", ".join(f"'{n}'" for n in missing)
        print(f"\n{_RED}❌ Function/class {names} not found in your notebook.{_RESET}")
        print(f"{_DIM}   Make sure you defined it in a cell above and ran that cell.{_RESET}\n")
        return

    user_objs = {n: user_ns[n] for n in fn_names}
    tests = task["tests"]
    passed = 0
    total = len(tests)

    print(f"\n{_BOLD}🧪 Testing: {task['title']} ({task['difficulty']}){_RESET}")
    print(f"{'─' * 50}")

    total_time = 0.0

    for i, test in enumerate(tests, 1):
        test_code = test["code"].replace("{fn}", fn_name)
        namespace: dict[str, Any] = dict(user_objs)

        t0 = time.perf_counter()
        try:
            exec(compile(test_code, f"<test:{test['name']}>", "exec"), namespace)  # noqa: S102
            elapsed = time.perf_counter() - t0
            total_time += elapsed
            passed += 1
            print(f"  {_GREEN}✅ [{i}/{total}] {test['name']}{_RESET} {_DIM}({elapsed*1000:.1f}ms){_RESET}")
        except AssertionError as e:
            elapsed = time.perf_counter() - t0
            msg = str(e) or "Assertion failed"
            print(f"  {_RED}❌ [{i}/{total}] {test['name']}{_RESET}")
            print(f"     {_RED}{msg}{_RESET}")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"  {_RED}💥 [{i}/{total}] {test['name']}{_RESET}")
            print(f"     {_RED}{type(e).__name__}: {e}{_RESET}")
            tb = traceback.format_exc()
            short_tb = "\n".join(tb.strip().split("\n")[-3:])
            print(f"     {_DIM}{short_tb}{_RESET}")

    print(f"{'─' * 50}")

    if passed == total:
        print(f"  {_GREEN}{_BOLD}🎉 All {total} tests passed! ({total_time*1000:.1f}ms total){_RESET}")
        mark_solved(task_id, total_time)
        print(f"  {_DIM}Progress saved. Run status() to see your dashboard.{_RESET}\n")
    else:
        print(f"  {_YELLOW}📊 {passed}/{total} tests passed.{_RESET}")
        mark_attempted(task_id)
        print(f"  {_DIM}Keep going! Use hint(\"{task_id}\") if you're stuck.{_RESET}\n")


def hint(task_id: str) -> None:
    """Show a hint for the given task."""
    task = get_task(task_id)
    if task is None:
        print(f"{_RED}Unknown task '{task_id}'.{_RESET}")
        return
    print(f"\n{_YELLOW}💡 Hint for {task['title']}:{_RESET}")
    print(f"   {task['hint']}\n")
