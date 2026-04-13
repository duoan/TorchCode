import io
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, List, Optional

from torch_judge.tasks import get_task, TASKS

def execute_code(task_id: str, user_code: str) -> Dict[str, Any]:
    """
    Executes user code in a clean namespace and runs the tests for a specific task.
    Returns a structured dictionary with the results.
    """
    task = get_task(task_id)
    if task is None:
        return {
            "success": False,
            "error": f"Unknown task '{task_id}'.",
            "passed": 0,
            "total": 0,
            "tests": []
        }

    fn_name = task["function_name"]
    tests = task["tests"]
    total = len(tests)
    passed = 0
    total_time = 0.0

    namespace: Dict[str, Any] = {}
    
    # 1. Execute the user's code to populate the namespace
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
        try:
            exec(compile(user_code, "<user_code>", "exec"), namespace)
        except Exception as e:
            tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
            clean_tb = []
            for line in tb_lines:
                if 'torch_judge/web_engine.py' in line or 'exec(compile(' in line:
                    continue
                clean_tb.append(line)
            
            return {
                "success": False,
                "error": "Syntax or execution error in your code.",
                "traceback": "".join(clean_tb).strip(),
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "passed": 0,
                "total": total,
                "tests": []
            }

    if fn_name not in namespace:
        return {
            "success": False,
            "error": f"Function or class '{fn_name}' not found. Did you define it?",
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
            "passed": 0,
            "total": total,
            "tests": []
        }

    user_fn = namespace[fn_name]
    test_results: List[Dict[str, Any]] = []

    # 2. Run each test against the user's function
    for i, test in enumerate(tests, 1):
        test_code = test["code"].replace("{fn}", fn_name)
        test_namespace: Dict[str, Any] = {fn_name: user_fn}
        
        test_stdout = io.StringIO()
        test_stderr = io.StringIO()
        
        t0 = time.perf_counter()
        test_passed = False
        error_msg = None
        error_traceback = None
        
        with redirect_stdout(test_stdout), redirect_stderr(test_stderr):
            try:
                exec(compile(test_code, f"<test:{test['name']}>", "exec"), test_namespace)
                test_passed = True
                passed += 1
            except AssertionError as e:
                error_msg = str(e) or "Assertion failed"
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                
                # Format traceback and filter out internal engine frames
                tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
                clean_tb = []
                for line in tb_lines:
                    # Skip frames from our own engine files
                    if 'torch_judge/web_engine.py' in line or 'torch_judge/engine.py' in line:
                        continue
                    # Skip the generic 'exec(compile(...))' line
                    if 'exec(compile(' in line:
                        continue
                    # Simplify the "<test:...>" framing
                    if 'in <module>' in line and '<test:' in line:
                        continue
                    clean_tb.append(line)
                
                # If the traceback only contains the "Traceback (most recent...)" header and the exception string, 
                # just clear the traceback completely, as it provides no additional code context.
                if len(clean_tb) <= 2:
                    error_traceback = ""
                else:
                    error_traceback = "".join(clean_tb).strip()
                
        elapsed = time.perf_counter() - t0
        total_time += elapsed
        
        test_results.append({
            "name": test["name"],
            "code": test["code"].strip(),
            "passed": test_passed,
            "time_ms": elapsed * 1000,
            "error_msg": error_msg,
            "error_traceback": error_traceback,
            "stdout": test_stdout.getvalue(),
            "stderr": test_stderr.getvalue()
        })

    return {
        "success": passed == total,
        "passed": passed,
        "total": total,
        "total_time_ms": total_time * 1000,
        "tests": test_results,
        "stdout": stdout_capture.getvalue(),
        "stderr": stderr_capture.getvalue()
    }
