from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import sys
import os

# Add the root directory to PYTHONPATH so we can import torch_judge
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_judge.tasks import TASKS, get_task
from torch_judge.web_engine import execute_code
from api.parser import get_all_templates
from api.practice import get_store, router as practice_router
from api.solutions import get_solution


class _IgnoreDebuggerProbes(logging.Filter):
    """Hide 404s from editors/browsers probing localhost for a Chrome DevTools endpoint."""

    PROBE_PATHS = {"/", "/json/version", "/json", "/json/list"}

    def filter(self, record: logging.LogRecord) -> bool:
        # uvicorn.access record args: (client_addr, method, path, http_version, status_code)
        args = record.args
        if isinstance(args, tuple) and len(args) == 5:
            path, status = args[2], args[4]
            if status == 404 and path in self.PROBE_PATHS:
                return False
        return True


logging.getLogger("uvicorn.access").addFilter(_IgnoreDebuggerProbes())

app = FastAPI(title="TorchCode UI Backend")
app.include_router(practice_router)

# Allow CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load templates on startup
TEMPLATES = get_all_templates()

class SubmitRequest(BaseModel):
    code: str
    user_id: str = "local"
    hint_used: bool = False

@app.get("/api/tasks")
def list_tasks():
    tasks_list = []
    for task_id, task_data in TASKS.items():
        tasks_list.append({
            "id": task_id,
            "title": task_data["title"],
            "difficulty": task_data.get("difficulty", "Unknown")
        })
    return tasks_list

@app.get("/api/tasks/{task_id}")
def get_task_details(task_id: str):
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
        
    template = TEMPLATES.get(task_id, {})
    return {
        "id": task_id,
        "title": task["title"],
        "difficulty": task.get("difficulty", "Unknown"),
        "hint": task.get("hint", ""),
        "description": template.get("description", "Description not found."),
        "initial_code": template.get("initial_code", "# Write your code here.")
    }

@app.get("/api/tasks/{task_id}/solution")
def get_task_solution(task_id: str):
    return get_solution(task_id)


@app.post("/api/submit/{task_id}")
def submit_code(task_id: str, request: SubmitRequest):
    if not get_task(task_id):
        raise HTTPException(status_code=404, detail="Task not found")
    # Validate profile before executing code so invalid requests have no side effects.
    get_store().profile(request.user_id)
    result = execute_code(task_id, request.code)
    get_store().record_attempt(request.user_id, task_id, result,
                               hint_used=request.hint_used, code=request.code)
    return result
