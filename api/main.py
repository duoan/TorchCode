from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
from pathlib import Path

# Add the root directory to PYTHONPATH so we can import torch_judge
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_judge.tasks import TASKS, get_task
from torch_judge.web_engine import execute_code
from api.parser import get_all_solutions, get_all_templates

app = FastAPI(title="TorchCode UI Backend")

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
SOLUTIONS = get_all_solutions()
ROOT_DIR = Path(__file__).resolve().parent.parent
USER_CODE_DIR = ROOT_DIR / "user_solutions"

class SubmitRequest(BaseModel):
    code: str

class CodeRequest(BaseModel):
    code: str

def get_user_code_path(task_id: str) -> Path:
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")
    return USER_CODE_DIR / f"{task_id}.py"

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
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    solution = SOLUTIONS.get(task_id)
    if not solution:
        raise HTTPException(status_code=404, detail="Solution not found")

    return solution

@app.get("/api/code/{task_id}")
def get_saved_code(task_id: str):
    code_path = get_user_code_path(task_id)
    if not code_path.exists():
        return {"code": None, "saved": False}
    return {"code": code_path.read_text(encoding="utf-8"), "saved": True}

@app.put("/api/code/{task_id}")
def save_code(task_id: str, request: CodeRequest):
    code_path = get_user_code_path(task_id)
    USER_CODE_DIR.mkdir(parents=True, exist_ok=True)
    code_path.write_text(request.code, encoding="utf-8")
    return {"saved": True, "path": str(code_path.relative_to(ROOT_DIR))}

@app.delete("/api/code/{task_id}")
def delete_saved_code(task_id: str):
    code_path = get_user_code_path(task_id)
    if code_path.exists():
        code_path.unlink()
    return {"deleted": True}

@app.post("/api/submit/{task_id}")
def submit_code(task_id: str, request: SubmitRequest):
    result = execute_code(task_id, request.code)
    return result
