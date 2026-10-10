"""Durable local learner profiles, attempts, and daily practice recommendations.

History contains real web submissions only. Older solved markers are imported as
progress, without inventing timestamps, test results, XP, or a practice streak.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone as dt_timezone
from functools import lru_cache
import json
import os
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from torch_judge.tasks import TASKS


ROOT = Path(__file__).resolve().parent.parent
# Define each course explicitly, then preserve the README's weekly order for
# recommendations, including harder lessons before later easy training lessons.
STUDY_WEEKS = [
    {"week": 1, "focus": "Foundations", "task_ids": [
        "relu", "softmax", "cross_entropy", "dropout", "embedding", "gelu",
        "linear", "linear_regression", "layernorm", "batchnorm", "rmsnorm", "mlp", "conv2d",
    ]},
    {"week": 2, "focus": "Attention Deep Dive", "task_ids": [
        "attention", "mha", "cross_attention", "causal_attention", "gqa",
        "kv_cache", "sliding_window", "rope", "linear_attention", "flash_attention",
    ]},
    {"week": 3, "focus": "Architecture + Training", "task_ids": [
        "gpt2_block", "lora", "moe", "vit_patch", "adam", "cosine_lr",
        "gradient_clipping", "gradient_accumulation", "weight_init",
    ]},
    {"week": 4, "focus": "Inference + Advanced", "task_ids": [
        "topk_sampling", "beam_search", "speculative_decoding", "bpe",
        "int8_quantization", "dpo_loss", "grpo_loss", "ppo_loss", "opd_loss",
    ]},
]
CURRICULUM = [task_id for week in STUDY_WEEKS for task_id in week["task_ids"]]
ATTENTION = {"attention", "mha", "cross_attention", "causal_attention", "gqa",
             "kv_cache", "sliding_window", "rope", "linear_attention", "flash_attention"}
TRAINING = {"adam", "cosine_lr", "gradient_clipping", "gradient_accumulation", "weight_init"}
ARCHITECTURE = {"gpt2_block", "lora", "moe", "vit_patch"}
INFERENCE = {"topk_sampling", "beam_search", "speculative_decoding", "int8_quantization", "bpe"}
INTERVALS = (2, 4, 7, 14, 30)


def study_for(task_id: str) -> dict[str, Any]:
    for week in STUDY_WEEKS:
        if task_id in week["task_ids"]:
            return {"study_week": week["week"], "study_focus": week["focus"]}
    return {"study_week": None, "study_focus": "Extra Practice"}


def utc_now() -> datetime:
    return datetime.now(dt_timezone.utc)


def timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt_timezone.utc)
    return value.astimezone(dt_timezone.utc).isoformat()


def parse_timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    return result if result.tzinfo else result.replace(tzinfo=dt_timezone.utc)


def topic_for(task_id: str) -> str:
    if task_id in ATTENTION:
        return "Attention"
    if task_id in TRAINING:
        return "Training"
    if task_id in ARCHITECTURE:
        return "Architecture"
    if task_id in INFERENCE:
        return "Inference"
    if task_id.endswith("_loss"):
        return "Alignment"
    return "Foundations"


class PracticeStore:
    def __init__(self, db_path: Path | str | None = None,
                 legacy_path: Path | str | None = None):
        self.path = Path(db_path or os.environ.get("PRACTICE_DB_PATH", ROOT / "data/practice.db"))
        self.legacy_path = Path(legacy_path or ROOT / "data/progress.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS profiles (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL REFERENCES profiles(id), task_id TEXT NOT NULL,
                    success INTEGER NOT NULL, passed INTEGER NOT NULL, total INTEGER NOT NULL,
                    hint_used INTEGER NOT NULL, failed_tests TEXT NOT NULL,
                    created_at TEXT NOT NULL, source TEXT NOT NULL DEFAULT 'web',
                    submitted_code TEXT
                );
                CREATE INDEX IF NOT EXISTS attempts_user_date ON attempts(user_id, created_at);
                CREATE INDEX IF NOT EXISTS attempts_user_task_id ON attempts(user_id, task_id, id DESC);
                CREATE TABLE IF NOT EXISTS progress (
                    user_id TEXT NOT NULL REFERENCES profiles(id), task_id TEXT NOT NULL,
                    solved INTEGER NOT NULL DEFAULT 0, needs_review INTEGER NOT NULL DEFAULT 0,
                    clean_passes INTEGER NOT NULL DEFAULT 0, review_due_at TEXT,
                    review_anchor_at TEXT, review_due_days INTEGER, review_reason TEXT,
                    imported_attempts INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (user_id, task_id)
                );
                CREATE TABLE IF NOT EXISTS daily_plans (
                    user_id TEXT NOT NULL REFERENCES profiles(id), day TEXT NOT NULL,
                    timezone TEXT NOT NULL, entries TEXT NOT NULL,
                    PRIMARY KEY (user_id, day, timezone)
                );
            """)
            connection.execute("INSERT OR IGNORE INTO profiles VALUES (?, ?, ?)",
                               ("local", "You", timestamp(utc_now())))
            # Existing attempts have metadata only; NULL means their code was
            # never saved. In particular, an empty submitted string is not NULL.
            attempt_columns = {row["name"] for row in connection.execute("PRAGMA table_info(attempts)")}
            if "submitted_code" not in attempt_columns:
                connection.execute("ALTER TABLE attempts ADD COLUMN submitted_code TEXT")
            # Preserve existing local databases when review scheduling improves.
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(progress)")}
            for column, kind in (("review_anchor_at", "TEXT"), ("review_due_days", "INTEGER"),
                                 ("review_reason", "TEXT")):
                if column not in columns:
                    connection.execute(f"ALTER TABLE progress ADD COLUMN {column} {kind}")
        self._import_notebook_progress()

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def profile(self, user_id: str) -> dict[str, str]:
        with self.connection() as connection:
            row = connection.execute("SELECT id, name FROM profiles WHERE id = ?", (user_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Learner profile not found")
        return dict(row)

    def profiles(self) -> list[dict[str, str]]:
        with self.connection() as connection:
            return [dict(row) for row in connection.execute(
                "SELECT id, name FROM profiles ORDER BY created_at, id")]

    def create_profile(self, name: str) -> dict[str, str]:
        name = name.strip()
        if not name:
            raise HTTPException(status_code=422, detail="Please enter a learner name")
        result = {"id": uuid4().hex, "name": name[:40]}
        with self.connection() as connection:
            connection.execute("INSERT INTO profiles VALUES (?, ?, ?)",
                               (result["id"], result["name"], timestamp(utc_now())))
        return result

    def _import_notebook_progress(self) -> None:
        # This file stores aggregate state, not a reliable dated attempt log.
        if not self.legacy_path.is_file():
            return
        try:
            data = json.loads(self.legacy_path.read_text())
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        with self.connection() as connection:
            for task_id, entry in data.items():
                if task_id not in TASKS or not isinstance(entry, dict):
                    continue
                status = entry.get("status")
                if status not in ("solved", "attempted"):
                    continue
                # First import wins; never overwrite newer web practice results.
                try:
                    count = max(0, int(entry.get("attempts", 0)))
                except (ValueError, TypeError):
                    count = 0
                connection.execute("""
                    INSERT OR IGNORE INTO progress
                    (user_id, task_id, solved, needs_review, review_due_at, imported_attempts)
                    VALUES ('local', ?, ?, ?, ?, ?)
                """, (task_id, int(status == "solved"), int(status == "attempted"),
                      timestamp(utc_now()), count))

    def import_solved(self, user_id: str, task_ids: list[str]) -> dict[str, int]:
        self.profile(user_id)
        imported = 0
        with self.connection() as connection:
            for task_id in dict.fromkeys(task_ids):
                if task_id not in TASKS:
                    continue
                current = connection.execute(
                    "SELECT solved FROM progress WHERE user_id = ? AND task_id = ?",
                    (user_id, task_id)).fetchone()
                # Browser markers describe older state. Actual submissions are
                # authoritative, including an unsuccessful first web attempt.
                if connection.execute("SELECT 1 FROM attempts WHERE user_id = ? AND task_id = ? LIMIT 1",
                                      (user_id, task_id)).fetchone():
                    continue
                if current is None:
                    connection.execute("""
                        INSERT INTO progress (user_id, task_id, solved, review_due_at)
                        VALUES (?, ?, 1, ?)
                    """, (user_id, task_id, timestamp(utc_now())))
                    imported += 1
                elif not current["solved"]:
                    connection.execute("UPDATE progress SET solved = 1 WHERE user_id = ? AND task_id = ?",
                                       (user_id, task_id))
                    imported += 1
        return {"imported": imported, "ignored": len(task_ids) - imported}

    def record_attempt(self, user_id: str, task_id: str, result: dict[str, Any],
                       hint_used: bool = False, at: datetime | None = None,
                       *, code: str | None = None) -> None:
        self.profile(user_id)
        if task_id not in TASKS:
            raise HTTPException(status_code=404, detail="Task not found")
        at = at or utc_now()
        success = bool(result.get("success", False))
        failed = [test["name"] for test in result.get("tests", []) if not test.get("passed")]
        with self.connection() as connection:
            last_success = connection.execute("""
                SELECT COALESCE(MAX(id), 0) AS id FROM attempts
                WHERE user_id = ? AND task_id = ? AND success = 1
            """, (user_id, task_id)).fetchone()["id"]
            prior_failures = connection.execute("""
                SELECT COUNT(*) AS count FROM attempts
                WHERE user_id = ? AND task_id = ? AND id > ? AND success = 0
            """, (user_id, task_id, last_success)).fetchone()["count"]
            connection.execute("""
                INSERT INTO attempts
                (user_id, task_id, success, passed, total, hint_used, failed_tests, created_at, submitted_code)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, task_id, int(success), int(result.get("passed", 0)),
                  int(result.get("total", 0)), int(hint_used), json.dumps(failed), timestamp(at), code))
            connection.execute("INSERT OR IGNORE INTO progress (user_id, task_id) VALUES (?, ?)",
                               (user_id, task_id))
            old = connection.execute("SELECT * FROM progress WHERE user_id = ? AND task_id = ?",
                                     (user_id, task_id)).fetchone()
            struggled = success and prior_failures >= 2
            clean_passes = old["clean_passes"] + 1 if success and not hint_used and not struggled else 0
            delay = INTERVALS[min(clean_passes - 1, len(INTERVALS) - 1)] if clean_passes else (1 if success else 0)
            due = at + timedelta(days=delay)
            reason = "failed" if not success else "hint" if hint_used else "retries" if struggled else "spaced"
            connection.execute("""
                UPDATE progress SET solved = ?, needs_review = ?, clean_passes = ?, review_due_at = ?,
                                    review_anchor_at = ?, review_due_days = ?, review_reason = ?
                WHERE user_id = ? AND task_id = ?
            """, (int(old["solved"] or success), int(not success or hint_used or struggled),
                  clean_passes, timestamp(due), timestamp(at), delay, reason, user_id, task_id))

    def _validate_submission_scope(self, user_id: str, task_id: str) -> None:
        self.profile(user_id)
        if task_id not in TASKS:
            raise HTTPException(status_code=404, detail="Task not found")

    @staticmethod
    def _submission_metadata(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"], "task_id": row["task_id"],
            "success": bool(row["success"]), "passed": row["passed"], "total": row["total"],
            "created_at": row["created_at"], "hint_used": bool(row["hint_used"]),
            "has_code": bool(row["has_code"]),
        }

    def submissions(self, user_id: str, task_id: str, limit: int = 20,
                    before_id: int | None = None) -> dict[str, Any]:
        self._validate_submission_scope(user_id, task_id)
        if not 1 <= limit <= 100:
            raise HTTPException(status_code=422, detail="Choose between 1 and 100 submissions")
        if before_id is not None and before_id <= 0:
            raise HTTPException(status_code=422, detail="Submission cursor must be positive")
        query = """
            SELECT id, task_id, success, passed, total, created_at, hint_used,
                   submitted_code IS NOT NULL AS has_code
            FROM attempts WHERE user_id = ? AND task_id = ?
        """
        parameters: list[Any] = [user_id, task_id]
        if before_id is not None:
            query += " AND id < ?"
            parameters.append(before_id)
        query += " ORDER BY id DESC LIMIT ?"
        parameters.append(limit + 1)
        with self.connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        submissions = [self._submission_metadata(row) for row in rows[:limit]]
        return {
            "submissions": submissions,
            "next_before_id": submissions[-1]["id"] if len(rows) > limit else None,
        }

    def submission(self, submission_id: int, user_id: str, task_id: str) -> dict[str, Any]:
        self._validate_submission_scope(user_id, task_id)
        with self.connection() as connection:
            row = connection.execute("""
                SELECT id, task_id, success, passed, total, created_at, hint_used,
                       submitted_code IS NOT NULL AS has_code, submitted_code
                FROM attempts WHERE id = ? AND user_id = ? AND task_id = ?
            """, (submission_id, user_id, task_id)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Submission not found")
        return {**self._submission_metadata(row), "code": row["submitted_code"]}

    def mark_review(self, user_id: str, task_id: str, needs_review: bool) -> dict[str, bool]:
        self.profile(user_id)
        if task_id not in TASKS:
            raise HTTPException(status_code=404, detail="Task not found")
        with self.connection() as connection:
            connection.execute("INSERT OR IGNORE INTO progress (user_id, task_id) VALUES (?, ?)",
                               (user_id, task_id))
            at = utc_now()
            delay = 0 if needs_review else 2
            due = at + timedelta(days=delay)
            connection.execute("""
                UPDATE progress SET needs_review = ?, review_due_at = ?, review_anchor_at = ?,
                                    review_due_days = ?, review_reason = ?
                WHERE user_id = ? AND task_id = ?
            """, (int(needs_review), timestamp(due), timestamp(at), delay,
                  "manual" if needs_review else "spaced", user_id, task_id))
        return {"ok": True}

    def dashboard(self, user_id: str = "local", count: int = 3,
                  day: str | None = None, timezone: str = "America/Los_Angeles",
                  now: datetime | None = None) -> dict[str, Any]:
        profile = self.profile(user_id)
        if not 1 <= count <= 10:
            raise HTTPException(status_code=422, detail="Choose between 1 and 10 problems")
        try:
            zone = ZoneInfo(timezone)
        except (ZoneInfoNotFoundError, ValueError):
            raise HTTPException(status_code=422, detail="Unknown timezone")
        try:
            today = date.fromisoformat(day) if day else (now or utc_now()).astimezone(zone).date()
        except ValueError:
            raise HTTPException(status_code=422, detail="Use YYYY-MM-DD for the day")
        with self.connection() as connection:
            attempts = [dict(row) for row in connection.execute(
                """SELECT id, user_id, task_id, success, passed, total, hint_used,
                          failed_tests, created_at, source
                   FROM attempts WHERE user_id = ? ORDER BY created_at, id""", (user_id,))]
            states = {row["task_id"]: dict(row) for row in connection.execute(
                "SELECT * FROM progress WHERE user_id = ?", (user_id,)) if row["task_id"] in TASKS}

            by_task: dict[str, list[dict[str, Any]]] = {}
            web_by_task: dict[str, list[dict[str, Any]]] = {}
            latest_today: dict[str, dict[str, Any]] = {}
            day_counts: dict[date, int] = {}
            for attempt in attempts:
                by_task.setdefault(attempt["task_id"], []).append(attempt)
                if attempt["source"] == "web":
                    web_by_task.setdefault(attempt["task_id"], []).append(attempt)
                attempt_day = parse_timestamp(attempt["created_at"]).astimezone(zone).date()
                day_counts[attempt_day] = day_counts.get(attempt_day, 0) + 1
                if attempt_day == today:
                    latest_today[attempt["task_id"]] = attempt
            def review_due_day(task_id: str) -> date | None:
                state = states.get(task_id, {})
                # Calendar days, rather than elapsed UTC hours, keep a next-day
                # review next day across daylight saving transitions.
                if state.get("review_anchor_at") and state.get("review_due_days") is not None:
                    anchor_day = parse_timestamp(state["review_anchor_at"]).astimezone(zone).date()
                    return anchor_day + timedelta(days=state["review_due_days"])
                if state.get("review_due_at"):
                    return parse_timestamp(state["review_due_at"]).astimezone(zone).date()
                return None

            completed = {
                task_id for task_id, attempt in latest_today.items()
                if attempt["success"] and not (
                    states.get(task_id, {}).get("needs_review")
                    and review_due_day(task_id) is not None
                    and review_due_day(task_id) <= today
                )
            }
            order = {task_id: index for index, task_id in enumerate(CURRICULUM)}
            task_order = sorted(TASKS, key=lambda task_id: (order.get(task_id, 999), task_id))

            def card(task_id: str, kind: str, reason: str | None = None) -> dict[str, Any]:
                task = TASKS[task_id]
                entries = by_task.get(task_id, [])
                latest = entries[-1] if entries else None
                state = states.get(task_id, {})
                if reason is None:
                    if latest and not latest["success"]:
                        reason = f"Your last attempt passed {latest['passed']} of {latest['total']} tests. Let's close the gap."
                    elif state.get("review_reason") == "retries":
                        reason = "You got there after a few retries. Try it again to make it stick."
                    elif state.get("review_reason") == "manual":
                        reason = "You marked this lesson for another pass. Let's build confidence."
                    elif state.get("review_reason") == "hint":
                        reason = "You used a hint last time. Try it again without hints."
                    elif state.get("needs_review"):
                        reason = "Build confidence with a fresh attempt."
                    else:
                        reason = "A quick revisit will help this lesson stick."
                return {
                    "id": task_id, "title": task["title"], "difficulty": task.get("difficulty", "Unknown"),
                    "topic": topic_for(task_id), "kind": kind, "reason": reason,
                    "completed": task_id in completed,
                    "estimated_minutes": {"Easy": 10, "Medium": 20, "Hard": 30}.get(task.get("difficulty"), 15),
                    "passed": latest["passed"] if latest else None,
                    "total": latest["total"] if latest else None,
                    "last_attempt_at": latest["created_at"] if latest else None,
                    "needs_review": bool(state.get("needs_review")),
                    **study_for(task_id),
                }

            due_ids = [task_id for task_id in states
                       if review_due_day(task_id) is not None
                       and review_due_day(task_id) <= today
                       and task_id not in completed]
            due_ids.sort(key=lambda task_id: (
                0 if states[task_id]["needs_review"] else 1,
                0 if by_task.get(task_id) and not by_task[task_id][-1]["success"] else 1,
                review_due_day(task_id), order.get(task_id, 999)))
            reviews = [card(task_id, "review") for task_id in due_ids]
            existing = connection.execute("""
                SELECT entries FROM daily_plans WHERE user_id = ? AND day = ? AND timezone = ?
            """, (user_id, today.isoformat(), timezone)).fetchone()
            planned = json.loads(existing["entries"]) if existing else []
            planned = [entry for entry in planned if entry["id"] in TASKS]
            planned_ids = {entry["id"] for entry in planned}
            review_slots = max(1, count // 3)
            chosen_reviews = sum(entry["kind"] == "review" for entry in planned)
            candidates = [card(task_id, "review") for task_id in due_ids
                          if task_id not in planned_ids][:max(0, review_slots - chosen_reviews)]
            candidates += [card(task_id, "new", (
                f"Next in Week {study_for(task_id)['study_week']} · {study_for(task_id)['study_focus']}. "
                "Build momentum with the README study plan." if study_for(task_id)["study_week"] else
                "Keep growing with an extra lesson after the README study plan."))
                           for task_id in task_order
                           if task_id not in planned_ids and not states.get(task_id, {}).get("solved")
                           and task_id not in due_ids and task_id not in completed]
            # Once all lessons are solved, an all-review daily plan still works.
            candidates += [card(task_id, "review") for task_id in due_ids
                           if task_id not in planned_ids and task_id not in {entry["id"] for entry in candidates}]
            if len(planned) < count:
                planned += [{"id": entry["id"], "kind": entry["kind"], "reason": entry["reason"]}
                            for entry in candidates[:count - len(planned)]]
                connection.execute("""
                    INSERT INTO daily_plans (user_id, day, timezone, entries) VALUES (?, ?, ?, ?)
                    ON CONFLICT (user_id, day, timezone) DO UPDATE SET entries = excluded.entries
                """, (user_id, today.isoformat(), timezone, json.dumps(planned)))
            plan = [card(entry["id"], entry["kind"], entry["reason"]) for entry in planned[:count]]

        streak_day = today if day_counts.get(today) else today - timedelta(days=1)
        streak = 0
        while day_counts.get(streak_day):
            streak += 1
            streak_day -= timedelta(days=1)
        monday = today - timedelta(days=today.weekday())
        def history_entry(attempt: dict[str, Any]) -> dict[str, Any]:
            task = TASKS[attempt["task_id"]]
            return {
                "id": attempt["id"], "task_id": attempt["task_id"], "title": task["title"],
                "difficulty": task.get("difficulty", "Unknown"), "success": bool(attempt["success"]),
                "passed": attempt["passed"], "total": attempt["total"],
                "created_at": attempt["created_at"], "hint_used": bool(attempt["hint_used"]),
                "failed_tests": json.loads(attempt["failed_tests"]),
            }

        history = [{**history_entry(attempt), "source": attempt["source"]}
                   for attempt in reversed(attempts[-100:]) if attempt["task_id"] in TASKS]
        # Summarize the complete submission log, independently of history's
        # latest-100 window. Undated imported progress is never a submission.
        latest_web = sorted((entries[-1] for task_id, entries in web_by_task.items()
                             if task_id in TASKS),
                            key=lambda attempt: (attempt["created_at"], attempt["id"]), reverse=True)
        problem_history = [{
            **history_entry(attempt),
            "attempts": len(web_by_task[attempt["task_id"]]),
            "solved": bool(states.get(attempt["task_id"], {}).get("solved")),
            "needs_review": bool(states.get(attempt["task_id"], {}).get("needs_review")),
        } for attempt in latest_web]
        return {
            "profile": profile, "day": today.isoformat(), "timezone": timezone,
            "goal": {"target": count, "completed": sum(entry["completed"] for entry in plan)},
            "stats": {
                "solved": sum(state["solved"] for state in states.values()), "total": len(TASKS),
                "streak": streak, "attempts": len(attempts) + sum(state["imported_attempts"] for state in states.values()),
                "xp": 20 * sum(bool(attempt["success"]) for attempt in attempts),
                "today_completed": len(completed),
            },
            "week": [{"day": (monday + timedelta(days=index)).isoformat(),
                      "label": (monday + timedelta(days=index)).strftime("%a"),
                      "count": day_counts.get(monday + timedelta(days=index), 0),
                      "is_today": monday + timedelta(days=index) == today} for index in range(7)],
            "plan": plan, "reviews": reviews, "history": history, "problem_history": problem_history,
            "progress": [{"id": task_id,
                          **study_for(task_id),
                          "status": "solved" if states.get(task_id, {}).get("solved") else
                                    "attempted" if task_id in states or task_id in by_task else "unseen",
                          "needs_review": bool(states.get(task_id, {}).get("needs_review")),
                          "attempts": len(by_task.get(task_id, [])) + states.get(task_id, {}).get("imported_attempts", 0),
                          "last_attempt_at": by_task[task_id][-1]["created_at"] if task_id in by_task else None}
                         for task_id in task_order],
            "history_limit": 100,
            "curriculum": [{**week,
                            "solved": sum(bool(states.get(task_id, {}).get("solved"))
                                          for task_id in week["task_ids"]),
                            "total": len(week["task_ids"])} for week in STUDY_WEEKS],
        }


@lru_cache(maxsize=1)
def get_store() -> PracticeStore:
    return PracticeStore()


router = APIRouter(prefix="/api/practice", tags=["practice"])


class ProfileRequest(BaseModel):
    name: str = Field(min_length=1, max_length=40)


class ImportRequest(BaseModel):
    user_id: str = "local"
    solved_task_ids: list[str] = Field(default_factory=list, max_length=1000)


class ReviewRequest(BaseModel):
    user_id: str = "local"
    task_id: str
    needs_review: bool = True


@router.get("/profiles")
def list_profiles():
    return {"profiles": get_store().profiles()}


@router.post("/profiles")
def create_profile(request: ProfileRequest):
    return get_store().create_profile(request.name)


@router.get("/dashboard")
def dashboard(user_id: str = "local", count: int = Query(default=3, ge=1, le=10),
              day: str | None = None, timezone: str = "America/Los_Angeles"):
    return get_store().dashboard(user_id=user_id, count=count, day=day, timezone=timezone)


@router.get("/submissions")
def list_submissions(task_id: str, user_id: str = "local",
                     limit: int = Query(default=20, ge=1, le=100),
                     before_id: int | None = Query(default=None, gt=0)):
    return get_store().submissions(user_id, task_id, limit=limit, before_id=before_id)


@router.get("/submissions/{submission_id}")
def get_submission(submission_id: int, task_id: str, user_id: str = "local"):
    return get_store().submission(submission_id, user_id, task_id)


@router.post("/import")
def import_progress(request: ImportRequest):
    return get_store().import_solved(request.user_id, request.solved_task_ids)


@router.post("/review")
def mark_review(request: ReviewRequest):
    return get_store().mark_review(request.user_id, request.task_id, request.needs_review)
