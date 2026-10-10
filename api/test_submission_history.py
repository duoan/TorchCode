"""Run with: .venv/bin/python -m unittest api.test_submission_history"""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException

from api import main
from api.practice import PracticeStore


SUCCESS = {"success": True, "passed": 4, "total": 4,
           "tests": [{"name": "shape", "passed": True}]}
FAILURE = {"success": False, "passed": 2, "total": 4,
           "tests": [{"name": "gradient", "passed": False}]}
NOW = datetime(2026, 10, 10, 18, 0, tzinfo=timezone.utc)
METADATA_KEYS = {"id", "task_id", "success", "passed", "total", "created_at",
                 "hint_used", "has_code"}


class SubmissionHistoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name)
        self.db_path = self.path / "practice.db"
        self.legacy_path = self.path / "progress.json"
        self.store = PracticeStore(self.db_path, self.legacy_path)

    def tearDown(self):
        self.directory.cleanup()

    def record(self, task_id="relu", user_id="local", result=None, **kwargs):
        self.store.record_attempt(user_id, task_id, result or SUCCESS, **kwargs)
        with self.store.connection() as connection:
            return connection.execute("SELECT MAX(id) FROM attempts").fetchone()[0]

    def assert_http_error(self, status, operation):
        with self.assertRaises(HTTPException) as caught:
            operation()
        self.assertEqual(caught.exception.status_code, status)

    def test_progress_markers_do_not_fabricate_submission_history(self):
        self.assertEqual(self.store.submissions("local", "relu"),
                         {"submissions": [], "next_before_id": None})
        self.store.import_solved("local", ["relu"])
        self.store.mark_review("local", "embedding", True)
        self.legacy_path.write_text(json.dumps({
            "softmax": {"status": "solved", "attempts": 7},
        }))
        reopened = PracticeStore(self.db_path, self.legacy_path)
        for task_id in ("relu", "embedding", "softmax"):
            with self.subTest(task_id=task_id):
                self.assertEqual(reopened.submissions("local", task_id),
                                 {"submissions": [], "next_before_id": None})
        self.assert_http_error(404, lambda: reopened.submission(1, "local", "relu"))

    def test_list_has_only_complete_metadata_in_descending_id_order(self):
        first = self.record(hint_used=True, at=NOW, code="first source")
        second = self.record(result=FAILURE, at=NOW - timedelta(days=1))
        self.record(task_id="softmax", code="different task")
        other = self.store.create_profile("Ada")
        self.record(user_id=other["id"], code="different learner")

        page = self.store.submissions("local", "relu")
        self.assertEqual(set(page), {"submissions", "next_before_id"})
        self.assertIsNone(page["next_before_id"])
        self.assertEqual(page["submissions"], [
            {"id": second, "task_id": "relu", "success": False, "passed": 2,
             "total": 4, "created_at": (NOW - timedelta(days=1)).isoformat(),
             "hint_used": False, "has_code": False},
            {"id": first, "task_id": "relu", "success": True, "passed": 4,
             "total": 4, "created_at": NOW.isoformat(),
             "hint_used": True, "has_code": True},
        ])
        for row in page["submissions"]:
            self.assertEqual(set(row), METADATA_KEYS)
            for field in ("success", "hint_used", "has_code"):
                self.assertIsInstance(row[field], bool)

    def test_source_text_empty_string_and_missing_code_survive_reopening(self):
        original = "  # π and indentation\r\ndef relu(x):\r\n\treturn x.clamp(min=0)\r\n\r\n"
        cases = [(self.record(at=NOW), None),
                 (self.record(at=NOW, code=None), None),
                 (self.record(at=NOW, code=""), ""),
                 (self.record(at=NOW, code=original), original)]
        reopened = PracticeStore(self.db_path, self.legacy_path)
        for submission_id, code in cases:
            with self.subTest(submission_id=submission_id):
                detail = reopened.submission(submission_id, "local", "relu")
                self.assertEqual(set(detail), METADATA_KEYS | {"code"})
                self.assertEqual(detail["code"], code)
                self.assertEqual(detail["has_code"], code is not None)
                listed = next(row for row in reopened.submissions("local", "relu")["submissions"]
                              if row["id"] == submission_id)
                self.assertEqual({key: detail[key] for key in METADATA_KEYS}, listed)

    def test_default_page_size_and_exclusive_cursor(self):
        ids = [self.record(code="source " + str(index)) for index in range(21)]
        first = self.store.submissions("local", "relu")
        self.assertEqual([row["id"] for row in first["submissions"]], list(reversed(ids[1:])))
        self.assertEqual(first["next_before_id"], ids[1])
        second = self.store.submissions("local", "relu", before_id=first["next_before_id"])
        self.assertEqual([row["id"] for row in second["submissions"]], ids[:1])
        self.assertIsNone(second["next_before_id"])
        self.assertNotIn("source ", json.dumps(first))

    def test_paginated_history_counts_only_the_requested_learner_and_task(self):
        other = self.store.create_profile("Grace")
        ids = []
        for index in range(3):
            ids.append(self.record(code=str(index)))
            self.record(task_id="softmax")
            self.record(user_id=other["id"])
        first = self.store.submissions("local", "relu", limit=2)
        self.assertEqual([row["id"] for row in first["submissions"]], list(reversed(ids[1:])))
        self.assertEqual(first["next_before_id"], ids[1])
        second = self.store.submissions("local", "relu", limit=2,
                                        before_id=first["next_before_id"])
        self.assertEqual([row["id"] for row in second["submissions"]], ids[:1])
        self.assertIsNone(second["next_before_id"])
        empty = self.store.submissions("local", "relu", limit=2, before_id=ids[0])
        self.assertEqual(empty, {"submissions": [], "next_before_id": None})

    def test_single_item_page_has_cursor_only_when_more_rows_exist(self):
        ids = [self.record(), self.record()]
        first = self.store.submissions("local", "relu", limit=1)
        self.assertEqual([row["id"] for row in first["submissions"]], ids[1:])
        self.assertEqual(first["next_before_id"], ids[1])
        last = self.store.submissions("local", "relu", limit=1, before_id=ids[1])
        self.assertEqual([row["id"] for row in last["submissions"]], ids[:1])
        self.assertIsNone(last["next_before_id"])

    def test_page_limit_accepts_100_and_returns_all_available_rows(self):
        ids = [self.record() for _ in range(101)]
        page = self.store.submissions("local", "relu", limit=100)
        self.assertEqual([row["id"] for row in page["submissions"]], list(reversed(ids[1:])))
        self.assertEqual(page["next_before_id"], ids[1])

    def test_invalid_page_limits_and_cursors_are_rejected(self):
        for limit in (0, -1, 101):
            with self.subTest(limit=limit):
                self.assert_http_error(422, lambda: self.store.submissions("local", "relu", limit=limit))
        for before_id in (0, -1):
            with self.subTest(before_id=before_id):
                self.assert_http_error(422, lambda: self.store.submissions("local", "relu", before_id=before_id))
        self.assertEqual(self.store.submissions("local", "relu", before_id=1),
                         {"submissions": [], "next_before_id": None})

    def test_list_and_detail_validate_known_learners_and_tasks(self):
        submission_id = self.record()
        for user_id, task_id in (("missing", "relu"), ("local", "missing")):
            with self.subTest(user_id=user_id, task_id=task_id):
                self.assert_http_error(404, lambda: self.store.submissions(user_id, task_id))
                self.assert_http_error(404, lambda: self.store.submission(submission_id, user_id, task_id))

    def test_detail_enforces_both_learner_and_task_scopes(self):
        other = self.store.create_profile("Ada")
        local_id = self.record(code="private local source")
        other_id = self.record(user_id=other["id"], code="private Ada source")
        softmax_id = self.record(task_id="softmax", code="private softmax source")
        for submission_id, user_id, task_id in (
            (other_id, "local", "relu"),
            (local_id, other["id"], "relu"),
            (softmax_id, "local", "relu"),
            (local_id, "local", "softmax"),
            (softmax_id + 1, "local", "relu"),
        ):
            with self.subTest(submission_id=submission_id, user_id=user_id, task_id=task_id):
                self.assert_http_error(404, lambda: self.store.submission(submission_id, user_id, task_id))
        self.assertEqual(self.store.submission(other_id, other["id"], "relu")["code"],
                         "private Ada source")

    def test_dashboard_never_exposes_submitted_source(self):
        source = "# a unique secret source marker\nreturn torch.maximum(x, torch.zeros_like(x))\n"
        self.record(code=source, at=NOW)
        dashboard = self.store.dashboard(day="2026-10-10")
        self.assertEqual(len(dashboard["history"]), 1)
        serialized = json.dumps(dashboard)
        self.assertNotIn("unique secret source marker", serialized)
        self.assertNotIn('"submitted_code"', serialized)
        self.assertNotIn('"code"', serialized)

    def test_migration_preserves_old_rows_and_adds_nullable_source(self):
        old_path = self.path / "old-practice.db"
        with sqlite3.connect(old_path) as connection:
            connection.executescript("""
                CREATE TABLE profiles (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL REFERENCES profiles(id), task_id TEXT NOT NULL,
                    success INTEGER NOT NULL, passed INTEGER NOT NULL, total INTEGER NOT NULL,
                    hint_used INTEGER NOT NULL, failed_tests TEXT NOT NULL,
                    created_at TEXT NOT NULL, source TEXT NOT NULL DEFAULT 'web'
                );
            """)
            connection.execute("INSERT INTO profiles VALUES (?, ?, ?)",
                               ("local", "Existing learner", NOW.isoformat()))
            old_values = (41, "local", "relu", 0, 2, 4, 1, '["gradient"]', NOW.isoformat(), "web")
            connection.execute("INSERT INTO attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", old_values)

        migrated = PracticeStore(old_path, self.legacy_path)
        reopened = PracticeStore(old_path, self.legacy_path)
        self.assertEqual(reopened.profile("local")["name"], "Existing learner")
        with reopened.connection() as connection:
            columns = {row["name"]: dict(row) for row in connection.execute("PRAGMA table_info(attempts)")}
            self.assertEqual(columns["submitted_code"]["type"].upper(), "TEXT")
            self.assertEqual(columns["submitted_code"]["notnull"], 0)
            row = connection.execute("SELECT * FROM attempts WHERE id = 41").fetchone()
            self.assertEqual(tuple(row)[:-1], old_values)
            self.assertIsNone(row["submitted_code"])
        detail = migrated.submission(41, "local", "relu")
        self.assertIsNone(detail["code"])
        self.assertFalse(detail["has_code"])
        self.assertEqual(detail["passed"], 2)
        self.assertTrue(detail["hint_used"])
        old_list = reopened.submissions("local", "relu")["submissions"]
        self.assertEqual(len(old_list), 1)
        self.assertEqual(old_list[0]["id"], 41)
        self.assertFalse(old_list[0]["has_code"])
        reopened.record_attempt("local", "relu", SUCCESS, at=NOW, code="new source")
        latest = reopened.submissions("local", "relu")["submissions"][0]
        self.assertGreater(latest["id"], 41)
        self.assertEqual(reopened.submission(latest["id"], "local", "relu")["code"], "new source")
        first_page = reopened.submissions("local", "relu", limit=1)
        self.assertEqual(first_page["next_before_id"], latest["id"])
        legacy_page = reopened.submissions("local", "relu", limit=1,
                                           before_id=first_page["next_before_id"])
        self.assertEqual(legacy_page, {"submissions": old_list, "next_before_id": None})

    def test_submit_handler_records_the_exact_code_and_result(self):
        learner = self.store.create_profile("Lin")
        for source, result in (("", FAILURE), ("\n  # π\r\nreturn x\n", SUCCESS)):
            with self.subTest(source=source):
                store_spy = Mock(wraps=self.store)
                request = main.SubmitRequest(code=source, user_id=learner["id"], hint_used=True)
                with patch.object(main, "get_store", return_value=store_spy), \
                        patch.object(main, "execute_code", return_value=result) as execute:
                    response = main.submit_code("relu", request)
                self.assertIs(response, result)
                execute.assert_called_once_with("relu", source)
                store_spy.profile.assert_called_once_with(learner["id"])
                store_spy.record_attempt.assert_called_once_with(
                    learner["id"], "relu", result, hint_used=True, code=source)
                row = self.store.submissions(learner["id"], "relu")["submissions"][0]
                self.assertEqual(self.store.submission(row["id"], learner["id"], "relu")["code"], source)

    def test_submit_handler_rejects_invalid_profile_before_execution(self):
        store_spy = Mock(wraps=self.store)
        request = main.SubmitRequest(code="unexecuted source", user_id="missing")
        with patch.object(main, "get_store", return_value=store_spy), \
                patch.object(main, "execute_code") as execute:
            self.assert_http_error(404, lambda: main.submit_code("relu", request))
        execute.assert_not_called()
        store_spy.record_attempt.assert_not_called()
        self.assertEqual(self.store.submissions("local", "relu")["submissions"], [])


if __name__ == "__main__":
    unittest.main()
