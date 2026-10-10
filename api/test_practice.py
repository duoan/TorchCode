"""Run with: .venv/bin/python -m unittest api.test_practice"""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest

from fastapi import HTTPException

from api.practice import CURRICULUM, PracticeStore, STUDY_WEEKS
from torch_judge.tasks import TASKS


SUCCESS = {"success": True, "passed": 4, "total": 4,
           "tests": [{"name": "shape", "passed": True}]}
FAILURE = {"success": False, "passed": 2, "total": 4,
           "tests": [{"name": "shape", "passed": True}, {"name": "gradient", "passed": False}]}
NOW = datetime(2026, 10, 10, 18, 0, tzinfo=timezone.utc)


class PracticeStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name)
        self.store = PracticeStore(self.path / "practice.db", self.path / "progress.json")

    def tearDown(self):
        self.directory.cleanup()

    def dashboard(self, **kwargs):
        return self.store.dashboard(day="2026-10-10", **kwargs)

    def test_empty_profile_has_real_curriculum_and_no_invented_history(self):
        dashboard = self.dashboard()
        self.assertEqual([entry["id"] for entry in dashboard["plan"]], ["relu", "softmax", "cross_entropy"])
        self.assertEqual(dashboard["goal"], {"target": 3, "completed": 0})
        self.assertEqual(dashboard["history"], [])
        self.assertEqual(dashboard["problem_history"], [])
        self.assertEqual(dashboard["stats"]["xp"], 0)
        self.assertEqual(dashboard["stats"]["streak"], 0)
        self.assertEqual(len(dashboard["week"]), 7)

    def test_failed_tests_are_durable_and_review_is_prioritized(self):
        self.store.record_attempt("local", "dropout", FAILURE, at=NOW)
        reopened = PracticeStore(self.path / "practice.db", self.path / "progress.json")
        dashboard = reopened.dashboard(day="2026-10-10")
        self.assertEqual(dashboard["history"][0]["failed_tests"], ["gradient"])
        self.assertEqual(dashboard["history"][0]["passed"], 2)
        self.assertEqual(dashboard["plan"][0]["id"], "dropout")
        self.assertEqual(dashboard["plan"][0]["kind"], "review")
        self.assertTrue(dashboard["reviews"][0]["needs_review"])
        self.assertEqual(dashboard["stats"]["attempts"], 1)

    def test_daily_plan_is_stable_and_can_expand_then_shrink(self):
        original = self.dashboard()["plan"]
        self.store.record_attempt("local", "relu", SUCCESS, at=NOW)
        repeated = self.dashboard()["plan"]
        self.assertEqual([entry["id"] for entry in repeated], [entry["id"] for entry in original])
        self.assertTrue(repeated[0]["completed"])
        expanded = self.dashboard(count=7)["plan"]
        self.assertEqual(len(expanded), 7)
        self.assertEqual(len({entry["id"] for entry in expanded}), 7)
        self.assertEqual([entry["id"] for entry in expanded[:3]], [entry["id"] for entry in original])
        self.assertEqual(len(self.dashboard(count=1)["plan"]), 1)
        self.assertEqual(self.dashboard()["goal"]["completed"], 1)

    def test_new_lessons_follow_readme_week_order_including_difficulty_changes(self):
        self.store.import_solved("local", CURRICULUM[:21])
        # Ignore undated imported markers in the due list by using a date before
        # their import time; this test checks new-lesson curriculum selection.
        plan = self.store.dashboard(day="2000-01-01", count=10)["plan"]
        self.assertEqual([entry["id"] for entry in plan], [
            "linear_attention", "flash_attention", "gpt2_block", "lora", "moe",
            "vit_patch", "adam", "cosine_lr", "gradient_clipping", "gradient_accumulation",
        ])
        self.assertEqual(plan[1]["id"], "flash_attention")
        self.assertEqual(plan[2]["id"], "gpt2_block")
        self.assertEqual(plan[3]["id"], "lora")
        self.assertEqual(plan[2]["study_week"], 3)
        self.assertIn("Week 3", plan[2]["reason"])
        self.assertEqual(STUDY_WEEKS[3]["task_ids"], CURRICULUM[32:41])

    def test_curriculum_has_all_41_real_tasks_once_across_four_weeks(self):
        flattened = [task_id for week in STUDY_WEEKS for task_id in week["task_ids"]]
        self.assertEqual(CURRICULUM, flattened)
        self.assertEqual(len(CURRICULUM), 41)
        self.assertEqual(len(CURRICULUM), len(set(CURRICULUM)))
        self.assertEqual(set(CURRICULUM), set(TASKS))
        self.assertEqual([week["week"] for week in STUDY_WEEKS], [1, 2, 3, 4])
        self.assertEqual([week["total"] for week in self.dashboard()["curriculum"]], [13, 10, 9, 9])
        self.assertTrue(all(entry["study_week"] in (1, 2, 3, 4)
                            for entry in self.dashboard()["progress"]))

    def test_study_weeks_match_the_readme_plan(self):
        names = {
            "ReLU": "relu", "Softmax": "softmax", "CE Loss": "cross_entropy",
            "Dropout": "dropout", "Embedding": "embedding", "GELU": "gelu",
            "Linear": "linear", "Linear Regression": "linear_regression",
            "LayerNorm": "layernorm", "BatchNorm": "batchnorm", "RMSNorm": "rmsnorm",
            "SwiGLU MLP": "mlp", "Conv2d": "conv2d", "SDPA": "attention", "MHA": "mha",
            "Cross-Attn": "cross_attention", "Causal": "causal_attention", "GQA": "gqa",
            "KV Cache": "kv_cache", "Sliding Window": "sliding_window", "RoPE": "rope",
            "Linear Attn": "linear_attention", "Flash Attn": "flash_attention",
            "GPT-2 Block": "gpt2_block", "LoRA": "lora", "MoE": "moe", "ViT Patch": "vit_patch",
            "Adam": "adam", "Cosine LR": "cosine_lr", "Grad Clip": "gradient_clipping",
            "Grad Accumulation": "gradient_accumulation", "Kaiming Init": "weight_init",
            "Top-k/p Sampling": "topk_sampling", "Beam Search": "beam_search",
            "Speculative Decoding": "speculative_decoding", "BPE": "bpe",
            "INT8 Quant": "int8_quantization", "DPO Loss": "dpo_loss", "GRPO Loss": "grpo_loss",
            "PPO Loss": "ppo_loss", "OPD Loss": "opd_loss",
        }
        readme = (Path(__file__).resolve().parent.parent / "README.md").read_text()
        section = readme.split("## 📅 Suggested Study Plan", 1)[1].split("\n---", 1)[0]
        rows = [line for line in section.splitlines() if line.startswith("| **")]
        self.assertEqual(len(rows), 4)
        for week, row in zip(STUDY_WEEKS, rows):
            cells = [cell.strip() for cell in row.strip("|").split("|")]
            with self.subTest(week=week["week"]):
                self.assertEqual(int(cells[0].replace("*", "")), week["week"])
                self.assertEqual(cells[1].split(" ", 1)[1], week["focus"])
                lessons = cells[2].removesuffix(" + speed run").split(" → ")
                self.assertEqual([names[lesson] for lesson in lessons], week["task_ids"])

    def test_linear_regression_is_recommended_immediately_after_linear(self):
        self.store.import_solved("local", CURRICULUM[:7])
        plan = self.store.dashboard(day="2000-01-01")["plan"]
        self.assertEqual([entry["id"] for entry in plan], ["linear_regression", "layernorm", "batchnorm"])
        self.assertEqual(plan[0]["study_week"], 1)
        self.assertEqual(plan[0]["study_focus"], "Foundations")
        self.assertIn("Week 1", plan[0]["reason"])

    def test_problem_history_reports_latest_result_and_ever_solved_state(self):
        source = "# private journey source\nreturn x"
        self.store.record_attempt("local", "relu", SUCCESS, at=NOW, code=source)
        self.store.record_attempt("local", "relu", FAILURE, hint_used=True,
                                  at=NOW + timedelta(minutes=1), code=source)
        self.store.import_solved("local", ["softmax"])
        self.store.mark_review("local", "embedding", True)
        dashboard = self.dashboard()
        latest = {key: value for key, value in dashboard["history"][0].items() if key != "source"}
        self.assertEqual(dashboard["problem_history"], [{
            **latest, "attempts": 2, "solved": True, "needs_review": True,
        }])
        summary = dashboard["problem_history"][0]
        self.assertFalse(summary["success"])
        self.assertEqual(summary["failed_tests"], ["gradient"])
        self.assertTrue(summary["hint_used"])
        self.assertNotIn("private journey source", json.dumps(dashboard))
        self.assertNotIn("submitted_code", json.dumps(dashboard))
        self.assertNotIn('"code"', json.dumps(dashboard))

    def test_problem_history_groups_the_full_log_and_orders_by_latest_time_then_id(self):
        self.store.record_attempt("local", "dropout", SUCCESS, at=NOW - timedelta(days=3))
        for index in range(105):
            self.store.record_attempt("local", "relu", FAILURE, at=NOW + timedelta(minutes=index))
        # A timestamp tie puts the greater ID first.
        self.store.record_attempt("local", "softmax", SUCCESS, at=NOW + timedelta(minutes=104))
        self.store.record_attempt("local", "gelu", SUCCESS, at=NOW + timedelta(days=2))
        # A later insertion with an older timestamp does not become the latest result.
        self.store.record_attempt("local", "gelu", FAILURE, at=NOW - timedelta(days=2))
        dashboard = self.dashboard()
        self.assertEqual(len(dashboard["history"]), 100)
        self.assertNotIn("dropout", [entry["task_id"] for entry in dashboard["history"]])
        summaries = dashboard["problem_history"]
        self.assertEqual([entry["task_id"] for entry in summaries], ["gelu", "softmax", "relu", "dropout"])
        self.assertEqual([entry["attempts"] for entry in summaries], [2, 1, 105, 1])
        self.assertTrue(summaries[0]["success"])
        self.assertEqual(summaries[0]["created_at"], (NOW + timedelta(days=2)).isoformat())
        self.assertEqual(sum(entry["attempts"] for entry in summaries), 109)
        self.assertEqual(dashboard["stats"]["attempts"], 109)
        with self.store.connection() as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0], 109)

    def test_problem_history_counts_only_web_submissions(self):
        self.store.record_attempt("local", "relu", SUCCESS, at=NOW)
        self.store.record_attempt("local", "relu", FAILURE, at=NOW + timedelta(minutes=1))
        self.store.record_attempt("local", "softmax", SUCCESS, at=NOW + timedelta(minutes=2))
        with self.store.connection() as connection:
            connection.execute("UPDATE attempts SET source = 'notebook' WHERE id > 1")
        dashboard = self.dashboard()
        self.assertEqual(len(dashboard["history"]), 3)
        self.assertEqual(len(dashboard["problem_history"]), 1)
        summary = dashboard["problem_history"][0]
        self.assertEqual(summary["task_id"], "relu")
        self.assertEqual(summary["attempts"], 1)
        self.assertTrue(summary["success"])
        self.assertEqual(summary["created_at"], NOW.isoformat())

    def test_failed_reattempt_removes_today_completion_and_keeps_ever_solved(self):
        self.dashboard()
        self.store.record_attempt("local", "relu", SUCCESS, at=NOW)
        self.store.record_attempt("local", "relu", FAILURE, at=NOW + timedelta(minutes=1))
        dashboard = self.dashboard()
        self.assertFalse(dashboard["plan"][0]["completed"])
        self.assertEqual(dashboard["goal"]["completed"], 0)
        self.assertEqual(dashboard["stats"]["today_completed"], 0)
        self.assertEqual(dashboard["stats"]["solved"], 1)
        self.assertEqual(dashboard["progress"][0]["status"], "solved")
        self.assertEqual(dashboard["reviews"][0]["id"], "relu")

    def test_hint_usage_and_spaced_reviews(self):
        self.store.record_attempt("local", "softmax", SUCCESS, hint_used=True, at=NOW)
        self.assertEqual(self.dashboard()["reviews"], [])
        tomorrow = self.store.dashboard(day="2026-10-11")
        self.assertEqual(tomorrow["reviews"][0]["id"], "softmax")
        self.assertTrue(tomorrow["history"][0]["hint_used"])
        self.store.record_attempt("local", "softmax", SUCCESS, at=NOW + timedelta(days=1))
        self.assertEqual(self.store.dashboard(day="2026-10-12")["reviews"], [])
        self.assertEqual(self.store.dashboard(day="2026-10-13")["reviews"][0]["id"], "softmax")
        self.store.record_attempt("local", "softmax", SUCCESS, at=NOW + timedelta(days=3))
        self.assertEqual(self.store.dashboard(day="2026-10-16")["reviews"], [])
        self.assertEqual(self.store.dashboard(day="2026-10-17")["reviews"][0]["id"], "softmax")

    def test_hint_review_uses_calendar_days_across_daylight_saving_fall_back(self):
        at = datetime(2026, 11, 1, 7, 30, tzinfo=timezone.utc)  # Nov 1, 00:30 PDT
        self.store.record_attempt("local", "relu", SUCCESS, hint_used=True, at=at)
        today = self.store.dashboard(day="2026-11-01", timezone="America/Los_Angeles")
        self.assertEqual(today["stats"]["today_completed"], 1)
        self.assertEqual(today["reviews"], [])
        tomorrow = self.store.dashboard(day="2026-11-02", timezone="America/Los_Angeles")
        self.assertEqual(tomorrow["reviews"][0]["id"], "relu")

    def test_clean_pass_after_repeated_failures_is_a_weak_solve_due_tomorrow(self):
        for index in range(3):
            self.store.record_attempt("local", "relu", FAILURE, at=NOW + timedelta(minutes=index))
        self.store.record_attempt("local", "relu", SUCCESS, at=NOW + timedelta(minutes=3))
        today = self.dashboard()
        self.assertEqual(today["stats"]["today_completed"], 1)
        self.assertEqual(today["stats"]["solved"], 1)
        self.assertEqual(today["reviews"], [])
        self.assertTrue(today["progress"][0]["needs_review"])
        tomorrow = self.store.dashboard(day="2026-10-11")
        self.assertEqual(tomorrow["reviews"][0]["id"], "relu")
        self.assertIn("retries", tomorrow["reviews"][0]["reason"])
        self.assertNotIn("hints", tomorrow["reviews"][0]["reason"])
        self.store.record_attempt("local", "relu", SUCCESS, at=NOW + timedelta(days=1))
        self.assertFalse(self.store.dashboard(day="2026-10-11")["progress"][0]["needs_review"])
        self.assertEqual(self.store.dashboard(day="2026-10-12")["reviews"], [])
        self.assertEqual(self.store.dashboard(day="2026-10-13")["reviews"][0]["id"], "relu")

    def test_profiles_are_isolated(self):
        other = self.store.create_profile("  Ada  ")
        self.assertEqual(other["name"], "Ada")
        self.store.record_attempt(other["id"], "relu", FAILURE, at=NOW)
        self.store.mark_review(other["id"], "embedding", True)
        self.assertEqual(self.dashboard()["history"], [])
        self.assertEqual(self.dashboard()["problem_history"], [])
        dashboard = self.dashboard(user_id=other["id"])
        self.assertEqual(len(dashboard["history"]), 1)
        self.assertEqual(len(dashboard["problem_history"]), 1)
        self.assertEqual(dashboard["problem_history"][0]["attempts"], 1)
        self.assertFalse(dashboard["problem_history"][0]["solved"])
        self.store.record_attempt("local", "relu", SUCCESS, at=NOW)
        local = self.dashboard()["problem_history"][0]
        other_summary = self.dashboard(user_id=other["id"])["problem_history"][0]
        self.assertEqual(local["attempts"], 1)
        self.assertTrue(local["solved"])
        self.assertFalse(local["needs_review"])
        self.assertEqual(other_summary["attempts"], 1)
        self.assertFalse(other_summary["success"])
        self.assertTrue(other_summary["needs_review"])
        self.assertEqual(dashboard["profile"], other)
        self.assertEqual(len(self.store.profiles()), 2)

    def test_timezone_boundary_and_streak_allow_yesterday(self):
        at = datetime(2026, 10, 11, 1, tzinfo=timezone.utc)  # Oct 10 in Los Angeles
        self.store.record_attempt("local", "relu", SUCCESS, at=at)
        self.store.record_attempt("local", "softmax", FAILURE, at=at - timedelta(days=1))
        local = self.store.dashboard(now=at, timezone="America/Los_Angeles")
        self.assertEqual(local["day"], "2026-10-10")
        self.assertEqual(local["stats"]["streak"], 2)
        self.assertEqual(local["stats"]["today_completed"], 1)
        utc = self.store.dashboard(now=at, timezone="UTC")
        self.assertEqual(utc["day"], "2026-10-11")
        self.assertEqual(utc["stats"]["today_completed"], 1)
        tomorrow = self.store.dashboard(day="2026-10-11", timezone="America/Los_Angeles")
        self.assertEqual(tomorrow["stats"]["streak"], 2)
        later = self.store.dashboard(day="2026-10-12", timezone="America/Los_Angeles")
        self.assertEqual(later["stats"]["streak"], 0)

    def test_legacy_import_is_idempotent_and_does_not_invent_attempts(self):
        imported = self.store.import_solved("local", ["relu", "relu", "not-a-task"])
        self.assertEqual(imported["imported"], 1)
        self.assertEqual(self.store.import_solved("local", ["relu"])["imported"], 0)
        dashboard = self.dashboard()
        self.assertEqual(dashboard["stats"]["solved"], 1)
        self.assertEqual(dashboard["history"], [])
        self.assertEqual(dashboard["problem_history"], [])
        self.assertEqual(dashboard["stats"]["streak"], 0)
        self.assertEqual(dashboard["stats"]["xp"], 0)

    def test_old_browser_marker_cannot_override_real_web_failure(self):
        self.store.record_attempt("local", "relu", FAILURE, at=NOW)
        imported = self.store.import_solved("local", ["relu"])
        self.assertEqual(imported, {"imported": 0, "ignored": 1})
        self.assertEqual(self.dashboard()["progress"][0]["status"], "attempted")
        self.assertEqual(self.dashboard()["stats"]["solved"], 0)

    def test_notebook_aggregates_import_without_overwriting_later_web_progress(self):
        legacy = self.path / "progress.json"
        legacy.write_text(json.dumps({"relu": {"status": "solved", "attempts": 3},
                                      "softmax": {"status": "attempted", "attempts": 2}}))
        store = PracticeStore(self.path / "practice.db", legacy)
        dashboard = store.dashboard(day="2026-10-10")
        self.assertEqual(dashboard["stats"]["solved"], 1)
        self.assertEqual(dashboard["stats"]["attempts"], 5)
        self.assertEqual(dashboard["history"], [])
        self.assertEqual(dashboard["problem_history"], [])
        store.record_attempt("local", "softmax", SUCCESS, at=NOW)
        reopened = PracticeStore(self.path / "practice.db", legacy)
        dashboard = reopened.dashboard(day="2026-10-10")
        self.assertEqual(dashboard["stats"]["solved"], 2)
        self.assertEqual(len(dashboard["problem_history"]), 1)
        self.assertEqual(dashboard["problem_history"][0]["task_id"], "softmax")
        self.assertEqual(dashboard["problem_history"][0]["attempts"], 1)

    def test_manual_review_has_no_fabricated_submission(self):
        self.store.mark_review("local", "embedding", True)
        # The flag uses the current real clock; use that same day for this test.
        dashboard = self.store.dashboard(timezone="UTC")
        self.assertEqual(dashboard["reviews"][0]["id"], "embedding")
        self.assertEqual(dashboard["history"], [])
        self.store.mark_review("local", "embedding", False)
        self.assertEqual(self.store.dashboard(timezone="UTC")["reviews"], [])

    def test_manual_review_of_today_success_is_visible_until_a_clean_pass(self):
        self.store.record_attempt("local", "relu", SUCCESS)
        self.store.mark_review("local", "relu", True)
        dashboard = self.store.dashboard()
        self.assertEqual(dashboard["reviews"][0]["id"], "relu")
        self.assertEqual(dashboard["stats"]["today_completed"], 0)
        self.store.record_attempt("local", "relu", SUCCESS)
        dashboard = self.store.dashboard()
        self.assertEqual(dashboard["reviews"], [])
        self.assertEqual(dashboard["stats"]["today_completed"], 1)

    def test_invalid_profiles_tasks_timezone_day_and_count(self):
        cases = [lambda: self.dashboard(user_id="missing"),
                 lambda: self.dashboard(count=0), lambda: self.dashboard(count=11),
                 lambda: self.dashboard(timezone="unknown-zone"),
                 lambda: self.store.dashboard(day="not-a-date"),
                 lambda: self.store.record_attempt("local", "missing", FAILURE),
                 lambda: self.store.create_profile("   ")]
        for operation in cases:
            with self.subTest(operation=operation), self.assertRaises(HTTPException):
                operation()
        self.assertEqual(self.dashboard()["history"], [])


if __name__ == "__main__":
    unittest.main()
