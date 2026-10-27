from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from harbor.models.agent.context import AgentContext

from benchmarks.harbor_agent.agent import AgentHarnessAgent
from benchmarks.runner.run_one_task import (
    _is_completed_result as worker_result_is_completed,
    _normalize_exact_match,
    _score_answer,
)
from benchmarks.runner.run_subprocess import (
    _is_completed_result as master_result_is_completed,
)


class _FailingSession:
    async def run(self, *args, **kwargs):
        raise RuntimeError("provider request failed")


class _EmptySession:
    async def run(self, *args, **kwargs):
        return {"task_id": "task-1"}


class BenchmarkErrorHandlingTests(unittest.IsolatedAsyncioTestCase):
    def test_question_runtime_is_unlimited_by_default(self) -> None:
        from benchmarks.runner import run_one_task, run_subprocess

        self.assertIsNone(run_one_task._QUESTION_TIMEOUT)
        self.assertIsNone(run_subprocess._QUESTION_TIMEOUT)

    def test_generated_harbor_task_has_no_agent_timeout(self) -> None:
        from harbor.models.task.config import TaskConfig

        from benchmarks.core.harbor_task_generator import _render_task_toml

        rendered = _render_task_toml(
            question_id="1",
            ground_truth="answer",
            answer_type="exactMatch",
            pipeline_id="react_base",
            verifier_image="agent-harness/verifier:v1",
            image_path=None,
            category="",
        )
        task = TaskConfig.model_validate_toml(rendered)
        self.assertIsNone(task.agent.timeout_sec)

    def test_unordered_team_match_ignores_home_away_order(self) -> None:
        from benchmarks.runner.run_one_task import _unordered_team_match

        self.assertTrue(_unordered_team_match(
            "Romania vs Ireland", "Ireland v Romania",
        ))
        self.assertFalse(_unordered_team_match(
            "Romania vs England", "Ireland v Romania",
        ))
    async def test_pipeline_failure_is_not_converted_to_an_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = AgentHarnessAgent(logs_dir=Path(tmp), model_name=None)
            agent._session = _FailingSession()

            with self.assertRaisesRegex(
                RuntimeError,
                "provider request failed",
            ):
                await agent.run("question", MagicMock(), AgentContext())

            self.assertFalse(
                (Path(tmp) / "agent" / "final_answer.txt").exists()
            )

    async def test_missing_final_answer_is_a_pipeline_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = AgentHarnessAgent(logs_dir=Path(tmp), model_name=None)
            agent._session = _EmptySession()

            with self.assertRaisesRegex(
                RuntimeError,
                "pipeline completed without state.final_answer",
            ):
                await agent.run("question", MagicMock(), AgentContext())

    def test_legacy_missing_answer_result_is_rerunnable(self) -> None:
        result = {
            "predicted_answer": "<ANSWER_NOT_FOUND>",
            "error": None,
            "scoring_error": None,
        }

        self.assertFalse(worker_result_is_completed(result))
        self.assertFalse(master_result_is_completed(result))

    def test_exact_match_normalizes_final_answer_wrapper(self) -> None:
        self.assertEqual(
            _normalize_exact_match("**FINAL ANSWER:** Ireland  v Romania."),
            _normalize_exact_match("Ireland v Romania"),
        )

    async def test_exact_match_scoring_does_not_need_judge(self) -> None:
        score, method, error, rubric, details = await _score_answer(
            "Ireland v Romania",
            "Ireland v Romania",
            "question",
            "exactMatch",
            "1",
            "browsecomp_200",
            "exact_match",
        )
        self.assertEqual(score, 1)
        self.assertEqual(method, "exact_match")
        self.assertIsNone(error)
        self.assertIsNone(rubric)
        self.assertIsNone(details)


if __name__ == "__main__":
    unittest.main()
