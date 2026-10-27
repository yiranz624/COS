from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from benchmarks.runner.rejudge import _rejudge_one


class SemanticRejudgeTests(unittest.IsolatedAsyncioTestCase):
    @patch(
        "benchmarks.runner.rejudge.judge_semantic_equivalence",
        new_callable=AsyncMock,
    )
    async def test_semantic_rejudge_persists_structured_details(self, judge) -> None:
        judge.return_value = {
            "extracted_final_answer": "Stranger Things",
            "reasoning": "The extra wording is correct.",
            "correct": True,
            "confidence": 100.0,
            "parse_error": False,
            "raw_response": "correct: yes",
            "model": "test-model",
            "attempts": 1,
        }
        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "result.json"
            result_path.write_text(
                json.dumps({
                    "question_id": "23",
                    "predicted_answer": "The full name is Stranger Things.",
                    "rubric_score": None,
                }),
                encoding="utf-8",
            )
            question = SimpleNamespace(
                question="What is the full name of the series?",
                ground_truth="Stranger Things",
            )
            outcome = await _rejudge_one(
                result_path,
                "browsecomp",
                {"23": question},
                asyncio.Semaphore(1),
                True,
                "semantic_judge",
            )
            saved = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertEqual(outcome, "ok")
        self.assertTrue(saved["is_correct"])
        self.assertEqual(saved["reward"], 1)
        self.assertEqual(saved["judge_method"], "semantic_judge")
        self.assertEqual(saved["scoring_mode"], "semantic_judge")
        self.assertIsNone(saved["scoring_error"])
        self.assertEqual(
            saved["judge_details"]["extracted_final_answer"],
            "Stranger Things",
        )
