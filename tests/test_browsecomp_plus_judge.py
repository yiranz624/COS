from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from benchmarks.judges import score_answer_detailed
from benchmarks.judges.browsecomp_plus import (
    create_judge_prompt,
    get_semantic_judge_model,
    judge_semantic_equivalence,
    parse_judge_response,
    verdict_from_judgement,
)
from benchmarks.runner.run_one_task import _score_answer


class BrowseCompPlusJudgeTests(unittest.IsolatedAsyncioTestCase):
    @patch.dict(
        "os.environ",
        {
            "OPENAI_MODEL": "deepseek-v4-flash",
            "JUDGE_BASE_URL": "",
            "SEMANTIC_JUDGE_MODEL": "",
            "BROWSECOMP_PLUS_JUDGE_MODEL": "",
            "BROWSECOMP_JUDGE_MODEL": "",
        },
    )
    def test_model_follows_agent_endpoint_when_judge_is_not_dedicated(self) -> None:
        self.assertEqual(get_semantic_judge_model(), "deepseek-v4-flash")

    def test_prompt_explicitly_allows_semantic_variants(self) -> None:
        prompt = create_judge_prompt(
            "Which series?", "The full title is Stranger Things.", "Stranger Things",
        )
        self.assertIn("semantically equivalent", prompt)
        self.assertIn("more precise or verbose", prompt)
        self.assertIn("The full title is Stranger Things.", prompt)
        self.assertIn("[correct_answer]: Stranger Things", prompt)

    def test_parser_handles_markdown_fields_and_multiline_reasoning(self) -> None:
        parsed = parse_judge_response(
            "**extracted_final_answer:** Stranger Things\n\n"
            "**reasoning:** The response adds harmless wording.\n"
            "It names the same series.\n\n"
            "**correct:** yes\n"
            "**confidence:** 100%"
        )
        self.assertEqual(parsed["extracted_final_answer"], "Stranger Things")
        self.assertIn("same series", parsed["reasoning"])
        self.assertTrue(parsed["correct"])
        self.assertEqual(parsed["confidence"], 100.0)
        self.assertFalse(parsed["parse_error"])
        self.assertEqual(verdict_from_judgement(parsed), "CORRECT")

    def test_parser_marks_missing_correct_field_as_error(self) -> None:
        parsed = parse_judge_response(
            "extracted_final_answer: Stranger Things\nconfidence: 100%"
        )
        self.assertTrue(parsed["parse_error"])
        self.assertEqual(verdict_from_judgement(parsed), "NOT_ATTEMPTED")

    @patch(
        "benchmarks.judges.browsecomp_plus._judge_call_with_effort_fallback",
        new_callable=AsyncMock,
    )
    async def test_detailed_judge_preserves_response_metadata(self, call) -> None:
        call.return_value = (
            "extracted_final_answer: Stranger Things\n"
            "reasoning: Semantically identical.\n"
            "correct: yes\n"
            "confidence: 100%"
        )
        result = await judge_semantic_equivalence(
            "Which series?", "Stranger Things", "The answer is Stranger Things.",
            max_attempts=1,
        )
        self.assertTrue(result["correct"])
        self.assertEqual(result["attempts"], 1)
        self.assertIn("Semantically identical", result["raw_response"])
        self.assertTrue(result["model"])

    @patch(
        "benchmarks.runner.run_one_task.judge_semantic_equivalence",
        new_callable=AsyncMock,
    )
    async def test_semantic_scoring_mode_returns_details(self, judge) -> None:
        judge.return_value = {
            "extracted_final_answer": "Stranger Things",
            "reasoning": "Same series.",
            "correct": True,
            "confidence": 100.0,
            "parse_error": False,
            "raw_response": "correct: yes",
            "model": "test-model",
            "attempts": 1,
        }
        score, method, error, rubric, details = await _score_answer(
            "The answer is Stranger Things.",
            "Stranger Things",
            "Which series?",
            "exactMatch",
            "23",
            "browsecomp",
            "semantic_judge",
        )
        self.assertEqual(score, 1)
        self.assertEqual(method, "semantic_judge")
        self.assertIsNone(error)
        self.assertIsNone(rubric)
        self.assertEqual(details["extracted_final_answer"], "Stranger Things")

    @patch(
        "benchmarks.judges.judge_semantic_equivalence",
        new_callable=AsyncMock,
    )
    async def test_default_browsecomp_judge_uses_plus_details(self, judge) -> None:
        judge.return_value = {
            "extracted_final_answer": "Stranger Things",
            "reasoning": "Equivalent.",
            "correct": True,
            "confidence": 100.0,
            "parse_error": False,
            "raw_response": "correct: yes",
            "model": "test-model",
            "attempts": 1,
        }
        verdict, rubric, details = await score_answer_detailed(
            "browsecomp",
            "Which series?",
            "Stranger Things",
            "The answer is Stranger Things.",
        )
        self.assertEqual(verdict, "CORRECT")
        self.assertIsNone(rubric)
        self.assertEqual(details["model"], "test-model")


if __name__ == "__main__":
    unittest.main()
