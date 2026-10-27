from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_harness.orchestration.markdown_audit import write_trial_agent_report


class MarkdownAuditTests(unittest.TestCase):
    def test_writes_review_and_runtime_agent_inputs_outputs_and_rejections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trial = Path(directory)
            review = trial / "generated" / "plan" / "attempts" / "attempt_01" / "review"
            review.mkdir(parents=True)
            (review / "system_prompt.md").write_text("review system", encoding="utf-8")
            (review / "review_prompt.md").write_text("review input", encoding="utf-8")
            (review / "raw_response.md").write_text(json.dumps({
                "accepted": False,
                "critique": "缺少验证阶段",
                "issues": ["没有 verifier"],
                "suggested_changes": ["增加 verifier"],
            }, ensure_ascii=False), encoding="utf-8")

            traces = trial / "agent_traces"
            traces.mkdir()
            rows = [
                {
                    "t": "start", "role_id": "verifier",
                    "system_prompt": "runtime system", "user_message": "check candidate",
                },
                {
                    "t": "end", "最终输出": json.dumps({
                        "accepted": False,
                        "critique": "证据不足",
                        "failed_criteria": ["条件 4 未证明"],
                    }, ensure_ascii=False),
                },
            ]
            (traces / "run_job_1_verifier.jsonl").write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
                encoding="utf-8",
            )

            output = write_trial_agent_report(trial)
            rendered = output.read_text(encoding="utf-8")
            self.assertEqual(output.name, "researcher_verifier_report.md")
            self.assertIn("Plan verifier · attempt_01", rendered)
            self.assertIn("缺少验证阶段", rendered)
            self.assertIn("运行时 verifier #1", rendered)
            self.assertIn("check candidate", rendered)
            self.assertIn("条件 4 未证明", rendered)


if __name__ == "__main__":
    unittest.main()
