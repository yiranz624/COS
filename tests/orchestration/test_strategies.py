from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_harness.orchestration import (
    AgentResult,
    BudgetSpec,
    DynamicMultiAgentStrategy,
    FixedMultiAgentStrategy,
)
from agent_harness.orchestration.evolution import diagnose_trace
from agent_harness.orchestration.strategies import _collect_candidates
from agent_harness.orchestration.testing import FakeWorkerExecutor


def _handler(request):
    if request.role in {"solver", "researcher"}:
        return AgentResult(
            value=f"solution:{request.key}",
            usage={"input_tokens": 10, "output_tokens": 5},
        )
    if request.role == "verifier":
        return {
            "accepted": True,
            "critique": "candidate satisfies the supplied criteria",
        }
    if request.role == "synthesizer":
        return "final answer"
    raise AssertionError(request.role)


class StrategyTests(unittest.IsolatedAsyncioTestCase):
    def test_task_323_incomplete_result_keeps_concrete_answers(self) -> None:
        candidates = _collect_candidates([{
            "branch_index": 0,
            "branch": {"role": "researcher"},
            "solution": {
                "status": "incomplete",
                "answer": "Svetlana Gromenkova (likely 2008 winner)",
                "candidate_table": [
                    "Vanessa Hellebuyck (candidate X)",
                    "Svetlana Gromenkova (2008 winner)",
                ],
                "evidence": ["2008 event result"],
                "missing_constraints": ["grandparent clue"],
                "incorrect_claims": [],
            },
        }])

        self.assertEqual(len(candidates), 3)
        self.assertTrue(all(item["status"] == "incomplete" for item in candidates))
        self.assertTrue(any(
            "Svetlana Gromenkova" in item["answer"] for item in candidates
        ))

    async def test_dynamic_skips_verification_when_stage_budget_is_spent(self) -> None:
        roles = []
        synthesis_instructions = []

        def handler(request):
            roles.append(request.role)
            if request.role == "researcher":
                return AgentResult(value="candidate", usage={"tokens": 100})
            if request.role == "synthesizer":
                synthesis_instructions.append(request.instruction)
                return "best available answer"
            raise AssertionError(request.role)

        result = await DynamicMultiAgentStrategy().run(
            FakeWorkerExecutor(default_handler=handler),
            task="question",
            run_id="stage-budget",
            budget=BudgetSpec(max_tokens=100, max_jobs=8),
        )

        self.assertIn("candidate", synthesis_instructions[0])
        self.assertNotIn("verifier", roles)
        self.assertIn(
            "verification_skipped_budget",
            {event.stage for event in result.trace.events},
        )
        report = diagnose_trace(result.trace)
        self.assertIn("verification_skipped_budget", report.failure_categories)
        self.assertIn(
            "reserve_verification_budget",
            {item["change"] for item in report.candidate_changes},
        )

    async def test_failed_verifier_degrades_without_dropping_candidate(self) -> None:
        synthesis_instructions = []

        def handler(request):
            if request.role == "researcher":
                return "candidate"
            if request.role == "verifier":
                raise RuntimeError("verifier unavailable")
            if request.role == "synthesizer":
                synthesis_instructions.append(request.instruction)
                return "fallback synthesis"
            raise AssertionError(request.role)

        result = await DynamicMultiAgentStrategy().run(
            FakeWorkerExecutor(default_handler=handler),
            task="question",
            run_id="revision-degrade",
            budget=BudgetSpec(max_tokens=10_000, max_jobs=8),
        )

        self.assertIn("candidate", synthesis_instructions[0])
        self.assertIn(
            "verification_degraded",
            {event.stage for event in result.trace.events},
        )

    async def test_low_confidence_solution_remains_in_synthesis_pool(self) -> None:
        synthesis_instructions = []

        def handler(request):
            if request.role == "researcher":
                return "DISPROVEN_SECRET_CANDIDATE"
            if request.role == "verifier":
                return {
                    "accepted": False,
                    "critique": "identity is contradicted by the source",
                }
            if request.role == "synthesizer":
                synthesis_instructions.append(request.instruction)
                return "Insufficient information"
            raise AssertionError(request.role)

        result = await DynamicMultiAgentStrategy().run(
            FakeWorkerExecutor(default_handler=handler),
            task="question",
            run_id="admission-control",
            budget=BudgetSpec(max_tokens=10_000, max_jobs=10),
        )

        self.assertEqual(result.value, "Insufficient information")
        self.assertIn(
            "DISPROVEN_SECRET_CANDIDATE", synthesis_instructions[0],
        )
        self.assertIn("identity is contradicted", synthesis_instructions[0])

    async def test_web_branch_uses_researcher_and_preserves_full_task(self) -> None:
        requests = []

        def handler(request):
            requests.append(request)
            if request.role == "researcher":
                return "evidence-backed candidate"
            if request.role == "verifier":
                return {
                    "accepted": True,
                    "critique": "all required evidence is present",
                }
            if request.role == "synthesizer":
                return "final answer: evidence-backed candidate"
            raise AssertionError(request.role)

        result = await FixedMultiAgentStrategy().run(
            FakeWorkerExecutor(default_handler=handler),
            task="complete question with condition A and condition B",
            run_id="web-routing",
            budget=BudgetSpec(max_jobs=8, max_parallel_jobs=2),
        )

        self.assertEqual(result.value, "final answer: evidence-backed candidate")
        research = next(r for r in requests if r.role == "researcher")
        self.assertIn("complete question with condition A", research.instruction)
        self.assertIn("condition B", research.instruction)
        self.assertIn("web_search/web_fetch", research.instruction)
        self.assertNotIn("Kaggle", research.instruction)
        synthesis = next(r for r in requests if r.role == "synthesizer")
        self.assertIn("evidence-backed candidate", synthesis.instruction)

    async def test_fixed_strategy_audits_each_candidate_independently(self) -> None:
        verifier_requests = []

        def handler(request):
            if request.role in {"researcher", "solver"}:
                return {
                    "status": "candidate",
                    "answer": f"candidate:{request.key}",
                    "candidate_table": [],
                    "evidence": ["source"],
                    "missing_constraints": [],
                    "incorrect_claims": [],
                }
            if request.role == "verifier":
                verifier_requests.append(request)
                return {
                    "accepted": False,
                    "critique": "one criterion remains unresolved",
                }
            if request.role == "synthesizer":
                return "synthesized"
            raise AssertionError(request.role)

        executor = FakeWorkerExecutor(default_handler=handler)
        result = await FixedMultiAgentStrategy().run(
            executor,
            task="question",
            run_id="independent-audits",
            budget=BudgetSpec(max_jobs=16, max_parallel_jobs=3),
        )

        self.assertEqual(result.value, "synthesized")
        self.assertEqual(len(verifier_requests), 3)
        self.assertTrue(all(request.key.startswith("verify:") for request in verifier_requests))
        self.assertTrue(all("Candidate to audit" in request.instruction for request in verifier_requests))
        self.assertTrue(all("candidate set" not in request.instruction.casefold() for request in verifier_requests))

    async def test_fixed_strategy_produces_versioned_trace_and_usage(self) -> None:
        result = await FixedMultiAgentStrategy().run(
            FakeWorkerExecutor(default_handler=_handler),
            task="question",
            run_id="fixed",
            budget=BudgetSpec(max_jobs=16, max_parallel_jobs=3),
        )

        self.assertEqual(result.value, "final answer")
        # The fixed graph uses three bounded static subtasks.
        self.assertEqual(result.trace.usage.tokens, 45)
        self.assertIn(
            "subtask_list_prepared",
            {event.stage for event in result.trace.events},
        )

        with tempfile.TemporaryDirectory() as directory:
            path = result.trace.write_jsonl(Path(directory) / "trace.jsonl")
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(
            [row["sequence"] for row in rows],
            list(range(1, len(rows) + 1)),
        )
        self.assertTrue(all(row["schema_version"] == "1.0" for row in rows))

    async def test_dynamic_strategy_records_applied_normalization_patch(self) -> None:
        result = await DynamicMultiAgentStrategy(max_subproblems=2).run(
            FakeWorkerExecutor(default_handler=_handler),
            task="question",
            run_id="dynamic",
            budget=BudgetSpec(max_jobs=16, max_parallel_jobs=2),
        )

        self.assertEqual(result.value, "final answer")
        self.assertEqual(result.patches_proposed, 1)
        self.assertEqual(result.patches_applied, 1)
        self.assertFalse(result.budget_violations)
        report = diagnose_trace(result.trace)
        self.assertFalse(report.auto_deploy)
        self.assertTrue(report.replay_validation["schema_valid"])

    async def test_budget_violation_is_reported(self) -> None:
        result = await FixedMultiAgentStrategy().run(
            FakeWorkerExecutor(default_handler=_handler),
            task="question",
            run_id="over-budget",
            budget=BudgetSpec(
                max_tokens=1, max_jobs=16, max_parallel_jobs=3,
            ),
        )

        self.assertEqual(result.budget_violations, ("tokens:45>1",))
