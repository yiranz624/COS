from __future__ import annotations

import json
import unittest

from agent_harness.orchestration.runtime import ScriptRuntime
from agent_harness.orchestration.testing import FakeWorkerExecutor
from orchestrations.contracts import CandidateV2, EvidenceRefV2, SolverResultV2, SubtaskPlanV1, SubtaskSpec
from orchestrations.reference_v3 import _apply_admission_policy, _descendant_ids, _is_stale
from orchestrations.registry import get_orchestration_script


def generated_plan_with_synthesis() -> dict:
    return {"original_task": "question", "subtasks": [
        {"id": "discover", "question": "Find up to five joint candidates",
         "dependencies": [], "difficulty": "hard", "success_criteria": ["candidate"],
         "required_evidence": ["URL"], "tools": ["web_search"]},
        {"id": "final_answer", "question": "Synthesize final answer",
         "dependencies": ["discover"], "difficulty": "easy", "success_criteria": ["answer"],
         "required_evidence": ["result"], "tools": []}],
        "synthesis_requirements": ["short answer"]}


def candidate_result(paper_only: bool = False) -> dict:
    evidence = [{"claim": "lead", "url": "https://aclanthology.org/paper.pdf",
                 "source": "paper", "status": "verified"}]
    if not paper_only:
        evidence.append({"claim": "match record", "url": "https://example.test/match",
                         "source": "archive", "status": "verified"})
    return {"status": "candidate", "candidates": [{"answer": "Romania vs Ireland",
        "constraint_status": {"referee": "verified", "cards": "verified",
                              "substitution": "verified"}, "evidence": evidence}],
        "verified_facts": [], "missing_constraints": [], "search_summary": "candidate"}


def verdict() -> dict:
    return {"accepted": True, "critique": "", "satisfied_criteria": ["done"],
            "failed_criteria": [], "verified_evidence": [], "confidence": 0.9}


class ReferenceV3Tests(unittest.IsolatedAsyncioTestCase):
    async def test_candidate_stops_discovery_and_tools_reach_job(self) -> None:
        discovery_calls = 0

        def solver(request):
            nonlocal discovery_calls
            if "candidate_discovery" in request.key:
                discovery_calls += 1
                if discovery_calls == 1:
                    return json.dumps({"status": "incomplete", "candidates": [],
                        "verified_facts": [], "missing_constraints": ["candidate"],
                        "search_summary": "none"})
            return json.dumps(candidate_result())

        executor = FakeWorkerExecutor(handlers={
            "decomposer": lambda _: generated_plan_with_synthesis(), "solver": solver,
            "verifier": lambda _: verdict(), "synthesizer": lambda _: "Ireland v Romania"})
        registered = get_orchestration_script("reference_v3")
        result = await ScriptRuntime(executor).run(
            registered.solve, task="question", run_id="v3",
            metadata={"script_source_hash": registered.manifest.source_hash,
                      "budget": {"max_tokens": 72_000}})
        self.assertEqual(result.value, "Ireland v Romania")
        self.assertEqual(discovery_calls, 2)
        self.assertFalse(any("execution:3" in item.key for item in executor.requests))
        solvers = [item for item in executor.requests if item.role == "solver"]
        self.assertEqual(solvers[0].metadata["allowed_tools"], ["web_search", "web_fetch"])
        self.assertEqual(len({item.key.split(":")[1] for item in solvers}), 2)
        event = next(e for e in result.events if e.payload.get("script_event") == "plan_validated")
        self.assertIn("redundant synthesis", event.payload["fallback_reason"])

    def test_dependency_versions_make_descendants_stale(self) -> None:
        plan = SubtaskPlanV1("q", [
            SubtaskSpec("a", "a", [], "easy", ["x"], [], []),
            SubtaskSpec("b", "b", ["a"], "easy", ["x"], [], []),
            SubtaskSpec("c", "c", ["b"], "easy", ["x"], [], []),
        ], [])
        states = {"a": {"version": 2},
                  "b": {"version": 1, "dependency_versions": {"a": 1}}}
        self.assertTrue(_is_stale(plan.subtasks[1], states["b"], states))
        self.assertEqual(_descendant_ids(plan, "a"), ["b", "c"])

    def test_paper_only_candidate_cannot_be_accepted(self) -> None:
        guarded = _apply_admission_policy(
            SolverResultV2.from_dict(candidate_result(True)), verdict())
        self.assertFalse(guarded["accepted"])
        self.assertIn("独立非论文来源", guarded["critique"])

    def test_unknown_constraint_cannot_be_accepted(self) -> None:
        candidate = CandidateV2("Romania vs Ireland", {"injury": "unknown"}, [
            EvidenceRefV2("match", "https://example.test", status="verified")])
        solution = SolverResultV2("candidate", [candidate], [], [], "")
        self.assertFalse(_apply_admission_policy(solution, verdict())["accepted"])
