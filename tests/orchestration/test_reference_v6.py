from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent_harness.orchestration.runtime import ScriptRuntime
from agent_harness.orchestration.testing import FakeWorkerExecutor
from orchestrations.registry import get_orchestration_script


def plan() -> dict:
    return {"original_task": "find the answer", "subtasks": [
        {"id": "discover", "question": "discover candidate", "dependencies": [],
         "difficulty": "hard", "success_criteria": ["candidate found"],
         "required_evidence": ["url"], "tools": ["web_search"]},
        {"id": "filter", "question": "filter candidate",
         "dependencies": ["discover"], "difficulty": "easy",
         "success_criteria": ["candidate filtered"],
         "required_evidence": ["url"], "tools": ["web_search", "web_fetch"]},
    ], "synthesis_requirements": ["short answer"]}


def review(accepted: bool = True) -> dict:
    return {"accepted": accepted, "critique": "remove invented enumeration",
            "missing_requirements": [],
            "invented_requirements": [] if accepted else ["twenty candidates"]}


def update(candidate: str, *, gain: bool = True) -> dict:
    return {"solution": candidate, "queries_attempted": [f"query {candidate}"],
            "observations": [f"observed {candidate}"], "candidates": [candidate],
            "verified_evidence": [{"claim": candidate,
                "url": "https://example.test/evidence", "source": "source"}],
            "rejected_candidates": [], "unresolved_criteria": [],
            "information_gain": gain}


def verdict(accepted: bool) -> dict:
    return {"accepted": accepted, "critique": "search a distinct route",
            "satisfied_criteria": ["done"] if accepted else [],
            "failed_criteria": [] if accepted else ["candidate found"],
            "verified_evidence": [], "confidence": 0.9}


class ReferenceV6Tests(unittest.IsolatedAsyncioTestCase):
    async def test_plan_review_fair_first_pass_and_stateful_revision(self):
        decompositions = 0
        solver_calls: dict[str, int] = {}

        def decomposer(_request):
            nonlocal decompositions
            decompositions += 1
            return plan()

        def reviewer(request):
            return review(accepted=":0" not in request.key)

        def solver(request):
            sid = request.key.split(":")[1]
            solver_calls[sid] = solver_calls.get(sid, 0) + 1
            if sid == "filter":
                self.assertIn("candidate-a", request.instruction)
                self.assertIn("即使incomplete也要主动补全", request.instruction)
                return update("FINAL")
            if solver_calls[sid] > 1:
                self.assertIn("query candidate-a", request.instruction)
                self.assertIn("search a distinct route", request.instruction)
            return update("candidate-a")

        def verifier(request):
            return verdict(request.key.startswith("reference_v6:filter"))

        executor = FakeWorkerExecutor(handlers={
            "decomposer": decomposer, "plan_reviewer": reviewer,
            "stateful_solver": solver, "verifier": verifier,
            "synthesizer": lambda _: "FINAL",
        })
        registered = get_orchestration_script("reference_v6")
        with tempfile.TemporaryDirectory() as directory:
            result = await ScriptRuntime(
                executor, checkpoint_path=Path(directory) / "checkpoint.json",
            ).run(registered.solve, task="find the answer", run_id="v6-flow",
                  metadata={"script_source_hash": registered.manifest.source_hash,
                            "max_rounds_per_subtask": 3})
        self.assertEqual(result.value, "FINAL")
        self.assertEqual(decompositions, 2)
        roles = [request.role for request in executor.requests]
        first_filter = next(i for i, request in enumerate(executor.requests)
                            if request.key == "reference_v6:filter:solver:0")
        second_discover = next(i for i, request in enumerate(executor.requests)
                               if request.key == "reference_v6:discover:solver:1")
        self.assertLess(first_filter, second_discover)
        self.assertIn("plan_reviewer", roles)

    async def test_two_stagnant_rounds_stop_revisions(self):
        calls = 0

        def solver(_request):
            nonlocal calls
            calls += 1
            return update("same")

        executor = FakeWorkerExecutor(handlers={
            "decomposer": lambda _: {**plan(), "subtasks": [plan()["subtasks"][0]]},
            "plan_reviewer": lambda _: review(),
            "stateful_solver": solver, "verifier": lambda _: verdict(False),
            "synthesizer": lambda _: "信息未查全",
        })
        registered = get_orchestration_script("reference_v6")
        result = await ScriptRuntime(executor).run(
            registered.solve, task="find", run_id="v6-stagnant",
            metadata={"max_rounds_per_subtask": 10},
        )
        self.assertEqual(result.value, "信息未查全")
        self.assertEqual(calls, 3)
