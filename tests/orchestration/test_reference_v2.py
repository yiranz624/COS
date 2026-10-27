from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_harness.orchestration.runtime import ScriptRuntime
from agent_harness.orchestration.testing import FakeWorkerExecutor
from agent_harness.orchestration.types import AgentResult, RunLimits
from orchestrations.registry import get_orchestration_script


def broad_invalid_plan() -> dict:
    return {
        "original_task": "question",
        "subtasks": [{
            "id": "all", "question": "找出某时期所有足球比赛",
            "dependencies": [], "difficulty": "hard",
            "success_criteria": ["all matches"],
            "required_evidence": ["database"],
            "tools": ["database_query", "web_search"],
        }],
        "synthesis_requirements": ["answer"],
    }


def solver_result() -> str:
    return json.dumps({
        "status": "candidate",
        "candidates": [{
            "answer": "Candidate A",
            "constraint_status": {"rare constraint": "verified"},
            "evidence": [{
                "claim": "supporting fact", "url": "https://example.test/a",
                "source": "Example", "status": "verified",
            }],
        }],
        "verified_facts": [], "missing_constraints": [],
        "search_summary": "bounded search",
    })


def verdict(accepted: bool) -> dict:
    return {
        "accepted": accepted, "critique": "more evidence" if not accepted else "",
        "satisfied_criteria": ["done"] if accepted else [],
        "failed_criteria": [] if accepted else ["cross-check"],
        "verified_evidence": [{
            "claim": "supporting fact", "url": "https://example.test/a",
            "source": "Example", "status": "verified",
        }],
        "confidence": 0.8,
    }


class ReferenceV2Tests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_exhaustive_plan_uses_candidate_funnel(self) -> None:
        executor = FakeWorkerExecutor(handlers={
            "decomposer": lambda _request: broad_invalid_plan(),
            "solver": lambda _request: solver_result(),
            "verifier": lambda _request: verdict(True),
            "synthesizer": lambda _request: "Candidate A",
        })
        registered = get_orchestration_script("reference_v2")
        with tempfile.TemporaryDirectory() as directory:
            result = await ScriptRuntime(
                executor, limits=RunLimits(max_total_jobs=20),
                checkpoint_path=Path(directory) / "checkpoint.json",
            ).run(
                registered.solve, task="question", run_id="v2-fallback",
                metadata={"script_source_hash": registered.manifest.source_hash,
                          "budget": {"max_tokens": 72_000}},
            )
        self.assertEqual(result.value, "Candidate A")
        solver_requests = [item for item in executor.requests if item.role == "solver"]
        self.assertEqual(len(solver_requests), 3)
        self.assertTrue(all("database_query" not in item.instruction
                            for item in solver_requests))
        plan_event = next(
            item for item in result.events
            if item.payload.get("script_event") == "plan_validated"
        )
        self.assertTrue(plan_event.payload["fallback"])
        self.assertIn("unavailable tools", plan_event.payload["fallback_reason"])

    async def test_coverage_pass_precedes_every_revision(self) -> None:
        executor = FakeWorkerExecutor(handlers={
            "decomposer": lambda _request: broad_invalid_plan(),
            "solver": lambda _request: AgentResult(
                value=solver_result(), usage={"total_tokens": 1000}
            ),
            "verifier": lambda _request: AgentResult(
                value=verdict(False), usage={"total_tokens": 500}
            ),
            "synthesizer": lambda _request: "信息未查全",
        })
        registered = get_orchestration_script("reference_v2")
        await ScriptRuntime(executor, limits=RunLimits(max_total_jobs=20)).run(
            registered.solve, task="question", run_id="v2-coverage",
            metadata={"script_source_hash": registered.manifest.source_hash,
                      "budget": {"max_tokens": 72_000}},
        )
        solver_keys = [item.key for item in executor.requests if item.role == "solver"]
        first_revision = next(
            index for index, key in enumerate(solver_keys) if key.endswith(":1")
        )
        self.assertEqual(first_revision, 3)
        self.assertEqual(
            solver_keys[:3],
            [
                "reference_v2:candidate_discovery:solver:0",
                "reference_v2:constraint_verification:solver:0",
                "reference_v2:cross_check:solver:0",
            ],
        )

    async def test_structured_evidence_is_forwarded_to_downstream(self) -> None:
        executor = FakeWorkerExecutor(handlers={
            "decomposer": lambda _request: broad_invalid_plan(),
            "solver": lambda _request: solver_result(),
            "verifier": lambda _request: verdict(True),
            "synthesizer": lambda _request: "Candidate A",
        })
        registered = get_orchestration_script("reference_v2")
        await ScriptRuntime(executor).run(
            registered.solve, task="question", run_id="v2-evidence",
            metadata={"script_source_hash": registered.manifest.source_hash,
                      "budget": {"max_tokens": 72_000}},
        )
        downstream = next(
            item for item in executor.requests
            if item.key == "reference_v2:constraint_verification:solver:0"
        )
        self.assertIn("https://example.test/a", downstream.instruction)
        self.assertIn("Candidate A", downstream.instruction)
