from __future__ import annotations

import json
import unittest

from agent_harness.orchestration.runtime import ScriptRuntime
from agent_harness.orchestration.testing import FakeWorkerExecutor
from orchestrations.reference_v4 import (
    ConstraintV4,
    _programmatic_admission,
)
from orchestrations.contracts import CandidateV2, EvidenceRefV2
from orchestrations.registry import get_orchestration_script


def plan() -> dict:
    return {
        "original_task": "question",
        "subtasks": [{
            "id": "constraints",
            "question": "find a matching game",
            "dependencies": [],
            "difficulty": "hard",
            "success_criteria": ["referee is Brazilian", "cards match"],
            "required_evidence": ["match report URL"],
            "tools": ["web_search", "web_fetch"],
        }],
        "synthesis_requirements": ["short answer"],
    }


def candidate_result() -> str:
    return json.dumps({
        "status": "candidate",
        "candidates": [{
            "answer": "Ireland v Romania",
            "constraint_status": {"C1": "unknown", "C2": "unknown"},
            "evidence": [{
                "claim": "match", "url": "https://example.test/match",
                "source": "archive", "status": "unverified",
            }],
        }],
        "verified_facts": [],
        "missing_constraints": ["C1", "C2"],
        "search_summary": "candidate found",
    })


def incomplete_result() -> str:
    return json.dumps({
        "status": "incomplete", "candidates": [], "verified_facts": [],
        "missing_constraints": ["candidate"], "search_summary": "none",
    })


def verified() -> dict:
    return {
        "accepted": False,
        "critique": "",
        "satisfied_criteria": ["C1: supported", "C2: supported"],
        "failed_criteria": [],
        "verified_evidence": [{
            "claim": "official record", "url": "https://fifa.test/match",
            "source": "FIFA", "status": "verified",
        }],
        "confidence": 0.9,
    }


class ReferenceV4Tests(unittest.IsolatedAsyncioTestCase):
    async def test_no_candidate_blocks_verifier_and_synthesizer(self) -> None:
        executor = FakeWorkerExecutor(handlers={
            "decomposer": lambda _: plan(),
            "solver": lambda _: incomplete_result(),
            "verifier": lambda _: self.fail("verifier must be blocked"),
            "synthesizer": lambda _: self.fail("synthesis must be blocked"),
        })
        registered = get_orchestration_script("reference_v4")
        result = await ScriptRuntime(executor).run(
            registered.solve, task="question", run_id="v4-empty",
            metadata={"budget": {"max_tokens": 72_000}},
        )
        self.assertEqual(result.value, "信息未查全")
        self.assertEqual([r.role for r in executor.requests].count("solver"), 3)
        self.assertNotIn("verifier", [r.role for r in executor.requests])
        blocked = [e for e in result.events
                   if e.payload.get("script_event") == "verification_blocked"]
        self.assertEqual(len(blocked), 1)

    async def test_search_fetch_output_turns_and_programmatic_acceptance(self) -> None:
        executor = FakeWorkerExecutor(handlers={
            "decomposer": lambda _: plan(),
            "solver": lambda _: candidate_result(),
            "verifier": lambda _: verified(),
            "synthesizer": lambda _: "Ireland v Romania",
        })
        registered = get_orchestration_script("reference_v4")
        result = await ScriptRuntime(executor).run(
            registered.solve, task="question", run_id="v4-success",
            metadata={"budget": {"max_tokens": 72_000}},
        )
        self.assertEqual(result.value, "Ireland v Romania")
        solvers = [request for request in executor.requests
                   if request.role == "solver"]
        self.assertEqual(len(solvers), 2)
        self.assertTrue(all(request.max_turns == 4 for request in solvers))
        self.assertTrue(all(request.metadata["allowed_tools"] ==
                            ["web_search", "web_fetch"] for request in solvers))
        verifier = next(request for request in executor.requests
                        if request.role == "verifier")
        self.assertEqual(verifier.max_turns, 3)

    def test_missing_criterion_and_paper_only_fail_closed(self) -> None:
        constraints = [ConstraintV4("C1", "a", "url"),
                       ConstraintV4("C2", "b", "url")]
        candidate = CandidateV2("answer", {}, [
            EvidenceRefV2("lead", "https://arxiv.org/abs/1", status="verified"),
        ])
        verdict = verified()
        verdict["satisfied_criteria"] = ["C1: supported"]
        verdict["verified_evidence"] = []
        accepted, statuses, reasons = _programmatic_admission(
            constraints, candidate, verdict,
        )
        self.assertFalse(accepted)
        self.assertEqual(statuses["C2"], "unknown")
        self.assertIn("缺少独立非论文事实来源", reasons)
