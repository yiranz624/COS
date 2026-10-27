from __future__ import annotations

import unittest

from agent_harness.orchestration.runtime import ScriptRuntime
from agent_harness.orchestration.testing import FakeWorkerExecutor
from orchestrations.registry import get_orchestration_script
from orchestrations.reference_v5 import (
    _sanitize_observation,
    _validate_queries,
    _validate_routes,
)
from orchestrations.reference_v4 import ConstraintV4


def plan() -> dict:
    return {"original_task": "1990 match with a Brazilian referee",
            "subtasks": [{"id": "constraints", "question": "find match",
                "dependencies": [], "difficulty": "medium",
                "success_criteria": ["match was in 1990", "referee was Brazilian"],
                "required_evidence": ["match report"], "tools": []}],
            "synthesis_requirements": ["teams"]}


def planner(request) -> dict:
    if "expansion" in request.key:
        return {"queries": [{"query": '"José Roberto Wright" 1990 matches',
                              "goal": "enumerate matches",
                              "expected_entity": "event"}]}
    route = request.key.rsplit(":", 1)[-1]
    expected = "person" if route == "identity_route" else "event"
    return {"queries": [{"query": f"Brazilian referee 1990 {route}",
                          "goal": "find referee", "expected_entity": expected}]}


def selector(request) -> dict:
    if ":1:" in request.key:
        return {"entities": [], "information_gain": True,
                "missing_constraints": [], "candidates": [{
                    "answer": "Ireland v Romania",
                    "constraint_status": {"C1": "unknown", "C2": "unknown"},
                    "evidence": [{"claim": "match result",
                        "url": "https://example.test/match", "source": "archive",
                        "status": "unverified"}]}]}
    if "identity_route" in request.key:
        entities = [{"type": "person", "name": "José Roberto Wright",
                     "url": "https://example.test/referee",
                     "matched_constraints": ["C2"]}]
    else:
        entities = []
    return {"entities": entities,
            "candidates": [], "information_gain": True,
            "missing_constraints": ["match identity"]}


def verifier(_request) -> dict:
    return {"accepted": False, "critique": "", "confidence": 0.9,
            "satisfied_criteria": ["C1: supported", "C2: supported"],
            "failed_criteria": [], "verified_evidence": [{
                "claim": "official record", "url": "https://example.test/match",
                "source": "archive", "status": "verified"}]}


def route_design(_request) -> dict:
    return {"routes": [
        {"route_id": "identity_route", "goal": "find referee",
         "pivot_entity_type": "person", "expansion_entity_type": "event",
         "active_constraints": ["C2"], "deferred_constraints": ["C1"]},
        {"route_id": "event_route", "goal": "find match",
         "pivot_entity_type": "event", "expansion_entity_type": "record",
         "active_constraints": ["C1"], "deferred_constraints": ["C2"]},
    ]}


class ReferenceV5Tests(unittest.IsolatedAsyncioTestCase):
    def test_route_types_are_ai_selected_not_person_hardcoded(self) -> None:
        constraints = [ConstraintV4("C1", "work clue", "url"),
                       ConstraintV4("C2", "place clue", "url")]
        routes = _validate_routes([{
            "route_id": "work_to_place", "goal": "identify the work",
            "pivot_entity_type": "work", "expansion_entity_type": "place",
            "active_constraints": ["C1"], "deferred_constraints": ["C2"],
        }], constraints)
        self.assertEqual(routes[0]["pivot_entity_type"], "work")
        self.assertEqual(routes[0]["expansion_entity_type"], "place")

    def test_route_type_and_constraint_evidence_are_hard_constraints(self) -> None:
        rejected = _validate_queries([{
            "query": "Brazil team match 1990", "goal": "wrong pivot",
            "expected_entity": "event",
        }], "question from 1990", [], 2, "person")
        self.assertEqual(rejected, [])
        sanitized = _sanitize_observation({
            "entities": [{"type": "event", "name": "Brazil v Sweden",
                          "url": "https://example.test/match",
                          "matched_constraints": ["C1"]}],
            "candidates": [], "information_gain": True,
            "missing_constraints": ["C2"],
        }, "entity_pivot", "person", [ConstraintV4("C2", "referee", "url")])
        self.assertEqual(sanitized["entities"], [])
        self.assertFalse(sanitized["information_gain"])

    async def test_entity_hop_direct_tools_and_programmatic_verification(self) -> None:
        executor = FakeWorkerExecutor(
            handlers={"decomposer": lambda _: plan(),
                      "route_designer": route_design, "query_planner": planner,
                      "result_selector": selector, "verifier": verifier,
                      "synthesizer": lambda _: "Ireland v Romania"},
            tool_handlers={
                "web_search": lambda request: (
                    "[1] Title: relevant record\n    Snippet: useful evidence\n"
                    "    URL: https://example.test/result"
                ),
                "web_fetch": lambda request: "C1 and C2 are supported",
            },
        )
        registered = get_orchestration_script("reference_v5")
        result = await ScriptRuntime(executor).run(
            registered.solve, task="1990 match with a Brazilian referee",
            run_id="v5-success", metadata={"budget": {"max_tokens": 72_000,
                                                        "max_tool_calls": 20}},
        )
        self.assertEqual(result.value, "Ireland v Romania")
        self.assertEqual(len(executor.tool_requests), 4)
        self.assertEqual([x["tool_name"] for x in executor.tool_requests].count(
            "web_search"), 3)
        self.assertTrue(all(request.max_turns == 1 for request in executor.requests))
        event_types = [event.event_type.value for event in result.events]
        self.assertEqual(event_types.count("tool_call_started"), 4)
        self.assertIn("tool_call_succeeded", event_types)

    async def test_no_information_gain_stops_without_verifier(self) -> None:
        empty = {"entities": [], "candidates": [], "information_gain": False,
                 "missing_constraints": ["candidate"]}
        executor = FakeWorkerExecutor(
            handlers={"decomposer": lambda _: plan(),
                      "route_designer": route_design,
                      "query_planner": planner,
                      "result_selector": lambda _: empty,
                      "verifier": lambda _: self.fail("must not verify"),
                      "synthesizer": lambda _: self.fail("must not synthesize")},
            tool_handlers={"web_search": lambda _: ""},
        )
        registered = get_orchestration_script("reference_v5")
        result = await ScriptRuntime(executor).run(
            registered.solve, task="1990 match with a Brazilian referee",
            run_id="v5-empty", metadata={"budget": {"max_tokens": 72_000}},
        )
        self.assertEqual(result.value, "信息未查全")
        self.assertFalse(any(request.role == "verifier"
                             for request in executor.requests))

    async def test_verifier_cannot_invent_an_unfetched_source(self) -> None:
        bad_verifier = verifier(None)
        bad_verifier["verified_evidence"][0]["url"] = "https://invented.test/x"
        executor = FakeWorkerExecutor(
            handlers={"decomposer": lambda _: plan(),
                      "route_designer": route_design, "query_planner": planner,
                      "result_selector": selector,
                      "verifier": lambda _: bad_verifier,
                      "synthesizer": lambda _: self.fail("must not synthesize")},
            tool_handlers={
                "web_search": lambda _: (
                    "[1] Title: record\n    URL: https://example.test/result"),
                "web_fetch": lambda _: "support",
            },
        )
        registered = get_orchestration_script("reference_v5")
        result = await ScriptRuntime(executor).run(
            registered.solve, task="1990 match with a Brazilian referee",
            run_id="v5-invented", metadata={"budget": {"max_tokens": 72_000}},
        )
        self.assertEqual(result.value, "信息未查全")
