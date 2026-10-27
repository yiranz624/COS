from __future__ import annotations

import unittest

from agent_harness.orchestration import (
    BudgetSpec,
    NodeState,
    OrchestrationPatchV1,
    OrchestrationPlan,
    PatchOperation,
    PatchValidator,
    PlanNode,
)


class PatchValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = PatchValidator(
            allowed_roles={"solver", "verifier", "synthesizer"},
            budget=BudgetSpec(max_jobs=4, max_parallel_jobs=2),
        )

    def test_valid_add_is_atomic_and_versions_plan(self) -> None:
        plan = OrchestrationPlan((
            PlanNode("solve", "solver", "solve"),
        ), max_parallel_jobs=2)
        patch = OrchestrationPatchV1(
            base_version=1,
            trigger_stage="solver",
            evidence=("rejected",),
            operations=(PatchOperation(
                op="add",
                node=PlanNode(
                    "verify", "verifier", "verify", dependencies=("solve",),
                ),
            ),),
            reason="verify the candidate",
        )

        decision = self.validator.apply(plan, patch)

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.plan.version, 2)
        self.assertEqual(len(decision.plan.nodes), 2)

    def test_started_node_cannot_be_changed(self) -> None:
        plan = OrchestrationPlan((
            PlanNode("solve", "solver", "solve", state=NodeState.SUCCEEDED),
        ))
        patch = OrchestrationPatchV1(
            base_version=1,
            trigger_stage="solver",
            evidence=("late",),
            operations=(PatchOperation(
                op="update", node_id="solve", changes={"max_turns": 8},
            ),),
            reason="too late",
        )

        decision = self.validator.apply(plan, patch)

        self.assertFalse(decision.accepted)
        self.assertIs(decision.plan, plan)

    def test_cycle_unknown_role_and_budget_are_rejected(self) -> None:
        base = OrchestrationPlan((
            PlanNode("a", "solver", "a"),
            PlanNode("b", "verifier", "b", dependencies=("a",)),
        ))
        cases = (
            PatchOperation(
                op="update", node_id="a", changes={"dependencies": ("b",)},
            ),
            PatchOperation(
                op="add", node=PlanNode("c", "invented", "c"),
            ),
            PatchOperation(
                op="set_parallelism", changes={"max_parallel_jobs": 3},
            ),
        )
        for index, operation in enumerate(cases):
            with self.subTest(index=index):
                decision = self.validator.apply(
                    base,
                    OrchestrationPatchV1(
                        base_version=1,
                        trigger_stage="test",
                        evidence=("test",),
                        operations=(operation,),
                        reason="invalid",
                    ),
                )
                self.assertFalse(decision.accepted)
                self.assertEqual(decision.plan.version, 1)
