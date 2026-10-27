"""Constrained patches for the not-yet-executed portion of a plan."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable

from agent_harness.orchestration.contracts import BudgetSpec


class NodeState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class PlanNode:
    node_id: str
    role: str
    instruction: str
    dependencies: tuple[str, ...] = ()
    max_turns: int = 4
    state: NodeState = NodeState.PENDING
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrchestrationPlan:
    nodes: tuple[PlanNode, ...]
    version: int = 1
    max_parallel_jobs: int = 4
    termination: str = "all_required_complete"


@dataclass(frozen=True)
class PatchOperation:
    op: str
    node_id: str | None = None
    node: PlanNode | None = None
    changes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrchestrationPatchV1:
    base_version: int
    trigger_stage: str
    evidence: tuple[str, ...]
    operations: tuple[PatchOperation, ...]
    reason: str
    estimated_added_jobs: int = 0
    schema_version: str = "1.0"


@dataclass(frozen=True)
class PatchDecision:
    accepted: bool
    reason: str
    plan: OrchestrationPlan


class PatchValidationError(ValueError):
    pass


class PatchValidator:
    def __init__(self, *, allowed_roles: Iterable[str], budget: BudgetSpec) -> None:
        self.allowed_roles = set(allowed_roles)
        self.budget = budget

    def apply(
        self,
        plan: OrchestrationPlan,
        patch: OrchestrationPatchV1,
    ) -> PatchDecision:
        try:
            updated = self._apply_or_raise(plan, patch)
        except PatchValidationError as exc:
            return PatchDecision(False, str(exc), plan)
        return PatchDecision(True, "validated and applied atomically", updated)

    def _apply_or_raise(
        self,
        plan: OrchestrationPlan,
        patch: OrchestrationPatchV1,
    ) -> OrchestrationPlan:
        if patch.base_version != plan.version:
            raise PatchValidationError("patch base_version does not match plan")
        nodes = {node.node_id: node for node in plan.nodes}
        parallel = plan.max_parallel_jobs
        termination = plan.termination
        for operation in patch.operations:
            if operation.op == "add":
                if operation.node is None or operation.node.node_id in nodes:
                    raise PatchValidationError("add requires a new node")
                nodes[operation.node.node_id] = operation.node
            elif operation.op == "delete":
                node = self._pending(nodes, operation.node_id)
                del nodes[node.node_id]
            elif operation.op in {"replace", "retry", "update"}:
                node = self._pending(nodes, operation.node_id)
                changes = dict(operation.changes)
                if operation.op == "retry":
                    changes.setdefault("state", NodeState.PENDING)
                nodes[node.node_id] = replace(node, **changes)
            elif operation.op == "set_parallelism":
                parallel = int(operation.changes.get("max_parallel_jobs", 0))
            elif operation.op == "set_termination":
                termination = str(operation.changes.get("termination", ""))
            else:
                raise PatchValidationError(f"unsupported operation {operation.op!r}")

        if len(nodes) > self.budget.max_jobs:
            raise PatchValidationError("patch exceeds max_jobs")
        if parallel <= 0 or parallel > self.budget.max_parallel_jobs:
            raise PatchValidationError("patch exceeds max_parallel_jobs")
        if not termination:
            raise PatchValidationError("termination condition cannot be empty")
        for node in nodes.values():
            if node.role not in self.allowed_roles:
                raise PatchValidationError(f"unknown role {node.role!r}")
            if node.max_turns <= 0:
                raise PatchValidationError("node max_turns must be positive")
            unknown = set(node.dependencies) - nodes.keys()
            if unknown:
                raise PatchValidationError(
                    f"node {node.node_id!r} has unknown dependencies {sorted(unknown)}"
                )
        self._assert_acyclic(nodes)
        return OrchestrationPlan(
            nodes=tuple(nodes.values()),
            version=plan.version + 1,
            max_parallel_jobs=parallel,
            termination=termination,
        )

    @staticmethod
    def _pending(nodes: dict[str, PlanNode], node_id: str | None) -> PlanNode:
        if node_id not in nodes:
            raise PatchValidationError(f"unknown node {node_id!r}")
        node = nodes[node_id]
        if node.state != NodeState.PENDING:
            raise PatchValidationError(
                f"node {node.node_id!r} has already started"
            )
        return node

    @staticmethod
    def _assert_acyclic(nodes: dict[str, PlanNode]) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise PatchValidationError("patch introduces a cycle")
            if node_id in visited:
                return
            visiting.add(node_id)
            for dependency in nodes[node_id].dependencies:
                visit(dependency)
            visiting.remove(node_id)
            visited.add(node_id)

        for node_id in nodes:
            visit(node_id)
