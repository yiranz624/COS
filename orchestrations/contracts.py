"""Data contracts owned by orchestration scripts, not by the runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Difficulty = Literal["easy", "medium", "hard"]

ALLOWED_SUBTASK_TOOLS = frozenset({
    "web_search", "web_fetch", "run_python_code",
})


@dataclass(frozen=True)
class SubtaskSpec:
    id: str
    question: str
    dependencies: list[str]
    difficulty: Difficulty
    success_criteria: list[str]
    required_evidence: list[str]
    tools: list[str]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SubtaskSpec":
        return cls(
            id=str(value["id"]).strip(),
            question=str(value["question"]).strip(),
            dependencies=[str(x) for x in value.get("dependencies", [])],
            difficulty=value.get("difficulty", "medium"),
            success_criteria=[
                str(x) for x in value.get("success_criteria", [])
            ],
            required_evidence=[
                str(x) for x in value.get("required_evidence", [])
            ],
            tools=[str(x) for x in value.get("tools", [])],
        )


@dataclass(frozen=True)
class SubtaskPlanV1:
    original_task: str
    subtasks: list[SubtaskSpec]
    synthesis_requirements: list[str]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SubtaskPlanV1":
        return cls(
            original_task=str(value.get("original_task", "")).strip(),
            subtasks=[SubtaskSpec.from_dict(x) for x in value["subtasks"]],
            synthesis_requirements=[
                str(x) for x in value.get("synthesis_requirements", [])
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SubproblemStateV1:
    subtask_id: str
    status: Literal["pending", "accepted", "incomplete", "rejected"] = "pending"
    solution: str = ""
    critique: str = ""
    satisfied_criteria: list[str] = field(default_factory=list)
    failed_criteria: list[str] = field(default_factory=list)
    verified_evidence: list[Any] = field(default_factory=list)
    confidence: float = 0.0
    rounds: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SubproblemStateV1":
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceRefV2:
    claim: str
    url: str
    source: str = ""
    status: Literal["verified", "refuted", "unverified"] = "unverified"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvidenceRefV2":
        status = str(value.get("status", "unverified"))
        if status not in {"verified", "refuted", "unverified"}:
            status = "unverified"
        return cls(
            claim=str(value.get("claim", "")).strip(),
            url=str(value.get("url", "")).strip(),
            source=str(value.get("source", "")).strip(),
            status=status,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class CandidateV2:
    answer: str
    constraint_status: dict[str, str]
    evidence: list[EvidenceRefV2]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CandidateV2":
        return cls(
            answer=str(value.get("answer", "")).strip(),
            constraint_status={
                str(key): str(item)
                for key, item in value.get("constraint_status", {}).items()
            },
            evidence=[
                EvidenceRefV2.from_dict(item)
                for item in value.get("evidence", [])
                if isinstance(item, dict)
            ],
        )


@dataclass(frozen=True)
class SolverResultV2:
    status: Literal["candidate", "incomplete"]
    candidates: list[CandidateV2]
    verified_facts: list[EvidenceRefV2]
    missing_constraints: list[str]
    search_summary: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SolverResultV2":
        status = "candidate" if value.get("status") == "candidate" else "incomplete"
        candidates = [
            CandidateV2.from_dict(item)
            for item in value.get("candidates", [])
            if isinstance(item, dict)
        ][:5]
        if status == "candidate" and not any(item.answer for item in candidates):
            status = "incomplete"
        return cls(
            status=status,
            candidates=candidates,
            verified_facts=[
                EvidenceRefV2.from_dict(item)
                for item in value.get("verified_facts", [])
                if isinstance(item, dict)
            ],
            missing_constraints=[
                str(item) for item in value.get("missing_constraints", [])
            ],
            search_summary=str(value.get("search_summary", "")).strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_plan(plan: SubtaskPlanV1) -> None:
    """Reject invalid IDs, dependencies, cycles, and oversized plans."""
    if not plan.original_task:
        raise ValueError("original_task must not be empty")
    if not 1 <= len(plan.subtasks) <= 3:
        raise ValueError("plan must contain between one and three subtasks")
    ids = [subtask.id for subtask in plan.subtasks]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("subtask IDs must be non-empty and unique")
    known = set(ids)
    graph: dict[str, list[str]] = {}
    for subtask in plan.subtasks:
        if not subtask.question or not subtask.success_criteria:
            raise ValueError("each subtask needs a question and success criteria")
        if subtask.difficulty not in {"easy", "medium", "hard"}:
            raise ValueError(f"invalid difficulty: {subtask.difficulty}")
        unknown_tools = set(subtask.tools) - ALLOWED_SUBTASK_TOOLS
        if unknown_tools:
            raise ValueError(f"unavailable tools: {sorted(unknown_tools)}")
        if subtask.id in subtask.dependencies:
            raise ValueError("a subtask cannot depend on itself")
        unknown = set(subtask.dependencies) - known
        if unknown:
            raise ValueError(f"unknown dependencies: {sorted(unknown)}")
        graph[subtask.id] = subtask.dependencies

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError("subtask dependency graph contains a cycle")
        if node in visited:
            return
        visiting.add(node)
        for dependency in graph[node]:
            visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for node in ids:
        visit(node)


def fallback_plan(task: str) -> SubtaskPlanV1:
    return SubtaskPlanV1(
        original_task=task,
        subtasks=[SubtaskSpec(
            id="complete_task",
            question=task,
            dependencies=[],
            difficulty="hard",
            success_criteria=["直接回答原始问题并满足其中全部显式约束"],
            required_evidence=["对最终答案关键事实提供可核验来源"],
            tools=["web_search", "web_fetch", "run_python_code"],
        )],
        synthesis_requirements=["输出 benchmark 所需的简洁最终答案"],
    )
