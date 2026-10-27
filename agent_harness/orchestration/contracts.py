"""Versioned contracts for budgeted orchestration experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class BudgetSpec:
    """Hard, comparable limits shared by orchestration strategies."""

    max_tokens: int | None = None
    max_llm_calls: int | None = None
    max_tool_calls: int | None = None
    max_jobs: int = 128
    max_parallel_jobs: int = 4
    wall_time_seconds: float | None = None

    def __post_init__(self) -> None:
        for name in ("max_tokens", "max_llm_calls", "max_tool_calls"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set")
        if self.max_jobs <= 0 or self.max_parallel_jobs <= 0:
            raise ValueError("job limits must be positive")
        if self.wall_time_seconds is not None and self.wall_time_seconds <= 0:
            raise ValueError("wall_time_seconds must be positive when set")


@dataclass(frozen=True)
class BudgetUsage:
    tokens: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    jobs: int = 0
    duration_seconds: float = 0.0

    def add(self, other: "BudgetUsage") -> "BudgetUsage":
        return BudgetUsage(
            tokens=self.tokens + other.tokens,
            llm_calls=self.llm_calls + other.llm_calls,
            tool_calls=self.tool_calls + other.tool_calls,
            jobs=self.jobs + other.jobs,
            duration_seconds=self.duration_seconds + other.duration_seconds,
        )


class StrategyKind(str, Enum):
    SINGLE_AGENT = "single_agent"
    FIXED_MULTI_AGENT = "fixed_multi_agent"
    DYNAMIC_MULTI_AGENT = "dynamic_multi_agent"
    GENERATED_CODE_ORCHESTRATION = "generated_code_orchestration"
    ROLECHANGE_SELF_EVOLVING_V4 = "rolechange_self_evolving_v4"
    ROLECHANGE_SELF_EVOLVING_V6 = "rolechange_self_evolving_v6"
    ROLECHANGE_SELF_EVOLVING_V5 = "rolechange_self_evolving_v5"


@dataclass(frozen=True)
class EvaluationSummary:
    strategy: str
    total: int
    attempted: int
    correct: int
    accuracy: float
    mean_rubric_score: float | None
    usage: BudgetUsage
    failures: int
    timeouts: int
    budget_violations: int
    patches_proposed: int = 0
    patches_applied: int = 0
    estimated_cost: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
