"""Python-as-orchestration primitives and structured script runtime."""

from agent_harness.orchestration.agent_executor import (
    AgentHarnessWorkerExecutor,
)
from agent_harness.orchestration.context import OrchestrationContext
from agent_harness.orchestration.contracts import (
    BudgetSpec,
    BudgetUsage,
    EvaluationSummary,
    StrategyKind,
)
from agent_harness.orchestration.events import (
    OrchestrationEvent,
    OrchestrationEventType,
)
from agent_harness.orchestration.exceptions import (
    IdempotencyConflict,
    InvalidJobRequest,
    InvalidRoleOutput,
    JobCancelled,
    JobFailed,
    JobLimitExceeded,
    JobTimedOut,
    OrchestrationError,
    RunTimedOut,
    UnknownRole,
)
from agent_harness.orchestration.executor import (
    WorkerEventEmitter,
    WorkerExecutor,
)
from agent_harness.orchestration.handles import JobHandle
from agent_harness.orchestration.runtime import (
    OrchestrationScript,
    ScriptRuntime,
)
from agent_harness.orchestration.patching import (
    NodeState,
    OrchestrationPatchV1,
    OrchestrationPlan,
    PatchDecision,
    PatchOperation,
    PatchValidator,
    PlanNode,
)
from agent_harness.orchestration.strategies import (
    DynamicMultiAgentStrategy,
    FixedMultiAgentStrategy,
    OrchestrationStrategy,
    SingleAgentStrategy,
    StrategyRunResult,
    create_strategy,
)
from agent_harness.orchestration.trace import RunTraceV1, TraceEventV1
from agent_harness.orchestration.generated_code import (
    EnvironmentManifestV1,
    GeneratedSolvePlanner,
    GeneratedSolveV1,
    GeneratedSolveValidator,
    SolveGenerationRequestV1,
    ValidationReport,
    write_generation_research_report,
)
from agent_harness.orchestration.roles import (
    BUILTIN_ORCHESTRATION_ROLES,
    register_builtin_orchestration_roles,
)
from agent_harness.orchestration.types import (
    AgentResult,
    FailurePolicy,
    JobFailure,
    JobOutcome,
    JobRequest,
    JobSnapshot,
    JobStatus,
    RunLimits,
    ScriptRunResult,
)

__all__ = [
    "AgentResult",
    "AgentHarnessWorkerExecutor",
    "BUILTIN_ORCHESTRATION_ROLES",
    "BudgetSpec",
    "BudgetUsage",
    "DynamicMultiAgentStrategy",
    "EvaluationSummary",
    "FailurePolicy",
    "FixedMultiAgentStrategy",
    "EnvironmentManifestV1",
    "GeneratedSolvePlanner",
    "GeneratedSolveV1",
    "GeneratedSolveValidator",
    "IdempotencyConflict",
    "InvalidJobRequest",
    "InvalidRoleOutput",
    "JobCancelled",
    "JobFailed",
    "JobFailure",
    "JobHandle",
    "JobLimitExceeded",
    "JobOutcome",
    "JobRequest",
    "JobSnapshot",
    "JobStatus",
    "JobTimedOut",
    "NodeState",
    "OrchestrationContext",
    "OrchestrationError",
    "OrchestrationEvent",
    "OrchestrationEventType",
    "OrchestrationPatchV1",
    "OrchestrationPlan",
    "OrchestrationScript",
    "OrchestrationStrategy",
    "PatchDecision",
    "PatchOperation",
    "PatchValidator",
    "PlanNode",
    "RunLimits",
    "RunTraceV1",
    "RunTimedOut",
    "SingleAgentStrategy",
    "ScriptRunResult",
    "ScriptRuntime",
    "StrategyKind",
    "StrategyRunResult",
    "SolveGenerationRequestV1",
    "TraceEventV1",
    "WorkerEventEmitter",
    "WorkerExecutor",
    "ValidationReport",
    "write_generation_research_report",
    "UnknownRole",
    "create_strategy",
    "register_builtin_orchestration_roles",
]
