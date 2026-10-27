"""Safe, per-task generation of Code-as-Orchestration ``solve`` functions.

The generated source is deliberately a small Python DSL.  It is validated
before compilation and receives only the public orchestration context plus a
minimal builtin namespace.  This is defence in depth; benchmark workers still
run each question in their existing subprocess.
"""

from __future__ import annotations

import ast
import asyncio
import copy
import hashlib
import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

from agent_harness.core.llm import LLMClient
from agent_harness.core.messages import system_msg, text_of, user_msg
from agent_harness.infra.retriable import is_transient_network
from agent_harness.orchestration.contracts import BudgetSpec, BudgetUsage
from agent_harness.orchestration.context import OrchestrationContext
from agent_harness.orchestration.rolechange_evolution import (
    evolution_system_prompt, with_evolution_context,
)
from agent_harness.orchestration.types import (
    AgentResult,
    FailurePolicy,
    JobOutcome,
    JobStatus,
)

DSL_VERSION = "1"
ALLOWED_CTX_METHODS = frozenset({
    "call", "spawn", "gather", "parallel", "map", "cancel",
    "checkpoint", "load_checkpoint", "remaining_budget", "emit",
})
SAFE_BUILTINS: dict[str, Any] = {
    "str": str, "int": int, "float": float, "bool": bool,
    "list": list, "dict": dict, "set": set, "tuple": tuple,
    "len": len, "range": range, "enumerate": enumerate, "zip": zip,
    "min": min, "max": max, "sum": sum, "sorted": sorted,
    "any": any, "all": all, "isinstance": isinstance,
    "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
}
SAFE_VALUE_METHODS = frozenset({
    "get", "items", "keys", "values", "setdefault", "append", "extend", "update",
    "add", "strip", "split", "splitlines", "join", "lower", "upper",
    "casefold", "startswith", "endswith", "replace", "format", "copy",
})
FORBIDDEN_NAMES = frozenset({
    "eval", "exec", "compile", "open", "input", "globals", "locals",
    "getattr", "setattr", "delattr", "__import__", "breakpoint", "help",
    "os", "sys", "subprocess", "socket", "pathlib", "builtins",
})

# These findings describe best-effort robustness rather than whether the
# restricted program can run with the declared DSL.  Keep collecting them for
# diagnostics, but do not spend planner repair attempts or block execution on
# them.  Syntax, sandbox, role, API, and result-contract incompatibilities
# remain blocking validation issues.
ROBUSTNESS_WARNING_CODES = frozenset({
    "unprotected_critical_call",
    "empty_failure_degradation",
    "missing_loop_budget_check",
    "loop_bound",
})

LOOP_JOB_KEY_GUIDANCE = (
    "Every ctx.call and ctx.spawn inside a for loop must use an explicit job "
    "key containing the logical stage, loop iteration, and stable entity identity. "
    "The logical stage must be a literal label and the loop iteration must be the "
    "real value bound by the loop. For `for item in items`, the key must "
    "reference the current item; do not use only a fixed unrelated label. For "
    "`for index, item in enumerate(items)`, include both the index and current "
    "item, for example `f'research:{index}:{item}'`. If an index is needed, "
    "bind it with enumerate instead of inventing or manually maintaining an "
    "unrelated counter. Different logical invocations must never reuse a key; "
    "reuse a key only when retrying the exact same role, instruction, and "
    "options. Never use 'unknown' alone as the entity identity."
)


@dataclass(frozen=True)
class EnvironmentManifestV1:
    runtime_version: str
    dsl_version: str
    roles: dict[str, dict[str, Any]]
    tools: dict[str, dict[str, Any]]
    ctx_api: tuple[str, ...]
    budget: dict[str, Any]


@dataclass(frozen=True)
class SolveGenerationRequestV1:
    task: str
    metadata: dict[str, Any]
    environment: EnvironmentManifestV1
    templates: tuple[dict[str, Any], ...]
    failure_lessons: tuple[dict[str, Any], ...]
    generation_constraints: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    line: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationReport:
    accepted: bool
    issues: tuple[ValidationIssue, ...]
    source_hash: str = ""
    warnings: tuple[ValidationIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "issues": [item.to_dict() for item in self.issues],
            "warnings": [item.to_dict() for item in self.warnings],
            "source_hash": self.source_hash,
        }


@dataclass
class RepairAttemptRecord:
    """Audit data shared with later generated-code repair attempts."""

    attempt: int
    source: str = ""
    generation_error: dict[str, Any] | None = None
    static_validation: dict[str, Any] | None = None
    compile_error: dict[str, Any] | None = None
    preflight_error: dict[str, Any] | None = None
    review: dict[str, Any] | None = None
    review_error: dict[str, Any] | None = None
    review_skipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReviewVerdict:
    accepted: bool
    critique: str
    suggested_changes: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "critique": self.critique,
            "issues": list(self.issues),
            "suggested_changes": list(self.suggested_changes),
        }


@dataclass
class GeneratedSolveV1:
    generation_id: str
    dsl_version: str
    planner_model: str
    source: str
    source_hash: str
    generation_attempt: int
    validation_status: str
    validation_errors: list[dict[str, Any]] = field(default_factory=list)
    validation_warnings: list[dict[str, Any]] = field(default_factory=list)
    natural_language_plan_hash: str = ""
    natural_language_plan_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlannerResult:
    generated: GeneratedSolveV1 | None
    usage: BudgetUsage
    events: tuple[dict[str, Any], ...]
    request_path: str
    validation_path: str
    source_path: str | None
    attempts_path: str
    natural_language_plan_path: str | None = None
    natural_language_plan_artifacts_path: str | None = None
    fallback_reason: str | None = None


class GeneratedCodeValidationError(ValueError):
    pass


class GeneratedSolveValidator:
    """Validate the restricted orchestration DSL and compile it safely."""

    def __init__(
        self,
        *,
        allowed_roles: set[str],
        max_parallel_jobs: int,
        max_loop_iterations: int = 8,
        require_evolution_proposal: bool = False,
    ) -> None:
        self.allowed_roles = set(allowed_roles)
        self.max_parallel_jobs = max_parallel_jobs
        self.max_loop_iterations = max_loop_iterations
        self.require_evolution_proposal = require_evolution_proposal
        self.allowed_ctx_methods = ALLOWED_CTX_METHODS | (
            {"facts_query", "facts_index"} if require_evolution_proposal else set()
        )

    def validate(self, source: str) -> ValidationReport:
        issues: list[ValidationIssue] = []
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return ValidationReport(False, (ValidationIssue(
                "syntax_error", exc.msg, exc.lineno,
            ),))

        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.AsyncFunctionDef):
            issues.append(ValidationIssue(
                "top_level", "source must contain exactly one async solve(ctx)",
            ))
        else:
            fn = tree.body[0]
            args = fn.args
            if fn.name != "solve" or [arg.arg for arg in args.args] != ["ctx"]:
                issues.append(ValidationIssue(
                    "signature", "entrypoint must be async def solve(ctx)", fn.lineno,
                ))
            if args.vararg or args.kwarg or args.kwonlyargs or args.defaults:
                issues.append(ValidationIssue(
                    "signature", "solve may not define optional or variadic arguments",
                    fn.lineno,
                ))

        if self.require_evolution_proposal:
            from agent_harness.orchestration.rolechange_evolution import validate_proposal_source
            for code, message in validate_proposal_source(tree):
                issues.append(ValidationIssue(code, message))

        self._local_callables = {
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for node in ast.walk(tree):
            line = getattr(node, "lineno", None)
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal,
                                 ast.ClassDef, ast.Lambda, ast.While, ast.With,
                                 ast.AsyncWith, ast.Delete, ast.Yield, ast.YieldFrom)):
                issues.append(ValidationIssue(
                    "forbidden_syntax", f"{type(node).__name__} is not allowed", line,
                ))
            if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                issues.append(ValidationIssue(
                    "implicit_loop", "comprehensions are not allowed; use a bounded for loop", line,
                ))
            if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
                issues.append(ValidationIssue(
                    "forbidden_name", f"name {node.id!r} is not allowed", line,
                ))
            if isinstance(node, ast.Attribute):
                if node.attr.startswith("_"):
                    issues.append(ValidationIssue(
                        "private_attribute", f"attribute {node.attr!r} is not allowed", line,
                    ))
                direct_ctx_attribute = (
                    isinstance(node.value, ast.Name) and node.value.id == "ctx"
                )
                if direct_ctx_attribute and node.attr not in self.allowed_ctx_methods | {"task", "metadata", "run_id"}:
                    issues.append(ValidationIssue(
                        "ctx_api", f"ctx.{node.attr} is not an allowed API", line,
                    ))
            if isinstance(node, ast.Call):
                self._validate_call(node, issues)
            if isinstance(node, ast.For):
                self._validate_loop(node, issues)
            if isinstance(node, ast.AsyncFor):
                issues.append(ValidationIssue(
                    "unbounded_loop", "async for is not allowed", line,
                ))

        self._validate_job_keys(tree, issues)
        self._collect_runtime_diagnostics(tree, issues)

        deduped = tuple({(x.code, x.message, x.line): x for x in issues}.values())
        warnings = tuple(
            item for item in deduped if item.code in ROBUSTNESS_WARNING_CODES
        )
        blocking = tuple(
            item for item in deduped if item.code not in ROBUSTNESS_WARNING_CODES
        )
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest() if not blocking else ""
        return ValidationReport(not blocking, blocking, digest, warnings)

    def _validate_job_keys(
        self, tree: ast.Module, issues: list[ValidationIssue],
    ) -> None:
        """Reject ambiguous loop keys and conflicting explicit key reuse."""
        parents = _parent_map(tree)
        seen_keys: dict[str, tuple[str, int | None]] = {}

        for node in ast.walk(tree):
            if not _is_ctx_call(node, methods={"call", "spawn"}):
                continue
            assert isinstance(node, ast.Call)
            line = getattr(node, "lineno", None)
            key_expression = _job_key_expression(node)
            resolved_key = _resolve_local_key_expression(
                key_expression, node, parents,
            )

            if resolved_key is not None:
                key_shape = ast.dump(
                    resolved_key, annotate_fields=True,
                    include_attributes=False,
                )
                request_shape = _job_request_shape(node)
                previous = seen_keys.get(key_shape)
                if previous is not None and previous[0] != request_shape:
                    issues.append(ValidationIssue(
                        "duplicate_job_key",
                        "the same explicit job key is used by different logical "
                        "ctx.call/ctx.spawn invocations; use the same key only when "
                        "retrying the exact same role, instruction, and options",
                        line,
                    ))
                else:
                    seen_keys[key_shape] = (request_shape, line)

            loops = _enclosing_for_loops(node, parents)
            if not loops:
                continue
            if resolved_key is None or _is_none_literal(resolved_key):
                issues.append(ValidationIssue(
                    "loop_job_key",
                    "ctx.call/ctx.spawn inside a loop requires an explicit key "
                    "containing the logical stage, the actual loop value, and a "
                    "stable entity identity",
                    line,
                ))
                continue

            key_names = _key_expression_names(resolved_key)
            missing_iterations = []
            missing_entities = []
            loop_names: set[str] = set()
            for loop in loops:
                iteration_names = _loop_iteration_names(loop)
                loop_names.update(iteration_names)
                if iteration_names and not key_names.intersection(iteration_names):
                    missing_iterations.append(
                        "/".join(sorted(iteration_names))
                    )
                entity_names = _loop_entity_names(loop)
                if entity_names and not _key_references_names(
                    resolved_key, entity_names, node, parents,
                ):
                    missing_entities.append("/".join(sorted(entity_names)))

            literal_tokens = _key_literal_tokens(resolved_key)
            stage_tokens = [
                token for token in literal_tokens
                if token.casefold() != "unknown" and not token.isdigit()
            ]
            identity_names = (
                key_names - loop_names - set(SAFE_BUILTINS) - {"ctx"}
            )
            has_ctx_identity = any(
                isinstance(item, ast.Attribute)
                and isinstance(item.value, ast.Name)
                and item.value.id == "ctx"
                and item.attr in {"run_id", "metadata"}
                for item in ast.walk(resolved_key)
            )
            # Collection loops must identify the current item. For range loops,
            # which have no separate item identity, retain the stable-label rule.
            has_entity_identity = bool(
                identity_names or has_ctx_identity or len(stage_tokens) >= 2
            )
            uses_unknown_identity = any(
                token.casefold() == "unknown" for token in literal_tokens
            )
            uses_unknown_identity = uses_unknown_identity or any(
                binding is not None and any(
                    token.casefold() == "unknown"
                    for token in _key_literal_tokens(binding)
                )
                for name in identity_names
                for binding in (
                    _resolve_local_name_binding(name, node, parents),
                )
            )

            defects = []
            if not stage_tokens:
                defects.append("a literal logical-stage label")
            if missing_iterations:
                defects.append(
                    "the loop iteration value(s) "
                    + ", ".join(missing_iterations)
                )
            if missing_entities:
                defects.append(
                    "the current loop item(s) "
                    + ", ".join(missing_entities)
                )
            if not has_entity_identity:
                defects.append("a stable entity identity")
            if uses_unknown_identity:
                defects.append(
                    "a concrete entity identity instead of 'unknown'"
                )
            if defects:
                issues.append(ValidationIssue(
                    "loop_job_key",
                    "loop job key must contain " + "; ".join(defects)
                    + ". Use a shape such as "
                    "f'<stage>:{iteration}:{entity-id}' and reference the "
                    "current item when iterating over a collection",
                    line,
                ))

    def _collect_runtime_diagnostics(
        self, tree: ast.Module, issues: list[ValidationIssue],
    ) -> None:
        """Collect API incompatibilities and non-blocking robustness findings."""
        parent: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parent[child] = node

        critical_roles = self.allowed_roles
        for node in ast.walk(tree):
            if not _is_ctx_call(node, methods={"call"}):
                continue
            assert isinstance(node, ast.Call)
            role = (
                str(node.args[0].value)
                if node.args and isinstance(node.args[0], ast.Constant)
                else ""
            )
            if role in critical_roles and not _inside_try_body(node, parent):
                issues.append(ValidationIssue(
                    "unprotected_critical_call",
                    f"ctx.call({role!r}, ...) is not protected by try/except; "
                    "exception degradation is optional robustness",
                    getattr(node, "lineno", None),
                ))

        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            enclosing_try = parent.get(node)
            protected_ctx_methods = {
                item.func.attr
                for statement in enclosing_try.body
                for item in ast.walk(statement)
                if (
                    isinstance(enclosing_try, ast.Try)
                    and isinstance(item, ast.Call)
                    and isinstance(item.func, ast.Attribute)
                    and isinstance(item.func.value, ast.Name)
                    and item.func.value.id == "ctx"
                )
            } if isinstance(enclosing_try, ast.Try) else set()
            # Checkpoints and trace events are best-effort observability. Their
            # failure must not invalidate an otherwise usable answer path.
            if protected_ctx_methods and protected_ctx_methods <= {"checkpoint", "emit"}:
                continue
            meaningful = any(
                isinstance(item, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Return))
                or (
                    isinstance(item, ast.Call)
                    and isinstance(item.func, ast.Attribute)
                    and item.func.attr in {"append", "extend", "update", "add", "emit"}
                )
                for statement in node.body for item in ast.walk(statement)
            )
            if not meaningful:
                issues.append(ValidationIssue(
                    "empty_failure_degradation",
                    "except handler has no fallback assignment or best-effort return; "
                    "failure degradation is optional robustness",
                    getattr(node, "lineno", None),
                ))

        for node in ast.walk(tree):
            if not isinstance(node, ast.For):
                continue
            contains_job = any(
                _is_ctx_call(child, methods={"call", "spawn"})
                for child in ast.walk(node)
            )
            contains_budget_check = any(
                _is_ctx_call(child, methods={"remaining_budget"})
                for child in ast.walk(node)
            )
            # A nested loop inherits a budget gate from its enclosing work
            # loop. Missing gates are diagnostic only; Runtime owns the hard
            # run-wide limits.
            ancestor = parent.get(node)
            while not contains_budget_check and ancestor is not None:
                if isinstance(ancestor, ast.For):
                    contains_budget_check = any(
                        _is_ctx_call(child, methods={"remaining_budget"})
                        for child in ast.walk(ancestor)
                    )
                ancestor = parent.get(ancestor)
            iterations = _static_loop_iterations(node)
            if contains_job and iterations is not None and iterations > 1 and not contains_budget_check:
                issues.append(ValidationIssue(
                    "missing_loop_budget_check",
                    "multi-round agent loop has no ctx.remaining_budget() check; "
                    "per-loop budget gating is optional robustness",
                    getattr(node, "lineno", None),
                ))

        synchronous_ctx_methods = {"spawn", "load_checkpoint", "remaining_budget", "emit"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Await) and _is_ctx_call(
                node.value, methods=synchronous_ctx_methods,
            ):
                assert isinstance(node.value, ast.Call)
                method = node.value.func.attr  # type: ignore[union-attr]
                issues.append(ValidationIssue(
                    "await_sync_ctx_api",
                    f"ctx.{method} is synchronous; call it without await",
                    getattr(node, "lineno", None),
                ))

            if not _is_ctx_call(node, methods={"gather"}):
                continue
            assert isinstance(node, ast.Call)
            if not node.args or not isinstance(node.args[0], (ast.List, ast.Tuple)):
                continue
            direct_jobs = [
                item for item in node.args[0].elts
                if _is_ctx_call(item, methods={"call", "parallel", "map"})
            ]
            if direct_jobs:
                issues.append(ValidationIssue(
                    "gather_requires_handles",
                    "ctx.gather accepts handles returned by ctx.spawn; use ctx.parallel for direct awaitables",
                    getattr(node, "lineno", None),
                ))

    def _validate_call(self, node: ast.Call, issues: list[ValidationIssue]) -> None:
        line = getattr(node, "lineno", None)
        if isinstance(node.func, ast.Name):
            if node.func.id not in SAFE_BUILTINS and node.func.id not in self._local_callables:
                issues.append(ValidationIssue(
                    "call_target", f"call to {node.func.id!r} is not allowed", line,
                ))
            return
        if not isinstance(node.func, ast.Attribute):
            issues.append(ValidationIssue("call_target", "dynamic call target is not allowed", line))
            return
        direct_ctx_call = (
            isinstance(node.func.value, ast.Name) and node.func.value.id == "ctx"
        )
        method = node.func.attr
        if direct_ctx_call:
            if method not in self.allowed_ctx_methods:
                return
            if method in {"call", "spawn"}:
                if not node.args or not isinstance(node.args[0], ast.Constant):
                    issues.append(ValidationIssue(
                        "dynamic_role", f"ctx.{method} role must be a string literal", line,
                    ))
                elif node.args[0].value not in self.allowed_roles:
                    issues.append(ValidationIssue(
                        "unknown_role", f"unknown role {node.args[0].value!r}", line,
                    ))
            if method == "parallel" and len(node.args) > self.max_parallel_jobs:
                issues.append(ValidationIssue(
                    "parallel_limit", "ctx.parallel exceeds max_parallel_jobs", line,
                ))
            if method == "map":
                valid_function = (
                    len(node.args) >= 2
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id in self._local_callables
                )
                if not valid_function:
                    issues.append(ValidationIssue(
                        "map_requires_local_async_function",
                        "ctx.map(function, items) requires a locally defined async function; "
                        "for direct agent calls use ctx.parallel(ctx.call(...), ctx.call(...))",
                        line,
                    ))
        elif method not in SAFE_VALUE_METHODS:
            issues.append(ValidationIssue(
                "method_call", f"method {method!r} is not allowed", line,
            ))

    def _validate_loop(self, node: ast.For, issues: list[ValidationIssue]) -> None:
        line = getattr(node, "lineno", None)
        bound: int | None = None
        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
            if node.iter.func.id == "range" and node.iter.args:
                constants = [x.value for x in node.iter.args if isinstance(x, ast.Constant)]
                if len(constants) == len(node.iter.args) and all(isinstance(x, int) for x in constants):
                    try:
                        bound = len(range(*constants))
                    except (TypeError, ValueError):
                        bound = None
        elif isinstance(node.iter, (ast.List, ast.Tuple, ast.Set)):
            bound = len(node.iter.elts)
        elif isinstance(node.iter, ast.Subscript) and isinstance(node.iter.slice, ast.Slice):
            upper = node.iter.slice.upper
            lower = node.iter.slice.lower
            step = node.iter.slice.step
            if (
                isinstance(upper, ast.Constant) and isinstance(upper.value, int)
                and (lower is None or isinstance(lower, ast.Constant))
                and (step is None or isinstance(step, ast.Constant))
            ):
                start = 0 if lower is None else lower.value
                stride = 1 if step is None else step.value
                if isinstance(start, int) and isinstance(stride, int):
                    try:
                        bound = len(range(start, upper.value, stride))
                    except (TypeError, ValueError):
                        bound = None
        if bound is None or bound > self.max_loop_iterations:
            issues.append(ValidationIssue(
                "loop_bound",
                f"for loop has no static bound <= {self.max_loop_iterations}; "
                "loop bounding is optional robustness",
                line,
            ))

    def compile(self, source: str) -> Callable[[OrchestrationContext], Awaitable[Any]]:
        report = self.validate(source)
        if not report.accepted:
            raise GeneratedCodeValidationError(json.dumps(report.to_dict()))
        namespace: dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
        exec(compile(source, "<generated-orchestration>", "exec"), namespace, namespace)
        return namespace["solve"]


class PreflightSchemaError(ValueError):
    """Raised when a declared role result schema cannot drive preflight."""


_PREFLIGHT_MAX_ITEMS = 8


def _exception_details(exc: BaseException) -> dict[str, Any]:
    return {
        "error_type": type(exc).__name__,
        "message": str(exc),
    }


def _schema_value(
    schema: Any,
    *,
    path: str,
    depth: int = 0,
) -> Any:
    """Build a small deterministic value accepted by a common JSON Schema subset."""
    if depth > 20:
        raise PreflightSchemaError(
            f"schema nesting is too deep at {path}; cannot generate preflight value",
        )
    if schema is None or schema == {}:
        return "preflight"
    if not isinstance(schema, Mapping):
        raise PreflightSchemaError(
            f"result schema at {path} must be a JSON object",
        )

    if "const" in schema:
        return copy.deepcopy(schema["const"])
    enum = schema.get("enum")
    if enum is not None:
        if not isinstance(enum, list) or not enum:
            raise PreflightSchemaError(
                f"result schema enum at {path} must contain at least one value",
            )
        return copy.deepcopy(enum[0])
    if "default" in schema:
        return copy.deepcopy(schema["default"])

    for keyword in ("anyOf", "oneOf"):
        alternatives = schema.get(keyword)
        if alternatives is not None:
            if not isinstance(alternatives, list) or not alternatives:
                raise PreflightSchemaError(
                    f"result schema {keyword} at {path} must be a non-empty list",
                )
            return _schema_value(
                alternatives[0], path=f"{path}.{keyword}[0]", depth=depth + 1,
            )

    all_of = schema.get("allOf")
    if all_of is not None:
        if not isinstance(all_of, list) or not all_of:
            raise PreflightSchemaError(
                f"result schema allOf at {path} must be a non-empty list",
            )
        if not all(isinstance(item, Mapping) for item in all_of):
            raise PreflightSchemaError(
                f"result schema allOf at {path} must contain objects",
            )
        merged: dict[str, Any] = {}
        merged_properties: dict[str, Any] = {}
        merged_required: list[str] = []
        for item in all_of:
            assert isinstance(item, Mapping)
            merged.update(item)
            properties = item.get("properties", {})
            if properties:
                if not isinstance(properties, Mapping):
                    raise PreflightSchemaError(
                        f"result schema properties at {path} must be an object",
                    )
                merged_properties.update(properties)
            for name in item.get("required", []):
                if name not in merged_required:
                    merged_required.append(name)
        if merged_properties:
            merged["properties"] = merged_properties
        if merged_required:
            merged["required"] = merged_required
        return _schema_value(merged, path=path, depth=depth + 1)

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        if not schema_type:
            raise PreflightSchemaError(
                f"result schema type at {path} must not be empty",
            )
        schema_type = next(
            (item for item in schema_type if item != "null"), schema_type[0],
        )
    if schema_type is None:
        if "properties" in schema or "required" in schema:
            schema_type = "object"
        elif "items" in schema:
            schema_type = "array"
        else:
            # A schema containing only annotations accepts any JSON value. A
            # string is the least surprising value for the text-oriented DSL.
            schema_type = "string"

    if schema_type == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, Mapping):
            raise PreflightSchemaError(
                f"result schema properties at {path} must be an object",
            )
        if not isinstance(required, list) or any(
            not isinstance(name, str) for name in required
        ):
            raise PreflightSchemaError(
                f"result schema required at {path} must be a list of strings",
            )
        value: dict[str, Any] = {}
        for name in required:
            if name not in properties:
                raise PreflightSchemaError(
                    f"required field {name!r} is missing from properties at {path}",
                )
            value[name] = _schema_value(
                properties[name], path=f"{path}.properties.{name}", depth=depth + 1,
            )
        return value

    if schema_type == "array":
        min_items = schema.get("minItems", 1)
        max_items = schema.get("maxItems")
        if not isinstance(min_items, int) or min_items < 0:
            raise PreflightSchemaError(
                f"result schema minItems at {path} must be a non-negative integer",
            )
        if max_items is not None and (
            not isinstance(max_items, int) or max_items < 0
        ):
            raise PreflightSchemaError(
                f"result schema maxItems at {path} must be a non-negative integer",
            )
        if max_items is not None and min_items > max_items:
            raise PreflightSchemaError(
                f"result schema minItems exceeds maxItems at {path}",
            )
        count = max(1, min_items)
        if max_items == 0:
            count = 0
        if max_items is not None:
            count = min(count, max_items)
        if count > _PREFLIGHT_MAX_ITEMS:
            raise PreflightSchemaError(
                f"result schema array at {path} requires {count} items, "
                f"exceeding the preflight limit {_PREFLIGHT_MAX_ITEMS}",
            )
        item_schema = schema.get("items", {})
        return [
            _schema_value(item_schema, path=f"{path}.items", depth=depth + 1)
            for _ in range(count)
        ]

    if schema_type == "string":
        value = "preflight"
        min_length = schema.get("minLength", 0)
        max_length = schema.get("maxLength")
        if not isinstance(min_length, int) or min_length < 0:
            raise PreflightSchemaError(
                f"result schema minLength at {path} must be a non-negative integer",
            )
        if max_length is not None and (
            not isinstance(max_length, int) or max_length < 0
        ):
            raise PreflightSchemaError(
                f"result schema maxLength at {path} must be a non-negative integer",
            )
        value = value.ljust(min_length, "x")
        if max_length is not None:
            value = value[:max_length]
        return value

    if schema_type == "integer":
        minimum = schema.get("minimum", 0)
        maximum = schema.get("maximum")
        value = minimum if isinstance(minimum, int) and not isinstance(minimum, bool) else 0
        if maximum is not None and isinstance(maximum, int) and value > maximum:
            value = maximum
        return value

    if schema_type == "number":
        minimum = schema.get("minimum", 0.0)
        maximum = schema.get("maximum")
        value = minimum if isinstance(minimum, (int, float)) and not isinstance(minimum, bool) else 0.0
        if maximum is not None and isinstance(maximum, (int, float)) and value > maximum:
            value = maximum
        return value

    if schema_type == "boolean":
        return False
    if schema_type == "null":
        return None
    raise PreflightSchemaError(
        f"unsupported result schema type {schema_type!r} at {path}",
    )


def _role_result_schema(
    role_definitions: Mapping[str, Mapping[str, Any]], role: str,
) -> Any:
    try:
        definition = role_definitions[role]
    except KeyError as exc:
        raise PreflightSchemaError(
            f"role {role!r} has no definition in the preflight environment",
        ) from exc
    if not isinstance(definition, Mapping):
        raise PreflightSchemaError(
            f"role definition for {role!r} must be an object",
        )
    if "result_schema" not in definition:
        raise PreflightSchemaError(
            f"role {role!r} has no result_schema in the preflight environment",
        )
    return definition["result_schema"]


class _PreflightHandle:
    def __init__(self, value: Any) -> None:
        self._value = value

    async def result(self) -> Any:
        return self._value


class _PreflightContext:
    task = "preflight task"
    metadata: dict[str, Any] = {}
    run_id = "preflight"

    def __init__(self, role_definitions: Mapping[str, Mapping[str, Any]]) -> None:
        self._role_definitions = role_definitions

    def _result_for_role(self, role: Any) -> Any:
        if not isinstance(role, str):
            raise PreflightSchemaError(
                "preflight role argument must be a string",
            )
        schema = _role_result_schema(self._role_definitions, role)
        value = _schema_value(schema, path=f"roles.{role}.result_schema")
        if role == "evidence_storesubagent" and isinstance(value, dict):
            value["revision"] = 0  # Executor-assigned field, never model authority.
            value["resolved_refs"] = {}
        return value

    async def facts_index(self, *, keyword: str = "", statuses: list[str] | None = None,
                          rounds: list[int] | None = None) -> dict[str, Any]:
        return {"revision": 0, "fact_index": []}

    async def facts_query(
        self, *, keyword: str = "", fact_ids: list[str] | None = None,
        statuses: list[str] | None = None,
        rounds: list[int] | None = None,
    ) -> dict[str, Any]:
        # Match the runtime signature so invalid arguments fail during preflight.
        return {"revision": 0, "facts": []}

    async def call(self, role: Any, *_args: Any, **_kwargs: Any) -> Any:
        return self._result_for_role(role)

    def spawn(self, role: Any, *_args: Any, **_kwargs: Any) -> _PreflightHandle:
        return _PreflightHandle(self._result_for_role(role))

    async def gather(
        self,
        *handles: Any,
        failure_policy: FailurePolicy = "fail_fast",
        **_kwargs: Any,
    ) -> list[Any] | list[JobOutcome]:
        values = [await handle.result() for handle in handles]
        if failure_policy == "collect":
            return [
                JobOutcome(
                    job_id=f"preflight-{index}",
                    status=JobStatus.SUCCEEDED,
                    result=AgentResult(value=value),
                )
                for index, value in enumerate(values)
            ]
        if failure_policy != "fail_fast":
            raise ValueError(f"unknown failure policy: {failure_policy!r}")
        return values

    async def parallel(self, *awaitables: Awaitable[Any], **_kwargs: Any) -> list[Any]:
        return list(await asyncio.gather(*awaitables))

    async def map(self, fn: Any, items: Any, **_kwargs: Any) -> list[Any]:
        return [await fn(item) for item in items]

    async def cancel(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    async def checkpoint(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def load_checkpoint(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def remaining_budget(self) -> Any:
        class Snapshot:
            usage = {"jobs": 0}
            remaining = {"jobs": 10, "wall_time_seconds": 10.0}
        return Snapshot()

    def emit(self, *_args: Any, **_kwargs: Any) -> None:
        return None


async def preflight_solve(
    solve: Callable[..., Awaitable[Any]],
    role_definitions: Mapping[str, Mapping[str, Any]],
    timeout: float = 2.0,
    *,
    require_evolution_proposal: bool = False,
) -> None:
    value = await asyncio.wait_for(
        solve(_PreflightContext(role_definitions)), timeout=timeout,
    )
    if value is None:
        raise ValueError("solve returned None during preflight")
    if require_evolution_proposal:
        from agent_harness.orchestration.rolechange_evolution import validate_proposal_shape
        validate_proposal_shape(value)


class GeneratedSolvePlanner:
    """Generate, validate, repair, and persist one per-task solve function."""

    def __init__(
        self,
        llm: LLMClient,
        validator: GeneratedSolveValidator,
        *,
        model: str,
        max_repair_attempts: int = 3,
        max_tokens_per_call: int = 8192,
        review_generated_code: bool = True,
        facts_query: Callable[..., Awaitable[dict]] | None = None,
    ) -> None:
        self.llm = llm
        self.validator = validator
        self.model = model
        self.max_repair_attempts = max(0, max_repair_attempts)
        self.max_tokens_per_call = max_tokens_per_call
        self.review_generated_code = review_generated_code
        self.facts_query = facts_query

    async def generate(
        self,
        request: SolveGenerationRequestV1,
        *,
        artifact_dir: Path,
        budget: BudgetSpec,
    ) -> PlannerResult:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        request_path = artifact_dir / "generation_request.json"
        validation_path = artifact_dir / "validation_report.json"
        attempts_dir = artifact_dir / "attempts"
        plan_dir = artifact_dir / "plan"
        plan_attempts_dir = plan_dir / "attempts"
        attempts_dir.mkdir(parents=True, exist_ok=True)
        plan_attempts_dir.mkdir(parents=True, exist_ok=True)
        request_path.write_text(
            json.dumps(request.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        events: list[dict[str, Any]] = []
        usage = BudgetUsage()
        source = ""
        report = ValidationReport(
            False, (ValidationIssue("not_started", "generation not started"),),
        )
        generation_id = uuid.uuid4().hex
        total_attempts = 1 + self.max_repair_attempts
        natural_language_plan = ""
        natural_language_plan_path = plan_dir / "plan.md"
        code_generation_handoff_path = plan_dir / "code_generation_handoff.md"
        plan_hash = ""
        previous_plan = ""
        plan_critique = ""

        # A complete successful path normally needs four model calls: plan,
        # plan review, code generation, and code review. Experiments may skip
        # only the final semantic code review while retaining deterministic
        # validation and preflight.
        required_success_calls = 4 if self.review_generated_code else 3
        if not _planner_budget_available(
            budget, usage, self.max_tokens_per_call,
            required_calls=required_success_calls,
        ):
            report = ValidationReport(False, (ValidationIssue(
                "planner_budget",
                "insufficient reserved budget for planning, review, and code generation",
            ),))
            _write_json(validation_path, report.to_dict())
            _write_json(plan_dir / "failure.json", {
                "stage": "natural_language_planning",
                "error_type": "planner_budget",
                "message": report.issues[0].message,
            })
            return PlannerResult(
                generated=None, usage=usage, events=tuple(events),
                request_path=str(request_path), validation_path=str(validation_path),
                source_path=None, attempts_path=str(attempts_dir),
                natural_language_plan_path=None,
                natural_language_plan_artifacts_path=str(plan_dir),
                fallback_reason="could not reserve code-generation budget for both reviewed stages",
            )

        # Stage 1: generate and independently review the high-level plan. A
        # rejected plan is regenerated here; code generation cannot start yet.
        for plan_attempt in range(1, total_attempts + 1):
            plan_attempt_dir = plan_attempts_dir / f"attempt_{plan_attempt:02d}"
            plan_attempt_dir.mkdir(parents=True, exist_ok=True)
            if not _planner_budget_available(
                budget, usage, self.max_tokens_per_call,
                required_calls=required_success_calls,
            ):
                report = ValidationReport(False, (ValidationIssue(
                    "planner_budget",
                    "insufficient budget to regenerate and review the plan while reserving code generation",
                ),))
                _write_json(plan_attempt_dir / "validation.json", report.to_dict())
                break

            if self.facts_query is not None:
                snapshot = await self.facts_query(statuses=["active", "disputed"])
                usage = usage.add(BudgetUsage(tool_calls=1))
                context = dict(request.metadata.get("evolution_context", {}))
                context.update({
                    "fact_store_revision": snapshot["revision"],
                    "active_facts": [f for f in snapshot["facts"] if f["status"] == "active"],
                    "disputed_facts": [f for f in snapshot["facts"] if f["status"] == "disputed"],
                })
                request = replace(request, metadata={**request.metadata, "evolution_context": context})
                _write_json(plan_attempt_dir / "facts_read.json", snapshot)
                _write_json(request_path, request.to_dict())
                events.append(_generation_event("planner_facts_read", attempt=plan_attempt,
                                                revision=snapshot["revision"], facts=snapshot["facts"]))
            plan_prompt = _render_natural_language_plan_prompt(
                request, previous_plan=previous_plan, critique=plan_critique,
            )
            plan_system_prompt = evolution_system_prompt(_natural_language_planner_system_prompt(), request)
            for path, content in (
                (plan_dir / "system_prompt.md", plan_system_prompt),
                (plan_dir / "planner_prompt.md", plan_prompt),
                (plan_attempt_dir / "system_prompt.md", plan_system_prompt),
                (plan_attempt_dir / "planner_prompt.md", plan_prompt),
            ):
                path.write_text(content, encoding="utf-8")
            events.append(_generation_event(
                "natural_language_plan_started", attempt=plan_attempt,
                model=self.model,
                model_role="planner",
                model_input={
                    "system_prompt": plan_system_prompt,
                    "user_message": plan_prompt,
                },
            ))
            started = time.monotonic()
            try:
                response = await self.llm.chat(
                    [system_msg(plan_system_prompt), user_msg(plan_prompt)],
                    temperature=0.1,
                    max_tokens=min(self.max_tokens_per_call, 16384),
                )
            except Exception as exc:
                report = ValidationReport(False, (ValidationIssue(
                    "natural_language_plan_call", f"{type(exc).__name__}: {exc}",
                ),))
                _write_json(plan_attempt_dir / "validation.json", report.to_dict())
                _write_json(plan_attempt_dir / "failure.json", {
                    "stage": "natural_language_planning",
                    "error_type": type(exc).__name__, "message": str(exc),
                    "elapsed_seconds": time.monotonic() - started,
                })
                events.append(_generation_event(
                    "natural_language_plan_rejected", attempt=plan_attempt,
                    validation_errors=[item.to_dict() for item in report.issues],
                ))
                if is_transient_network(exc) and plan_attempt < total_attempts:
                    backoff_seconds = min(2 ** (plan_attempt - 1), 8)
                    events.append(_generation_event(
                        "natural_language_plan_retry_scheduled",
                        attempt=plan_attempt,
                        next_attempt=plan_attempt + 1,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        backoff_seconds=backoff_seconds,
                    ))
                    await asyncio.sleep(backoff_seconds)
                    continue
                break

            call_usage = _usage_from_response(
                response.usage, time.monotonic() - started,
            )
            usage = usage.add(call_usage)
            raw_output = text_of(response.content)
            candidate_plan = raw_output.strip()
            for path in (plan_dir / "raw_response.md", plan_attempt_dir / "raw_response.md"):
                path.write_text(raw_output, encoding="utf-8")
            _write_json(plan_dir / "usage.json", asdict(call_usage))
            _write_json(plan_attempt_dir / "usage.json", asdict(call_usage))

            if not candidate_plan:
                issue = ValidationIssue(
                    "natural_language_plan_empty",
                    "natural-language planner returned an empty plan",
                )
            else:
                issue = _validate_natural_language_plan(candidate_plan)
            if issue is not None:
                report = ValidationReport(False, (issue,))
                _write_json(plan_attempt_dir / "validation.json", report.to_dict())
                plan_critique = issue.message
                previous_plan = (
                    "" if issue.code == "natural_language_plan_contains_code"
                    else candidate_plan
                )
                events.append(_generation_event(
                    "natural_language_plan_rejected", attempt=plan_attempt,
                    model=self.model, model_output=raw_output,
                    usage=asdict(call_usage),
                    validation_errors=[issue.to_dict()],
                ))
                continue

            review_prompt = _render_plan_review_prompt(request, candidate_plan)
            review_system_prompt = evolution_system_prompt(_plan_reviewer_system_prompt(), request)
            review_dir = plan_attempt_dir / "review"
            review_dir.mkdir(parents=True, exist_ok=True)
            (review_dir / "system_prompt.md").write_text(
                review_system_prompt, encoding="utf-8",
            )
            (review_dir / "review_prompt.md").write_text(
                review_prompt, encoding="utf-8",
            )
            events.append(_generation_event(
                "natural_language_plan_review_started", attempt=plan_attempt,
                model=self.model,
                model_input={
                    "system_prompt": review_system_prompt,
                    "user_message": review_prompt,
                },
            ))
            review_started = time.monotonic()
            try:
                review_response = await self.llm.chat(
                    [system_msg(review_system_prompt), user_msg(review_prompt)],
                    temperature=0.0,
                    max_tokens=min(self.max_tokens_per_call, 2048),
                )
            except Exception as exc:
                report = ValidationReport(False, (ValidationIssue(
                    "natural_language_plan_review_call",
                    f"{type(exc).__name__}: {exc}",
                ),))
                _write_json(plan_attempt_dir / "validation.json", report.to_dict())
                _write_json(review_dir / "failure.json", {
                    "stage": "natural_language_plan_review",
                    "error_type": type(exc).__name__, "message": str(exc),
                    "elapsed_seconds": time.monotonic() - review_started,
                })
                events.append(_generation_event(
                    "natural_language_plan_review_failed", attempt=plan_attempt,
                    validation_errors=[item.to_dict() for item in report.issues],
                ))
                if is_transient_network(exc) and plan_attempt < total_attempts:
                    backoff_seconds = min(2 ** (plan_attempt - 1), 8)
                    events.append(_generation_event(
                        "natural_language_plan_review_retry_scheduled",
                        attempt=plan_attempt,
                        next_attempt=plan_attempt + 1,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        backoff_seconds=backoff_seconds,
                    ))
                    await asyncio.sleep(backoff_seconds)
                    continue
                break

            review_usage = _usage_from_response(
                review_response.usage, time.monotonic() - review_started,
            )
            usage = usage.add(review_usage)
            review_raw = text_of(review_response.content)
            verdict = _parse_review_verdict(review_raw)
            (review_dir / "raw_response.md").write_text(review_raw, encoding="utf-8")
            _write_json(review_dir / "usage.json", asdict(review_usage))
            _write_json(review_dir / "verdict.json", verdict.to_dict())
            events.append(_generation_event(
                "natural_language_plan_review_completed", attempt=plan_attempt,
                model=self.model, model_output=review_raw,
                verdict=verdict.to_dict(), usage=asdict(review_usage),
            ))

            if not verdict.accepted:
                repair_feedback = _review_repair_feedback(verdict)
                issue = ValidationIssue(
                    "natural_language_plan_review_rejected", repair_feedback,
                )
                report = ValidationReport(False, (issue,))
                _write_json(plan_attempt_dir / "validation.json", report.to_dict())
                previous_plan = candidate_plan
                plan_critique = repair_feedback
                events.append(_generation_event(
                    "natural_language_plan_rejected", attempt=plan_attempt,
                    review=verdict.to_dict(), validation_errors=[issue.to_dict()],
                ))
                continue

            natural_language_plan = candidate_plan
            plan_hash = hashlib.sha256(
                natural_language_plan.encode("utf-8"),
            ).hexdigest()
            natural_language_plan_path.write_text(
                natural_language_plan + "\n", encoding="utf-8",
            )
            code_generation_handoff_path.write_text(
                _render_code_generation_handoff(request, natural_language_plan) + "\n",
                encoding="utf-8",
            )
            _write_json(plan_dir / "review.json", verdict.to_dict())
            _write_json(
                plan_attempt_dir / "validation.json",
                {"accepted": True, "issues": [], "plan_hash": plan_hash},
            )
            events.append(_generation_event(
                "natural_language_plan_completed", attempt=plan_attempt,
                model=self.model, model_output=natural_language_plan,
                model_role="planner",
                plan_hash=plan_hash, review=verdict.to_dict(),
            ))
            break

        if not natural_language_plan:
            _write_json(validation_path, report.to_dict())
            _write_json(plan_dir / "failure.json", {
                "stage": "natural_language_plan_review",
                "error_type": report.issues[0].code,
                "message": report.issues[0].message,
            })
            return PlannerResult(
                generated=None, usage=usage, events=tuple(events),
                request_path=str(request_path), validation_path=str(validation_path),
                source_path=None, attempts_path=str(attempts_dir),
                natural_language_plan_path=None,
                natural_language_plan_artifacts_path=str(plan_dir),
                fallback_reason=(
                    "natural-language plan did not pass review: "
                    + report.issues[0].message
                ),
            )

        # Stage 2: generate code from the approved, immutable plan. Static or
        # semantic code failures repair only this stage and never regenerate it.
        repair_history: list[dict[str, Any]] = []
        repair_history_path = artifact_dir / "repair_history.json"
        _write_json(repair_history_path, {"attempts": repair_history})
        for attempt in range(1, total_attempts + 1):
            attempt_dir = attempts_dir / f"attempt_{attempt:02d}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            is_repair = bool(repair_history)
            attempt_record = RepairAttemptRecord(attempt=attempt)
            event_name = "code_repair_started" if is_repair else "orchestrator_generation_started"
            required_code_calls = 2 if self.review_generated_code else 1
            if not _planner_budget_available(
                budget, usage, self.max_tokens_per_call,
                required_calls=required_code_calls,
            ):
                report = ValidationReport(False, (ValidationIssue(
                    "planner_budget",
                    "insufficient reserved budget for code generation",
                ),))
                _write_json(attempt_dir / "validation.json", report.to_dict())
                attempt_record.generation_error = {
                    "error_type": "PlannerBudgetError",
                    "message": report.issues[0].message,
                }
                _append_repair_attempt(
                    repair_history, attempt_record, repair_history_path,
                )
                break

            prompt = _render_generation_prompt(
                request, natural_language_plan, repair_history,
            )
            system_prompt = evolution_system_prompt(_orchestrator_system_prompt(), request)
            events.append(_generation_event(
                event_name, attempt=attempt, model=self.model,
                model_role="orchestrator",
                plan_hash=plan_hash,
                model_input={
                    "system_prompt": system_prompt, "user_message": prompt,
                },
            ))
            (attempt_dir / "orchestrator_prompt.md").write_text(prompt, encoding="utf-8")
            started = time.monotonic()
            try:
                response = await self.llm.chat(
                    [system_msg(system_prompt), user_msg(prompt)],
                    temperature=0.1, max_tokens=self.max_tokens_per_call,
                )
            except Exception as exc:
                report = ValidationReport(False, (ValidationIssue(
                    "orchestrator_call", f"{type(exc).__name__}: {exc}",
                ),))
                attempt_record.generation_error = _exception_details(exc)
                _write_json(attempt_dir / "validation.json", report.to_dict())
                _write_json(attempt_dir / "failure.json", {
                    "stage": "orchestrator_call", "error_type": type(exc).__name__,
                    "message": str(exc),
                    "elapsed_seconds": time.monotonic() - started,
                })
                events.append(_generation_event(
                    "generated_code_rejected", attempt=attempt,
                    validation_errors=[item.to_dict() for item in report.issues],
                ))
                _append_repair_attempt(
                    repair_history, attempt_record, repair_history_path,
                )
                if is_transient_network(exc) and attempt < total_attempts:
                    backoff_seconds = min(2 ** (attempt - 1), 8)
                    events.append(_generation_event(
                        "orchestrator_generation_retry_scheduled",
                        attempt=attempt,
                        next_attempt=attempt + 1,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        backoff_seconds=backoff_seconds,
                    ))
                    await asyncio.sleep(backoff_seconds)
                    continue
                break

            call_usage = _usage_from_response(
                response.usage, time.monotonic() - started,
            )
            usage = usage.add(call_usage)
            raw_output = text_of(response.content)
            source = extract_source(raw_output)
            attempt_record.source = source
            (attempt_dir / "raw_response.md").write_text(raw_output, encoding="utf-8")
            (attempt_dir / "solve.py").write_text(source, encoding="utf-8")
            _write_json(attempt_dir / "usage.json", asdict(call_usage))
            events.append(_generation_event(
                "code_repair_completed" if is_repair else "orchestrator_generation_completed",
                attempt=attempt, model=self.model, plan_hash=plan_hash,
                model_role="orchestrator",
                model_output=raw_output, usage=asdict(call_usage),
            ))

            static_report = self.validator.validate(source)
            attempt_record.static_validation = static_report.to_dict()
            report = static_report
            if static_report.accepted:
                try:
                    solve = self.validator.compile(source)
                except Exception as exc:  # noqa: BLE001
                    attempt_record.compile_error = _exception_details(exc)
                    report = ValidationReport(False, (ValidationIssue(
                        "compile", f"{type(exc).__name__}: {exc}",
                    ),), warnings=static_report.warnings)
                else:
                    try:
                        await preflight_solve(
                            solve, request.environment.roles,
                            require_evolution_proposal=self.validator.require_evolution_proposal,
                        )
                    except Exception as exc:  # noqa: BLE001
                        attempt_record.preflight_error = _exception_details(exc)
                        report = ValidationReport(False, (ValidationIssue(
                            "preflight", f"{type(exc).__name__}: {exc}",
                        ),), warnings=static_report.warnings)
            if not report.accepted:
                _write_json(attempt_dir / "validation.json", report.to_dict())
                _append_repair_attempt(
                    repair_history, attempt_record, repair_history_path,
                )
                events.append(_generation_event(
                    "generated_code_rejected", attempt=attempt,
                    plan_hash=plan_hash, rejection_stage="static_validation",
                    validation_errors=[item.to_dict() for item in report.issues],
                ))
                continue

            if not self.review_generated_code:
                attempt_record.review_skipped = True
                _write_json(attempt_dir / "validation.json", report.to_dict())
                events.append(_generation_event(
                    "generated_code_review_skipped", attempt=attempt,
                    plan_hash=plan_hash,
                    reason="disabled by runner configuration",
                ))
                events.append(_generation_event(
                    "generated_code_validated", attempt=attempt,
                    source_hash=report.source_hash, plan_hash=plan_hash,
                    review_skipped=True,
                    validation_warnings=[item.to_dict() for item in report.warnings],
                ))
                source_path = artifact_dir / "solve.py"
                source_path.write_text(source, encoding="utf-8")
                generated = GeneratedSolveV1(
                    generation_id=generation_id, dsl_version=DSL_VERSION,
                    planner_model=self.model, source=source,
                    source_hash=report.source_hash, generation_attempt=attempt,
                    validation_status="accepted", validation_errors=[],
                    validation_warnings=[item.to_dict() for item in report.warnings],
                    natural_language_plan_hash=plan_hash,
                    natural_language_plan_path="plan/plan.md",
                )
                _append_repair_attempt(
                    repair_history, attempt_record, repair_history_path,
                )
                _write_json(validation_path, report.to_dict())
                (artifact_dir / "manifest.json").write_text(
                    json.dumps(generated.to_dict(), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                return PlannerResult(
                    generated=generated, usage=usage, events=tuple(events),
                    request_path=str(request_path), validation_path=str(validation_path),
                    source_path=str(source_path), attempts_path=str(attempts_dir),
                    natural_language_plan_path=str(natural_language_plan_path),
                    natural_language_plan_artifacts_path=str(plan_dir),
                )

            review_prompt = _render_code_review_prompt(
                request, natural_language_plan, source, repair_history,
            )
            review_system_prompt = evolution_system_prompt(_code_reviewer_system_prompt(), request)
            review_dir = attempt_dir / "review"
            review_dir.mkdir(parents=True, exist_ok=True)
            (review_dir / "system_prompt.md").write_text(
                review_system_prompt, encoding="utf-8",
            )
            (review_dir / "review_prompt.md").write_text(
                review_prompt, encoding="utf-8",
            )
            events.append(_generation_event(
                "generated_code_review_started", attempt=attempt,
                model=self.model, plan_hash=plan_hash,
                model_input={
                    "system_prompt": review_system_prompt,
                    "user_message": review_prompt,
                },
            ))
            review_started = time.monotonic()
            try:
                review_response = await self.llm.chat(
                    [system_msg(review_system_prompt), user_msg(review_prompt)],
                    temperature=0.0,
                    max_tokens=min(self.max_tokens_per_call, 2048),
                )
            except Exception as exc:
                report = ValidationReport(False, (ValidationIssue(
                    "generated_code_review_call", f"{type(exc).__name__}: {exc}",
                ),), warnings=report.warnings)
                attempt_record.review_error = _exception_details(exc)
                _write_json(attempt_dir / "validation.json", report.to_dict())
                _write_json(review_dir / "failure.json", {
                    "stage": "generated_code_review",
                    "error_type": type(exc).__name__, "message": str(exc),
                    "elapsed_seconds": time.monotonic() - review_started,
                })
                events.append(_generation_event(
                    "generated_code_review_failed", attempt=attempt,
                    validation_errors=[item.to_dict() for item in report.issues],
                ))
                _append_repair_attempt(
                    repair_history, attempt_record, repair_history_path,
                )
                if is_transient_network(exc) and attempt < total_attempts:
                    backoff_seconds = min(2 ** (attempt - 1), 8)
                    events.append(_generation_event(
                        "generated_code_review_retry_scheduled",
                        attempt=attempt,
                        next_attempt=attempt + 1,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        backoff_seconds=backoff_seconds,
                    ))
                    await asyncio.sleep(backoff_seconds)
                    continue
                break

            review_usage = _usage_from_response(
                review_response.usage, time.monotonic() - review_started,
            )
            usage = usage.add(review_usage)
            review_raw = text_of(review_response.content)
            verdict = _parse_review_verdict(review_raw)
            (review_dir / "raw_response.md").write_text(review_raw, encoding="utf-8")
            _write_json(review_dir / "usage.json", asdict(review_usage))
            _write_json(review_dir / "verdict.json", verdict.to_dict())
            attempt_record.review = verdict.to_dict()
            events.append(_generation_event(
                "generated_code_review_completed", attempt=attempt,
                model=self.model, plan_hash=plan_hash,
                model_output=review_raw, verdict=verdict.to_dict(),
                usage=asdict(review_usage),
            ))
            if not verdict.accepted:
                repair_feedback = _review_repair_feedback(verdict)
                issue = ValidationIssue("generated_code_review_rejected", repair_feedback)
                report = ValidationReport(False, (issue,), warnings=report.warnings)
                _write_json(attempt_dir / "validation.json", report.to_dict())
                _append_repair_attempt(
                    repair_history, attempt_record, repair_history_path,
                )
                events.append(_generation_event(
                    "generated_code_rejected", attempt=attempt,
                    plan_hash=plan_hash, rejection_stage="semantic_review",
                    review=verdict.to_dict(), validation_errors=[issue.to_dict()],
                ))
                continue

            _write_json(attempt_dir / "validation.json", report.to_dict())
            _append_repair_attempt(
                repair_history, attempt_record, repair_history_path,
            )
            events.append(_generation_event(
                "generated_code_validated", attempt=attempt,
                source_hash=report.source_hash, plan_hash=plan_hash,
                review=verdict.to_dict(),
                validation_warnings=[item.to_dict() for item in report.warnings],
            ))
            source_path = artifact_dir / "solve.py"
            source_path.write_text(source, encoding="utf-8")
            generated = GeneratedSolveV1(
                generation_id=generation_id, dsl_version=DSL_VERSION,
                planner_model=self.model, source=source,
                source_hash=report.source_hash, generation_attempt=attempt,
                validation_status="accepted", validation_errors=[],
                validation_warnings=[item.to_dict() for item in report.warnings],
                natural_language_plan_hash=plan_hash,
                natural_language_plan_path="plan/plan.md",
            )
            _write_json(validation_path, report.to_dict())
            (artifact_dir / "manifest.json").write_text(
                json.dumps(generated.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return PlannerResult(
                generated=generated, usage=usage, events=tuple(events),
                request_path=str(request_path), validation_path=str(validation_path),
                source_path=str(source_path), attempts_path=str(attempts_dir),
                natural_language_plan_path=str(natural_language_plan_path),
                natural_language_plan_artifacts_path=str(plan_dir),
            )

        _write_json(validation_path, report.to_dict())
        return PlannerResult(
            generated=None, usage=usage, events=tuple(events),
            request_path=str(request_path), validation_path=str(validation_path),
            source_path=None, attempts_path=str(attempts_dir),
            natural_language_plan_path=str(natural_language_plan_path),
            natural_language_plan_artifacts_path=str(plan_dir),
            fallback_reason="generated code did not pass validation and review",
        )


def analyze_generated_source(source: str) -> dict[str, Any]:
    """Return stable structural features for later workflow evolution."""
    if not source.strip():
        return {"available": False}
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {"available": True, "syntax_valid": False, "syntax_error": exc.msg}
    role_calls: dict[str, int] = {}
    ctx_calls: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if _attribute_root(node.func) != "ctx":
            continue
        method = node.func.attr
        ctx_calls[method] = ctx_calls.get(method, 0) + 1
        if method in {"call", "spawn"} and node.args and isinstance(node.args[0], ast.Constant):
            role = str(node.args[0].value)
            role_calls[role] = role_calls.get(role, 0) + 1
    return {
        "available": True,
        "syntax_valid": True,
        "source_hash": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "line_count": len(source.splitlines()),
        "ctx_calls": ctx_calls,
        "role_calls": role_calls,
        "bounded_for_loops": sum(isinstance(node, ast.For) for node in ast.walk(tree)),
        "conditionals": sum(isinstance(node, ast.If) for node in ast.walk(tree)),
        "exception_handlers": sum(isinstance(node, ast.ExceptHandler) for node in ast.walk(tree)),
    }


def write_generation_research_report(
    artifact_dir: Path,
    *,
    question_id: str,
    predicted_answer: str,
    ground_truth: str,
    is_correct: bool,
    strict_exact_match: bool,
    unordered_team_match: bool,
    judge_method: str,
    usage: dict[str, Any],
    budget_violations: list[str],
    pipeline_error: str | None,
    fallback_reason: str | None,
) -> Path:
    """Persist a non-deploying success/failure diagnosis beside every script."""
    source_path = artifact_dir / "solve.py"
    source = source_path.read_text(encoding="utf-8") if source_path.exists() else ""
    validation_path = artifact_dir / "validation_report.json"
    validation = (
        json.loads(validation_path.read_text(encoding="utf-8"))
        if validation_path.exists() else {"accepted": False, "issues": []}
    )
    categories: list[str] = []
    if not source:
        categories.append("generation_failed")
    if not validation.get("accepted"):
        categories.append("validation_failed")
        categories.extend(
            f"validation:{item.get('code', 'unknown')}"
            for item in validation.get("issues", [])
        )
    if fallback_reason:
        categories.append("fallback_used")
        if "execution failed" in fallback_reason:
            categories.append("generated_execution_failed")
    if pipeline_error:
        categories.append("pipeline_failed")
    if budget_violations:
        categories.append("budget_violation")
    if source and validation.get("accepted") and not is_correct:
        categories.append("answer_incorrect")
    if is_correct and not fallback_reason and not budget_violations and not pipeline_error:
        classification = "promising"
    elif source:
        classification = "failed_or_inconclusive"
    else:
        classification = "generation_failure"
    report = {
        "schema_version": "generated_script_research_v1",
        "question_id": question_id,
        "classification": classification,
        "eligible_for_offline_evolution": classification == "promising",
        "auto_deploy": False,
        "failure_categories": sorted(set(categories)),
        "source_analysis": analyze_generated_source(source),
        "validation": validation,
        "outcome": {
            "predicted_answer": predicted_answer,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
            "strict_exact_match": strict_exact_match,
            "unordered_team_match": unordered_team_match,
            "judge_method": judge_method,
            "pipeline_error": pipeline_error,
            "fallback_reason": fallback_reason,
        },
        "usage": usage,
        "budget_violations": budget_violations,
        "artifacts": {
            "natural_language_plan": (
                "plan/plan.md" if (artifact_dir / "plan" / "plan.md").exists()
                else None
            ),
            "natural_language_plan_artifacts": "plan",
            "attempts": "attempts",
            "repair_history": (
                "repair_history.json"
                if (artifact_dir / "repair_history.json").exists()
                else None
            ),
            "final_source": "solve.py" if source else None,
            "generation_request": "generation_request.json",
            "validation_report": "validation_report.json",
            "manifest": "manifest.json" if (artifact_dir / "manifest.json").exists() else None,
        },
    }
    target = artifact_dir / "research_report.json"
    _write_json(target, report)
    return target


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _append_repair_attempt(
    history: list[dict[str, Any]],
    record: RepairAttemptRecord,
    history_path: Path,
) -> None:
    """Append one completed attempt and persist the append-only history."""
    history.append(record.to_dict())
    _write_json(history_path, {"attempts": history})


def _render_repair_history(history: Sequence[Mapping[str, Any]]) -> str:
    """Render every prior code attempt and its paired diagnostics for an LLM."""
    if not history:
        return ""
    sections = [
        "## Repair history",
        "The following attempts happened before the current generation. "
        "Preserve constraints that earlier attempts already satisfied. "
        "Make the smallest change that fixes the current material defect.",
    ]
    for entry in history:
        attempt = entry.get("attempt", "unknown")
        source = str(entry.get("source") or "")
        diagnostics = {
            key: value
            for key, value in entry.items()
            if key != "source"
        }
        sections.extend([
            f"### Attempt {attempt}",
            "#### Generated solve.py\n\n```python\n"
            + source.rstrip()
            + "\n```",
            "#### Validation and reviewer record\n\n```json\n"
            + json.dumps(diagnostics, ensure_ascii=False, indent=2, default=str)
            + "\n```",
        ])
    return "\n\n".join(sections)


def extract_source(raw: str) -> str:
    """Accept plain source and conservatively unwrap one Markdown code block."""
    value = raw.strip()
    if value.startswith("```") and value.endswith("```"):
        lines = value.splitlines()
        if len(lines) >= 3 and lines[0].strip() in {"```", "```python", "```py"}:
            value = "\n".join(lines[1:-1]).strip()
    marker = value.find("async def solve(ctx)")
    if marker > 0 and not value[:marker].strip().startswith("#"):
        value = value[marker:]
    return value.rstrip() + "\n" if value else ""


def build_environment_manifest(
    *, roles: dict[str, Any], tools: dict[str, Any], budget: BudgetSpec,
) -> EnvironmentManifestV1:
    role_data = {
        role_id: {
            "description": str(getattr(role, "description", "")),
            "responsibilities": str(getattr(role, "system_prompt", "")),
            "planner_usage": _planner_usage_for_role(role),
            "result_guidance": _result_guidance_for_role(role),
            "allowed_tools": list(getattr(role, "allowed_tools", []) or []),
            "result_schema": getattr(role, "result_schema", None) or {},
            "default_max_turns": (getattr(role, "metadata", {}) or {}).get("max_turns"),
        }
        for role_id, role in roles.items()
    }
    tool_data = {}
    for name, tool in tools.items():
        tool_data[name] = {
            "description": str(getattr(tool, "description", "")),
            "parameters": getattr(tool, "parameters", None) or getattr(tool, "input_schema", None) or {},
        }
    return EnvironmentManifestV1(
        runtime_version="1.1", dsl_version=DSL_VERSION,
        roles=role_data, tools=tool_data,
        ctx_api=tuple(sorted(ALLOWED_CTX_METHODS | (
            {"facts_query", "facts_index"} if any("facts_index" in role["allowed_tools"] for role in role_data.values()) else set()
        ))), budget=asdict(budget),
    )


def _planner_usage_for_role(role: Any) -> str:
    """Describe role selection without duplicating the role catalog."""
    tools = list(getattr(role, "allowed_tools", []) or [])
    tool_note = (
        " It may use only these tools: " + ", ".join(tools) + "."
        if tools
        else " It has no tools."
    )
    return (
        "Use when the delegated subtask directly matches this role's stated "
        "responsibilities. Pass the original task, exact subgoal, all relevant "
        "prior results, constraints, and required output format."
        + tool_note
    )


def _result_guidance_for_role(role: Any) -> str:
    """Derive result handling from the AgentDefinition result contract."""
    schema = getattr(role, "result_schema", None)
    if schema:
        return (
            "Returns a structured value conforming exactly to the declared JSON "
            "Schema below. Read only declared fields and preserve the value for "
            "the next orchestration stage."
        )
    return (
        "Returns plain text. Preserve it as text for the next orchestration stage; "
        "do not read dictionary fields from it."
    )


def build_generation_request(
    task: str, metadata: dict[str, Any], environment: EnvironmentManifestV1,
) -> SolveGenerationRequestV1:
    request = SolveGenerationRequestV1(
        task=task, metadata=dict(metadata), environment=environment,
        # The approved natural-language plan is the only source of workflow
        # semantics.  Keep this compatibility field empty so complete solve
        # templates cannot steer code generation; the ctx API appendix below
        # supplies interface usage cases instead.
        templates=(),
        failure_lessons=_failure_lessons(),
        generation_constraints=(
            "Return exactly one async def solve(ctx), without imports or markdown.",
            "Use only declared roles and ctx APIs.",
            "While loops, async-for loops, and comprehensions are unsupported by this DSL. Ordinary for loops do not need a static <=8 bound.",
            "ctx.gather(*handles) accepts only ctx.spawn handles; use ctx.parallel(ctx.call(...), ctx.call(...)) for direct calls.",
            "ctx.map(function, items) requires a locally defined async function and accepts no role name or job kwargs.",
            "ctx.spawn, ctx.load_checkpoint, ctx.remaining_budget, and ctx.emit are synchronous and must not be awaited.",
            LOOP_JOB_KEY_GUIDANCE,
            "Return a concise answer on the normal planned execution path.",
            "Choose roles only when their declared responsibilities match a stage in the approved plan; no role is mandatory for every task.",
            "Handle every role result according to its declared contract. Structured results must conform to their JSON Schema; roles without a schema return text.",
            "Do not read fields absent from a role's declared result schema, and route structured decisions according to the declared fields and meanings.",
            "Do not hard-code entities taken from unrelated reference material unless they occur in the current task.",
            "Robustness is not an acceptance requirement in this phase. Do not add or repair try/except fallback paths, static for-loop bounds, per-loop ctx.remaining_budget() checks, checkpointing, partial-result preservation, retries, tracing, or logging unless the approved plan itself requires them.",
            "Treat the approved plan as the source of orchestration semantics. Use only the ctx API documentation and usage cases for implementation mechanics.",
        ),
    )

    from agent_harness.orchestration.rolechange_evolution import adapt_generation_request
    return adapt_generation_request(request)


def _failure_lessons() -> tuple[dict[str, Any], ...]:
    return (
        {
            "lesson_id": "task_specific_orchestration_v1",
            "category": "planner_quality",
            "observed_failure": "the generated workflow did not reflect the current task's uncertainty, evidence routes, or dependency structure",
            "required_behavior": [
                "identify task-specific uncertainty and evidence routes first",
                "derive role selection, dependencies and parallelism from the current task",
                "treat the approved plan as the source of orchestration semantics and use only API documentation for implementation mechanics",
            ],
        },
    )


def _attribute_root(node: ast.Attribute) -> str | None:
    value: ast.AST = node
    while isinstance(value, ast.Attribute):
        value = value.value
    return value.id if isinstance(value, ast.Name) else None


def _is_ctx_call(node: ast.AST, *, methods: set[str]) -> bool:
    return bool(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "ctx"
        and node.func.attr in methods
    )


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _job_key_expression(node: ast.Call) -> ast.AST | None:
    return next(
        (keyword.value for keyword in node.keywords if keyword.arg == "key"),
        None,
    )


def _resolve_local_key_expression(
    expression: ast.AST | None,
    call: ast.Call,
    parents: dict[ast.AST, ast.AST],
) -> ast.AST | None:
    """Resolve a simple local ``job_key = f'...'`` binding for validation."""
    if not isinstance(expression, ast.Name):
        return expression

    return _resolve_local_name_binding(expression.id, call, parents) or expression


def _resolve_local_name_binding(
    name: str,
    call: ast.Call,
    parents: dict[ast.AST, ast.AST],
) -> ast.AST | None:
    """Return the nearest preceding simple assignment in the same function."""

    scope: ast.AST = call
    while scope in parents:
        scope = parents[scope]
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            break
    call_line = getattr(call, "lineno", 0)
    candidates: list[tuple[int, ast.AST]] = []
    for node in ast.walk(scope):
        line = getattr(node, "lineno", 0)
        if not line or line >= call_line:
            continue
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name)
                and target.id == name
                for target in node.targets
            ):
                candidates.append((line, node.value))
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and node.value is not None
        ):
            candidates.append((line, node.value))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _job_request_shape(node: ast.Call) -> str:
    values = [
        node.func.attr,
        *(ast.dump(item, include_attributes=False) for item in node.args),
        *(
            f"{keyword.arg}={ast.dump(keyword.value, include_attributes=False)}"
            for keyword in node.keywords if keyword.arg != "key"
        ),
    ]
    return "|".join(values)


def _enclosing_for_loops(
    node: ast.AST, parents: dict[ast.AST, ast.AST],
) -> tuple[ast.For, ...]:
    loops: list[ast.For] = []
    current = parents.get(node)
    while current is not None:
        if isinstance(current, ast.For):
            loops.append(current)
        current = parents.get(current)
    return tuple(loops)


def _target_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return {
            name
            for item in target.elts
            for name in _target_names(item)
        }
    return set()


def _loop_iteration_names(loop: ast.For) -> set[str]:
    if (
        isinstance(loop.iter, ast.Call)
        and isinstance(loop.iter.func, ast.Name)
        and loop.iter.func.id == "enumerate"
        and isinstance(loop.target, (ast.Tuple, ast.List))
        and loop.target.elts
    ):
        return _target_names(loop.target.elts[0])
    return _target_names(loop.target)


def _loop_entity_names(loop: ast.For) -> set[str]:
    """Return names that identify the current item for a loop job key."""
    if (
        isinstance(loop.iter, ast.Call)
        and isinstance(loop.iter.func, ast.Name)
        and loop.iter.func.id == "enumerate"
        and isinstance(loop.target, (ast.Tuple, ast.List))
        and len(loop.target.elts) >= 2
    ):
        return _target_names(loop.target.elts[1])
    if (
        isinstance(loop.iter, ast.Call)
        and isinstance(loop.iter.func, ast.Name)
        and loop.iter.func.id == "range"
    ):
        return set()
    return _target_names(loop.target)


def _key_references_names(
    expression: ast.AST,
    required_names: set[str],
    call: ast.Call,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    """Check direct names and simple local aliases derived from loop items."""
    expression_names = _key_expression_names(expression)
    if expression_names.intersection(required_names):
        return True
    return any(
        _local_binding_depends_on_names(
            name, required_names, call, parents, seen=set(),
        )
        for name in expression_names
        if name not in required_names
    )


def _local_binding_depends_on_names(
    name: str,
    required_names: set[str],
    call: ast.Call,
    parents: dict[ast.AST, ast.AST],
    *,
    seen: set[str],
) -> bool:
    if name in required_names:
        return True
    if name in seen or name in SAFE_BUILTINS or name == "ctx":
        return False
    seen.add(name)
    binding = _resolve_local_name_binding(name, call, parents)
    if binding is None:
        return False
    binding_names = _key_expression_names(binding)
    return bool(binding_names.intersection(required_names)) or any(
        _local_binding_depends_on_names(
            dependency, required_names, call, parents, seen=seen,
        )
        for dependency in binding_names
    )


def _key_expression_names(expression: ast.AST) -> set[str]:
    return {
        node.id for node in ast.walk(expression)
        if isinstance(node, ast.Name)
    }


def _key_literal_tokens(expression: ast.AST) -> list[str]:
    tokens: list[str] = []
    for node in ast.walk(expression):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            tokens.extend(re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", node.value))
    return tokens


def _is_none_literal(expression: ast.AST) -> bool:
    return isinstance(expression, ast.Constant) and expression.value is None


def _inside_try_body(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    child = node
    current = parents.get(child)
    while current is not None:
        if isinstance(current, ast.Try):
            return any(_is_descendant(child, statement) for statement in current.body)
        child = current
        current = parents.get(current)
    return False


def _is_descendant(node: ast.AST, ancestor: ast.AST) -> bool:
    return node is ancestor or any(item is node for item in ast.walk(ancestor))


def _static_loop_iterations(node: ast.For) -> int | None:
    if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
        if node.iter.func.id == "range" and node.iter.args:
            constants = [item.value for item in node.iter.args if isinstance(item, ast.Constant)]
            if len(constants) == len(node.iter.args) and all(isinstance(item, int) for item in constants):
                try:
                    return len(range(*constants))
                except (TypeError, ValueError):
                    return None
    if isinstance(node.iter, (ast.List, ast.Tuple, ast.Set)):
        return len(node.iter.elts)
    return None


def _usage_from_response(raw: dict[str, int], elapsed: float) -> BudgetUsage:
    tokens = int(raw.get("total_tokens", 0) or (
        raw.get("prompt_tokens", raw.get("input_tokens", 0))
        + raw.get("completion_tokens", raw.get("output_tokens", 0))
    ))
    return BudgetUsage(tokens=tokens, llm_calls=1, duration_seconds=elapsed)


def _validate_natural_language_plan(plan: str) -> ValidationIssue | None:
    lowered = plan.casefold()
    if (
        "async def solve" in lowered
        or "```" in plan
        or re.search(r"\b(?:await\s+ctx\.|ctx\.[a-z_]\w*)", plan, re.IGNORECASE)
        or re.search(
            r"^\s*(?:async\s+def|def|class|import|from\s+\S+\s+import)\b",
            plan,
            re.IGNORECASE | re.MULTILINE,
        )
    ):
        return ValidationIssue(
            "natural_language_plan_contains_code",
            "natural-language planning stage must not return code or implementation syntax",
        )
    return None


def _parse_review_verdict(raw: str) -> ReviewVerdict:
    text = raw.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```", text, re.IGNORECASE | re.DOTALL,
    )
    if fenced is not None:
        text = fenced.group(1).strip()
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return ReviewVerdict(
            False,
            "Reviewer returned malformed JSON.",
            ("Return an object with accepted, critique, issues, and suggested_changes fields.",),
            ("The reviewer response was not valid JSON.",),
        )
    if not isinstance(value, dict) or not isinstance(value.get("accepted"), bool):
        return ReviewVerdict(
            False,
            "Reviewer verdict must contain a boolean accepted field.",
            ("Return accepted as true or false in the required review object.",),
            ("The accepted field is missing or is not boolean.",),
        )
    critique = value.get("critique", "")
    if not isinstance(critique, str):
        return ReviewVerdict(
            False,
            "Reviewer critique must be a string.",
            ("Replace critique with a concise string describing the defect.",),
            ("The critique field is not a string.",),
        )
    raw_issues = value.get("issues", [])
    if not isinstance(raw_issues, list) or any(
        not isinstance(item, str) for item in raw_issues
    ):
        return ReviewVerdict(
            False,
            "Reviewer issues must be a list of strings.",
            ("Return issues as a JSON list containing every material defect found.",),
            ("The issues field is not a list of strings.",),
        )
    review_issues = tuple(item.strip() for item in raw_issues if item.strip())
    raw_changes = value.get("suggested_changes", [])
    if not isinstance(raw_changes, list) or any(
        not isinstance(item, str) for item in raw_changes
    ):
        return ReviewVerdict(
            False,
            "Reviewer suggested_changes must be a list of strings.",
            ("Return suggested_changes as a JSON list containing concrete repair steps.",),
            review_issues or ("The suggested_changes field is not a list of strings.",),
        )
    suggested_changes = tuple(
        item.strip() for item in raw_changes if item.strip()
    )
    if not value["accepted"] and not critique.strip():
        return ReviewVerdict(
            False,
            "Reviewer rejected the artifact without an actionable critique.",
            suggested_changes or ("Describe the concrete defect that caused rejection.",),
            review_issues or ("The rejection has no stated reason.",),
        )
    if not value["accepted"] and not review_issues:
        # Older reviewer responses had only a summary critique. Preserve
        # compatibility while normalizing every rejection to an issue list.
        review_issues = (critique.strip(),)
    if not value["accepted"] and not suggested_changes:
        suggested_changes = review_issues
    if value["accepted"]:
        suggested_changes = ()
        review_issues = ()
    return ReviewVerdict(
        value["accepted"], critique.strip(), suggested_changes, review_issues,
    )


def _review_repair_feedback(verdict: ReviewVerdict) -> str:
    """Render reviewer diagnosis and concrete repair steps for the next attempt."""
    sections = [verdict.critique.strip()]
    if verdict.issues:
        issues = "\n".join(
            f"{index}. {issue}"
            for index, issue in enumerate(verdict.issues, start=1)
        )
        sections.append("Issues found:\n" + issues)
    if verdict.suggested_changes:
        changes = "\n".join(
            f"{index}. {change}"
            for index, change in enumerate(verdict.suggested_changes, start=1)
        )
        sections.append("Suggested changes:\n" + changes)
    return "\n\n".join(section for section in sections if section)


def _planner_budget_available(
    budget: BudgetSpec,
    usage: BudgetUsage,
    estimate: int,
    *,
    required_calls: int = 1,
) -> bool:
    if (
        budget.max_llm_calls is not None
        and usage.llm_calls + required_calls > budget.max_llm_calls
    ):
        return False
    if budget.max_tokens is None:
        return True
    execution_reserve = max(4096, int(budget.max_tokens * 0.50))
    planning_estimate = min(estimate, 4096) * required_calls
    return usage.tokens + planning_estimate <= budget.max_tokens - execution_reserve


def _generation_event(event_type: str, **payload: Any) -> dict[str, Any]:
    return {
        "timestamp": time.time(), "stage": "code_generation",
        "event_type": event_type, "payload": payload,
    }


def _orchestrator_system_prompt() -> str:
    return (
        "You are the Orchestrator and orchestration-program author for an agent runtime. "
        "Your sole job is to translate the current task into a safe, task-specific "
        "Python control-flow program; you must not research or answer the task yourself. "
        "The program should decide which declared specialist roles to call, what precise "
        "instructions and context each role needs, which calls can run concurrently, how "
        "intermediate evidence should be checked or synthesized, and how to degrade safely "
        "when a call fails or the budget is nearly exhausted. Treat each role as an "
        "independent agent with the responsibilities, tools, and output contract stated in "
        "the role catalog. Invoke roles only through the declared ctx API and its documented "
        "calling forms. Never invent roles, tools, "
        "runtime methods, task facts, candidate answers, or output fields. Treat every role as optional "
        "and select it only when its declared responsibilities match the approved plan. Read structured "
        "results only through fields present in their declared schemas, and preserve plain-text results as "
        "text. Follow every DSL, validation, "
        "budget, and safety constraint in the user message. Return only executable source "
        "code beginning with async def solve(ctx), with no markdown or explanation."
    )


def _natural_language_planner_system_prompt() -> str:
    return (
        "You are the high-level coordinator for a team of agents. Produce a concise, "
        "task-specific plan in natural language. The Planner itself is responsible "
        "for decomposing the original task into stages and assigning work to agents. "
        "The plan should explain only when "
        "each available agent should be called, the high-level task it should receive, "
        "what earlier result it depends on, which independent agent calls may run in "
        "parallel, and when verification or final synthesis is actually useful. Select agents only from "
        "the supplied catalog and only when their stated responsibilities match the task; no agent is "
        "mandatory. Keep a "
        "big-picture view and do not solve the task. Do not include implementation "
        "mechanics such as programming languages, pseudocode, APIs, function or method "
        "names, variables, data structures, schemas, tool-call syntax, exception "
        "handling, or numerical resource calculations."
    )


def _plan_reviewer_system_prompt() -> str:
    return (
        "You are the independent reviewer of a high-level multi-agent orchestration plan. "
        "Check whether the plan interprets the task relationships correctly, avoids unsupported "
        "assumptions, assigns only available agents, uses sensible ordering and parallelism, and "
        "uses verification or synthesis only when justified by the task. Reject any plan that invents an "
        "agent, assigns work outside an agent's declared responsibilities or tools, assumes undeclared "
        "result fields, or requires every available agent without a task-specific reason. Also reject plans "
        "that drift "
        "into code, APIs, variables, schemas, detailed search commands, or other implementation "
        "mechanics. Before deciding, complete a full pass over every acceptance criterion and collect "
        "all material defects; do not stop reviewing after the first defect. Do not solve the task and "
        "do not rewrite the full plan. Return only JSON with exactly four fields: accepted (boolean), "
        "critique (string summary), issues (array of strings containing every material defect found), "
        "and suggested_changes (array of strings with a concrete repair for each issue in the same "
        "order). When rejecting, report all material issues and all corresponding changes in one "
        "response. When accepting, return an empty critique and empty arrays."
    )


def _code_reviewer_system_prompt() -> str:
    return (
        "You are the independent compatibility reviewer of a generated orchestration program. Static "
        "syntax, sandbox validation, compilation, and preflight execution have already passed. Apply "
        "a deliberately narrow standard: accept when the program can run with the declared API/result "
        "contracts and faithfully implements the approved high-level plan. Completely ignore robustness, "
        "fault tolerance, best-effort quality, defensive programming, recovery behavior, checkpointing, "
        "logging, preservation of partial results, and how gracefully exceptional or incomplete agent "
        "outputs are handled. Do not reject code merely because it may perform poorly or return a weak "
        "answer in edge cases. Reject only a concrete incompatibility that makes the normal planned path "
        "unable to run, or a material mismatch with an approved stage, dependency, role, parallelism, "
        "verification step, or synthesis step. Check loop job keys strictly: every ctx.call and ctx.spawn "
        "inside a for loop must identify the logical stage, the actual loop value, and a stable entity; "
        "for collection loops the current item must appear in the key, and an index must be bound with "
        "enumerate when used. Never accept a repair that removes an already-correct key component. Before "
        "deciding, complete a full pass over every acceptance criterion and collect all material "
        "defects; do not stop reviewing after the first defect. Do not solve the underlying factual "
        "task and do not return replacement code. Return only JSON with exactly four fields: accepted "
        "(boolean), critique (string summary), issues (array of strings containing every material "
        "defect found), and suggested_changes (array of strings with a concrete repair for each issue "
        "in the same order). Report all material issues and corresponding changes in one response. "
        "When accepting, return an empty critique and empty arrays."
    )


def _render_natural_language_plan_prompt(
    request: SolveGenerationRequestV1,
    *,
    previous_plan: str = "",
    critique: str = "",
) -> str:
    document = request.to_dict()
    environment = document.get("environment", {})
    task = _strip_leading_markdown_heading(
        str(document.get("task", "")), "task",
    )
    sections = [
        "# High-level agent orchestration planning request",
        (
            "Create a concise, big-picture plan for coordinating the available agents. "
            "The Planner itself is responsible for decomposing the original task into "
            "stages and assigning work to agents. Derive the plan directly from the "
            "original task. Do not solve the factual task."
        ),
        "## Task\n\n" + task,
        _render_planning_agents_markdown(environment.get("roles", {})),
        """## Required plan

For each stage, state only:

1. When the stage begins or what prior result triggers it.
2. Which agent is called and what high-level task it performs.
3. Which calls are independent and may run in parallel.
4. Which agent, if any, receives the result next.

Use only agents listed above. Choose each agent from its stated responsibilities and tool access, and include verification or synthesis only when the task actually needs it. Do not require every available agent. Preserve useful prior results across handoffs and do not assume any output fields beyond the role contracts supplied to the code-generation stage.

Keep the plan at the orchestration level. Do not include code or pseudocode, API or runtime names, variables or state structures, result schemas or field names, exact prompt wording, detailed search procedures, numerical budgets, retry settings, exception handling, or other implementation details. Do not include a candidate answer.""",
    ]
    if critique:
        sections.append(
            "## Review feedback to address\n\n"
            + critique.strip()
            + "\n\n## Previous rejected plan\n\n"
            + previous_plan.strip()
            + "\n\nReturn a complete replacement plan, not a patch or commentary."
        )
    return "\n\n".join(sections)


def _render_plan_review_prompt(
    request: SolveGenerationRequestV1,
    natural_language_plan: str,
) -> str:
    document = request.to_dict()
    environment = document.get("environment", {})
    task = _strip_leading_markdown_heading(
        str(document.get("task", "")), "task",
    )
    plan = _strip_leading_markdown_heading(
        natural_language_plan,
        "plan", "orchestration plan", "natural-language orchestration plan",
    )
    return "\n\n".join([
        "# High-level orchestration plan review",
        "## Task\n\n" + task,
        _render_planning_agents_markdown(environment.get("roles", {})),
        "## Proposed plan\n\n" + plan,
        """## Acceptance criteria

- The plan preserves the task's actual entity and dependency relationships instead of merging distinct entities or assuming unstated equivalences.
- Every stage says when an available agent is called, its high-level objective, and the next handoff.
- Independent work is parallelized only when its inputs do not depend on another stage.
- Every selected agent exists in the supplied catalog and receives work matching its declared responsibilities and tools.
- The plan does not require all available agents; decomposition, research, solving, verification, synthesis, and general reasoning appear only when justified by the task.
- Useful prior results and qualifications are preserved across handoffs.
- When a checking stage is needed, its agent receives the exact candidate or result to check and downstream routing follows the declared result contract.
- When a combining stage is needed, its agent receives all intermediate material required to produce the requested final format.
- The plan remains high-level natural language with no implementation mechanics or candidate answer.

Review every criterion before deciding; do not return immediately after finding the first problem. Return only {"accepted": true, "critique": "", "issues": [], "suggested_changes": []} when every criterion passes. Otherwise return accepted=false, list every material defect in issues, and provide a corresponding repair in suggested_changes at the same array position.""",
    ])


def _render_code_review_prompt(
    request: SolveGenerationRequestV1,
    natural_language_plan: str,
    source: str,
    repair_history: Sequence[Mapping[str, Any]] = (),
) -> str:
    document = request.to_dict()
    environment = document.get("environment", {})
    task = _strip_leading_markdown_heading(
        str(document.get("task", "")), "task",
    )
    plan = _strip_leading_markdown_heading(
        natural_language_plan,
        "plan", "orchestration plan", "natural-language orchestration plan",
    )
    role_contracts: dict[str, Any] = {}
    for role_id, role in environment.get("roles", {}).items():
        role_contracts[str(role_id)] = {
            "description": role.get("description"),
            "result_schema": role.get("result_schema", {}),
        }
    sections = [
        "# Generated orchestration code review",
        "## Task\n\n" + task,
        "## Approved plan\n\n" + plan,
        "## Declared agent result contracts\n\n```json\n"
        + json.dumps(role_contracts, ensure_ascii=False, indent=2)
        + "\n```",
    ]
    history_section = _render_repair_history(repair_history)
    if history_section:
        sections.append(history_section)
    sections.extend([
        "## Generated program\n\n```python\n" + source.rstrip() + "\n```",
        """## Acceptance criteria

- The normal execution path can run with the declared API and agent result contracts.
- The program implements the approved stages, dependencies, role assignments, parallelism, verification, and synthesis without materially changing the approved strategy.
- Every called role is declared in the catalog and is used consistently with its responsibilities and allowed tools.
- The program reads each role result according to its declared schema, and plain-text results remain text.
- Every ctx.call and ctx.spawn inside a for loop has an explicit job key containing a literal logical-stage label, the real value bound by the loop, and a stable entity identity. For `for item in items`, the key must reference the current item; do not use only a fixed unrelated label. For `for index, item in enumerate(items)`, include both the index and current item, for example `f'research:{index}:{item}'`. If an index is needed, bind it with enumerate instead of inventing or manually maintaining an unrelated counter. Different logical invocations never reuse a key, and "unknown" is not used alone as an entity identity.
- The program does not force decomposition, research, verification, synthesis, or a fallback role unless the approved plan contains that stage.
- Reject a result-contract mismatch only when it makes the normal planned execution path fail at runtime.
- Completely disregard robustness, edge cases, weak fallback behavior, partial-result preservation, defensive checks, retry quality, exception recovery, checkpointing, tracing, and logging.
- Do not judge whether the orchestration is likely to find the correct factual answer; judge only runnability and fidelity to the approved plan.

Review every criterion before deciding; do not return immediately after finding the first problem. Return only {"accepted": true, "critique": "", "issues": [], "suggested_changes": []} when every criterion passes. Otherwise return accepted=false, list every material defect in issues, and provide a corresponding repair in suggested_changes at the same array position.""",
    ])
    return "\n\n".join(sections)


def _render_generation_prompt(
    request: SolveGenerationRequestV1,
    natural_language_plan: str,
    repair_history: Sequence[Mapping[str, Any]] = (),
) -> str:
    document = request.to_dict()
    environment = document.get("environment", {})
    task = _strip_leading_markdown_heading(
        str(document.get("task", "")), "task",
    )
    sections = [
        "# Orchestration generation request",
        (
            "Translate the high-level plan below into a task-specific "
            "orchestration program. Do not "
            "answer it directly in this response. Preserve the plan's stages, role "
            "assignments, dependencies, parallelism, handoffs, verification, and final "
            "synthesis. Use the technical references below to supply the implementation "
            "mechanics required by the runtime."
        ),
        "## Task\n\n" + task,
        _render_code_generation_handoff(request, natural_language_plan),
        _render_roles_markdown(environment.get("roles", {})),
        _render_budget_markdown(environment.get("budget", {})),
        "## Generation constraints\n\n" + "\n".join(
            f"{index}. {constraint}"
            for index, constraint in enumerate(
                document.get("generation_constraints", ()), 1,
            )
        ),
    ]
    history_section = _render_repair_history(repair_history)
    if history_section:
        sections.append(history_section)
        sections.append(
            "## Repair instructions\n\n"
            "Fix the current material defect while preserving every constraint "
            "already satisfied by earlier attempts. Do not remove a correct role, "
            "job key component, ctx API call, dependency, handoff, or result-format "
            "contract to solve an unrelated error. Change only the smallest portion "
            "directly related to the current diagnostics, then re-check the complete "
            "program against all declared constraints."
        )
    sections.append("## Required output\n\nReturn only the complete Python source beginning with `async def solve(ctx)`. Do not use a Markdown code fence or add explanatory text.")
    return "\n\n".join(sections)


def _render_code_generation_handoff(
    request: SolveGenerationRequestV1,
    natural_language_plan: str,
) -> str:
    """Combine the approved plan with its authoritative runtime API appendix."""
    document = request.to_dict()
    environment = document.get("environment", {})
    plan = _strip_leading_markdown_heading(
        natural_language_plan,
        "plan",
        "orchestration plan",
        "natural-language orchestration plan",
    )
    return "\n\n".join([
        "## Plan\n\n" + plan,
        """## Appendix A: Orchestration context (`ctx`) API

This appendix is part of the planner handoff and is the authoritative implementation interface for the plan. It documents only the context values and methods available in this run. Each entry gives the exact calling form, its purpose, its result, and the constraints that matter when translating the plan. Do not infer additional `ctx` members from the usage cases or general Python knowledge."""
        + "\n\n"
        + _ctx_api_reference(environment.get("ctx_api", ())),
    ])


def _strip_leading_markdown_heading(text: str, *titles: str) -> str:
    """Remove a redundant leading heading before a standard prompt heading."""
    stripped = text.strip()
    if not stripped:
        return stripped
    lines = stripped.splitlines()
    heading = re.fullmatch(r"\s{0,3}#{1,6}\s+(.+?)\s*#*\s*", lines[0])
    if heading is None:
        return stripped
    normalized = heading.group(1).strip().casefold()
    if normalized not in {title.strip().casefold() for title in titles}:
        return stripped
    return "\n".join(lines[1:]).lstrip()


def _render_planning_agents_markdown(roles: dict[str, Any]) -> str:
    """Render compact capability information needed for planning."""
    lines = [
        "## Available agents",
        "",
        "Available roles are optional rather than mandatory.",
        "",
    ]
    for role_id, role in roles.items():
        description = str(
            role.get("description")
            or "Handle work matching this agent's role."
        ).strip()
        responsibilities = str(
            role.get("responsibilities")
            or "Follow the delegated instruction carefully."
        ).strip()
        planner_usage = str(role.get("planner_usage") or "").strip()
        lines.append(f"- `{role_id}`")
        lines.append(f"  - Capability: {description}")
        lines.append(f"  - Instructions: {responsibilities}")
        if planner_usage:
            lines.append(f"  - Best used when: {planner_usage}")
    return "\n".join(lines).rstrip()


def _render_roles_markdown(roles: dict[str, Any]) -> str:
    lines = ["## Available roles", ""]
    for role_id, role in roles.items():
        tools = role.get("allowed_tools") or []
        result_schema = role.get("result_schema") or {}
        lines.extend([
            f"#### `{role_id}`", "",
            str(role.get("description") or "No short description provided."), "",
            "**Responsibilities and operating boundaries**", "",
            str(role.get("responsibilities") or "Follow the delegated instruction carefully."), "",
            "**When and how the planner should use this role**", "",
            str(role.get("planner_usage") or "Delegate only work matching this role's responsibilities."), "",
            "- Allowed tools: " + (", ".join(f"`{item}`" for item in tools) if tools else "none"),
            "- Default maximum turns: `" + str(role.get("default_max_turns")) + "`",
        ])
        lines.extend([
            "", "**Expected result and how to use it**", "",
            str(role.get("result_guidance") or "Use the returned value in the next orchestration stage."),
        ])
        if result_schema:
            lines.extend([
                "",
                "**Normative Result Schema**",
                "",
                "The worker output must conform to this JSON Schema. Use these exact "
                "field names and value types when reading the result:",
                "",
                "```json",
                json.dumps(result_schema, ensure_ascii=False, indent=2, default=str),
                "```",
            ])
        lines.append("")
    return "\n".join(lines).rstrip()


def _render_budget_markdown(budget: dict[str, Any]) -> str:
    def limit(name: str, suffix: str = "") -> str:
        value = budget.get(name)
        return "no explicit limit" if value is None else f"{value:,}{suffix}"
    return "\n".join([
        "## Budget", "",
        f"- Maximum LLM calls: {limit('max_llm_calls')}.",
        f"- Maximum tool calls: {limit('max_tool_calls')}.",
        f"- Maximum Jobs: {limit('max_jobs')}.",
        f"- Maximum concurrently running Jobs: {limit('max_parallel_jobs')}.",
        f"- Maximum wall-clock time: {limit('wall_time_seconds', ' seconds')}.",
        "- These are run-wide limits shared by planning and execution and are enforced by Runtime. `ctx.remaining_budget()` is available for optional adaptive control but is not required for acceptance.",
    ])


def _ctx_api_reference(allowed_methods: Any) -> str:
    """Render model-oriented documentation for the generated-code API."""
    allowed = set(allowed_methods or ())
    values = """### Read-only context values

- `ctx.task` (`str`)
  - **Purpose:** supplies the complete original problem that the worker agents must solve.
  - **How to use it:** include it in each self-contained worker instruction together with that worker's objective, relevant prior results, constraints, and expected output form.
  - **Constraint:** it is not a prewritten worker prompt and must not be assigned to or mutated.
- `ctx.metadata` (`dict[str, Any]`)
  - **Purpose:** exposes optional operational run metadata such as question id, strategy, budget, trace directory, and script identity.
  - **How to use it:** read optional values with `ctx.metadata.get("key", default)` only when they affect orchestration control.
  - **Constraint:** it is not factual evidence for the task; do not modify it or assume any optional key exists.
- `ctx.run_id` (`str`)
  - **Purpose:** provides a stable identifier for correlating this orchestration execution with trace labels or job keys.
  - **Constraint:** it contains neither the task nor an answer and must not be parsed as task data.

Every role call receives only the `instruction` and job metadata that the script supplies. Therefore, make each instruction self-contained: explain the complete problem, what this role must solve, what prior evidence or state it may rely on, which constraints matter, and what shape of result the next stage needs. Methods whose headings begin with `await` are asynchronous; all other methods are synchronous and must not be awaited."""
    async_methods = [
        name for name in (
            "call", "gather", "parallel", "map", "cancel", "checkpoint", "facts_query", "facts_index",
        ) if name in allowed
    ]
    sync_methods = [
        name for name in (
            "spawn", "load_checkpoint", "remaining_budget", "emit",
        ) if name in allowed
    ]
    concurrency_rules = [
        "### Parallel and await rules",
        "",
        "The asynchronous methods "
        + (", ".join(f"`ctx.{name}`" for name in async_methods) or "none")
        + " must be awaited before their results are consumed.",
        "The synchronous methods "
        + (", ".join(f"`ctx.{name}`" for name in sync_methods) or "none")
        + " must never be awaited.",
    ]
    if "parallel" in allowed:
        concurrency_rules.extend([
            "`ctx.parallel` accepts independent awaitable orchestration branches and returns their results in branch order.",
            "Its branches must not depend on one another, and a branch must not be replaced by a completed result or a JobHandle.",
        ])
    if "spawn" in allowed and "gather" in allowed:
        concurrency_rules.extend([
            "`ctx.spawn` schedules a Job and returns a JobHandle immediately; the handle is not the Job result.",
            "`ctx.gather` accepts only JobHandles previously returned by `ctx.spawn`; it waits for those Jobs and returns results or collected JobOutcome objects according to its failure policy.",
            "Do not pass `ctx.call` awaitables to `ctx.gather`, and do not pass `ctx.spawn` handles to `ctx.parallel`.",
        ])
    concurrency_rules.append(
        "Do not start dependent branches in parallel; wait for the upstream result before constructing the downstream call."
    )
    concurrency_rules.append(
        "Job-key rule: "
        + LOOP_JOB_KEY_GUIDANCE.replace("'unknown'", "`unknown`")
    )
    docs = {
        "facts_index": """### `await ctx.facts_index(keyword='', statuses=None, rounds=None)`
Browse the shared summary directory: revision and fact_index entries containing only fact_id,
summary and status. Includes active/disputed by default; statuses=[] includes all states.
Use facts_query(fact_ids=[...], statuses=[]) to read full details. Read before relying on a summary.
""",
        "facts_query": """### `await ctx.facts_query(keyword='', fact_ids=None, statuses=None, rounds=None)`
Read shared local facts. Returns {"revision": int, "facts": list[dict]}. Defaults to active facts;
statuses=[] returns all states. Filters combine with AND. Fact fields: id, claim, source_urls,
trace_refs, origin_round, updated_round, status (active or disputed), conflicts_with.
Use ctx.call("evidence_storesubagent", instruction) for summaries or changes. It returns
summary, fact_refs (stable IDs), changes, resolved_refs (new local_ref to stable ID), and revision.
Only this role can maintain facts.
""",
        "call": """### `await ctx.call(role, instruction, *, key=None, max_turns=None, timeout=None, metadata=None)`

**Purpose:** run one declared worker role and wait for its answer.

**Arguments:** `role` must exactly match a role id in the catalog. `instruction` is the worker's complete task and context. `key` is a stable trace/reuse label, not a role name. Inside a `for` loop it must contain a literal logical stage, the actual loop value, and a stable entity identity. When iterating over a collection, include the current item; when using an index, bind it with `enumerate`. Different logical invocations must use different keys; reuse a key only for the exact same role, instruction, and options. Never use `unknown` alone as the entity identity. `max_turns` overrides the role default, `timeout` is a wall-clock limit in seconds, and `metadata` carries optional per-job control data but never replaces the instruction.

**Returns:** the role's plain Python value. A role with a declared result schema normally returns a matching dict/list; otherwise expect text.

**Use constraints:** the role and instruction must satisfy the declared runtime contract. Exception degradation is optional and is not an acceptance requirement.""",
        "spawn": """### `ctx.spawn(role, instruction, *, key=None, max_turns=None, timeout=None, metadata=None)`

**Purpose:** start one declared worker without blocking so independent jobs can overlap.

**Arguments:** the parameters have the same meanings and constraints as `ctx.call`.

**Returns:** a `JobHandle`, not the worker result.

**Use constraints:** do not await `ctx.spawn` itself. Save every handle and later pass the handles as separate positional arguments to `ctx.gather`. Spawn only jobs whose inputs are already available and independent.""",
        "gather": """### `await ctx.gather(*handles, failure_policy="fail_fast", cancel_remaining=True)`

**Purpose:** wait concurrently for jobs previously created with `ctx.spawn`.

**Arguments:** pass previously returned JobHandles as separate positional arguments, never as one list and never as direct role-call awaitables. `cancel_remaining=True` cancels unfinished siblings after a fail-fast error.

**Returns:** results in the same order as the input handles. With `failure_policy="fail_fast"`, it returns plain values and raises on a failed job. With `failure_policy="collect"`, it returns `JobOutcome` objects; check `outcome.succeeded`, then read `outcome.result.value`, or inspect `outcome.failure`.

**Use constraints:** choose the failure policy required by the approved plan.

Only `collect` mode returns `JobOutcome` objects with `.succeeded`, `.result`, and `.failure`.""",
        "parallel": """### `await ctx.parallel(*branches)`

**Purpose:** run direct async orchestration branches concurrently without manually managing handles.

**Arguments:** pass each independent awaitable orchestration branch separately. Do not pass completed values or JobHandles.

**Returns:** a list of branch results in argument order.

**Use constraints:** branches must be independent. Failure is fail-fast.

These are plain role results, not `JobOutcome` objects; do not read `.succeeded` or `.result.value` from them.""",
        "map": """### `await ctx.map(function, items)`

**Purpose:** apply the same local async orchestration branch concurrently to a set of inputs.

**Arguments:** `function` must be a locally defined async function, not a role name; `items` must be an iterable available to the orchestration program.

**Returns:** a list of function results in item order.

**Use constraints:** each item must be independent. The local function may call `ctx.call`; map failures follow fail-fast parallel behavior.""",
        "cancel": """### `await ctx.cancel(handle, *, reason="cancelled by orchestration script")`

**Purpose:** stop a spawned job whose result is no longer needed and wait for cleanup.

**Arguments:** `handle` must be the `JobHandle` returned by `ctx.spawn`; `reason` is recorded in the trace.

**Returns:** `None` after cancellation cleanup completes.

**Use constraints:** do not pass a completed plain value or a result returned by `ctx.call`.""",
        "checkpoint": """### `await ctx.checkpoint(name, state)`

**Purpose:** atomically persist compact orchestration state at a safe stage boundary so a resumed run can continue.

**Arguments:** `name` is a stable checkpoint key; `state` must be JSON-compatible and may contain completed-stage markers, candidates, or compact intermediate results.

**Returns:** `None` after persistence completes.

**Use constraints:** await the call. Never store handles, coroutines, secrets, or other non-JSON runtime objects.""",
        "load_checkpoint": """### `ctx.load_checkpoint(name)`

**Purpose:** recover the most recently persisted state for a named checkpoint when a run resumes.

**Arguments:** `name` must match the stable key used with `ctx.checkpoint`.

**Returns:** the saved JSON-compatible value, or `None` when no checkpoint exists.

**Use constraints:** this method is synchronous; do not await it. Validate the returned shape and fall back to a clean initial state.""",
        "remaining_budget": """### `ctx.remaining_budget()`

**Purpose:** inspect current resource consumption before starting optional jobs, loops, or follow-up rounds.

**Returns:** an immutable snapshot with `.usage` and `.remaining` mappings. Inspect limits such as `snapshot.remaining.get("jobs", 0)`.

**Use constraints:** this method is synchronous and the result is only a point-in-time snapshot, not a reservation. Do not await or mutate it. Calling it is optional and is not required for acceptance.""",
        "emit": """### `ctx.emit(event_type, **payload)`

**Purpose:** append a semantic decision or stage marker to the orchestration trace.

**Arguments:** `event_type` should be concise and stable; every payload value must be JSON-compatible.

**Returns:** `None`; it neither calls an agent nor changes control flow.

**Use constraints:** this method is synchronous. Do not await it, and do not emit secrets or large worker outputs.

It is a best-effort trace marker, so its `None` return value must not be assigned or used for control flow.""",
    }
    rendered = [values, "\n".join(concurrency_rules)]
    order = ("call", "spawn", "gather", "parallel", "map", "cancel", "checkpoint", "load_checkpoint", "remaining_budget", "emit")
    rendered.extend(docs[name] for name in order if name in allowed)
    return "\n\n".join(rendered)


__all__ = [
    "ALLOWED_CTX_METHODS", "DSL_VERSION", "EnvironmentManifestV1",
    "GeneratedCodeValidationError", "GeneratedSolvePlanner",
    "GeneratedSolveV1", "GeneratedSolveValidator", "PlannerResult",
    "PreflightSchemaError", "RepairAttemptRecord", "ReviewVerdict",
    "SolveGenerationRequestV1", "ValidationIssue", "ValidationReport",
    "build_environment_manifest", "build_generation_request", "extract_source",
    "analyze_generated_source", "preflight_solve",
    "write_generation_research_report",
]


# Add the fusion contract to every planning/review stage, without changing base prompts.
_render_natural_language_plan_prompt = with_evolution_context(
    _render_natural_language_plan_prompt, planning=True,
)
_render_plan_review_prompt = with_evolution_context(_render_plan_review_prompt, planning=True)
_render_generation_prompt = with_evolution_context(_render_generation_prompt)
_render_code_review_prompt = with_evolution_context(_render_code_review_prompt)
