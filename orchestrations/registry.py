"""Allowlisted registry for repository-owned orchestration scripts."""

from __future__ import annotations

import hashlib
import importlib
import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from agent_harness.orchestration.context import OrchestrationContext

ScriptEntrypoint = Callable[[OrchestrationContext], Awaitable[Any]]


@dataclass(frozen=True)
class OrchestrationScriptManifest:
    script_id: str
    version: int
    entrypoint: str
    input_schema: str
    output_schema: str
    compatible_runtime: str
    roles: tuple[str, ...]
    source_hash: str


@dataclass(frozen=True)
class RegisteredOrchestrationScript:
    manifest: OrchestrationScriptManifest
    solve: ScriptEntrypoint


_ALLOWLIST = {
    "reference_v1": "orchestrations.reference_v1",
    "reference_v2": "orchestrations.reference_v2",
    "reference_v3": "orchestrations.reference_v3",
    "reference_v4": "orchestrations.reference_v4",
    "reference_v5": "orchestrations.reference_v5",
    "reference_v7": "orchestrations.reference_v7",
}

_RUN_SCOPED: dict[str, RegisteredOrchestrationScript] = {}


def get_orchestration_script(script_id: str) -> RegisteredOrchestrationScript:
    if script_id in _RUN_SCOPED:
        return _RUN_SCOPED[script_id]
    try:
        module_name = _ALLOWLIST[script_id]
    except KeyError as exc:
        raise KeyError(f"unknown orchestration script: {script_id!r}") from exc
    module = importlib.import_module(module_name)
    entrypoint = getattr(module, "solve")
    source = inspect.getsource(module).encode("utf-8")
    declared = module.MANIFEST
    manifest = OrchestrationScriptManifest(
        **declared,
        source_hash=hashlib.sha256(source).hexdigest(),
    )
    return RegisteredOrchestrationScript(manifest=manifest, solve=entrypoint)


def register_generated_script(
    *,
    generation_id: str,
    source: str,
    source_hash: str,
    solve: ScriptEntrypoint,
    roles: tuple[str, ...],
) -> RegisteredOrchestrationScript:
    """Register already-validated source for the lifetime of this worker.

    This deliberately does not modify the repository allowlist.  Benchmark
    questions run in separate processes, so the registration cannot leak to
    another task.
    """
    script_id = f"generated:{generation_id}"
    calculated = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if calculated != source_hash:
        raise ValueError("generated source hash mismatch")
    registered = RegisteredOrchestrationScript(
        manifest=OrchestrationScriptManifest(
            script_id=script_id,
            version=1,
            entrypoint="solve",
            input_schema="string",
            output_schema="string",
            compatible_runtime=">=1.1",
            roles=roles,
            source_hash=source_hash,
        ),
        solve=solve,
    )
    _RUN_SCOPED[script_id] = registered
    return registered


def unregister_generated_script(script_id: str) -> None:
    _RUN_SCOPED.pop(script_id, None)


__all__ = [
    "OrchestrationScriptManifest", "RegisteredOrchestrationScript",
    "get_orchestration_script", "register_generated_script",
    "unregister_generated_script",
]
