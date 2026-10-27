"""Rolechange execution with the v4 outer evolution loop."""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent_harness.orchestration.rolechange_evolution import (
    ROLECHANGE_STRATEGY, ROLECHANGE_V5_STRATEGY, ROLECHANGE_V6_STRATEGY, SCRIPT_ROLES, archive_previous_attempt,
    attach_worker_evidence, bind_executed_verifier, validate_proposal_shape,
)

async def run_self_evolving_strategy(
    instruction: str,
    *,
    strategy: str = ROLECHANGE_STRATEGY,
    trial_dir: Path,
    question_id: str,
    orchestration_profile: str,
    planner_model: str | None,
    planner_max_repair_attempts: int,
    skip_generated_code_review: bool,
    planner_only: bool,
    spec: Any,
    fact_flow_mode: str | None = None,
) -> dict[str, Any]:
    """Run rolechange scripts under the v4 terminal evolution gate."""
    is_v6 = strategy == ROLECHANGE_V6_STRATEGY
    is_v5 = strategy in {ROLECHANGE_V5_STRATEGY, ROLECHANGE_V6_STRATEGY}
    flow_mode = (fact_flow_mode or os.environ.get("ROLECHANGE_V6_MODE", "slice")) if is_v6 else "off"
    if flow_mode not in {"off", "flow_only", "slice"}:
        raise ValueError(f"Unknown V6 mode: {flow_mode}")
    if strategy not in {ROLECHANGE_STRATEGY, ROLECHANGE_V5_STRATEGY, ROLECHANGE_V6_STRATEGY}:
        raise ValueError(f"unsupported strategy: {strategy}")
    from agent_harness.core.runtime import registry as service_registry
    from agent_harness.core.runtime.resources.manager import ResourceManager
    from agent_harness.infra.config import get_config
    from agent_harness.infra.llm_adapter import create_llm
    from agent_harness.orchestration import (
        AgentHarnessWorkerExecutor,
        BudgetSpec,
        BudgetUsage,
        RunLimits,
        ScriptRuntime,
    )
    from agent_harness.orchestration.evidence_store import (
        EvidenceStore, EvidenceScope, EVIDENCE_ROLE,
        usage_from_evidence_run as usage_from_run,
    )
    from agent_harness.orchestration.evidence_judge import (
        local_judge_role, normalize_local_judge, local_failure_report, fallback_answer, answer_text,
        MAX_ROLECHANGE_EVOLUTION_ROUNDS,
    )
    def report_role(number):
        definition = local_judge_role(number, stage="report", reflector=is_v5)
        if flow_mode == "slice":
            from agent_harness.orchestration.reflection import SLICE_PROMPT
            definition = definition.model_copy(update={"system_prompt": definition.system_prompt + SLICE_PROMPT})
        return definition

    from agent_harness.orchestration.evolution import diagnose_trace
    from agent_harness.orchestration.evolution_judge import (
        build_prior_round_failures,
        build_structured_round_trace,
        normalize_round_proposal,
        render_synthesis_instruction,
    )
    from agent_harness.orchestration.generated_code import (
        GeneratedSolvePlanner,
        GeneratedSolveValidator,
        build_environment_manifest,
        build_generation_request,
    )
    from agent_harness.orchestration.roles import BUILTIN_ORCHESTRATION_ROLES
    from agent_harness.orchestration.self_evolving import (
        evolution_context,
    )
    from agent_harness.orchestration.strategies import (
        budget_violations,
        build_trace,
    )
    from benchmarks.core.kernel_adapter import BenchmarkSession
    from orchestrations.registry import register_generated_script, unregister_generated_script

    generated_root = trial_dir / "generated"
    archive_previous_attempt(generated_root, trial_dir)
    generated_root.mkdir(parents=True, exist_ok=True)
    from agent_harness.orchestration.trace_store import TraceStore
    from agent_harness.orchestration.reflection import build_diagnostic_bundle, normalize_reflection
    trace_store = TraceStore(generated_root) if is_v5 else None
    from agent_harness.orchestration.diagnostic_slice import build_diagnostic_slice
    from agent_harness.orchestration.fact_flow import FactFlowLedger
    flow_ledger = FactFlowLedger(generated_root, trace_store.attempt_id) if flow_mode != "off" else None
    previous_snapshot = {"facts": [], "revision": 0}
    config = get_config()
    history_enabled = True
    evidence_store = EvidenceStore(generated_root / "evidence_store.json")
    facts: list[dict[str, Any]] = []
    trace_history: list[dict[str, Any]] = []
    output_status = "failed"
    fallback_used = False
    previous_failure_report: dict[str, Any] | None = None
    round_records: list[dict[str, Any]] = []
    stage_events: list[dict[str, Any]] = []
    native_events: list[Any] = []
    native_jobs: list[Any] = []
    total_usage = BudgetUsage()
    answer = ""
    final_manifest: dict[str, Any] | None = None
    completed_rounds = 0
    last_round_dir: Path | None = None
    terminal_error: str | None = None
    terminal_judge_method: str | None = None

    if int(spec.max_jobs) < 5:
        raise ValueError(
            "rolechange V4 requires max_jobs >= 5 to reserve evidence maintenance, assessment, report and synthesis jobs",
        )
    script_budget = BudgetSpec(**{
        **asdict(spec),
        "max_jobs": int(spec.max_jobs) - 4,
    })

    async with BenchmarkSession(
        orchestration_profile=orchestration_profile,
    ) as session:
        resolved_planner_model = (
            planner_model or config.planner_model
            or getattr(session.llm, "model", "") or config.openai_model
        )
        if config.llm_provider.lower() == "openai":
            planner_config = config.model_copy(update={
                "openai_model": resolved_planner_model,
            })
        elif config.llm_provider.lower() == "qwen":
            planner_config = config.model_copy(update={
                "qwen_model": resolved_planner_model,
            })
        else:
            planner_config = config.model_copy(update={
                "anthropic_model": resolved_planner_model,
            })
        definitions = {
            item.role_id: item for item in BUILTIN_ORCHESTRATION_ROLES
        }
        definitions[EVIDENCE_ROLE.role_id] = EVIDENCE_ROLE
        script_definitions = {
            role_id: definition
            for role_id, definition in definitions.items()
            if role_id in SCRIPT_ROLES
        }
        # Local catalog augmentation only; global role definitions remain unchanged.
        script_definitions = {
            role: definition.model_copy(deep=True, update={"allowed_tools": list(dict.fromkeys(
                [*(definition.allowed_tools or []), "facts_index", "facts_query", "facts_submit"]
            ))}) for role, definition in script_definitions.items()
        }
        resources = service_registry.get(ResourceManager)
        environment = build_environment_manifest(
            roles=script_definitions,
            tools={**resources.all_tools, **{tool.name: tool
                   for tool in EvidenceScope(evidence_store, 1).tools("researcher")}},
            budget=script_budget,
        )

        for round_number in range(1, MAX_ROLECHANGE_EVOLUTION_ROUNDS + 1):
            completed_rounds = round_number
            round_dir = generated_root / f"evolution_round_{round_number:02d}"
            last_round_dir = round_dir
            round_dir.mkdir(parents=True, exist_ok=True)
            script_scope = EvidenceScope(evidence_store, round_number, flow_ledger=flow_ledger)
            context = evolution_context(
                facts,
                previous_failure_report,
                round_number=round_number,
            )
            context["max_rounds"] = MAX_ROLECHANGE_EVOLUTION_ROUNDS
            context["disputed_facts"] = [
                dict(item) for item in facts
                if item.get("status") == "disputed"
            ]
            request = build_generation_request(
                instruction,
                {
                    "question_id": question_id,
                    "orchestration_mode": "self_evolving_generated_code",
                    "prompt_profile": (
                        strategy
                    ),
                    "evolution_context": context,
                },
                environment,
            )
            validator = GeneratedSolveValidator(
                allowed_roles=set(script_definitions),
                max_parallel_jobs=script_budget.max_parallel_jobs,
                require_evolution_proposal=True,
            )
            planner = GeneratedSolvePlanner(
                create_llm(planner_config),
                validator,
                model=resolved_planner_model,
                max_repair_attempts=planner_max_repair_attempts,
                max_tokens_per_call=config.planner_max_tokens,
                review_generated_code=not skip_generated_code_review,
                facts_query=evidence_store.query,
            )
            generation = None
            generation_error = ""
            try:
                generation = await planner.generate(
                    request,
                    artifact_dir=round_dir,
                    budget=script_budget,
                )
            except Exception as exc:  # noqa: BLE001
                generation_error = f"{type(exc).__name__}: {exc}"
            generation_usage = (
                generation.usage if generation is not None else BudgetUsage()
            )
            round_usage = generation_usage
            if generation is not None:
                stage_events.extend(_round_stage_events(
                    generation.events, round_number,
                ))
            generated = generation.generated if generation is not None else None
            natural_language_plan_path = round_dir / "plan" / "plan.md"
            natural_language_plan = (
                natural_language_plan_path.read_text(encoding="utf-8")
                if natural_language_plan_path.exists() else ""
            )
            generation_reason = (
                (generation.fallback_reason if generation is not None else None)
                or generation_error
                or "generation did not produce an accepted script"
            )
            generation_status = {
                "accepted": generated is not None,
                "error": "" if generated is not None else generation_reason,
                "request_path": _relative_if_exists(
                    round_dir, "generation_request.json", trial_dir,
                ),
                "validation_path": _relative_if_exists(
                    round_dir, "validation_report.json", trial_dir,
                ),
                "events": _safe_generation_events(
                    generation.events if generation is not None else (),
                ),
            }
            if generated is not None:
                stage_events.append({
                    "timestamp": time.time(),
                    "stage": f"evolution_round_{round_number:02d}",
                    "event_type": "rolechange_v4_script_generated",
                    "payload": {
                        "round": round_number,
                        "generation_id": generated.generation_id,
                        "source_hash": generated.source_hash,
                    },
                })
            if planner_only and generated is not None:
                total_usage = total_usage.add(round_usage)
                round_records.append({
                    "round": round_number,
                    "status": "generated_planner_only",
                    "generation_id": generated.generation_id,
                    "source_hash": generated.source_hash,
                    "usage": asdict(round_usage),
                })
                break

            runtime = None
            run_id = (
                f"{question_id}:{strategy}:"
                f"round-{round_number:02d}"
            )
            runtime_result = None
            execution_error = ""
            round_events: tuple[Any, ...] = ()
            round_jobs: tuple[Any, ...] = ()
            registered = None
            if generated is not None:
                solve = validator.compile(generated.source)
                registered = register_generated_script(
                    generation_id=generated.generation_id,
                    source=generated.source,
                    source_hash=generated.source_hash,
                    solve=solve,
                    roles=tuple(sorted(script_definitions)),
                )
                runtime = ScriptRuntime(
                    AgentHarnessWorkerExecutor.from_runtime_services(
                        capture_evidence=True,
                        evidence_scope=script_scope,
                        trace_dir=(
                            trial_dir / "agent_traces"
                            / f"evolution_round_{round_number:02d}"
                        ),
                    ),
                    limits=RunLimits(
                        max_parallel_jobs=script_budget.max_parallel_jobs,
                        max_total_jobs=script_budget.max_jobs,
                        wall_time_seconds=script_budget.wall_time_seconds,
                    ),
                    live_trace_path=round_dir / "orchestration_live.jsonl",
                )
                try:
                    runtime_result = await runtime.run(
                        registered.solve,
                        task=instruction,
                        run_id=run_id,
                        metadata={
                            "question_id": question_id,
                            "strategy": strategy,
                            "evolution_round": round_number,
                            "budget": asdict(script_budget),
                            "script_id": registered.manifest.script_id,
                            "script_version": registered.manifest.version,
                            "script_source_hash": registered.manifest.source_hash,
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    execution_error = f"{type(exc).__name__}: {exc}"
                    round_events = tuple(runtime.get_events(run_id))
                    round_jobs = tuple(runtime.get_jobs(run_id))
                    failed_run = SimpleNamespace(
                        events=round_events,
                        jobs=round_jobs,
                    )
                    round_usage = round_usage.add(usage_from_run(failed_run))
                else:
                    round_events = tuple(runtime_result.events)
                    round_jobs = tuple(runtime_result.jobs)
                    round_usage = round_usage.add(
                        usage_from_run(runtime_result),
                    )
                native_events.extend(round_events)
                native_jobs.extend(round_jobs)

            if registered is not None:
                unregister_generated_script(registered.manifest.script_id)

            raw_proposal = (
                runtime_result.value if runtime_result is not None else {
                    "status": "incomplete",
                    "selected_candidate": "",
                    "selected_verifier": {},
                    "facts": [],
                    "issues": [execution_error or generation_reason],
                    "synthesis_context": {},
                }
            )
            try:
                validate_proposal_shape(raw_proposal)
            except (TypeError, ValueError) as exc:
                execution_error = execution_error or f"Invalid candidate proposal: {exc}"
                raw_proposal = {
                    "status": "incomplete", "selected_candidate": "",
                    "selected_verifier": {}, "facts": [],
                    "issues": [execution_error], "synthesis_context": {},
                }
            proposal = normalize_round_proposal(raw_proposal)
            bind_executed_verifier(proposal, round_events, round_jobs, execution_error)
            trace_document = build_structured_round_trace(
                task=instruction,
                round_number=round_number,
                evolution_context=context,
                natural_language_plan=natural_language_plan,
                generation_status=generation_status,
                proposal=proposal,
                events=round_events,
                jobs=round_jobs,
                execution_error=execution_error,
                usage=asdict(round_usage),
            )
            attach_worker_evidence(trace_document, round_events)
            terminal_dir = round_dir / "terminal_review"
            trace_document["prior_round_failures"] = build_prior_round_failures(round_records)
            frozen_facts = evidence_store.snapshot()
            trace_document["fact_store_snapshot"] = frozen_facts
            _write_json(round_dir / "evidence_maintenance" / "input_trace.json", trace_document)
            maintenance_scope = EvidenceScope(evidence_store, round_number, trace=trace_document,
                                              terminal_maintenance=True, flow_ledger=flow_ledger, stage="evidence")
            maintenance_runtime = ScriptRuntime(
                AgentHarnessWorkerExecutor.from_runtime_services(
                    capture_evidence=True, evidence_scope=maintenance_scope,
                    trace_dir=trial_dir / "agent_traces" / f"evolution_round_{round_number:02d}" / "evidence",
                ),
                limits=RunLimits(max_parallel_jobs=1, max_total_jobs=1,
                                 wall_time_seconds=spec.wall_time_seconds),
                live_trace_path=round_dir / "evidence_maintenance" / "live.jsonl",
            )
            maintenance_id = f"{run_id}:evidence"

            async def run_maintenance(ctx: Any) -> Any:
                return await ctx.call("evidence_storesubagent",
                                      "Review the complete committed fact table and stored submission evidence. Reconcile duplicates "
                                      "and conflicts, synchronize summaries, and do not import ordinary outputs or proposals.",
                                      key="terminal:evidence_store", max_turns=25)

            async def maintain_round(
                maintenance_runtime=maintenance_runtime, maintenance_id=maintenance_id,
                round_number=round_number,
            ) -> dict:
                try:
                    result = await maintenance_runtime.run(
                        run_maintenance, task=instruction, run_id=maintenance_id,
                        metadata={"evolution_round": round_number, "budget": asdict(spec)},
                    )
                    return {"result": result.value, "error": None}
                except Exception as exc:  # noqa: BLE001
                    return {"result": None, "error": f"{type(exc).__name__}: {exc}"}

            maintenance = await maintain_round()
            maintenance_events = maintenance_runtime.get_events(maintenance_id)
            for event in maintenance_events:
                if event.payload.get("name") == "evidence_compact_input":
                    _write_json(round_dir / "evidence_maintenance" / "compact_input.json", event.payload["data"])
            maintenance_jobs = maintenance_runtime.get_jobs(maintenance_id)
            native_events.extend(maintenance_events)
            native_jobs.extend(maintenance_jobs)
            round_usage = round_usage.add(usage_from_run(SimpleNamespace(
                events=maintenance_events, jobs=maintenance_jobs,
            )))
            facts = evidence_store.snapshot()["facts"]
            _write_json(round_dir / "evidence_maintenance" / "result.json", maintenance)
            _write_json(round_dir / "evidence_maintenance" / "snapshot.json", evidence_store.snapshot())
            # Terminal review is causally after the maintenance commit (or its recorded failure).
            frozen_facts = evidence_store.snapshot()
            trace_document["fact_store_snapshot"] = frozen_facts
            trace_document["evidence_maintenance"] = maintenance
            _write_json(terminal_dir / "structured_trace.json", trace_document)
            judge_scope = EvidenceScope(evidence_store, round_number,
                                        read_snapshot=frozen_facts, flow_ledger=flow_ledger, stage="assessment")
            judge_executor_options = {
                "role_overrides": {"evolution_judge": local_judge_role(round_number)},
                "evidence_scope": judge_scope,
            }
            judge_record = {
                "task": instruction,
                "facts": frozen_facts["facts"],
                "candidate_answer": proposal.get("selected_candidate", ""),
            }
            judge_instruction = (
                "Audit this local record according to your system policy.\n"
                + json.dumps(judge_record, ensure_ascii=False)
            )
            _write_json(terminal_dir / "assessment_input.json", judge_record)
            trace_history.append(trace_document)
            judge_value: Any = {}
            judge_error = ""
            judge_runtime = ScriptRuntime(
                AgentHarnessWorkerExecutor.from_runtime_services(
                    capture_evidence=True,
                    trace_dir=(
                        trial_dir / "agent_traces"
                        / f"evolution_round_{round_number:02d}" / "terminal"
                    ),
                    **judge_executor_options,
                ),
                limits=RunLimits(
                    max_parallel_jobs=1,
                    max_total_jobs=1,
                    wall_time_seconds=spec.wall_time_seconds,
                ),
                live_trace_path=terminal_dir / "judge_live.jsonl",
            )

            async def run_judge(ctx: Any, judge_instruction=judge_instruction) -> Any:
                return await ctx.call(
                    "evolution_judge",
                    judge_instruction,
                    key="terminal:evolution_judge",
                    max_turns=25,
                )

            judge_run_id = f"{run_id}:terminal-judge"
            judge_result = None
            try:
                judge_result = await judge_runtime.run(
                    run_judge,
                    task=instruction,
                    run_id=judge_run_id,
                    metadata={
                        "question_id": question_id,
                        "strategy": strategy,
                        "evolution_round": round_number,
                        "budget": asdict(spec),
                        "terminal_gate": True, "judge_stage": "assessment",
                        "fact_store_revision": frozen_facts["revision"],
                    },
                )
            except Exception as exc:  # noqa: BLE001
                judge_error = f"{type(exc).__name__}: {exc}"
                judge_events = tuple(judge_runtime.get_events(judge_run_id))
                judge_jobs = tuple(judge_runtime.get_jobs(judge_run_id))
                judge_usage = usage_from_run(SimpleNamespace(
                    events=judge_events, jobs=judge_jobs,
                ))
            else:
                judge_value = judge_result.value
                judge_events = tuple(judge_result.events)
                judge_jobs = tuple(judge_result.jobs)
                judge_usage = usage_from_run(judge_result)
            native_events.extend(judge_events)
            native_jobs.extend(judge_jobs)
            round_usage = round_usage.add(judge_usage)
            failure_stage = (
                "generation" if generated is None else (
                    "execution" if execution_error else (
                        "evolution_judge" if judge_error else "verification"
                    )
                )
            )
            summary = (
                judge_error or execution_error
                or (generation_reason if generated is None else "")
                or "; ".join(str(item) for item in proposal["issues"][:10])
                or str(
                    proposal["selected_verifier"].get("critique", "")
                )
                or "round requires terminal review"
            )
            judge = normalize_local_judge(
                {} if judge_error else judge_value, round_number=round_number,
                proposal=proposal, failure_stage=failure_stage, summary=summary,
                previous_report=previous_failure_report,
            )
            judge["fact_store_revision"] = frozen_facts["revision"]
            _write_json(terminal_dir / "assessment.json", judge)
            _write_json(terminal_dir / "decision.json", judge)
            stage_events.append({
                "timestamp": time.time(),
                "stage": f"evolution_round_{round_number:02d}/terminal_gate",
                "event_type": "evolution_judge_completed",
                "payload": {
                    "decision": judge["decision"],
                    "requested_decision": judge["requested_decision"],
                    "eligible_before_judge": (
                        judge["eligible_before_judge"]
                    ),
                    "error": judge_error or None,
                },
            })

            synthesis_error = ""
            if judge["decision"] == "approve_synthesis":
                tentative_facts = facts
                synthesis_instruction = render_synthesis_instruction(
                    instruction,
                    proposal,
                    judge,
                    [
                        item for item in tentative_facts
                        if item.get("status") == "active"
                    ],
                )
                synth_runtime = ScriptRuntime(
                    AgentHarnessWorkerExecutor.from_runtime_services(
                        capture_evidence=True,
                        evidence_scope=EvidenceScope(evidence_store, round_number, flow_ledger=flow_ledger, stage="synthesis"),
                        trace_dir=(
                            trial_dir / "agent_traces"
                            / f"evolution_round_{round_number:02d}"
                            / "terminal"
                        ),
                    ),
                    limits=RunLimits(
                        max_parallel_jobs=1,
                        max_total_jobs=max(1, int(spec.max_jobs) - round_usage.jobs - 1),
                        wall_time_seconds=spec.wall_time_seconds,
                    ),
                    live_trace_path=terminal_dir / "synthesis_live.jsonl",
                )

                async def run_synthesizer(ctx: Any, synthesis_instruction=synthesis_instruction) -> Any:
                    return await ctx.call(
                        "synthesizer",
                        synthesis_instruction,
                        key="terminal:synthesizer",
                    )

                synth_run_id = f"{run_id}:terminal-synthesis"
                synth_result = None
                try:
                    synth_result = await synth_runtime.run(
                        run_synthesizer,
                        task=instruction,
                        run_id=synth_run_id,
                        metadata={
                            "question_id": question_id,
                            "strategy": strategy,
                            "evolution_round": round_number,
                            "budget": asdict(spec),
                            "terminal_synthesis": True,
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    synthesis_error = f"{type(exc).__name__}: {exc}"
                    synth_events = tuple(
                        synth_runtime.get_events(synth_run_id)
                    )
                    synth_jobs = tuple(synth_runtime.get_jobs(synth_run_id))
                    synth_usage = usage_from_run(SimpleNamespace(
                        events=synth_events, jobs=synth_jobs,
                    ))
                else:
                    synth_events = tuple(synth_result.events)
                    synth_jobs = tuple(synth_result.jobs)
                    synth_usage = usage_from_run(synth_result)
                    answer = str(synth_result.value or "").strip()
                    if not answer:
                        synthesis_error = (
                            "Synthesizer returned an empty final answer"
                        )
                native_events.extend(synth_events)
                native_jobs.extend(synth_jobs)
                round_usage = round_usage.add(synth_usage)
                if not answer and round_number == MAX_ROLECHANGE_EVOLUTION_ROUNDS:
                    answer = answer_text(proposal.get("selected_candidate"))
                    fallback_used = bool(answer)
                if answer:
                    output_status = "approved"
                    _write_json(terminal_dir / "final_output.json", {
                        "status": output_status, "answer": answer, "fallback_used": fallback_used,
                        "synthesis_error": synthesis_error or None,
                    })
                    facts = evidence_store.snapshot()["facts"]
                    total_usage = total_usage.add(round_usage)
                    final_manifest = (
                        asdict(registered.manifest)
                        if registered is not None else None
                    )
                    round_records.append({
                        "round": round_number,
                        "status": "approved",
                        "evidence_maintenance": maintenance,
                        "generation_id": generated.generation_id,
                        "source_hash": generated.source_hash,
                        "proposal": proposal,
                        "evolution_judge": judge,
                        "usage": asdict(round_usage),
                        "active_fact_count": sum(
                            item.get("status") == "active" for item in facts
                        ),
                        "budget_violations": list(
                            budget_violations(spec, round_usage),
                        ),
                    })
                    break

            # A rejected assessment or failed synthesis gets a separate report inference.
            report_record = {
                **judge_record, "assessment": {
                    key: judge_value[key]
                    for key in ("decision", "rationale", "constraints", "issues")
                    if not judge_error and isinstance(judge_value, dict) and key in judge_value
                },
            }
            trace_scope = None
            if is_v5:
                trace_errors = []
                try:
                    trace_store.ingest(trial_dir, round_number, frozen_facts)
                except (OSError, ValueError, TypeError) as exc:
                    trace_errors.append({"error": f"{type(exc).__name__}: {exc}"})
                trace_store.add(round_number=round_number, stage="diagnostics", runtime=run_id,
                                sequence="outcome", content={"generation": generation_status,
                                    "execution_error": execution_error, "synthesis_error": synthesis_error,
                                    "gate": judge}, event_type="round_outcome")
                trace_store.persist()
                flow_snapshot = flow_ledger.freeze(round_number) if flow_ledger else None
                diagnostic_slice = None
                slice_error = ""
                if flow_ledger:
                    flow_ledger.persist_snapshot(round_dir, round_number)
                    try:
                        if flow_mode == "slice":
                            trace_store.ingest_flow(flow_snapshot)
                            diagnostic_slice = build_diagnostic_slice(
                                round_number=round_number, assessment=report_record["assessment"],
                                gate=judge, proposal=proposal, frozen_facts=frozen_facts,
                                fact_flow=flow_snapshot, trace_store=trace_store)
                            _write_json(terminal_dir / "diagnostic_slice.json", diagnostic_slice)
                    except Exception as exc:  # noqa: BLE001 - diagnostics must not change the business outcome.
                        slice_error = f"{type(exc).__name__}: {exc}"
                        trace_errors.append({"component": "diagnostic_slice", "error": slice_error})
                preferred = diagnostic_slice.get("trace_refs") if diagnostic_slice else None
                if flow_mode == "slice" and not preferred and not slice_error:
                    slice_error = "No usable slice; falling back to current-round retrieval."
                trace_scope = trace_store.freeze(round_number, preferred_trace_ids=preferred)
                diagnostic_bundle = build_diagnostic_bundle(
                    task=instruction, round_number=round_number, proposal=proposal,
                    assessment=report_record["assessment"], gate=judge, trace=trace_document,
                    frozen_facts=frozen_facts, previous_snapshot=previous_snapshot,
                    previous_report=previous_failure_report, usage=asdict(round_usage),
                    synthesis_error=synthesis_error, trace_limits=trace_scope.limits,
                    errors=trace_errors + trace_store.errors)
                if flow_ledger:
                    events = flow_snapshot["events"]
                    diagnostic_bundle["fact_flow_summary"] = {
                        "event_count": len(events), "fact_count": len(frozen_facts["facts"]),
                        "facts_with_readers": len({e["fact_id"] for e in events if e["event_type"] in {"READ_FULL", "READ_SUMMARY"}}),
                        "facts_with_transfers": len({e["fact_id"] for e in events if e["event_type"] == "TRANSFERRED"}),
                        "facts_with_producers": len({e["fact_id"] for e in events if e["event_type"] == "COMMITTED"}),
                        "fact_reads": sum(e["event_type"] in {"READ_FULL", "READ_SUMMARY"} for e in events),
                        "explicit_transfers": sum(e["event_type"] == "TRANSFERRED" for e in events),
                        "coverage_warnings": flow_snapshot["coverage_warnings"][:10]}
                if flow_mode == "slice":
                    diagnostic_bundle["diagnostic_slice_summary"] = {
                        "slice_id": diagnostic_slice.get("slice_id") if diagnostic_slice else None,
                        "fact_ids": diagnostic_slice.get("fact_ids", []) if diagnostic_slice else [],
                        "job_ids": diagnostic_slice.get("job_ids", []) if diagnostic_slice else [],
                        "trace_count": len(preferred or []), "fallback_reason": slice_error,
                        "seed": diagnostic_slice.get("seed", {}) if diagnostic_slice else {},
                        "truncation": diagnostic_slice.get("truncation", {}) if diagnostic_slice else {},
                        "full_round_trace_count": diagnostic_slice.get("full_round_trace_count", 0) if diagnostic_slice else 0,
                        "coverage_warnings": diagnostic_slice.get("coverage_warnings", [])[:10] if diagnostic_slice else [],
                        "unresolved_seeds": diagnostic_slice.get("unresolved_seeds", [])[:10] if diagnostic_slice else []}
                report_record["diagnostic_bundle"] = diagnostic_bundle
                _write_json(terminal_dir / "diagnostic_bundle.json", diagnostic_bundle)
                previous_snapshot = frozen_facts
            report_runtime = ScriptRuntime(
                AgentHarnessWorkerExecutor.from_runtime_services(
                    capture_evidence=True, evidence_scope=EvidenceScope(evidence_store, round_number,
                        read_snapshot=frozen_facts, stage="report"), trace_scope=trace_scope,
                    role_overrides={"evolution_judge": report_role(round_number)},
                    trace_dir=trial_dir / "agent_traces" / f"evolution_round_{round_number:02d}" / "report",
                ),
                limits=RunLimits(max_parallel_jobs=1, max_total_jobs=1,
                                 wall_time_seconds=spec.wall_time_seconds),
                live_trace_path=terminal_dir / "report_live.jsonl",
            )
            _write_json(terminal_dir / "report_input.json", report_record)
            report_instruction = (
                "Explain the evidence gaps and direct the next Planner according to your system policy.\n"
                + json.dumps(report_record, ensure_ascii=False)
            )

            async def run_report(ctx: Any, report_instruction=report_instruction) -> Any:
                return await ctx.call("evolution_judge", report_instruction,
                                      key="terminal:evolution_report", max_turns=25)

            report_run_id = f"{run_id}:terminal-report"
            report_value = {}
            report_error = ""
            try:
                report_result = await report_runtime.run(
                    run_report, task=instruction, run_id=report_run_id,
                    metadata={"question_id": question_id, "strategy": strategy,
                              "evolution_round": round_number, "budget": asdict(spec),
                              "judge_stage": "report", "fact_store_revision": frozen_facts["revision"]},
                )
                report_value = report_result.value
            except Exception as exc:  # noqa: BLE001
                report_error = f"{type(exc).__name__}: {exc}"
            report_events = report_runtime.get_events(report_run_id)
            report_jobs = report_runtime.get_jobs(report_run_id)
            native_events.extend(report_events)
            native_jobs.extend(report_jobs)
            round_usage = round_usage.add(usage_from_run(SimpleNamespace(
                events=report_events, jobs=report_jobs,
            )))
            report = local_failure_report(
                report_value.get("failure_report"),
                failure_stage="synthesis" if synthesis_error else failure_stage,
                summary=synthesis_error or judge["rationale"], previous_report=previous_failure_report,
            )
            if is_v5:
                report = normalize_reflection(report, report_value.get("failure_report"),
                    round_number=round_number, scope=trace_scope, snapshot=frozen_facts, error=report_error)
                _write_json(terminal_dir / "trace_usage.json", trace_scope.usage(report.get("trace_refs", [])))
            report["continue_evolution"] = round_number < MAX_ROLECHANGE_EVOLUTION_ROUNDS
            if report_error:
                report["report_error"] = report_error
                report["failed_routes"] = list(dict.fromkeys([*report["failed_routes"], *judge["issues"]]))
            judge.update(
                decision=report_value.get("decision", "evolve"),
                rationale=report_value.get("rationale", judge["rationale"]),
                final_answer=report_value.get("final_answer", ""), failure_report=report,
                report_error=report_error or None,
            )
            _write_json(terminal_dir / "report.json", {
                "result": report_value, "error": report_error or None,
                "fact_store_revision": frozen_facts["revision"],
            })
            _write_json(terminal_dir / "decision.json", judge)
            _write_json(round_dir / "failure_report.json", report)
            previous_failure_report = report
            stage_events.append({
                "timestamp": time.time(), "stage": f"evolution_round_{round_number:02d}/terminal_report",
                "event_type": "evolution_report_completed",
                "payload": {"error": report_error or None, "fact_store_revision": frozen_facts["revision"]},
            })

            if round_number == MAX_ROLECHANGE_EVOLUTION_ROUNDS:
                answer = (judge["final_answer"] if judge["decision"] == "best_effort" else "")
                answer = answer or fallback_answer([r["proposal"] for r in round_records] + [proposal], trace_history)
                output_status = "best_effort" if answer else "failed"
                fallback_used = True
                if not answer:
                    answer = "Unable to produce an answer: all three rounds lacked candidates and terminal model output failed."
                    terminal_error = answer
                    terminal_judge_method = "self_evolving_no_candidate_failure"
                total_usage = total_usage.add(round_usage)
                round_records.append({
                    "round": round_number, "status": output_status, "proposal": proposal,
                    "evolution_judge": judge, "evidence_maintenance": maintenance,
                    "failure_report": judge["failure_report"], "usage": asdict(round_usage),
                    "budget_violations": list(budget_violations(spec, round_usage)),
                })
                _write_json(terminal_dir / "final_output.json", {
                    "status": output_status, "answer": answer, "fallback_used": fallback_used,
                    "judge_error": judge_error or None,
                })
                break

            judge_status = "synthesis_failed" if synthesis_error else (
                "generation_failed" if generated is None else (
                    "execution_failed" if execution_error else "vetoed"
                )
            )
            facts = evidence_store.snapshot()["facts"]
            previous_failure_report = report
            _write_json(round_dir / "failure_report.json", report)
            total_usage = total_usage.add(round_usage)
            round_records.append({
                "round": round_number,
                "status": judge_status,
                "evidence_maintenance": maintenance,
                "generation_id": (
                    generated.generation_id if generated is not None else None
                ),
                "source_hash": (
                    generated.source_hash if generated is not None else None
                ),
                "execution_error": execution_error or None,
                "proposal": proposal,
                "evolution_judge": judge,
                "failure_report": report,
                "usage": asdict(round_usage),
                "active_fact_count": sum(
                    item.get("status") == "active" for item in facts
                ),
                "budget_violations": list(
                    budget_violations(spec, round_usage),
                ),
            })

    if not answer and not planner_only:
        terminal_error = (
            "self-evolving orchestration did not produce an Evolution-Judge-"
            f"approved answer after {completed_rounds} round(s)"
        )
        terminal_judge_method = "self_evolving_rounds_exhausted"

    if is_v5 and not planner_only:
        try:
            trace_store.ingest(trial_dir, completed_rounds, evidence_store.snapshot())
        except (OSError, ValueError, TypeError) as exc:
            trace_store.errors.append({"error": f"{type(exc).__name__}: {exc}"})
            trace_store.persist()

    if flow_ledger and not planner_only:
        flow_ledger.persist_snapshot(generated_root / f"evolution_round_{completed_rounds:02d}", completed_rounds)
        try:
            if flow_mode == "slice":
                trace_store.ingest_flow(flow_ledger.freeze(completed_rounds))
        except (OSError, ValueError, TypeError) as exc:
            flow_ledger.errors.append({"operation": "final_trace", "error": str(exc)})
    manifest_document = {
        "schema_version": "self_evolving_orchestration_v6_flow_slice_v1" if is_v6 else "self_evolving_orchestration_v5_reflector_v1" if is_v5 else "self_evolving_orchestration_v4_evidence_v3",
        "fact_flow_mode": flow_mode,
        "stagnation_relaxation_enabled": history_enabled,
        "strategy": strategy,
        "question_id": question_id,
        "max_rounds": MAX_ROLECHANGE_EVOLUTION_ROUNDS,
        "completed_rounds": completed_rounds,
        "status": "planner_only" if planner_only else output_status,
        "accepted": output_status == "approved",
        "answer_emitted": bool(answer),
        "evidence_store_path": "generated/evidence_store.json",
        "fact_store_revision": evidence_store.snapshot()["revision"],
        "rounds": round_records,
        "fact_memory": facts,
        "last_failure_report": previous_failure_report,
        "usage": asdict(total_usage),
        "fallback_used": fallback_used,
    }
    manifest_path = generated_root / "evolution_manifest.json"
    _write_json(manifest_path, manifest_document)
    combined = SimpleNamespace(
        run_id=f"{question_id}:{strategy}",
        value=answer,
        jobs=tuple(native_jobs),
        events=tuple(native_events),
    )
    trace = build_trace(
        combined,
        strategy=strategy,
        usage=total_usage,
        stage_events=stage_events,
    )
    trace_path = trace.write_jsonl(trial_dir / "trace.jsonl")
    diagnosis = diagnose_trace(trace)
    _write_json(trial_dir / "evolution.json", diagnosis.to_dict())
    if answer:
        agent_dir = trial_dir / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        (agent_dir / "final_answer.txt").write_text(
            answer, encoding="utf-8",
        )
    per_round_violations = [
        f"round_{record['round']}:{violation}"
        for record in round_records
        for violation in record.get("budget_violations", [])
    ]
    return {
        "trace_path": trace_path.name,
        "orchestration_live_trace_path": None,
        "stage_trace_path": None,
        "agent_trace_directory": "agent_traces",
        "researcher_verifier_report_path": None,
        "trace_summary": trace.to_summary(),
        "usage": asdict(total_usage),
        "budget": asdict(spec),
        "budget_violations": per_round_violations,
        "patches_proposed": 0,
        "patches_applied": 0,
        "orchestration_version": trace.orchestration_version,
        "evolution_path": "evolution.json",
        "orchestration_script": final_manifest,
        "checkpoint_path": None,
        "generation": {
            "request_path": _relative_if_exists(
                last_round_dir, "generation_request.json", trial_dir,
            ),
            "validation_path": _relative_if_exists(
                last_round_dir, "validation_report.json", trial_dir,
            ),
            "source_path": _relative_if_exists(
                last_round_dir, "solve.py", trial_dir,
            ),
            "attempts_path": _relative_if_exists(
                last_round_dir, "attempts", trial_dir,
            ),
            "natural_language_plan_path": _relative_if_exists(
                last_round_dir, "plan/plan.md", trial_dir,
            ),
            "natural_language_plan_artifacts_path": _relative_if_exists(
                last_round_dir, "plan", trial_dir,
            ),
            "fallback_reason": None,
            "planner_usage": asdict(total_usage),
            "evolution_manifest_path": str(
                manifest_path.relative_to(trial_dir),
            ),
            "round_count": completed_rounds,
            "fact_flow_mode": flow_mode,
        "stagnation_relaxation_enabled": history_enabled,
        },
        **({
            "_terminal_error": terminal_error,
            "_terminal_judge_method": terminal_judge_method,
        } if terminal_error is not None else {}),
    }


def _safe_generation_events(events: Any) -> list[dict[str, Any]]:
    """Keep generation control outcomes without exposing prompts or source."""
    result = []
    for item in events:
        if not isinstance(item, dict):
            continue
        raw_payload = item.get("payload", {})
        payload = dict(raw_payload) if isinstance(raw_payload, dict) else {}
        for key in (
            "model_input", "model_output", "source", "previous_source",
        ):
            payload.pop(key, None)
        result.append({
            "timestamp": item.get("timestamp"),
            "stage": item.get("stage"),
            "event_type": item.get("event_type"),
            "payload": payload,
        })
    return result


def _round_stage_events(
    events: Any, round_number: int,
) -> list[dict[str, Any]]:
    result = []
    for item in events:
        payload = dict(item.get("payload", {}))
        payload["evolution_round"] = round_number
        result.append({
            "timestamp": item.get("timestamp", time.time()),
            "stage": f"evolution_round_{round_number:02d}/"
                     + str(item.get("stage", "planner")),
            "event_type": str(item.get("event_type", "planner_event")),
            "payload": payload,
        })
    return result


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _relative_if_exists(
    base: Path | None, child: str, trial_dir: Path,
) -> str | None:
    if base is None:
        return None
    path = base / child
    return str(path.relative_to(trial_dir)) if path.exists() else None

