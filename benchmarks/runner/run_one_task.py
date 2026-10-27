#!/usr/bin/env python3
"""Single-question worker — spawned by run_harbor_subprocess.py.

Usage:
    uv run python benchmarks/run_one_task.py <qid> <out_dir> [<benchmark>]

Argv:
    qid       : question id (matches trials/<qid>/ dir under out_dir)
    out_dir   : path to the run's output dir (contains trials/, tasks/)
    benchmark : dataset key (e.g. "browsecomp", "widesearch"),
                defaults to "browsecomp"

Exit codes:
    0  — success, result.json written
    1  — fatal bootstrap or dataset error (no result.json)
    2  — question id not found in dataset

This script is self-contained: each invocation is a fresh Python process
with its own kernel + agent. Master relies on SIGKILL for timeouts
because sync-in-async tool calls (e2b sandbox) cannot be cancelled via
asyncio.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv  # noqa: E402
load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scorer"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
)
logger = logging.getLogger("run_one_task")


# ── Scoring ───────────────────────────────────────────────────────────

from benchmarks.judges import (  # noqa: E402
    has_judge,
    judge_semantic_equivalence,
    score_answer_detailed,
    verdict_from_judgement,
)

_QUESTION_TIMEOUT: float | None = None  # No outer per-question timeout by default.
_MISSING_ANSWER = "<ANSWER_NOT_FOUND>"


def _normalize_exact_match(value: str) -> str:
    """Conservatively normalize a short final answer for local scoring."""
    value = unicodedata.normalize("NFKC", value).strip()
    value = re.sub(
        r"^(?:\*{0,2})final\s+answer(?:\*{0,2})\s*:\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = value.strip().strip("`*_")
    value = value.rstrip(" \t\r\n.!?。！？")
    return " ".join(value.casefold().split())


def _unordered_team_match(predicted: str, ground_truth: str) -> bool:
    """Compare two-team answers without treating home/away order as semantic."""
    def teams(value: str) -> tuple[str, ...]:
        concise = value.splitlines()[0].split(",", 1)[0]
        parts = re.split(r"\s+(?:v(?:s\.?)*|versus)\s+", concise, flags=re.I)
        normalized = tuple(
            sorted(_normalize_exact_match(part) for part in parts if part.strip())
        )
        return normalized if len(normalized) == 2 else ()

    expected = teams(ground_truth)
    return bool(expected and teams(predicted) == expected)


def _is_completed_result(result: dict) -> bool:
    if result.get("planner_only") and not result.get("error"):
        return True
    predicted = str(result.get("predicted_answer") or "").strip()
    return bool(
        predicted
        and predicted != _MISSING_ANSWER
        and not result.get("error")
        and not result.get("scoring_error")
    )


async def _score_answer(
    predicted: str,
    ground_truth: str,
    question: str,
    answer_type: str,
    q_id: str,
    benchmark: str,
    scoring: str = "llm_judge",
) -> tuple[int, str, str | None, float | None, dict | None]:
    """Score locally or with the configured benchmark LLM judge."""
    if not predicted or not predicted.strip():
        logger.info("[%s] SCORE: empty prediction → WRONG", q_id)
        return 0, "empty", None, None, None

    if scoring == "exact_match":
        matched = (
            _normalize_exact_match(predicted)
            == _normalize_exact_match(ground_truth)
        )
        logger.info(
            "[%s] SCORE: local exact_match=%s | pred=%r gt=%r",
            q_id, matched, predicted[:50], ground_truth[:50],
        )
        return int(matched), "exact_match", None, None, None
    if scoring == "semantic_judge":
        logger.info(
            "[%s] SCORE: calling BrowseComp-Plus semantic judge | pred=%r gt=%r",
            q_id, predicted[:50], ground_truth[:50],
        )
        try:
            details = await judge_semantic_equivalence(
                question, ground_truth, predicted,
            )
        except Exception as exc:  # noqa: BLE001
            detail = f"{type(exc).__name__}: {exc}"
            logger.error(
                "[%s] SCORE: semantic judge raised — WRONG (%s). "
                "NEEDS MANUAL REVIEW.", q_id, detail,
            )
            return 0, "scoring_failed", detail, None, None
        verdict = verdict_from_judgement(details)
        if verdict == "CORRECT":
            logger.info("[%s] SCORE: semantic_judge=CORRECT", q_id)
            return 1, "semantic_judge", None, None, details
        if verdict == "INCORRECT":
            logger.info("[%s] SCORE: semantic_judge=WRONG", q_id)
            return 0, "semantic_judge", None, None, details
        return (
            0,
            "scoring_failed",
            "semantic judge NOT_ATTEMPTED after retries",
            None,
            details,
        )
    if scoring != "llm_judge":
        raise ValueError(f"unknown scoring mode: {scoring!r}")

    if not has_judge(benchmark):
        msg = f"no judge registered for benchmark {benchmark!r}"
        logger.error("[%s] SCORE: %s → WRONG", q_id, msg)
        return 0, "no_judge", msg, None, None

    logger.info(
        "[%s] SCORE: calling %s judge | pred=%r gt=%r",
        q_id, benchmark, predicted[:50], ground_truth[:50],
    )
    try:
        verdict, rubric_score, judge_details = await score_answer_detailed(
            benchmark, question, ground_truth, predicted,
        )
    except Exception as exc:  # noqa: BLE001
        detail = f"{type(exc).__name__}: {exc}"
        logger.error(
            "[%s] SCORE: judge raised — WRONG (%s). NEEDS MANUAL REVIEW.",
            q_id, detail,
        )
        return 0, "scoring_failed", detail, None, None

    score_tag = f" rubric_score={rubric_score:.2f}" if rubric_score is not None else ""
    if verdict == "CORRECT":
        logger.info("[%s] SCORE: llm=CORRECT%s", q_id, score_tag)
        return 1, "llm", None, rubric_score, judge_details
    if verdict == "INCORRECT":
        logger.info("[%s] SCORE: llm=WRONG%s", q_id, score_tag)
        return 0, "llm", None, rubric_score, judge_details

    logger.error(
        "[%s] SCORE: judge NOT_ATTEMPTED after retries → WRONG. "
        "NEEDS MANUAL REVIEW.", q_id,
    )
    return (
        0,
        "scoring_failed",
        "judge NOT_ATTEMPTED after retries",
        rubric_score,
        judge_details,
    )


# ── Single question runner ────────────────────────────────────────────

async def run_one_question(
    agent,
    question: dict,
    trial_dir: Path,
    task_src_dir: Path,
    benchmark: str,
    strategy: str = "single_agent",
    budget: dict | None = None,
    scoring: str = "llm_judge",
    orchestration_script: str = "reference_v1",
    orchestration_profile: str = "deepseek_v4_flash",
    planner_model: str | None = None,
    planner_max_repair_attempts: int = 3,
    planner_only: bool = False,
    skip_generated_code_review: bool = False,
    reuse_generated_code: bool = False,
) -> dict:
    """Run one question through AgentHarnessAgent and score it.

    Returns a result dict with question_id, predicted, ground_truth,
    is_correct, duration, etc.
    """
    from harbor.models.agent.context import AgentContext

    q_id = question["id"]
    trial_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume: skip if already scored ──────────────────────────
    existing_result = trial_dir / "result.json"
    if existing_result.exists():
        try:
            cached = json.loads(existing_result.read_text(encoding="utf-8"))
            if _is_completed_result(cached):
                logger.debug("[%s] Skipping - completed result exists", q_id)
                return cached
            logger.info("[%s] Re-running incomplete result", q_id)
        except (json.JSONDecodeError, OSError):
            pass  # corrupted, re-run

    # Do not let artifacts from an incomplete attempt become the new result
    # when the retry fails before producing an answer.
    for stale_name in ("final_answer.txt", "error.log"):
        (trial_dir / "agent" / stale_name).unlink(missing_ok=True)

    # Write config.json so agent can find task.toml metadata
    (trial_dir / "config.json").write_text(
        json.dumps({"task": {"path": str(task_src_dir)}}),
        encoding="utf-8",
    )

    instruction_path = task_src_dir / "instruction.md"
    instruction = instruction_path.read_text(encoding="utf-8")

    t0 = time.monotonic()
    error = None
    orchestration: dict = {}
    try:
        if strategy == "single_agent":
            # Create a fresh agent wrapper pointing to this trial.
            agent.logs_dir = trial_dir
            context = AgentContext()
            fake_env = MagicMock()
            await asyncio.wait_for(
                agent.run(instruction, fake_env, context),
                timeout=_QUESTION_TIMEOUT,
            )
        else:
            orchestration = await asyncio.wait_for(
                _run_orchestration_strategy(
                    instruction,
                    trial_dir=trial_dir,
                    question_id=q_id,
                    strategy=strategy,
                    budget=budget or {},
                    orchestration_script=orchestration_script,
                    orchestration_profile=orchestration_profile,
                    planner_model=planner_model,
                    planner_max_repair_attempts=planner_max_repair_attempts,
                    planner_only=planner_only,
                    skip_generated_code_review=skip_generated_code_review,
                    reuse_generated_code=reuse_generated_code,
                ),
                timeout=_QUESTION_TIMEOUT,
            )
    except asyncio.TimeoutError:
        if _QUESTION_TIMEOUT is None:
            error = "pipeline operation timed out"
            logger.error("[%s] Pipeline operation timed out", q_id)
        else:
            error = f"timeout after {_QUESTION_TIMEOUT}s"
            logger.error("[%s] Pipeline TIMEOUT after %ss", q_id, _QUESTION_TIMEOUT)
    except Exception as exc:
        error = str(exc)
        logger.error("[%s] Pipeline error: %s", q_id, exc)
    if orchestration.get("_terminal_error"):
        error = orchestration["_terminal_error"]
    duration = time.monotonic() - t0

    # Read answer
    answer_file = trial_dir / "agent" / "final_answer.txt"
    predicted = ""
    if answer_file.exists():
        predicted = answer_file.read_text(encoding="utf-8").strip()

    ground_truth = question["ground_truth"]
    answer_type = question["answer_type"]
    if planner_only:
        reward, judge_method, scoring_error, rubric_score, judge_details = (
            None, "not_scored_planner_only", None, None, None,
        )
        strict_exact_match = False
        unordered_team_match = False
    elif orchestration.get("_terminal_error"):
        reward, judge_method, scoring_error, rubric_score, judge_details = (
            0, orchestration["_terminal_judge_method"], None, None, None,
        )
        strict_exact_match = False
        unordered_team_match = False
    else:
        reward, judge_method, scoring_error, rubric_score, judge_details = await _score_answer(
            predicted, ground_truth, question["question"], answer_type, q_id,
            benchmark, scoring,
        )
        strict_exact_match = (
            _normalize_exact_match(predicted)
            == _normalize_exact_match(ground_truth)
        )
        unordered_team_match = _unordered_team_match(predicted, ground_truth)
    result = {
        "question_id": q_id,
        "question": question["question"][:200],
        "answer_type": answer_type,
        "ground_truth": ground_truth,
        "predicted_answer": predicted,
        "is_correct": None if planner_only else reward == 1,
        "reward": reward,
        "strict_exact_match": strict_exact_match,
        "unordered_team_match": unordered_team_match,
        "rubric_score": rubric_score,  # None = binary judge; float = rubric value
        "judge_details": judge_details,
        "judge_method": judge_method,
        "scoring_mode": scoring,
        "scoring_error": scoring_error,  # None = OK, string = needs manual review
        "duration_seconds": round(duration, 2),
        "error": error,
        "category": question.get("category", ""),
        "strategy": strategy,
        "planner_only": planner_only,
        "realtime_trace_paths": [
            str(path.relative_to(trial_dir))
            for path in sorted(trial_dir.rglob("*.jsonl"))
        ],
        **orchestration,
    }

    if strategy == "generated_code_orchestration":
        from agent_harness.orchestration.generated_code import (
            write_generation_research_report,
        )

        generation_data = result.get("generation") or {}
        research_path = write_generation_research_report(
            trial_dir / "generated",
            question_id=q_id,
            predicted_answer=predicted,
            ground_truth=ground_truth,
            is_correct=False if planner_only else reward == 1,
            strict_exact_match=strict_exact_match,
            unordered_team_match=unordered_team_match,
            judge_method=judge_method,
            usage=dict(result.get("usage") or {}),
            budget_violations=list(result.get("budget_violations") or []),
            pipeline_error=error,
            fallback_reason=generation_data.get("fallback_reason"),
        )
        generation_data["research_report_path"] = str(
            research_path.relative_to(trial_dir)
        )
        result["generation"] = generation_data

    # Write per-question result
    (trial_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    status = "PLANNER_ONLY" if planner_only else ("CORRECT" if reward == 1 else "WRONG")
    method_tag = f" [{judge_method}]" if judge_method != "rule" else ""
    logger.info(
        "[%s] %s%s | predicted=%r truth=%r (%.1fs)",
        q_id, status, method_tag, predicted[:40], ground_truth[:40], duration,
    )
    return result


async def _run_orchestration_strategy(
    instruction: str,
    *,
    trial_dir: Path,
    question_id: str,
    strategy: str,
    budget: dict,
    orchestration_script: str,
    orchestration_profile: str,
    planner_model: str | None,
    planner_max_repair_attempts: int,
    planner_only: bool = False,
    skip_generated_code_review: bool = False,
    reuse_generated_code: bool = False,
) -> dict:
    """Run a multi-agent strategy and persist its trace beside the answer."""
    from dataclasses import asdict

    from agent_harness.orchestration import (
        AgentHarnessWorkerExecutor,
        BudgetSpec,
        BudgetUsage,
        RunLimits,
        ScriptRuntime,
        create_strategy,
    )
    from agent_harness.orchestration.evolution import diagnose_trace
    from benchmarks.core.kernel_adapter import BenchmarkSession

    spec = BudgetSpec(**budget)
    if strategy in {"rolechange_self_evolving_v4", "rolechange_self_evolving_v5", "rolechange_self_evolving_v6"}:
        if reuse_generated_code:
            raise ValueError("rolechange self-evolution requires freshly generated scripts")
        from benchmarks.runner.rolechange_evolving_runner import run_self_evolving_strategy
        return await run_self_evolving_strategy(
            instruction, trial_dir=trial_dir, question_id=question_id,
            strategy=strategy, orchestration_profile=orchestration_profile,
            planner_model=planner_model,
            planner_max_repair_attempts=planner_max_repair_attempts,
            skip_generated_code_review=skip_generated_code_review,
            planner_only=planner_only, spec=spec,
        )
    orchestration_started = time.monotonic()

    def remaining_wall_time() -> float | None:
        if spec.wall_time_seconds is None:
            return None
        return max(
            0.001,
            spec.wall_time_seconds - (time.monotonic() - orchestration_started),
        )

    generation = None
    if strategy in {"code_orchestration", "generated_code_orchestration"}:
        from agent_harness.orchestration.strategies import (
            budget_violations,
            build_trace,
            usage_from_run,
        )
        from orchestrations.registry import get_orchestration_script

        planner_usage = None
        stage_events = []
        selected_script = orchestration_script
        live_path = trial_dir / "orchestration_live.jsonl"
        live_path.unlink(missing_ok=True)
        async with BenchmarkSession(
            orchestration_profile=orchestration_profile,
        ) as session:
            if strategy == "generated_code_orchestration":
                from agent_harness.core.runtime.resources.manager import ResourceManager
                from agent_harness.core.runtime import registry as service_registry
                from agent_harness.infra.config import get_config
                from agent_harness.infra.llm_adapter import create_llm
                from agent_harness.orchestration.generated_code import (
                    GeneratedSolveV1,
                    GeneratedSolvePlanner,
                    GeneratedSolveValidator,
                    PlannerResult,
                    build_environment_manifest,
                    build_generation_request,
                    preflight_solve,
                )
                from agent_harness.orchestration.roles import BUILTIN_ORCHESTRATION_ROLES
                from orchestrations.registry import register_generated_script

                config = get_config()
                resolved_planner_model = (
                    planner_model or config.planner_model or
                    getattr(session.llm, "model", "") or config.openai_model
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
                resources = service_registry.get(ResourceManager)
                environment = build_environment_manifest(
                    roles=definitions, tools=resources.all_tools, budget=spec,
                )
                request = build_generation_request(
                    instruction,
                    {"question_id": question_id},
                    environment,
                )
                validator = GeneratedSolveValidator(
                    allowed_roles=set(definitions),
                    max_parallel_jobs=spec.max_parallel_jobs,
                )
                generated_dir = trial_dir / "generated"
                if reuse_generated_code:
                    manifest_path = generated_dir / "manifest.json"
                    source_path = generated_dir / "solve.py"
                    if not manifest_path.exists() or not source_path.exists():
                        raise FileNotFoundError(
                            "--reuse-generated-code requires generated/manifest.json "
                            "and generated/solve.py"
                        )
                    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                    persisted_source = source_path.read_text(encoding="utf-8")
                    if manifest_data.get("source") != persisted_source:
                        raise ValueError(
                            "generated/solve.py does not match generated/manifest.json"
                        )
                    reuse_report = validator.validate(persisted_source)
                    if not reuse_report.accepted:
                        details = "; ".join(
                            f"{issue.code}: {issue.message}"
                            for issue in reuse_report.issues
                        )
                        raise ValueError(
                            f"persisted generated code failed validation: {details}"
                        )
                    reused_solve = validator.compile(persisted_source)
                    await preflight_solve(reused_solve, environment.roles)
                    generated = GeneratedSolveV1(**manifest_data)
                    if generated.source_hash != reuse_report.source_hash:
                        raise ValueError(
                            "persisted generated code source hash does not match validation"
                        )
                    generation = PlannerResult(
                        generated=generated,
                        usage=BudgetUsage(),
                        events=({
                            "timestamp": time.time(),
                            "stage": "code_generation",
                            "event_type": "generated_code_reused",
                            "payload": {
                                "generation_id": generated.generation_id,
                                "source_hash": generated.source_hash,
                            },
                        },),
                        request_path=str(generated_dir / "generation_request.json"),
                        validation_path=str(generated_dir / "validation_report.json"),
                        source_path=str(source_path),
                        attempts_path=str(generated_dir / "attempts"),
                        natural_language_plan_path=str(generated_dir / "plan" / "plan.md"),
                        natural_language_plan_artifacts_path=str(generated_dir / "plan"),
                    )
                else:
                    planner = GeneratedSolvePlanner(
                        create_llm(planner_config), validator,
                        model=resolved_planner_model,
                        max_repair_attempts=planner_max_repair_attempts,
                        max_tokens_per_call=config.planner_max_tokens,
                        review_generated_code=not skip_generated_code_review,
                    )
                    generation = await asyncio.wait_for(
                        planner.generate(
                            request,
                            artifact_dir=generated_dir,
                            budget=spec,
                        ),
                        timeout=remaining_wall_time(),
                    )
                if generation is not None:
                    planner_usage = generation.usage
                    stage_events.extend(generation.events)
                if generation is not None and generation.generated is not None:
                    generated = generation.generated
                    solve = validator.compile(generated.source)
                    registered = register_generated_script(
                        generation_id=generated.generation_id,
                        source=generated.source,
                        source_hash=generated.source_hash,
                        solve=solve,
                        roles=tuple(sorted(definitions)),
                    )
                    selected_script = registered.manifest.script_id
                    stage_events.append({
                        "timestamp": time.time(), "stage": "code_generation",
                        "event_type": "generated_script_registered",
                        "payload": {"script_id": selected_script,
                                    "source_hash": generated.source_hash},
                    })
                else:
                    reason = (
                        generation.fallback_reason
                        if generation is not None
                        else "planner did not return a generation result"
                    )
                    raise RuntimeError(
                        "generated orchestration was not accepted after all "
                        f"allowed attempts: {reason}"
                    )
            else:
                registered = get_orchestration_script(orchestration_script)
            if strategy == "code_orchestration":
                checkpoint_file = trial_dir / "checkpoint.json"
            else:
                checkpoint_file = trial_dir / "generated" / "checkpoint.json"
            if planner_only:
                from types import SimpleNamespace
                stage_events.append({
                    "timestamp": time.time(),
                    "stage": "execution",
                    "event_type": "planner_only_skipped_runtime",
                    "payload": {"selected_script": selected_script},
                })
                runtime_result = SimpleNamespace(
                    run_id=f"{question_id}:{selected_script}:planner-only",
                    value="",
                    jobs=(),
                    events=(),
                )
            else:
                runtime = ScriptRuntime(
                    AgentHarnessWorkerExecutor.from_runtime_services(
                        trace_dir=trial_dir / "agent_traces",
                    ),
                    limits=RunLimits(
                        max_parallel_jobs=spec.max_parallel_jobs,
                        max_total_jobs=spec.max_jobs,
                        wall_time_seconds=remaining_wall_time(),
                    ),
                    live_trace_path=live_path,
                    checkpoint_path=checkpoint_file,
                )
                stage_events.append({
                    "timestamp": time.time(), "stage": "execution",
                    "event_type": "generated_script_started" if strategy == "generated_code_orchestration" else "script_started",
                    "payload": {"script_id": selected_script},
                })
                runtime_result = await runtime.run(
                    registered.solve,
                    task=instruction,
                    run_id=f"{question_id}:{selected_script}",
                    metadata={
                        "question_id": question_id,
                        "strategy": strategy,
                        "trace_dir": str(trial_dir),
                        "budget": asdict(spec),
                        "script_id": registered.manifest.script_id,
                        "script_version": registered.manifest.version,
                        "script_source_hash": registered.manifest.source_hash,
                    },
                )
        usage = usage_from_run(runtime_result)
        if planner_usage is not None:
            usage = usage.add(planner_usage)
        trace = build_trace(
            runtime_result,
            strategy=strategy,
            usage=usage,
            stage_events=stage_events,
        )
        from types import SimpleNamespace
        outcome = SimpleNamespace(
            value=runtime_result.value,
            budget_violations=budget_violations(spec, usage),
            patches_proposed=0,
            patches_applied=0,
            trace=trace,
        )
        manifest = asdict(registered.manifest)
    else:
        async with BenchmarkSession():
            outcome = await create_strategy(strategy).run(
                AgentHarnessWorkerExecutor.from_runtime_services(
                    trace_dir=trial_dir / "agent_traces",
                ),
                task=instruction,
                run_id=f"{question_id}:{strategy}",
                budget=spec,
                metadata={
                    "question_id": question_id,
                    "strategy": strategy,
                    "trace_dir": str(trial_dir),
                },
            )
        manifest = None
    agent_dir = trial_dir / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    answer = str(outcome.value).strip()
    (agent_dir / "final_answer.txt").write_text(answer, encoding="utf-8")
    trace_path = outcome.trace.write_jsonl(trial_dir / "trace.jsonl")
    agent_report_path = None
    if strategy == "generated_code_orchestration":
        try:
            from agent_harness.orchestration.markdown_audit import (
                write_trial_agent_report,
            )
            agent_report_path = write_trial_agent_report(trial_dir)
        except Exception:  # noqa: BLE001
            logger.exception("failed to render researcher/verifier Markdown report")
    evolution = diagnose_trace(outcome.trace)
    (trial_dir / "evolution.json").write_text(
        json.dumps(evolution.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "trace_path": trace_path.name,
        "orchestration_live_trace_path": "orchestration_live.jsonl",
        "stage_trace_path": (
            None if strategy in {"code_orchestration", "generated_code_orchestration"} else "stage_trace.jsonl"
        ),
        "agent_trace_directory": "agent_traces",
        "researcher_verifier_report_path": (
            agent_report_path.name if agent_report_path is not None else None
        ),
        "trace_summary": outcome.trace.to_summary(),
        "usage": asdict(outcome.trace.usage),
        "budget": asdict(spec),
        "budget_violations": list(outcome.budget_violations),
        "patches_proposed": outcome.patches_proposed,
        "patches_applied": outcome.patches_applied,
        "orchestration_version": outcome.trace.orchestration_version,
        "evolution_path": "evolution.json",
        "orchestration_script": manifest,
        "checkpoint_path": (
            str(checkpoint_file.relative_to(trial_dir))
            if strategy in {"code_orchestration", "generated_code_orchestration"}
            else None
        ),
        "generation": (
            {
                "request_path": (
                    str(Path(generation.request_path).relative_to(trial_dir))
                    if generation is not None else None
                ),
                "validation_path": (
                    str(Path(generation.validation_path).relative_to(trial_dir))
                    if generation is not None else None
                ),
                "source_path": (
                    str(Path(generation.source_path).relative_to(trial_dir))
                    if generation is not None and generation.source_path else None
                ),
                "attempts_path": (
                    str(Path(generation.attempts_path).relative_to(trial_dir))
                    if generation is not None else "generated/attempts"
                ),
                "natural_language_plan_path": (
                    str(Path(generation.natural_language_plan_path).relative_to(trial_dir))
                    if generation is not None and generation.natural_language_plan_path
                    else None
                ),
                "natural_language_plan_artifacts_path": (
                    str(Path(generation.natural_language_plan_artifacts_path).relative_to(trial_dir))
                    if generation is not None and generation.natural_language_plan_artifacts_path
                    else "generated/plan"
                ),
                "fallback_reason": None,
                "planner_usage": (
                    asdict(generation.usage) if generation is not None else {}
                ),
            }
            if strategy == "generated_code_orchestration" else None
        ),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("qid", help="Question id (matches trials/<qid>/ dir)")
    p.add_argument("out_dir", help="Run output directory (contains trials/, tasks/)")
    p.add_argument("benchmark", nargs="?", default="browsecomp")
    p.add_argument(
        "--strategy",
        choices=(
            "single_agent", "fixed_multi_agent", "dynamic_multi_agent",
            "code_orchestration", "generated_code_orchestration", "rolechange_self_evolving_v4", "rolechange_self_evolving_v5", "rolechange_self_evolving_v6",
        ),
        default="single_agent",
    )
    p.add_argument("--budget-json", default="{}")
    p.add_argument("--orchestration-script", default="reference_v1")
    p.add_argument("--orchestration-profile", default="deepseek_v4_flash")
    p.add_argument("--planner-model", default=None)
    p.add_argument("--planner-max-repair-attempts", type=int, default=3)
    p.add_argument("--planner-only", action="store_true",
                   help="Only run planner generation/validation for generated_code_orchestration and skip runtime execution.")
    p.add_argument(
        "--skip-generated-code-review", action="store_true",
        help="Skip the generated-code semantic reviewer; static validation and preflight still run.",
    )
    p.add_argument(
        "--reuse-generated-code", action="store_true",
        help="Reuse generated/solve.py and manifest.json after revalidating them; do not call the planner.",
    )
    p.add_argument(
        "--scoring",
        choices=("llm_judge", "semantic_judge", "exact_match"),
        default="llm_judge",
    )
    args = p.parse_args()
    if args.planner_only and args.strategy not in {"generated_code_orchestration", "rolechange_self_evolving_v4", "rolechange_self_evolving_v5", "rolechange_self_evolving_v6"}:
        p.error("--planner-only requires --strategy generated_code_orchestration")
    if args.skip_generated_code_review and args.strategy not in {"generated_code_orchestration", "rolechange_self_evolving_v4", "rolechange_self_evolving_v5", "rolechange_self_evolving_v6"}:
        p.error(
            "--skip-generated-code-review requires "
            "--strategy generated_code_orchestration"
        )
    if args.reuse_generated_code and args.strategy != "generated_code_orchestration":
        p.error(
            "--reuse-generated-code requires --strategy generated_code_orchestration"
        )
    return args


async def _run_async(args: argparse.Namespace) -> int:
    from benchmarks.core.registry import load_questions
    from benchmarks.harbor_agent.agent import AgentHarnessAgent

    out_dir = Path(args.out_dir).resolve()
    trial_dir = out_dir / "trials" / args.qid
    task_src_dir = out_dir / "tasks" / args.qid

    # Resume: if already scored, no-op (master should have skipped us, but be safe).
    existing_result = trial_dir / "result.json"
    if existing_result.exists():
        try:
            cached = json.loads(existing_result.read_text(encoding="utf-8"))
            if _is_completed_result(cached):
                logger.info("[%s] completed result already present", args.qid)
                return 0
        except (json.JSONDecodeError, OSError):
            pass

    if not task_src_dir.exists():
        logger.error("[%s] task source dir not found: %s", args.qid, task_src_dir)
        return 1

    # Locate the single question by qid.
    all_questions = load_questions(args.benchmark)
    match = next((q for q in all_questions if q.id == args.qid), None)
    if match is None:
        logger.error("[%s] question id not found in %s dataset", args.qid, args.benchmark)
        return 2

    question = {
        "id": match.id,
        "question": match.question,
        "ground_truth": match.ground_truth,
        "answer_type": match.answer_type,
        "category": match.metadata.get("category", ""),
        "image_path": match.image_path,
    }

    # Bootstrap agent (one per subprocess — own :memory: DB, own kernel).
    agent = None
    run_db = str(out_dir / f".worker_kernel_{args.qid}.db")
    if args.strategy == "single_agent":
        agent = AgentHarnessAgent(logs_dir=trial_dir.parent, model_name=None)
        setup_env = MagicMock(spec=[])
        await agent.setup(setup_env, db_path=run_db)

    await run_one_question(
        agent, question, trial_dir, task_src_dir, args.benchmark,
        strategy=args.strategy,
        budget=json.loads(args.budget_json),
        scoring=args.scoring,
        orchestration_script=args.orchestration_script,
        orchestration_profile=args.orchestration_profile,
        planner_model=args.planner_model,
        planner_max_repair_attempts=args.planner_max_repair_attempts,
        planner_only=args.planner_only,
        skip_generated_code_review=args.skip_generated_code_review,
        reuse_generated_code=args.reuse_generated_code,
    )

    # Cleanup the per-worker kernel db file (best-effort).
    try:
        Path(run_db).unlink()
    except FileNotFoundError:
        pass

    return 0


def main() -> None:
    args = _parse_args()
    # An opt-in lock lets an externally started trial share its result with
    # the queued worker. Trials without this file keep their normal behavior.
    trial_dir = Path(args.out_dir).resolve() / "trials" / args.qid
    external_lock_path = trial_dir / ".external-worker.lock"
    external_lock = None
    if external_lock_path.exists():
        import fcntl

        external_lock = external_lock_path.open("a+")
        logger.info("[%s] acquiring external worker lock", args.qid)
        fcntl.flock(external_lock.fileno(), fcntl.LOCK_EX)
        result_path = trial_dir / "result.json"
        if result_path.exists():
            json.loads(result_path.read_text(encoding="utf-8"))
            logger.info("[%s] external worker result already present", args.qid)
            return
    exit_code = asyncio.run(_run_async(args))
    # Force-exit: agent_harness kernel keeps a non-daemon aiosqlite worker
    # thread alive, which blocks threading._shutdown() forever. result.json
    # has already been flushed to disk by run_one_question; nothing else
    # in this one-shot worker process is worth waiting to clean up.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    main()
