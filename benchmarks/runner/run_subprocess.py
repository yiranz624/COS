#!/usr/bin/env python3
"""Run a benchmark via subprocess-per-question isolation.

Each question runs in its own Python subprocess — fresh kernel, in-memory
DB, asyncio loop. Lets the master SIGKILL hung workers.

Output:

  Single-run (``--runs 1``):
    <out>/{tasks,trials}/<qid>/...
    <out>/{results.json,summary.txt}

  Multi-run (``--runs N``):
    <out>/config.json
    <out>/run_<i>/{tasks,trials,results.json,summary.txt}

Inspect progress with ``python -m benchmarks.runner.check_progress``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
)
logger = logging.getLogger("run_harbor_subprocess")

_QUESTION_TIMEOUT: float | None = None  # No master-side worker timeout by default.
_WORKER_SCRIPT = Path(__file__).resolve().parent / "run_one_task.py"
_MISSING_ANSWER = "<ANSWER_NOT_FOUND>"


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


# ── Subprocess runner ────────────────────────────────────────────────

async def _tee(reader, log, mirror, prefix=""):
    while line := await reader.readline():
        s = line.decode("utf-8", errors="replace")
        log.write(s)
        log.flush()
        if mirror:
            sys.stdout.write(prefix + s if prefix else s)
            sys.stdout.flush()


async def run_task_subprocess(
    qid: str,
    out_dir: Path,
    benchmark: str,
    sem: asyncio.Semaphore,
    mirror: bool = False,
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
    """Spawn a worker subprocess for one question. Returns the result dict
    from result.json, or a synthetic error dict on timeout/crash."""
    trial_dir = out_dir / "trials" / qid
    result_path = trial_dir / "result.json"

    # Resume: skip if already scored.
    if result_path.exists():
        try:
            cached = json.loads(result_path.read_text(encoding="utf-8"))
            if _is_completed_result(cached):
                return cached
        except (json.JSONDecodeError, OSError):
            pass  # corrupted, re-run

    async with sem:
        trial_dir.mkdir(parents=True, exist_ok=True)
        log_path = trial_dir / "agent.log"

        log_file = open(log_path, "w", encoding="utf-8")
        tee = None
        try:
            t0 = time.monotonic()
            proc = await asyncio.create_subprocess_exec(
                sys.executable, str(_WORKER_SCRIPT),
                qid, str(out_dir), benchmark,
                "--strategy", strategy,
                "--budget-json", json.dumps(budget or {}),
                "--scoring", scoring,
                "--orchestration-script", orchestration_script,
                "--orchestration-profile", orchestration_profile,
                "--planner-max-repair-attempts", str(planner_max_repair_attempts),
                *(["--planner-model", planner_model] if planner_model else []),
                *(["--planner-only"] if planner_only else []),
                *(
                    ["--skip-generated-code-review"]
                    if skip_generated_code_review else []
                ),
                *(["--reuse-generated-code"] if reuse_generated_code else []),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            tee = asyncio.create_task(_tee(proc.stdout, log_file, mirror, prefix=f"[{qid}] "))
            exit_code: int | None = None
            try:
                exit_code = await asyncio.wait_for(
                    proc.wait(), timeout=_QUESTION_TIMEOUT,
                )
            except asyncio.TimeoutError:
                if _QUESTION_TIMEOUT is None:
                    raise
                proc.kill()
                await proc.wait()
                duration = time.monotonic() - t0
                logger.error(
                    "[%s] subprocess SIGKILLed after %ss timeout",
                    qid, _QUESTION_TIMEOUT,
                )
                synth = {
                    "question_id": qid,
                    "is_correct": False,
                    "reward": 0,
                    "error": f"subprocess_timeout_{_QUESTION_TIMEOUT}s",
                    "duration_seconds": round(duration, 2),
                    "judge_method": "subprocess_timeout",
                }
                result_path.write_text(
                    json.dumps(synth, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                return synth

            duration = time.monotonic() - t0
        finally:
            if tee is not None:
                await tee
            log_file.close()

    if result_path.exists():
        try:
            return json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("[%s] result.json unreadable: %s", qid, exc)

    logger.error(
        "[%s] worker exit=%s but no result.json (duration=%.1fs)",
        qid, exit_code, duration,
    )
    synth = {
        "question_id": qid,
        "is_correct": False,
        "reward": 0,
        "error": f"worker_exit_{exit_code}_no_result",
        "duration_seconds": round(duration, 2),
        "judge_method": "worker_crash",
    }
    result_path.write_text(
        json.dumps(synth, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return synth


# ── One-run evaluator ────────────────────────────────────────────────

async def run_eval(
    args: argparse.Namespace,
    *,
    out_dir: Path,
    seed: int,
) -> dict:
    """Generate tasks + dispatch subprocesses for one run pass."""
    from benchmarks.core.registry import load_questions, get_config
    from benchmarks.core.harbor_task_generator import generate_task_dirs

    cfg = get_config(args.benchmark)
    pipeline = args.pipeline or cfg.default_pipeline or "react_base"

    questions = load_questions(
        args.benchmark,
        limit=args.limit,
        offset=args.offset,
        answer_type=args.answer_type,
        category=args.category,
    )

    if args.question_ids:
        requested_ids = [
            item.strip() for item in args.question_ids.split(",") if item.strip()
        ]
        questions_by_id = {question.id: question for question in questions}
        missing_ids = [qid for qid in requested_ids if qid not in questions_by_id]
        if missing_ids:
            raise ValueError(
                "Question IDs not found in the selected dataset: "
                + ", ".join(missing_ids)
            )
        questions = [questions_by_id[qid] for qid in requested_ids]

    if not args.no_shuffle:
        import random
        random.Random(seed).shuffle(questions)
        logger.info("Shuffled questions with seed=%d", seed)

    if args.limit:
        questions = questions[:args.limit]

    total = len(questions)
    logger.info("Loaded %d %s questions", total, cfg.name)
    if total == 0:
        logger.error("No questions to run")
        return {}

    tasks_dir = out_dir / "tasks"
    q_dicts = [
        {
            "id": q.id,
            "question": q.question,
            "answer": q.ground_truth,
            "answer_type": q.answer_type,
            "category": q.metadata.get("category", ""),
            "image_path": q.image_path,
            "file_path": q.file_path,
            "file_name": q.file_name,
            # Inject workflow-profile so it lands in task.toml's [metadata]
            # section. Read back by AgentHarnessAgent._read_task_metadata
            # and forwarded into the pipeline state as ``state.metadata.profile``.
            "metadata": {"profile": args.profile},
        }
        for q in questions
    ]
    generate_task_dirs(q_dicts, tasks_dir, pipeline_id=pipeline)

    (out_dir / "trials").mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(args.concurrency)
    mirror = sys.stdout.isatty()
    t_start = time.monotonic()
    completed = 0
    correct_count = 0

    async def wrap(q):
        nonlocal completed, correct_count
        result = await run_task_subprocess(
            q.id, out_dir, args.benchmark, sem, mirror,
            strategy=args.strategy,
            scoring=args.scoring,
            budget={
                "max_tokens": args.max_tokens,
                "max_llm_calls": args.max_llm_calls,
                "max_tool_calls": args.max_tool_calls,
                "max_jobs": args.max_jobs,
                "max_parallel_jobs": args.max_parallel_jobs,
                "wall_time_seconds": args.wall_time_seconds,
            },
            orchestration_script=args.orchestration_script,
            orchestration_profile=args.orchestration_profile,
            planner_model=args.planner_model,
            planner_max_repair_attempts=args.planner_max_repair_attempts,
            planner_only=args.planner_only,
            skip_generated_code_review=args.skip_generated_code_review,
            reuse_generated_code=args.reuse_generated_code,
        )
        completed += 1
        if result.get("is_correct"):
            correct_count += 1
        if completed % 10 == 0 or completed == total:
            elapsed = time.monotonic() - t_start
            logger.info(
                "PROGRESS: %d/%d done, %d correct (%.1f%%) (%.0fs elapsed)",
                completed, total, correct_count,
                correct_count * 100 / completed, elapsed,
            )
        return result

    results = await asyncio.gather(
        *(wrap(q) for q in questions),
        return_exceptions=True,
    )

    final: list[dict] = []
    for q, r in zip(questions, results):
        if isinstance(r, Exception):
            logger.error("[%s] orchestrator error: %s", q.id, r)
            final.append({
                "question_id": q.id,
                "is_correct": False,
                "reward": 0,
                "error": f"orchestrator_{type(r).__name__}: {r}",
                "duration_seconds": 0,
                "judge_method": "orchestrator_exception",
            })
        else:
            final.append(r)

    # Compatibility field: maxtoken records consumed tokens, not a budget limit.
    for result_item in final:
        result_item["maxtoken"] = int(result_item.get("usage", {}).get("tokens", 0))

    total_duration = time.monotonic() - t_start

    correct = sum(1 for r in final if r.get("is_correct"))
    strict_exact = sum(1 for r in final if r.get("strict_exact_match"))
    unordered_team = sum(1 for r in final if r.get("unordered_team_match"))
    attempted = sum(1 for r in final if r.get("predicted_answer"))
    accuracy = correct / total if total else 0.0

    by_method: dict[str, int] = {}
    for r in final:
        m = r.get("judge_method", "unknown")
        by_method[m] = by_method.get(m, 0) + 1

    summary = (
        f"\n{'=' * 60}\n"
        f"  {cfg.name} Results\n"
        f"{'=' * 60}\n"
        f"  Pipeline:     {pipeline}\n"
        f"  Profile:      {args.profile}\n"
        f"  Total:        {total}\n"
        f"  Attempted:    {attempted}\n"
        f"  Correct:      {correct}\n"
        f"  Accuracy:     {accuracy:.2%}\n"
        f"  Strict EM:    {strict_exact}/{total}\n"
        f"  Team-set EM:  {unordered_team}/{total}\n"
        f"  Duration:     {total_duration:.1f}s\n"
        f"  Concurrency:  {args.concurrency}\n"
        f"\n  Scoring method breakdown:\n"
    )
    for m, cnt in sorted(by_method.items()):
        summary += f"    {m}: {cnt}\n"
    summary += f"{'=' * 60}\n"
    logger.info(summary)

    report = {
        "benchmark": args.benchmark,
        "pipeline": pipeline,
        "profile": args.profile,
        "strategy": args.strategy,
        "total": total,
        "attempted": attempted,
        "correct": correct,
        "accuracy": accuracy,
        "strict_exact_match_count": strict_exact,
        "unordered_team_match_count": unordered_team,
        "by_method": by_method,
        "duration_seconds": round(total_duration, 2),
        "config": {
            "limit": args.limit,
            "seed": seed,
            "concurrency": args.concurrency,
            "offset": args.offset,
        },
        "started_at": datetime.now(timezone.utc).isoformat(),
        "results": final,
    }
    from agent_harness.orchestration.evaluation import summarize_results
    report["evaluation"] = summarize_results(
        final, strategy=args.strategy,
    ).to_dict()
    report["maxtoken"] = report["evaluation"]["usage"]["tokens"]
    if args.strategy == "generated_code_orchestration":
        script_records = []
        for result_item in final:
            qid = str(result_item.get("question_id", ""))
            report_path = out_dir / "trials" / qid / "generated" / "research_report.json"
            if not report_path.exists():
                continue
            try:
                item = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            item["artifact_directory"] = str(
                report_path.parent.relative_to(out_dir)
            )
            script_records.append(item)
        archive_index = {
            "schema_version": "generated_script_archive_v1",
            "auto_deploy": False,
            "total": len(script_records),
            "promising": sum(
                item.get("classification") == "promising"
                for item in script_records
            ),
            "failed_or_inconclusive": sum(
                item.get("classification") != "promising"
                for item in script_records
            ),
            "scripts": script_records,
        }
        (out_dir / "generated_scripts.json").write_text(
            json.dumps(archive_index, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        report["generated_script_archive"] = "generated_scripts.json"
    (out_dir / "results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")
    logger.info("Run finished — %d/%d correct (%.1f%%)", correct, total, accuracy * 100)
    return report


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--benchmark", required=True,
                   help="Dataset key registered in benchmarks.dataset_registry.REGISTRY")
    p.add_argument("--pipeline", default=None,
                   help="AgentHarness pipeline (default: dataset's default_pipeline)")
    p.add_argument("--profile", default="default",
                   help="Workflow profile name (default: 'default'). Written into "
                        "task.toml metadata; read back as state.metadata.profile.")
    p.add_argument(
        "--strategy",
        choices=("single_agent", "fixed_multi_agent", "dynamic_multi_agent", "code_orchestration", "generated_code_orchestration", "rolechange_self_evolving_v4", "rolechange_self_evolving_v5", "rolechange_self_evolving_v6"),
        default="single_agent",
        help="Comparable orchestration strategy (default: single_agent).",
    )
    p.add_argument(
        "--orchestration-script", default="reference_v1",
        help="Allowlisted Python workflow used by code_orchestration.",
    )
    p.add_argument(
        "--orchestration-profile", default="deepseek_v4_flash",
        help="Per-role model routing YAML profile.",
    )
    p.add_argument("--planner-model", default=None,
                   help="Independent model used to generate solve(ctx).")
    p.add_argument("--planner-max-repair-attempts", type=int, default=3)
    p.add_argument("--planner-only", action="store_true",
                   help="Run only planner generation/validation for generated_code_orchestration and skip runtime execution.")
    p.add_argument(
        "--skip-generated-code-review", action="store_true",
        help="Skip the generated-code semantic reviewer; static validation and preflight still run.",
    )
    p.add_argument(
        "--reuse-generated-code", action="store_true",
        help="Reuse each trial's accepted generated code instead of calling the planner.",
    )
    p.add_argument(
        "--scoring",
        choices=("llm_judge", "semantic_judge", "exact_match"),
        default="llm_judge",
        help=(
            "Scoring method. semantic_judge uses the BrowseComp-Plus "
            "semantic-equivalence protocol for any benchmark; exact_match "
            "is local and makes no judge-model API calls."
        ),
    )
    p.add_argument(
        "--max-tokens", type=int, default=None,
        help="Optional per-question total token budget (default: unlimited).",
    )
    p.add_argument("--max-llm-calls", type=int, default=None)
    p.add_argument("--max-tool-calls", type=int, default=None)
    p.add_argument("--max-jobs", type=int, default=128)
    p.add_argument("--max-parallel-jobs", type=int, default=4)
    p.add_argument(
        "--wall-time-seconds", type=float, default=None,
        help="Optional per-question wall-clock limit in seconds (default: unlimited).",
    )
    p.add_argument("--runs", type=int, default=1,
                   help="Number of independent runs over the question set. "
                        "Each lands in <out>/run_<i>/. Default: 1.")
    p.add_argument("--concurrency", type=int, default=1,
                   help="Per-run concurrent subprocess workers (default: 1). When --runs N, all N runs launch concurrently; total in-flight = runs × concurrency.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument(
        "--question-ids",
        default=None,
        help="Comma-separated question IDs to run, preserved in the given order.",
    )
    p.add_argument("--seed", type=int, default=None,
                   help="Shuffle seed (default: 42). For --runs N, each "
                        "run uses seed + run_index - 1.")
    p.add_argument("--no-shuffle", action="store_true",
                   help="Disable question shuffling (run in dataset order)")
    p.add_argument("--answer-type", default=None)
    p.add_argument("--category", default=None)
    p.add_argument("--out", default=None,
                   help="Output directory (default: benchmarks/results/<benchmark>/<ts>)")
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

    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = f"benchmarks/results/{args.benchmark}/{ts}"

    base_out = Path(args.out).resolve()
    base_out.mkdir(parents=True, exist_ok=True)
    (base_out / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str),
        encoding="utf-8",
    )

    base_seed = args.seed if args.seed is not None else 42

    if args.runs == 1:
        asyncio.run(run_eval(args, out_dir=base_out, seed=base_seed))
    else:
        # All N runs concurrently. Each run has its own ``asyncio.Semaphore``
        # of ``args.concurrency`` — total subprocesses in-flight =
        # args.runs × args.concurrency.
        async def _run_all() -> None:
            coros = []
            for run_idx in range(1, args.runs + 1):
                run_out = base_out / f"run_{run_idx}"
                run_out.mkdir(parents=True, exist_ok=True)
                seed = base_seed + run_idx - 1
                logger.info("queue run %d / %d  (seed=%d, out=%s)",
                            run_idx, args.runs, seed, run_out)
                coros.append(run_eval(args, out_dir=run_out, seed=seed))
            await asyncio.gather(*coros)

        asyncio.run(_run_all())


if __name__ == "__main__":
    main()
