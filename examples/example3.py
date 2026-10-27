#!/usr/bin/env python3
"""Runtime-backed try2 orchestration example.

This file keeps the try2 policy visible while using AgentHarness' structured
Runtime API. The real mode uses the built-in solver, synthesizer and
verifier roles. --fake is an offline smoke test that exercises the same
call/parallel/map/runtime path without model or network access.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from typing import Any

from dotenv import load_dotenv

from agent_harness.orchestration import (
    AgentHarnessWorkerExecutor,
    RunLimits,
    ScriptRuntime,
)
from agent_harness.orchestration.testing import FakeWorkerExecutor
from benchmarks.core.kernel_adapter import BenchmarkSession


AGENT_COUNT = 3
RESEARCH_TURNS = 8
SYNTHESIS_TURNS = 4
VERIFIER_TURNS = 10
REPAIR_ROUNDS = 2


def _clip(value: Any, limit: int = 7_000) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit] + "\n[truncated]"


def _extract_final_answer(value: Any) -> str:
    """Extract a marked answer while retaining raw text for auditability."""
    text = str(value or "").strip()
    for pattern in (
        r"(?im)^\s*FINAL[_ ]ANSWER\s*:\s*(.+?)\s*$",
        r"(?im)^\s*ANSWER\s*:\s*(.+?)\s*$",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip().strip(" \"'")
    return ""


def _answer_or_raw(value: Any) -> str:
    text = str(value or "").strip()
    return _extract_final_answer(text) or text


def _peer_block(answers: list[str], omit: int | None = None) -> str:
    blocks = []
    for index, answer in enumerate(answers):
        if omit is not None and index == omit:
            continue
        blocks.append(f"Peer candidate {index + 1}:\n{_clip(answer, 4_500)}")
    return "\n\n".join(blocks) or "(no peer candidates)"


async def _research_round(
    ctx: Any,
    prompts: list[str],
    round_index: int,
) -> list[str]:
    """Run one debate round through Runtime-managed solver jobs."""

    async def run_one(item: tuple[int, str]) -> str:
        index, prompt = item
        value = await ctx.call(
            "solver",
            prompt,
            key=f"try2:research:{round_index}:{index}",
            max_turns=RESEARCH_TURNS,
            timeout=900,
        )
        return str(value or "").strip()

    # ctx.map creates structured runtime branches; do not replace this with
    # unmanaged asyncio tasks in a Planner-generated orchestration script.
    return await ctx.map(run_one, enumerate(prompts))


def _round_one_prompts(task: str) -> list[str]:
    return [
        (
            "Solve this open-web question independently. Build a clue table, "
            "search each hop, and fetch sources for decisive facts. Do not "
            "assume another agent's answer. End with a concise candidate and "
            "source URLs; mark unresolved links explicitly.\n\nQUESTION:\n"
            f"{task}"
        )
        for _ in range(AGENT_COUNT)
    ]


def _round_two_prompts(task: str, first_round: list[str]) -> list[str]:
    prompts = []
    for index in range(AGENT_COUNT):
        prompts.append(
            "Review the peer research below, but do not accept it merely by "
            "consensus. Re-check every date, entity, relationship and numeric "
            "constraint. Use targeted web search/fetch to resolve the largest "
            "gap. Return your corrected candidate, not just a critique, and "
            "include supporting URLs.\n\nQUESTION:\n"
            f"{task}\n\nYOUR FIRST-ROUND RESEARCH:\n"
            f"{_clip(first_round[index])}\n\nPEER RESEARCH:\n"
            f"{_peer_block(first_round, omit=index)}"
        )
    return prompts


def _evidence_block(round_one: list[str], round_two: list[str]) -> str:
    blocks = []
    for index, (first, second) in enumerate(zip(round_one, round_two), 1):
        blocks.append(
            f"=== AGENT {index} / ROUND 1 ===\n{_clip(first)}\n\n"
            f"=== AGENT {index} / ROUND 2 ===\n{_clip(second)}"
        )
    return "\n\n".join(blocks)


async def _synthesize(
    ctx: Any,
    task: str,
    candidates: list[str],
    evidence: str,
    key: str,
) -> str:
    value = await ctx.call(
        "synthesizer",
        (
            "Combine independent research into one candidate answer. Candidate "
            "number and repetition are not evidence. Check every constraint "
            "in the original question and do not fill missing links with a "
            "guess. If a candidate is best supported, return it; otherwise "
            "return the most defensible concrete hypothesis and state what is "
            "unverified. Use exactly:\n"
            "FINAL_ANSWER: <concise answer>\n"
            "RATIONALE: <brief clue-chain rationale>\n\n"
            f"ORIGINAL QUESTION:\n{task}\n\n"
            f"CANDIDATES:\n{_peer_block(candidates)}\n\n"
            f"EVIDENCE:\n{_clip(evidence, 18_000)}"
        ),
        key=key,
        max_turns=SYNTHESIS_TURNS,
        timeout=600,
    )
    return str(value or "").strip()


async def _verify(
    ctx: Any,
    task: str,
    candidate: str,
    evidence: str,
    round_index: int,
) -> dict[str, Any]:
    try:
        value = await ctx.call(
            "verifier",
            (
                "Independently fact-check this candidate. Check every clue in "
                "the original question, not just whether the candidate sounds "
                "plausible. Use web_search/web_fetch for missing or conflicting "
                "facts. Accept only when the answer and the complete clue chain "
                "are supported. If rejected, give targeted next searches in "
                "critique. Return the role's required JSON verdict.\n\n"
                f"ORIGINAL QUESTION:\n{task}\n\n"
                f"CANDIDATE:\n{_clip(candidate, 8_000)}\n\n"
                f"RESEARCH RECORD:\n{_clip(evidence, 18_000)}"
            ),
            key=f"try2:verifier:{round_index}",
            max_turns=VERIFIER_TURNS,
            timeout=900,
        )
    except Exception as exc:
        return {"accepted": False, "critique": f"verifier runtime failure: {exc}"}
    if not isinstance(value, dict):
        return {"accepted": False, "critique": "verifier returned a non-object result"}
    return {
        "accepted": bool(value.get("accepted")),
        "critique": str(value.get("critique", "")),
    }


async def orchestrate_try2(ctx: Any) -> dict[str, Any]:
    """The try2 policy expressed only through the Runtime API."""
    task = str(ctx.task)
    round_one = await _research_round(ctx, _round_one_prompts(task), 1)
    round_two = await _research_round(ctx, _round_two_prompts(task, round_one), 2)
    evidence = _evidence_block(round_one, round_two)
    candidate = await _synthesize(
        ctx,
        task,
        round_two,
        evidence,
        key="try2:synthesis:initial",
    )

    verdict = await _verify(ctx, task, candidate, evidence, 0)
    repair_log: list[str] = []
    for repair_index in range(REPAIR_ROUNDS):
        if verdict["accepted"]:
            break
        repair = await ctx.call(
            "solver",
            (
                "Perform targeted repair research. Address every objection in "
                "the verifier critique, search for primary or independent "
                "sources, and return only newly established facts and URLs. "
                "Do not simply repeat the current candidate.\n\n"
                f"QUESTION:\n{task}\n\nCURRENT CANDIDATE:\n{_clip(candidate)}\n\n"
                f"VERIFIER CRITIQUE:\n{_clip(verdict['critique'], 8_000)}"
            ),
            key=f"try2:repair:research:{repair_index}",
            max_turns=RESEARCH_TURNS,
            timeout=900,
        )
        repair_text = str(repair or "").strip()
        repair_log.append(repair_text)
        evidence = (
            f"{evidence}\n\n=== TARGETED REPAIR {repair_index + 1} ===\n"
            f"{repair_text}"
        )
        candidate = await _synthesize(
            ctx,
            task,
            round_two + repair_log,
            evidence,
            key=f"try2:synthesis:repair:{repair_index}",
        )
        verdict = await _verify(
            ctx,
            task,
            candidate,
            evidence,
            repair_index + 1,
        )

    final_answer = _answer_or_raw(candidate) or "ANSWER_NOT_FOUND"
    return {
        "final_answer": final_answer,
        "candidate_raw": candidate,
        "verdict": verdict,
        "selection_method": (
            "runtime_verifier" if verdict["accepted"] else "best_revised_candidate"
        ),
        "round_one": round_one,
        "round_two": round_two,
        "repair_count": len(repair_log),
    }


def _fake_handler(request: Any) -> Any:
    """Deterministic handler used only to validate Runtime API wiring."""
    if request.role == "verifier":
        return {"accepted": True, "critique": ""}
    if request.role == "synthesizer":
        return "FINAL_ANSWER: fake smoke-test candidate\nRATIONALE: runtime wiring"
    return f"fake {request.role} result for {request.key}"


def _limits() -> RunLimits:
    return RunLimits(
        max_parallel_jobs=AGENT_COUNT,
        max_total_jobs=20,
        max_depth=4,
        wall_time_seconds=3_600,
    )


async def _run_fake(task: str, run_id: str) -> Any:
    executor = FakeWorkerExecutor(
        default_handler=_fake_handler,
        allowed_roles={"solver", "synthesizer", "verifier"},
    )
    runtime = ScriptRuntime(executor, limits=_limits())
    return await runtime.run(orchestrate_try2, task=task, run_id=run_id)


async def _run_real(task: str, run_id: str) -> Any:
    async with BenchmarkSession():
        runtime = ScriptRuntime(
            AgentHarnessWorkerExecutor.from_runtime_services(),
            limits=_limits(),
        )
        return await runtime.run(orchestrate_try2, task=task, run_id=run_id)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True, help="Question for the try2 planner.")
    parser.add_argument("--run-id", default="try2-runtime-demo")
    parser.add_argument(
        "--fake",
        action="store_true",
        help="Run an offline deterministic Runtime API smoke test.",
    )
    args = parser.parse_args()
    load_dotenv()
    run = await (
        _run_fake(args.question, args.run_id)
        if args.fake
        else _run_real(args.question, args.run_id)
    )
    print(json.dumps({
        "run_id": run.run_id,
        "value": run.value,
        "jobs": [
            {
                "key": job.key,
                "role": job.role,
                "status": job.status.value,
                "elapsed_seconds": round(job.elapsed_seconds, 3),
            }
            for job in run.jobs
        ],
        "event_count": len(run.events),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
