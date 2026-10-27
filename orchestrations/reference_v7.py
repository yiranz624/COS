"""Static-subtask Solver-Verifier-Synthesizer fallback workflow."""

from __future__ import annotations

from typing import Any

from agent_harness.orchestration.context import OrchestrationContext

MANIFEST = {
    "script_id": "reference_v7", "version": 7, "entrypoint": "solve",
    "input_schema": "string", "output_schema": "string",
    "compatible_runtime": ">=1.1",
    "roles": ("solver", "verifier", "synthesizer"),
}

_SUBTASKS = (
    "Discover concrete candidates using the rarest conjunctions of constraints.",
    "Independently identify candidates from structured records and primary sources.",
    "Resolve likely evidence gaps and test every required condition.",
)
_SUBTASK_IDS = (
    "candidate-discovery",
    "constraint-verification",
    "evidence-gap-resolution",
)


async def solve(ctx: OrchestrationContext) -> str:
    task = ctx.task
    restored = ctx.load_checkpoint("reference_v7_state") or {}
    ctx.emit("script_started", subtask_count=len(_SUBTASKS))
    if restored.get("final_answer"):
        return str(restored["final_answer"])

    async def investigate(index: int) -> str:
        try:
            return str(await ctx.call(
                "solver",
                "Solve the complete task through this assigned subtask. Build a "
                "constraint table, preserve source URLs, reject candidates that "
                "fail any required condition, and mark missing evidence.\n\n"
                f"Task:\n{task}\n\nSubtask:\n{_SUBTASKS[index]}",
                key=f"reference_v7:solver:{index}:{_SUBTASK_IDS[index]}",
                max_turns=8,
            ))
        except Exception as exc:
            return f"Insufficient information: solver unavailable: {type(exc).__name__}: {exc}"

    candidates = await ctx.parallel(
        investigate(0), investigate(1), investigate(2),
    )
    reviewed: list[dict[str, Any]] = []
    for index in range(3):
        if ctx.remaining_budget().remaining.get("jobs", 0) < 1:
            break
        try:
            verdict = await ctx.call(
                "verifier",
                "Independently check every material requirement. Accept only a "
                "fully supported candidate; otherwise provide actionable critique."
                f"\n\nTask:\n{task}\n\nCandidate:\n{candidates[index]}",
                key=f"reference_v7:verifier:{index}:{_SUBTASK_IDS[index]}",
                max_turns=3,
            )
        except Exception as exc:
            verdict = {"accepted": False, "critique": (
                f"verification unavailable: {type(exc).__name__}: {exc}"
            )}
        reviewed.append({"candidate": candidates[index], "verdict": verdict})
        await ctx.checkpoint("reference_v7_state", {"reviewed": reviewed})

    accepted = [item for item in reviewed
                if isinstance(item.get("verdict"), dict)
                and item["verdict"].get("accepted", False)]
    evidence = accepted or reviewed
    try:
        answer = str(await ctx.call(
            "synthesizer",
            "Return the concise final answer requested by the task. Use only "
            "supplied evidence, preserve qualifications, and do not invent facts."
            f"\n\nTask:\n{task}\n\nReviewed work:\n{evidence}",
            key="reference_v7:synthesis", max_turns=3,
            metadata={"allowed_tools": []},
        )).strip()
    except Exception:
        answer = str(evidence[0]["candidate"]) if evidence else "Insufficient information"
    answer = answer or "Insufficient information"
    await ctx.checkpoint(
        "reference_v7_state", {"reviewed": reviewed, "final_answer": answer},
    )
    ctx.emit("script_completed", accepted_count=len(accepted))
    return answer
