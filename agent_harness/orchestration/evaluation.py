"""Quality/cost aggregation shared by benchmark strategies."""

from __future__ import annotations

from agent_harness.orchestration.contracts import (
    BudgetUsage,
    EvaluationSummary,
)


def summarize_results(
    results: list[dict],
    *,
    strategy: str,
) -> EvaluationSummary:
    total = len(results)
    scores = [
        float(result["rubric_score"])
        for result in results if result.get("rubric_score") is not None
    ]
    usage = BudgetUsage(
        tokens=sum(int(result.get("usage", {}).get("tokens", 0)) for result in results),
        llm_calls=sum(
            int(result.get("usage", {}).get("llm_calls", 0)) for result in results
        ),
        tool_calls=sum(
            int(result.get("usage", {}).get("tool_calls", 0)) for result in results
        ),
        jobs=sum(int(result.get("usage", {}).get("jobs", 0)) for result in results),
        duration_seconds=sum(
            float(result.get("duration_seconds", 0)) for result in results
        ),
    )
    generated_runs = [
        result for result in results if isinstance(result.get("generation"), dict)
    ]
    first_pass = repaired = generated_success = fallback = 0
    planner_tokens = planner_calls = 0
    for result in generated_runs:
        generation = result["generation"]
        usage_item = generation.get("planner_usage") or {}
        planner_tokens += int(usage_item.get("tokens", 0))
        planner_calls += int(usage_item.get("llm_calls", 0))
        source_path = generation.get("source_path")
        if source_path:
            calls = int(usage_item.get("llm_calls", 0))
            first_pass += calls == 1
            repaired += calls > 1
            generated_success += not bool(generation.get("fallback_reason"))
        fallback += bool(generation.get("fallback_reason"))

    return EvaluationSummary(
        strategy=strategy,
        total=total,
        attempted=sum(bool(result.get("predicted_answer")) for result in results),
        correct=sum(bool(result.get("is_correct")) for result in results),
        accuracy=(
            sum(bool(result.get("is_correct")) for result in results) / total
            if total else 0.0
        ),
        mean_rubric_score=sum(scores) / len(scores) if scores else None,
        usage=usage,
        failures=sum(bool(result.get("error")) for result in results),
        timeouts=sum(
            "timeout" in str(result.get("error", "")).lower() for result in results
        ),
        budget_violations=sum(
            len(result.get("budget_violations", [])) for result in results
        ),
        patches_proposed=sum(
            int(result.get("patches_proposed", 0)) for result in results
        ),
        patches_applied=sum(
            int(result.get("patches_applied", 0)) for result in results
        ),
        metadata={
            "strict_exact_match_count": sum(
                bool(result.get("strict_exact_match")) for result in results
            ),
            "unordered_team_match_count": sum(
                bool(result.get("unordered_team_match")) for result in results
            ),
            "generation": {
                "runs": len(generated_runs),
                "first_pass_valid": first_pass,
                "repaired_valid": repaired,
                "generated_source_available": generated_success,
                "fallbacks": fallback,
                "planner_tokens": planner_tokens,
                "planner_llm_calls": planner_calls,
            },
        },
    )
