"""LLM-as-judge dispatch for benchmark answer grading.

Each family lives in its own module under ``benchmarks/judges/``:
    - hle.py            → verify_hle (also covers superchem_text)
    - browsecomp.py     → verify_browsecomp, verify_browsecomp_zh
    - frontier_science.py → verify_frontier_science(_olympiad), score_frontier_science
    - deepsearchqa.py   → verify_deepsearchqa
    - xbench.py         → verify_xbench
    - widesearch.py     → score_widesearch (returns (verdict, f1_by_item))

Shared infra (LLM client, retry helpers, types) is in _common.py.

Endpoint comes from ``JUDGE_API_KEY`` / ``JUDGE_BASE_URL`` env (with
``OPENAI_*`` fallback). Most judge models and reasoning effort are selected
inside each family module. BrowseComp-Plus semantic scoring can override its
model with ``SEMANTIC_JUDGE_MODEL``.

Adding a new benchmark's judge:
    1. Create benchmarks/judges/<family>.py implementing
       ``async def verify_<name>(question, target, predicted) -> Verdict``.
    2. Register it in JUDGE_REGISTRY below.
"""

from __future__ import annotations

from benchmarks.judges._common import JudgeFn, Verdict
from benchmarks.judges.browsecomp import (
    JUDGE_PROMPT_BC_ZH,
    JUDGE_PROMPT_BROWSECOMP_OFFICIAL,
    verify_browsecomp,
    verify_browsecomp_zh,
)
from benchmarks.judges.browsecomp_plus import (
    BROWSECOMP_PLUS_GRADER_TEMPLATE,
    get_semantic_judge_model,
    judge_semantic_equivalence,
    parse_judge_response,
    verdict_from_judgement,
    verify_semantic_equivalence,
)
from benchmarks.judges.deepsearchqa import (
    JUDGE_PROMPT_DEEPSEARCHQA,
    score_deepsearchqa,
    verify_deepsearchqa,
)
from benchmarks.judges.frontier_science import (
    JUDGE_PROMPT_FS,
    JUDGE_PROMPT_FS_OLYMPIAD,
    score_frontier_science,
    verify_frontier_science,
    verify_frontier_science_olympiad,
)
from benchmarks.judges.hle import (
    JUDGE_PROMPT_HLE,
    verify_hle,
)
from benchmarks.judges.widesearch import score_widesearch
from benchmarks.judges.xbench import verify_xbench

JUDGE_REGISTRY: dict[str, JudgeFn] = {
    "browsecomp":                  verify_browsecomp,
    "browsecomp_200":              verify_browsecomp,
    "browsecomp_zh":               verify_browsecomp_zh,
    "xbench_dr_202510":            verify_xbench,
    "frontier_science_research":   verify_frontier_science,
    "frontier_science_olympiad":   verify_frontier_science_olympiad,
    "hle_text":                    verify_hle,
    "deepsearchqa":                verify_deepsearchqa,
    # SUPERChem-Text is single-letter MCQ (A-J); HLE's extract-final-answer
    # + correct: yes|no schema handles it cleanly — judge pulls the chosen
    # letter from the model's natural-language reply and matches it against
    # the gold letter in ground_truth.
    "superchem_text":              verify_hle,
    # WideSearch uses score_widesearch (structural row/item F1 + judge LLM
    # for column alignment). It returns (verdict, f1_by_item) so it is
    # dispatched via a separate score_* path in the runner, NOT through
    # JUDGE_REGISTRY's standard Verdict-returning dispatch.
}


def has_judge(benchmark: str) -> bool:
    """True if ``benchmark`` has an LLM judge configured (rubric or binary)."""
    return benchmark in JUDGE_REGISTRY or benchmark in SCORE_REGISTRY


async def verify_answer(
    benchmark: str, question: str, target: str, predicted: str,
) -> Verdict:
    """Dispatch to the registered judge. Raises ``KeyError`` if unregistered."""
    fn = JUDGE_REGISTRY.get(benchmark)
    if fn is None:
        raise KeyError(
            f"No LLM judge registered for benchmark {benchmark!r}. "
            f"Registered: {sorted(JUDGE_REGISTRY)}"
        )
    return await fn(question, target, predicted)


# Benchmarks whose judge surfaces a raw rubric value (0-10 for FS,
# F1 0-1 for WideSearch). Stored in result.json's ``rubric_score`` field.
# WideSearch has no binary verify_* sibling — it only exists here, so
# SCORE_REGISTRY is also the dispatch root for it.
SCORE_REGISTRY: dict[str, JudgeFn] = {
    "deepsearchqa":              score_deepsearchqa,
    "frontier_science_research": score_frontier_science,
    "widesearch":                score_widesearch,
}


async def score_answer(
    benchmark: str, question: str, target: str, predicted: str,
) -> tuple[Verdict, float | None]:
    """Verdict plus optional raw rubric score (``None`` for binary judges)."""
    verdict, score, _details = await score_answer_detailed(
        benchmark, question, target, predicted,
    )
    return verdict, score


async def score_answer_detailed(
    benchmark: str, question: str, target: str, predicted: str,
) -> tuple[Verdict, float | None, dict | None]:
    """Score an answer and preserve structured judge output when available."""
    if benchmark in {"browsecomp", "browsecomp_200"}:
        details = await judge_semantic_equivalence(question, target, predicted)
        return verdict_from_judgement(details), None, details
    if (fn := SCORE_REGISTRY.get(benchmark)) is not None:
        verdict, score = await fn(question, target, predicted)
        return verdict, score, None
    verdict = await verify_answer(benchmark, question, target, predicted)
    return verdict, None, None


__all__ = [
    "BROWSECOMP_PLUS_GRADER_TEMPLATE",
    "JUDGE_PROMPT_BC_ZH",
    "JUDGE_PROMPT_BROWSECOMP_OFFICIAL",
    "JUDGE_PROMPT_DEEPSEARCHQA",
    "JUDGE_PROMPT_FS",
    "JUDGE_PROMPT_FS_OLYMPIAD",
    "JUDGE_PROMPT_HLE",
    "JUDGE_REGISTRY",
    "JudgeFn",
    "SCORE_REGISTRY",
    "Verdict",
    "has_judge",
    "get_semantic_judge_model",
    "score_answer",
    "score_answer_detailed",
    "score_deepsearchqa",
    "score_frontier_science",
    "score_widesearch",
    "verify_answer",
    "verify_browsecomp",
    "verify_browsecomp_zh",
    "verify_deepsearchqa",
    "verify_frontier_science",
    "verify_frontier_science_olympiad",
    "verify_hle",
    "verify_xbench",
    "judge_semantic_equivalence",
    "parse_judge_response",
    "verdict_from_judgement",
    "verify_semantic_equivalence",
]
