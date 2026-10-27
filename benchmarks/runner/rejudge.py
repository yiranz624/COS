#!/usr/bin/env python3
"""Re-judge existing trial result.json files in place.

Backfills the ``rubric_score`` field (and refreshes ``is_correct`` /
``reward``) for trials that were scored before the field was tracked.
The agent's prediction is reused from result.json — only the judge is
re-run, so this is much cheaper than a full restart.

Idempotent: trials that already carry the ``rubric_score`` key are
skipped unless ``--force``.

Usage:
    uv run python -m benchmarks.runner.rejudge <out_dir> [--concurrency N]

``<out_dir>`` is the directory you passed as ``--out`` to run_subprocess.
Single-run and multi-run layouts are both supported.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.core.registry import load_questions  # noqa: E402
from benchmarks.judges import (  # noqa: E402
    judge_semantic_equivalence,
    score_answer_detailed,
    verdict_from_judgement,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
)
logger = logging.getLogger("rejudge")


def _collect_results(out_dir: Path) -> list[Path]:
    """Find result.json files under both single-run and multi-run layouts."""
    return sorted(out_dir.glob("**/trials/*/result.json"))


async def _rejudge_one(
    path: Path,
    benchmark: str,
    questions: dict[str, object],
    sem: asyncio.Semaphore,
    force: bool,
    scoring: str,
) -> str:
    r = json.loads(path.read_text(encoding="utf-8"))
    if not force and "rubric_score" in r:
        return "skip"
    pred = (r.get("predicted_answer") or "").strip()
    if not pred:
        r["rubric_score"] = None
        _atomic_write(path, r)
        return "empty"
    # result.json's ``question`` is truncated to 200 chars for display; reload
    # the full question + ground truth from the dataset so the judge sees the
    # same payload that the original run did.
    q = questions.get(r["question_id"])
    if q is None:
        logger.error("[%s] question_id not in dataset — skipping", path.parent.name)
        return "fail"

    async with sem:
        try:
            if scoring == "semantic_judge":
                details = await judge_semantic_equivalence(
                    q.question, q.ground_truth, pred,
                )
                verdict = verdict_from_judgement(details)
                score = None
            else:
                verdict, score, details = await score_answer_detailed(
                    benchmark, q.question, q.ground_truth, pred,
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] judge failed: %s", path.parent.name, exc)
            return "fail"

    r["rubric_score"] = score
    r["is_correct"] = verdict == "CORRECT"
    r["reward"] = 1 if verdict == "CORRECT" else 0
    r["judge_method"] = "semantic_judge" if scoring == "semantic_judge" else "llm"
    r["scoring_mode"] = scoring
    r["judge_details"] = details
    r["scoring_error"] = (
        None if verdict != "NOT_ATTEMPTED"
        else f"{scoring} judge NOT_ATTEMPTED after retries"
    )
    _atomic_write(path, r)
    return "ok"


def _atomic_write(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


async def _amain(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).resolve()
    cfg = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    benchmark = cfg["benchmark"]
    results = _collect_results(out_dir)
    questions = {q.id: q for q in load_questions(benchmark)}
    logger.info(
        "Rejudging %d trial(s) under %s (benchmark=%s, concurrency=%d, force=%s)",
        len(results), out_dir, benchmark, args.concurrency, args.force,
    )

    sem = asyncio.Semaphore(args.concurrency)
    outcomes = await asyncio.gather(
        *(
            _rejudge_one(
                p, benchmark, questions, sem, args.force, args.scoring,
            )
            for p in results
        ),
    )
    counts = {k: outcomes.count(k) for k in ("ok", "skip", "empty", "fail")}
    logger.info(
        "Done: %(ok)d updated, %(skip)d skipped, %(empty)d empty-pred, %(fail)d failed",
        counts,
    )
    return 0 if counts["fail"] == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out_dir", help="The --out dir passed to run_subprocess")
    p.add_argument("--concurrency", type=int, default=10,
                   help="Parallel judge calls (default: 10)")
    p.add_argument("--force", action="store_true",
                   help="Re-judge even if rubric_score key already present")
    p.add_argument(
        "--scoring",
        choices=("llm_judge", "semantic_judge"),
        default="llm_judge",
        help="Judge protocol to apply (default: benchmark llm_judge).",
    )
    return asyncio.run(_amain(p.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
