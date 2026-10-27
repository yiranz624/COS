#!/usr/bin/env python3
"""Matched first10 runs with identical model configuration and budgets across modes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import dotenv_values


def summarize(root):
    rows = []
    for mode in ("off", "flow_only", "slice"):
        for trial in sorted((root / mode / "trials").glob("*")):
            if not trial.is_dir():
                continue
            row = {"mode": mode, "question": trial.name, "reflection_usage": []}
            result = trial / "result.json"
            if result.exists():
                value = json.loads(result.read_text())
                row["result"] = {k: value[k] for k in ("status", "is_correct", "score", "correct", "error", "wall_time_seconds", "duration_seconds", "scoring_error") if k in value}
            manifest = trial / "generated/evolution_manifest.json"
            if manifest.exists():
                value = json.loads(manifest.read_text())
                row["round_statuses"] = [r.get("status") for r in value.get("rounds", [])]
                row["usage"] = value.get("usage", value.get("total_usage", {}))
            for review in sorted((trial / "generated").glob("evolution_round_*/terminal_review")):
                u = review / "trace_usage.json"
                if not u.exists():
                    continue
                item = json.loads(u.read_text())
                item["round"] = review.parent.name
                bundle = review / "diagnostic_bundle.json"
                if bundle.exists():
                    item["flow_summary"] = json.loads(bundle.read_text()).get("fact_flow_summary", {})
                slice_file = review / "diagnostic_slice.json"
                if slice_file.exists():
                    sliced = json.loads(slice_file.read_text())
                    item["slice_trace_count"] = len(sliced["trace_refs"])
                    item["full_round_trace_count"] = sliced["full_round_trace_count"]
                row["reflection_usage"].append(item)
            rows.append(row)
    (root / "comparison.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--question-ids", default="1,6,13,14,39,43,51,54,61,64")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize(output)
        return
    if any((output / mode).exists() for mode in ("off", "flow_only", "slice")):
        parser.error("Use a fresh output directory to preserve prior experiments")
    env = os.environ.copy()
    if args.env_file:
        env.update({k: v for k, v in dotenv_values(args.env_file).items() if v is not None})
    dataset = root / "benchmarks/datasets/BrowseComp/standardized_data.jsonl"
    config = {"question_ids": args.question_ids, "profile": "deepseek_v4_flash", "concurrency": args.concurrency,
              "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
              "models": {k: env.get(k, "") for k in ("OPENAI_MODEL", "PLANNER_MODEL", "LLM_PROVIDER")},
              "modes": ["off", "flow_only", "slice"], "budget": "runner defaults, identical across modes"}
    (output / "experiment.json").write_text(json.dumps(config, indent=2))
    for mode in config["modes"]:
        target = output / mode
        target.mkdir()
        command = [sys.executable, "-u", "-m", "benchmarks.runner.run_subprocess", "--benchmark", "browsecomp",
                   "--strategy", "rolechange_self_evolving_v6", "--question-ids", args.question_ids,
                   "--no-shuffle", "--concurrency", str(args.concurrency), "--orchestration-profile", "deepseek_v4_flash",
                   "--out", str(target)]
        (target / "command.json").write_text(json.dumps(command, indent=2))
        print(f"Starting {mode}", flush=True)
        with (target / "runner.log").open("w") as log:
            completed = subprocess.run(command, cwd=root, env={**env, "ROLECHANGE_V6_MODE": mode},
                                       stdout=log, stderr=subprocess.STDOUT, check=False)
        (target / "exit_code.txt").write_text(str(completed.returncode))
        summarize(output)
        print(f"Finished {mode}: exit={completed.returncode}", flush=True)
        if completed.returncode:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
