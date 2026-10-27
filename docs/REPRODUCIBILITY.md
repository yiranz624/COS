# Reproducing COS experiments

## Environment

The supplied implementation requires Python 3.12 or later. Docker is the primary entry point and installs the versions recorded in `uv.lock`.

```bash
./scripts/docker-rolechange-v6.sh build
./scripts/docker-rolechange-v6.sh unit
./scripts/docker-rolechange-v6.sh lint
```

Alternatively, in a separate environment with Python 3.12 and uv:

```bash
uv sync --frozen --extra dev
uv run python -m benchmarks.runner.run_subprocess --help
```

Docker is needed for the Docker-based development commands. Linux or WSL2 is recommended for the provided Bash scripts and host-network Compose configuration. Live tools also require access to their configured services.

## Model and tool configuration

Copy `.env.example` to `.env`; configure the model endpoint, provider credentials, planner model, judge, and any search/reader/sandbox services used by the selected workflow. Never commit `.env` or real keys. Model names in the supplied profiles are configuration examples, not a guarantee of current provider availability.

Profiles under `orchestrations/profiles/` assign models to worker roles. Changing `OPENAI_MODEL` alone does not rewrite those assignments. Inspect and adjust the profile when changing providers. `PLANNER_MODEL` separately controls the planner. Keep the provider API compatible with the configured client.

## Data preparation

Datasets are excluded from this release. Acquire them from their original distributors under their own licenses and access conditions. The registry in `benchmarks/families/` selects each directory and schema; `benchmarks/core/registry.py` implements loading.

For `--benchmark browsecomp`, prepare:

```text
benchmarks/datasets/BrowseComp/standardized_data.jsonl
```

Each line is a JSON object. Standard families use `task_id`, `task_question`, and `ground_truth`; optional metadata includes `answer_type`, `category`, `image_path`, and `file_name`. A synthetic schema example:

```json
{"task_id":"demo-1","task_question":"What is 2 + 2?","ground_truth":"4","answer_type":"exactMatch","category":"demo"}
```

This example illustrates parsing only. It is not an official benchmark item, and a trivial arithmetic item does not validate research-agent performance. Keep attachments in the location expected by the selected family/workflow.

## Single run

```bash
./scripts/docker-rolechange-v6.sh run env ROLECHANGE_V6_MODE=slice \
  python -m benchmarks.runner.run_subprocess \
  --benchmark browsecomp --strategy rolechange_self_evolving_v6 \
  --orchestration-profile deepseek_v4_flash \
  --limit 1 --no-shuffle --concurrency 1 --out results/COS-smoke
```

Use `--question-ids` for an explicit comma-separated split. The runner also exposes `--runs`, `--seed`, `--max-tokens`, `--max-llm-calls`, `--max-tool-calls`, `--max-jobs`, `--max-parallel-jobs`, and `--wall-time-seconds`. Confirm defaults with `--help`; record the actual flags used for every reported result.

Compose maps container `results/`, `logs/`, and `data/` to host directories scoped by the local username. For example, container `results/COS-smoke` appears under host `results/<user>/COS-smoke`.

## Ablations and run artifacts

The supported V6 modes are `off`, `flow_only`, and `slice` (default). The planner preserves exact fact-ID prompts in all three modes; `off` is not an exact V5 prompt ablation.

```bash
./scripts/docker-rolechange-v6.sh run python scripts/run_v6_ablation.py \
  --out results/COS-ablation
```

The script has a predefined question list and budgets. Review `scripts/run_v6_ablation.py --help` and the source before a full run. Use a new output directory. `--summarize-only` summarizes existing artifacts without starting new model calls.

Depending on the run, diagnostic outputs include:

- `generated/fact_flow/events.jsonl`: fact-flow events.
- `generated/evolution_round_XX/fact_flow/snapshot.json`: round flow snapshot.
- `terminal_review/diagnostic_slice.json`: selected facts, jobs, and trace references.
- `terminal_review/diagnostic_bundle.json`: bounded diagnostic summary.
- `terminal_review/trace_usage.json`: retrieval and expansion statistics.
- `generated/evolution_manifest.json`: run version and mode.

These paths are relative to the corresponding attempt/round directories. A dependency slice prioritizes evidence retrieval; it is not proof of causality. Recording failures or missing events limit coverage.

Offline diagnostic replay writes to a separate directory:

```bash
./scripts/docker-rolechange-v6.sh run python scripts/replay_v6_diagnostics.py \
  --source results/existing-run --out results/replayed-diagnostics
```

Replay can recover only information available in saved artifacts. It does not measure the accuracy or recovery rate of a new online reflector.

## Reporting results

For a comparable experiment, record the commit, dataset version and split, question IDs/order, all provider/model IDs, prompts and profiles, seeds, run count, concurrency, budgets, tool availability, judge configuration, failure/retry policy, and aggregation procedure. Network and model nondeterminism can affect results even with the same seed.

This code release does not assert independent reproduction of all paper results. Full online evaluations require external credentials and data. Local syntax checks do not establish runtime or numerical equivalence.

## Source references

The retained Chinese implementation references describe the original harness and earlier orchestration design:

- [AgentHarness code guide](AGENTHARNESS_CODE_GUIDE_ZH.md)
- [Code-as-orchestration reference](CODE_AS_ORCHESTRATION_REFERENCE_V1_ZH.md)
- [Developer guide](DEVELOPER_GUIDE.md)

Some historical names in these documents differ from the COS repository name. The current source and runner arguments take precedence for execution.
