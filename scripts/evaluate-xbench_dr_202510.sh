#!/usr/bin/env bash
set -euo pipefail

./scripts/docker.sh run python -m benchmarks.runner.run_subprocess \
  --benchmark xbench_dr_202510 \
  --pipeline react_base \
  --profile default \
  --runs 1 \
  --concurrency 50 \
  --out ./results/$(date +%F)_xbench_dr_202510


# ./scripts/docker.sh run python -m benchmarks.runner.check_progress ./results/$(date +%F)_xbench_dr_202510
